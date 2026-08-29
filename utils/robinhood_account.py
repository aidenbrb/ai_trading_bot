"""
Robinhood account state reader + shared login helper.

Robinhood has NO paper-trading sandbox - ROBINHOOD_ENABLED routes real stock
orders to a real account. Falls back to a disconnected/zero-balance state if
credentials are missing or login fails (never a pretend balance, unlike the
Alpaca paper fallback defaults in utils/account.py).

Login uses robin_stocks with TOTP-based 2FA (via pyotp) so it can run
unattended from Task Scheduler once a session is cached. The session pickle
is pinned to ai_trading_bot/.robinhood_session/ (git-ignored) rather than
robin_stocks' library-default location, so its storage location is explicit.

nodes/execution_node.py and nodes/monitor_node.py import `login()` directly
to get the robin_stocks module for placing/cancelling orders and reading
positions - robin_stocks is function-based (module-level session state), not
client-object-based like alpaca-py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings

_SESSION_DIR = Path(__file__).parent.parent / ".robinhood_session"
_PICKLE_NAME = "robinhood"

_DEFAULTS = {
    "equity":         0.0,
    "cash":           0.0,
    "open_positions": 0,
    "trades_today":   0,
    "daily_pnl":      0.0,
    "daily_pnl_pct":  0.0,
    "positions":      [],
    "connected":      False,
}

_logged_in = False


def login():
    """
    Log in to Robinhood (idempotent within a process - reuses the cached
    session after the first successful call). Returns the
    robin_stocks.robinhood module so callers can use its functions directly.

    Raises RuntimeError if credentials are missing, or whatever robin_stocks
    raises on a failed login - callers should catch and fall back to safe
    defaults (see get_robinhood_account_state below for the pattern).
    """
    global _logged_in
    import robin_stocks.robinhood as r

    if _logged_in:
        return r

    if not settings.ROBINHOOD_USERNAME or not settings.ROBINHOOD_PASSWORD:
        raise RuntimeError("ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD not set")

    mfa_code = None
    if settings.ROBINHOOD_TOTP_SECRET:
        import pyotp
        mfa_code = pyotp.TOTP(settings.ROBINHOOD_TOTP_SECRET).now()

    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    r.login(
        username=settings.ROBINHOOD_USERNAME,
        password=settings.ROBINHOOD_PASSWORD,
        mfa_code=mfa_code,
        store_session=True,
        expiresIn=60 * 60 * 24 * 60,   # 60 days - minimizes re-auth on scheduled runs
        pickle_path=str(_SESSION_DIR),
        pickle_name=_PICKLE_NAME,
    )
    _logged_in = True
    return r


def get_robinhood_account_state() -> dict:
    """
    Returns:
        equity          - total account value
        cash            - cash available
        open_positions  - number of currently held (non-zero qty) symbols
        trades_today    - number of buy orders filled today
        daily_pnl       - unrealized + realized P&L for today
        daily_pnl_pct   - daily P&L as fraction of equity
        positions       - list of {symbol, qty, entry_price, current_price, unrealized_pnl}
        connected       - True if Robinhood responded
    """
    if not settings.ROBINHOOD_USERNAME or not settings.ROBINHOOD_PASSWORD:
        print("  [Robinhood] No credentials - using disconnected defaults ($0, 0 positions)")
        return dict(_DEFAULTS)

    try:
        return _fetch_from_robinhood()
    except Exception as exc:
        print(f"  [Robinhood] Connection failed: {exc}")
        print("  [Robinhood] Falling back to disconnected defaults ...")
        return dict(_DEFAULTS)


def _fetch_from_robinhood() -> dict:
    r = login()

    account_profile   = r.profiles.load_account_profile()   or {}
    portfolio_profile = r.profiles.load_portfolio_profile()  or {}

    equity = float(portfolio_profile.get("equity") or 0.0)
    cash   = float(account_profile.get("cash") or 0.0)

    holdings = r.account.build_holdings() or {}
    positions = [
        {
            "symbol":         symbol,
            "qty":            float(h.get("quantity", 0) or 0),
            "entry_price":    float(h.get("average_buy_price", 0) or 0),
            "current_price":  float(h.get("price", 0) or 0),
            "unrealized_pnl": float(h.get("equity_change", 0) or 0),
        }
        for symbol, h in holdings.items()
        if float(h.get("quantity", 0) or 0) > 0
    ]

    # Buy orders filled today (best-effort - falls back to 0 on any shape
    # mismatch rather than failing the whole account read).
    trades_today = 0
    try:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for o in (r.orders.get_all_stock_orders() or []):
            if (
                o.get("side") == "buy"
                and o.get("state") == "filled"
                and str(o.get("created_at", "")).startswith(today_str)
            ):
                trades_today += 1
    except Exception:
        pass

    # Daily P&L from broker equity vs previous-close equity - same approach as
    # utils/account.py's Alpaca last_equity comparison, so the daily-loss
    # circuit breaker in risk_node.py works identically regardless of broker.
    try:
        prev_equity = float(portfolio_profile.get("equity_previous_close") or 0.0)
    except (TypeError, ValueError):
        prev_equity = 0.0

    if prev_equity > 0:
        daily_pnl     = equity - prev_equity
        daily_pnl_pct = daily_pnl / prev_equity
    else:
        daily_pnl     = sum(p["unrealized_pnl"] for p in positions)
        daily_pnl_pct = daily_pnl / equity if equity else 0.0

    return {
        "equity":         equity,
        "cash":           cash,
        "open_positions": len(positions),
        "trades_today":   trades_today,
        "daily_pnl":      daily_pnl,
        "daily_pnl_pct":  daily_pnl_pct,
        "positions":      positions,
        "connected":      True,
    }
