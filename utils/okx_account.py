"""
OKX account state reader.

Reads balance and open positions from OKX (demo or live).
Returns the same dict shape as utils/account.py so callers can treat
both brokers uniformly.

OKX_DEMO=true  uses the paper/demo environment (simulated funds).
OKX_DEMO=false uses the live environment - real money.
"""
from __future__ import annotations

from config.settings import settings

_DEFAULTS = {
    "equity":         10_000.0,   # default demo USDT balance
    "cash":           10_000.0,
    "open_positions": 0,
    "trades_today":   0,
    "daily_pnl":      0.0,
    "daily_pnl_pct":  0.0,
    "positions":      [],
    "connected":      False,
}


def get_okx_account_state() -> dict:
    """
    Returns:
        equity          - USDT total balance (funding + trading)
        cash            - available USDT
        open_positions  - number of open spot positions
        trades_today    - filled orders today (approximate)
        daily_pnl       - unrealized P&L in USDT
        daily_pnl_pct   - daily P&L as fraction of equity
        positions       - list of {symbol, qty, entry_price, current_price, unrealized_pnl}
        connected       - True if OKX responded
    """
    if not settings.OKX_API_KEY or not settings.OKX_SECRET_KEY or not settings.OKX_PASSPHRASE:
        print("  [OKX] No credentials - using defaults ($10k demo balance)")
        return dict(_DEFAULTS)

    try:
        return _fetch_from_okx()
    except Exception as exc:
        print(f"  [OKX] Connection failed: {exc}")
        print("  [OKX] Falling back to defaults ...")
        return dict(_DEFAULTS)


def _fetch_from_okx() -> dict:
    from okx.Account import AccountAPI

    flag   = "1" if settings.OKX_DEMO else "0"
    client = AccountAPI(
        api_key=settings.OKX_API_KEY,
        api_secret_key=settings.OKX_SECRET_KEY,
        passphrase=settings.OKX_PASSPHRASE,
        flag=flag,
        debug=False,
    )

    # -- Total USDT balance ----------------------------------------------------
    bal_resp = client.get_account_balance(ccy="USDT")
    if bal_resp["code"] != "0":
        raise RuntimeError(f"OKX balance error: {bal_resp['msg']}")

    details = bal_resp["data"][0]["details"] if bal_resp["data"] else []
    usdt    = next((d for d in details if d["ccy"] == "USDT"), {})
    equity  = float(usdt.get("eq", 0))
    cash    = float(usdt.get("availEq", equity))

    # -- Open spot positions ---------------------------------------------------
    pos_resp = client.get_positions(instType="SPOT")
    positions = []
    unrealized_total = 0.0

    if pos_resp["code"] == "0":
        for p in pos_resp.get("data", []):
            qty        = float(p.get("pos", 0))
            avg_px     = float(p.get("avgPx", 0))
            last_px    = float(p.get("last", avg_px))
            unreal_pnl = float(p.get("upl", 0))
            unrealized_total += unreal_pnl
            if qty > 0:
                positions.append({
                    "symbol":         p.get("instId", ""),
                    "qty":            qty,
                    "entry_price":    avg_px,
                    "current_price":  last_px,
                    "unrealized_pnl": unreal_pnl,
                })

    daily_pnl_pct = unrealized_total / equity if equity else 0.0

    return {
        "equity":         equity,
        "cash":           cash,
        "open_positions": len(positions),
        "trades_today":   0,   # OKX order history requires separate call; omitted for speed
        "daily_pnl":      unrealized_total,
        "daily_pnl_pct":  daily_pnl_pct,
        "positions":      positions,
        "connected":      True,
    }
