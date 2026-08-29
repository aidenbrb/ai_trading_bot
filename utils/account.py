"""
Alpaca paper account state reader.

Reads live account data from Alpaca paper trading API so the risk node
has accurate equity, open position count, and today's trade count.

Falls back to safe defaults if credentials are missing (offline/test mode).
paper=True is hardcoded - cannot connect to a live broker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Optional

from config.settings import settings

_DEFAULTS = {
    "equity":                      100_000.0,
    "cash":                        100_000.0,
    "buying_power":                0.0,
    "non_marginable_buying_power": 0.0,
    "available_cash":              0.0,
    "available_cash_valid":        False,
    "available_cash_reason":       "not connected to Alpaca",
    "open_positions":              0,
    "trades_today":                0,
    "daily_pnl":                   0.0,
    "daily_pnl_pct":               0.0,
    "positions":                   [],
    "connected":                   False,
}


@dataclass(frozen=True)
class AvailableCash:
    value: float                  # always >= 0.0, safe to use directly for sizing
    valid: bool                   # False if the underlying data could not be trusted
    reason: Optional[str] = None  # explanation when value == 0.0


def _available_cash(cash, non_marginable_buying_power) -> AvailableCash:
    """Conservative, fail-closed sizing budget - never a fallback to a MORE
    permissive number. Validates BOTH arguments independently of whatever
    the caller already believes about them: either being missing,
    non-numeric, NaN, or infinite returns (0.0, valid=False). A valid but
    negative value on either side floors to (0.0, valid=True) - a negative
    budget is meaningless, but the data itself wasn't untrustworthy.
    Otherwise never exceeds either input (so margin buying power can never
    leak into sizing). Never raises, so a malformed field only zeroes this
    one value - it must not trip get_account_state()'s outer fallback to
    the permissive defaults.
    """
    try:
        cash_value = float(cash)
        nonmargin_value = float(non_marginable_buying_power)
    except (TypeError, ValueError, OverflowError):
        return AvailableCash(0.0, False, "cash or non_marginable_buying_power is missing or non-numeric")
    if not math.isfinite(cash_value) or not math.isfinite(nonmargin_value):
        return AvailableCash(0.0, False, "cash or non_marginable_buying_power is NaN or infinite")
    capped = max(0.0, min(cash_value, nonmargin_value))
    if capped == 0.0:
        return AvailableCash(0.0, True, "cash or non-margin buying power is zero or negative")
    return AvailableCash(capped, True, None)


def _lenient_float(value, default: float = 0.0) -> float:
    """Best-effort parse for audit/display-only fields. Never raises, never
    gates anything - used only for the buying_power/non_marginable_buying_power
    AUDIT copies, neither of which affects sizing."""
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return default if not math.isfinite(parsed) else parsed


def _canonical_alpaca_symbol(symbol: str) -> str:
    """Translate Alpaca crypto pair variants (SOLUSD/SOL/USD) to SOL-USD."""
    upper = str(symbol or "").upper()
    compact = upper.replace("/", "").replace("-", "")
    from config.universe import CRYPTO
    for crypto_symbol in CRYPTO:
        if compact == crypto_symbol.replace("-", ""):
            return crypto_symbol
    return upper


def get_account_state() -> dict:
    """
    Returns:
        equity                      - total account value
        cash                        - raw Alpaca ledger cash balance (display only)
        buying_power                - margin buying power (audit only, NEVER used for sizing -
                                       can be several times cash on a margin account)
        non_marginable_buying_power - Alpaca's margin-free available capital (audit copy)
        available_cash              - conservative sizing budget: min(cash,
                                       non_marginable_buying_power), floored at 0.0.
                                       This is what risk_node.py sizes positions against.
        available_cash_valid        - False if available_cash is 0.0 because the
                                       underlying data was untrustworthy (missing/
                                       non-numeric/NaN/infinite), not because the
                                       account genuinely has no capital
        available_cash_reason       - explanation when available_cash == 0.0
        open_positions              - number of currently open positions
        trades_today                - number of orders filled today
        daily_pnl                   - unrealized + realized P&L for today
        daily_pnl_pct               - daily P&L as fraction of equity
        positions                   - list of {symbol, qty, entry_price, current_price, unrealized_pnl}
        connected                   - True if Alpaca responded
    """
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        print("  [Account] No Alpaca credentials - using defaults ($100k, 0 positions)")
        return dict(_DEFAULTS)

    try:
        return _fetch_from_alpaca()
    except Exception as exc:
        print(f"  [Account] Alpaca connection failed: {exc}")
        print(f"  [Account] Falling back to defaults ...")
        return dict(_DEFAULTS)


def _fetch_from_alpaca() -> dict:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import OrderStatus, OrderSide

    client = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=True,   # HARDCODED - never live
    )

    account   = client.get_account()
    equity    = float(account.equity)

    # cash must not be converted with a bare float() before _available_cash()
    # sees it - raising here would send the ENTIRE account read through the
    # outer except-fallback below (blanking out real positions/open_positions/
    # connected, not just the cash figure). The raw value goes to BOTH the
    # lenient display parser and the strict sizing validator, independently.
    cash_raw = getattr(account, "cash", None)
    cash = _lenient_float(cash_raw)                                    # display only, never raises

    buying_power_raw = getattr(account, "buying_power", None)          # audit only
    non_marginable_raw = getattr(account, "non_marginable_buying_power", None)
    sizing = _available_cash(cash_raw, non_marginable_raw)             # sizing-critical - RAW inputs

    # Open positions
    raw_positions = client.get_all_positions()
    positions = [
        {
            "symbol":        _canonical_alpaca_symbol(p.symbol),
            # Crypto positions are fractional; stock quantities still arrive
            # as whole-number strings and safely convert to float.
            "qty":           float(p.qty),
            "entry_price":   float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price else 0.0,
            "unrealized_pnl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
        }
        for p in raw_positions
    ]

    # Buy orders filled today (use 'closed' status - Alpaca only accepts open/closed/all)
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    closed_today = client.get_orders(
        GetOrdersRequest(
            status="closed",
            after=today_start,
            side=OrderSide.BUY,
        )
    )
    trades_today = sum(
        1 for o in (closed_today or [])
        if str(getattr(o, "status", "")).lower() == "filled"
    )

    # Daily P&L from broker equity vs previous-close equity. This captures BOTH
    # realized and unrealized P&L for the day, so a position closed at a loss
    # earlier today still counts toward the daily loss limit. (The old version
    # summed only open-position unrealized P&L and silently ignored realized
    # losses, which could let the daily-loss circuit breaker be bypassed.)
    try:
        last_equity = float(account.last_equity)
    except (TypeError, ValueError, AttributeError):
        last_equity = 0.0

    if last_equity > 0:
        daily_pnl     = equity - last_equity
        daily_pnl_pct = daily_pnl / last_equity
    else:
        # Fallback when last_equity is unavailable: unrealized across positions
        daily_pnl     = sum(p["unrealized_pnl"] for p in positions)
        daily_pnl_pct = daily_pnl / equity if equity else 0.0

    return {
        "equity":                      equity,
        "cash":                        cash,
        "buying_power":                _lenient_float(buying_power_raw),
        "non_marginable_buying_power": _lenient_float(non_marginable_raw),
        "available_cash":              sizing.value,
        "available_cash_valid":        sizing.valid,
        "available_cash_reason":       sizing.reason,
        "open_positions":              len(positions),
        "trades_today":                trades_today,
        "daily_pnl":                   daily_pnl,
        "daily_pnl_pct":               daily_pnl_pct,
        "positions":                   positions,
        "connected":                   True,
    }
