"""Price construction helpers shared by stock and crypto strategies."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP


def _decimal(value: float) -> Decimal:
    return Decimal(str(value))


def _to_tick(value: Decimal, tick: Decimal, rounding: str) -> Decimal:
    units = (value / tick).to_integral_value(rounding=rounding)
    return units * tick


def rounded_long_bracket(
    *,
    close: float,
    atr: float,
    stop_atr: float,
    target_atr: float,
    min_rr: float,
    tick_size: float,
) -> dict[str, float]:
    """
    Build rounded long-entry prices while preserving minimum reward/risk.

    Stops round down and targets round up to valid price ticks. The target is
    raised when necessary so tick rounding can never turn a valid setup into a
    strategy BUY that the risk node immediately rejects.
    """
    tick = _decimal(tick_size)
    if tick <= 0:
        raise ValueError("tick_size must be positive")

    entry = _to_tick(_decimal(close), tick, ROUND_HALF_UP)
    atr_d = _decimal(atr)
    if entry <= 0 or atr_d <= 0:
        raise ValueError("close and ATR must be positive")

    stop = _to_tick(
        entry - _decimal(stop_atr) * atr_d,
        tick,
        ROUND_FLOOR,
    )
    if stop <= 0 or stop >= entry:
        raise ValueError(
            f"invalid rounded stop: entry={entry} stop={stop}"
        )

    risk = entry - stop
    atr_target = entry + _decimal(target_atr) * atr_d
    rr_target = entry + _decimal(min_rr) * risk
    target = _to_tick(max(atr_target, rr_target), tick, ROUND_CEILING)
    if target <= entry:
        raise ValueError(
            f"invalid rounded target: entry={entry} target={target}"
        )

    rr = (target - entry) / risk
    return {
        "entry": float(entry),
        "stop": float(stop),
        "target": float(target),
        "rr": round(float(rr), 8),
    }
