"""
Deterministic portfolio-wide ranking and capacity-gated selection.

A cycle must process every symbol and collect every confirmation before
submitting anything - never submit symbol-by-symbol as bars arrive, since
that would make outcomes depend on API response order or iteration order.
The ranking tie-break here is not a reworded equivalent - it's the exact
tuple already implemented in the frozen backtest engine, verified directly
at backtest/run_slc_backtest.py:238-241:

    sorted(day_signals, key=lambda s: (
        s.entry_time, -s.level_active_time.value, -s.impulse_atr, s.symbol
    ))

i.e. entry/confirmation time ascending, then most-recently-activated level
first, then greatest impulse/ATR first, then symbol alphabetically.
"""
from __future__ import annotations

from typing import Callable

from live_slc.reducer import Confirmation

CapacityCheckFn = Callable[[Confirmation, list], tuple[bool, str]]


def rank_confirmations(confirmations: list[Confirmation]) -> list[Confirmation]:
    return sorted(
        confirmations,
        key=lambda c: (
            c.entry_time, -c.level_active_time.value, -c.impulse_atr, c.symbol,
        ),
    )


def select_within_capacity(
    ranked: list[Confirmation],
    *,
    remaining_daily_entries: int,
    capacity_check_fn: CapacityCheckFn,
) -> tuple[list[Confirmation], list[tuple[Confirmation, str]]]:
    """Walks the ranked list in order (never any other order), admitting a
    confirmation only while daily-entry slots remain and capacity_check_fn
    approves it. `admitted` accumulates so capacity_check_fn can account
    for confirmations already admitted earlier in this same ranked walk
    (e.g. two confirmations in one cycle both consuming exposure).
    Returns (admitted, [(skipped_confirmation, reason), ...])."""
    admitted: list[Confirmation] = []
    skipped: list[tuple[Confirmation, str]] = []
    for confirmation in ranked:
        if remaining_daily_entries <= 0:
            skipped.append((confirmation, "skipped_capacity_daily_trades"))
            continue
        ok, reason = capacity_check_fn(confirmation, admitted)
        if ok:
            admitted.append(confirmation)
            remaining_daily_entries -= 1
        else:
            skipped.append((confirmation, reason))
    return admitted, skipped
