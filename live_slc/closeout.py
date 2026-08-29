"""
Calendar-driven closeout guardian.

Timing is calendar-derived, not a hardcoded clock time (rev. 6 amendment_004):
begins at the official session close minus 5 minutes, using
utils/market_calendar.session_for() - the same function
utils/slc_signals.py already relies on for early-close handling - so an
early-close session (e.g. the day after Thanksgiving) is flattened
correctly without a special case. The Windows Task Scheduler trigger for
this stage is a fixed, deliberately wide window (12:55-4:05 PM ET, every
minute); should_begin_closeout() is what makes each invocation within that
window a no-op until the real, calendar-derived moment arrives.

Runs under live_slc.guardrails.assert_closeout_preconditions() - always
reachable regardless of deployment status, the daily-loss halt, or Tier-2
signal-fidelity drift (rev. 6 Step 2/3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from live_slc.execution import CloseOutcome, cancel_then_confirm_then_close, client_order_id
from utils.market_calendar import session_for

CLOSEOUT_LEAD_MINUTES = 5


def should_begin_closeout(now_utc: pd.Timestamp, day: date) -> bool:
    session = session_for(day)
    if session is None:
        return False
    close = pd.Timestamp(session["close"])
    return now_utc >= close - timedelta(minutes=CLOSEOUT_LEAD_MINUTES)


@dataclass
class CloseoutResult:
    # (position, actual_close_fill_price) - never just `position`, so the
    # caller can record REAL P&L instead of the position's theoretical
    # target price (found via review: a closeout flatten exits at
    # whatever the market is, not the frozen strategy's target).
    flattened: list = field(default_factory=list)
    skipped_ambiguous: list = field(default_factory=list)
    failed: list = field(default_factory=list)
    ambiguous_close: list = field(default_factory=list)  # exit side went ambiguous (rev. 11 point 7) - never assumed flat


def run_closeout(client, positions: list, *, side_for) -> CloseoutResult:
    """positions: SLC-owned SlcPosition-like records (must expose .symbol,
    .level_id, .confirmation_time, .direction, .qty, .status,
    .protective_order_id). Ambiguous-ownership positions are skipped
    entirely - never force-flattened, since some of that broker-side
    quantity may not be SLC's (rev. 6 Step 7). side_for(direction) ->
    OrderSide, injected so this module doesn't need to import the SDK enum
    just to pick BUY/SELL for the closing leg."""
    result = CloseoutResult()
    for position in positions:
        if getattr(position, "status", None) == "ambiguous":
            result.skipped_ambiguous.append(position)
            continue
        closing_side = side_for("sell" if position.direction == "long" else "buy")
        exit_client_id = client_order_id(
            position.symbol, getattr(position, "level_id", None) or position.symbol,
            getattr(position, "confirmation_time", None) or position.opened_at, "exit",
        )
        close_result = cancel_then_confirm_then_close(
            client, position.symbol,
            getattr(position, "protective_order_id", None),
            position.qty, closing_side,
            close_client_order_id=exit_client_id,
        )
        if close_result.outcome == CloseOutcome.CONFIRMED_FLAT:
            result.flattened.append((position, close_result.fill_price))
        elif close_result.outcome in (CloseOutcome.AMBIGUOUS_CANCEL, CloseOutcome.AMBIGUOUS_SUBMISSION):
            # Never inferred flat from a submission response alone - an
            # ambiguous exit is its own state, resolved by a later
            # invocation of this same guardian, not assumed either way.
            result.ambiguous_close.append(position)
        else:
            result.failed.append(position)
    return result
