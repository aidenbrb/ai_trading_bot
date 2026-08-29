"""
Account-wide sizing and exposure gates.

Deliberately takes an already-fetched AccountSnapshot rather than calling
the broker itself - keeps this module pure and independently testable;
execution.py is responsible for building a fresh snapshot immediately
before every submission (rev. 6 Step 7's "re-read broker state
immediately before every submission").

No second Alpaca account (rev. 6 Step 7): every check here is account-wide
- it reads ALL positions and pending opening orders, from any source, not
just live_slc's own, so SLC's sizing can never double-count capital the
swing/day/crypto pipelines already committed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import floor

SLC_RISK_PER_TRADE = 0.0025      # 0.25% of equity
SLC_MAX_POSITIONS = 5
SLC_MAX_NEW_TRADES_PER_DAY = 2
SLC_MAX_POSITION_PCT = 0.20      # 20% concentration cap
SLC_DAILY_LOSS_LIMIT = 0.02      # 2% account-wide daily equity-change halt


@dataclass(frozen=True)
class OrderInfo:
    symbol: str
    classification: str  # "entry" | "exit_leg" | "unclassifiable"
    notional: float = 0.0
    is_slc: bool = False


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    qty: float  # signed: positive long, negative short
    market_value: float
    is_slc: bool = False


@dataclass(frozen=True)
class AccountSnapshot:
    account_id: str
    equity: float
    cash: float
    non_marginable_buying_power: float
    start_of_day_equity: float
    daily_realized_pnl: float
    daily_unrealized_pnl: float
    positions: list = field(default_factory=list)          # list[PositionInfo]
    today_orders: list = field(default_factory=list)        # list[OrderInfo] - filled + pending, today only


def classify_order(order: dict) -> str:
    """order: a raw broker order record (dict-like). Any order that can't
    be confidently identified as an opening entry or a bracket exit leg is
    conservatively counted as an entry (rev. 6 Step 7)."""
    client_order_id = order.get("client_order_id") or ""
    if client_order_id.startswith("slc-"):
        # live_slc's own deterministic ID encodes the leg explicitly.
        if (
            client_order_id.endswith("-protect") or "-protect-" in client_order_id
            or client_order_id.endswith("-exit") or "-exit-" in client_order_id
        ):
            return "exit_leg"
        return "entry"
    if order.get("parent_id") or order.get("parent_order_id"):
        return "exit_leg"
    order_class = (order.get("order_class") or "").lower()
    order_type = (order.get("type") or "").lower()
    if order_class in ("oco", "oto") and order_type in ("stop", "stop_limit", "limit"):
        # A resting stop/limit under an OCO/OTO umbrella that isn't
        # self-identified as SLC's own entry leg is very likely a
        # protective exit leg - but this is a heuristic on foreign orders,
        # not a certainty, so anything not clearly one or the other still
        # falls through to "unclassifiable" below rather than guessing here.
        legs = order.get("legs")
        if legs:
            return "entry"  # a parent order carrying its own child legs is the opening order
        return "exit_leg" if order.get("position_intent", "").endswith("_close") else "unclassifiable"
    if order.get("position_intent", "").endswith("_open"):
        return "entry"
    if order.get("position_intent", "").endswith("_close"):
        return "exit_leg"
    return "unclassifiable"


def account_wide_entries_today(snapshot: AccountSnapshot) -> int:
    """Opening parent orders count once; exit legs are ignored;
    unclassifiable fills count conservatively as entries."""
    return sum(1 for o in snapshot.today_orders if o.classification in ("entry", "unclassifiable"))


def remaining_daily_entry_slots(snapshot: AccountSnapshot) -> int:
    return max(0, SLC_MAX_NEW_TRADES_PER_DAY - account_wide_entries_today(snapshot))


def gross_exposure(snapshot: AccountSnapshot) -> float:
    return sum(abs(p.market_value) for p in snapshot.positions)


def pending_opening_notional(snapshot: AccountSnapshot) -> float:
    return sum(o.notional for o in snapshot.today_orders if o.classification in ("entry", "unclassifiable"))


def available_buying_power(snapshot: AccountSnapshot) -> float:
    """Floored at zero (rev. 6): the unfloored formula can go negative if
    other strategies' exposure exceeds equity, which would corrupt
    downstream sizing math if used directly."""
    return max(0.0, min(
        snapshot.cash,
        snapshot.non_marginable_buying_power,
        snapshot.equity - gross_exposure(snapshot) - pending_opening_notional(snapshot),
    ))


def daily_loss_breached(snapshot: AccountSnapshot) -> bool:
    """Account-wide, realized + unrealized, stricter than relying on a
    broker-reported 'realized loss' figure alone."""
    if snapshot.start_of_day_equity <= 0:
        return True  # fail closed
    change = snapshot.daily_realized_pnl + snapshot.daily_unrealized_pnl
    return (change / snapshot.start_of_day_equity) <= -SLC_DAILY_LOSS_LIMIT


def open_position_count(snapshot: AccountSnapshot) -> int:
    return sum(1 for p in snapshot.positions if p.qty != 0)


def size_order(
    snapshot: AccountSnapshot,
    *,
    reference_price: float,
    stop_price: float,
    hypothetical_additional_notional: float = 0.0,
) -> int:
    """Whole-share quantity from the fixed-fractional 0.25%-of-equity
    formula, capped by the 20% concentration limit and available buying
    power (net of any confirmations already admitted earlier in the same
    cycle's ranked walk, via hypothetical_additional_notional)."""
    risk_per_share = abs(reference_price - stop_price)
    if risk_per_share <= 0 or reference_price <= 0:
        return 0
    by_risk = snapshot.equity * SLC_RISK_PER_TRADE / risk_per_share
    by_concentration = snapshot.equity * SLC_MAX_POSITION_PCT / reference_price
    available = max(0.0, available_buying_power(snapshot) - hypothetical_additional_notional)
    by_cash = available / reference_price
    return max(0, floor(min(by_risk, by_concentration, by_cash)))


# The 5 conditions that block ALL new SLC entries system-wide (any
# symbol), never closeout or management of other unambiguous positions
# (rev. 11 Step 3). The first four are SlcOrder.status values; the fifth
# is an SlcPosition.status value.
_BLOCKING_ORDER_STATUSES = (
    "submission_intent_pending", "submitted_unresolved", "ambiguous_submission",
    "existing_broker_order_detected",
)


def system_wide_entry_block_reasons(session) -> list[str]:
    """Queries the live_slc DB directly via an already-open `session`
    (kept as a plain parameter, not this module's own DB access, so
    risk.py's other functions stay pure/independently testable) for any
    of the 6 blocking conditions. Non-empty return blocks every new-entry
    candidate this cycle; never affects closeout or management of other,
    unambiguous positions."""
    from sqlmodel import select as _select

    from live_slc.models import SlcAuditEvent, SlcOrder, SlcPosition

    reasons: list[str] = []
    if session.exec(_select(SlcPosition).where(SlcPosition.status == "protected_degraded")).first():
        reasons.append("protected_degraded_position_exists")
    for status in _BLOCKING_ORDER_STATUSES:
        if session.exec(_select(SlcOrder).where(SlcOrder.status == status)).first():
            reasons.append(f"unresolved_{status}")
    if session.exec(_select(SlcPosition).where(SlcPosition.status == "ambiguous")).first():
        reasons.append("unresolved_quantity_mismatch")
    # An orphan broker position that couldn't be confidently adopted
    # (run_slc_live._discover_orphan_slc_positions) was previously only
    # ever recorded as an SlcAuditEvent - a historical log entry nothing
    # re-reads, never an active gate (found via review: a later cycle
    # could resume entries while that broker position stayed genuinely
    # unresolved). resolved defaults False and nothing in this codebase
    # ever flips it True automatically - clearing it is a deliberate
    # manual/operator action, matching "never guessed at, flagged for
    # manual review" everywhere else this pattern appears.
    if session.exec(
        _select(SlcAuditEvent).where(
            SlcAuditEvent.event_type == "orphan_broker_position_unresolved",
            SlcAuditEvent.resolved == False,  # noqa: E712
        )
    ).first():
        reasons.append("unresolved_orphan_broker_position")
    return reasons


def check_new_entry_capacity(snapshot: AccountSnapshot, symbol: str, admitted_this_cycle: list) -> tuple[bool, str]:
    """The capacity_check_fn passed to live_slc.ranking.select_within_capacity."""
    if daily_loss_breached(snapshot):
        return False, "skipped_daily_loss_halt"
    if open_position_count(snapshot) + len(admitted_this_cycle) >= SLC_MAX_POSITIONS:
        return False, "skipped_capacity_max_positions"
    if any(p.symbol == symbol for p in snapshot.positions):
        return False, "skipped_symbol_already_has_position"
    if any(o.symbol == symbol for o in snapshot.today_orders if o.classification in ("entry", "unclassifiable")):
        return False, "skipped_symbol_already_has_order_today"
    return True, ""
