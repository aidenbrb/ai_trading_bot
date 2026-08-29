"""
Atomic bracket submission and post-fill recovery for live_slc.

Deliberately separate from nodes/execution_node.py (never imported from,
never touched): that module is long-only and its bracket entry uses a
LIMIT order at a precomputed price. live_slc needs both directions, and
its entry price is unknowable before the fill (amendment_004), so entry
is a MARKET order under an atomic OrderClass.BRACKET - the stop leg is
fixed at submission (computed from the level, independent of entry); the
take_profit leg carries a provisional price at submission and is PATCHed
to the exact 2R value once the real fill price is known.

Every function here takes an explicit `client` parameter rather than
constructing one internally, so tests can inject a fake client - only
`get_alpaca_client()` ever builds a real one, and only that one call site
needs `paper=True` verified.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from enum import Enum
from typing import Callable, Optional

import pandas as pd

from live_slc.settings import live_slc_settings

PARTIAL_FILL_DEADLINE_SECONDS = 15.0
QUOTE_FRESHNESS_DEADLINE_SECONDS = 5.0
EASY_TO_BORROW_FALLBACK_EXPIRES = date(2026, 9, 22)


def enum_value(value) -> str:
    """Return an Alpaca enum's wire value, or normalize a string mock.

    alpaca-py enums are ``str`` subclasses, but ``str(OrderStatus.FILLED)``
    is ``"OrderStatus.FILLED"`` rather than ``"filled"``.  Production
    order objects therefore must be unwrapped through ``.value`` before
    any comparison.  The fallback preserves the plain-string fakes used
    by the test suite and unknown values remain fail-closed at callers.
    """
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def order_status_value(order) -> str:
    """Canonical lowercase status for a broker order-like object."""
    return enum_value(getattr(order, "status", ""))


def is_confirmed_not_found(exc: Exception) -> bool:
    """True only for an Alpaca APIError backed by an actual HTTP 404.

    A 404 from ``get_open_position`` is Alpaca's documented/observed
    representation of a flat symbol (``position does not exist``).  A
    timeout, transport failure, or any other HTTP status proves nothing
    and must stay ambiguous.  Centralizing the check keeps position and
    order reconciliation on the same fail-closed SDK boundary.
    """
    try:
        from alpaca.common.exceptions import APIError
    except Exception:  # noqa: BLE001
        return False
    return isinstance(exc, APIError) and getattr(exc, "status_code", None) == 404

# -- Decimal-based tick precision (amendment_005) ----------------------------
# Alpaca's documented price-precision rule: $0.01 at/above $1.00, $0.0001
# below $1.00. Decimal throughout, never binary float - see
# to_decimal()'s docstring for why Decimal(str(x)) is mandatory.

_TICK_HIGH = Decimal("0.01")
_TICK_LOW = Decimal("0.0001")
_ONE_DOLLAR = Decimal("1.00")


def to_decimal(value) -> Decimal:
    """`Decimal(str(value))`, never `Decimal(value)` directly - the latter
    imports the exact binary floating-point representation's rounding
    noise (e.g. Decimal(0.1) -> Decimal('0.100000000000000005551...'))
    before any tick normalization runs, silently defeating the entire
    point of using Decimal arithmetic."""
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def tick_size(price: Decimal) -> Decimal:
    return _TICK_HIGH if price >= _ONE_DOLLAR else _TICK_LOW


def round_stop(price, direction: str) -> Decimal:
    """Away from entry, never let rounding silently tighten protection:
    a long stop (below entry) rounds down; a short stop (above entry)
    rounds up."""
    d = to_decimal(price)
    tick = tick_size(d)
    rounding = ROUND_FLOOR if direction == "long" else ROUND_CEILING
    return d.quantize(tick, rounding=rounding)


def round_target(price, direction: str) -> Decimal:
    """Rounds to guarantee >=2R is achieved, never less (amendment_005's
    selected 'practical fidelity'): a long target rounds up; a short
    target rounds down."""
    d = to_decimal(price)
    tick = tick_size(d)
    rounding = ROUND_CEILING if direction == "long" else ROUND_FLOOR
    return d.quantize(tick, rounding=rounding)


def validate_rounded_bracket(fill, stop: Decimal, target: Decimal, direction: str) -> bool:
    """Applied at two different lifecycle points by the caller, with two
    different consequences (amendment_005 item 5) - this function only
    answers "is the geometry valid," never decides what to do about it:
    pre-submission (nothing to flatten, skip the trade) vs. post-fill
    (protected_degraded + guarded emergency exit)."""
    fill_d = to_decimal(fill)
    if stop <= 0 or target <= 0 or fill_d <= 0:
        return False
    if not (stop.is_finite() and target.is_finite() and fill_d.is_finite()):
        return False
    if direction == "long":
        return stop < fill_d < target
    return target < fill_d < stop


def effective_reward_risk(fill, stop: Decimal, target: Decimal, direction: str) -> Decimal:
    """From the actual submitted broker prices - always >= 2.0 under
    amendment_005's practical-fidelity rounding, never labeled "exactly
    2R" anywhere it's recorded."""
    fill_d = to_decimal(fill)
    if direction == "long":
        return (target - fill_d) / (fill_d - stop)
    return (fill_d - target) / (stop - fill_d)


def get_alpaca_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=live_slc_settings.ALPACA_API_KEY,
        secret_key=live_slc_settings.ALPACA_SECRET_KEY,
        paper=True,
    )


def client_order_id(symbol: str, level_id: str, confirmation_time, leg: str) -> str:
    """Deterministic from signal identity - a restarted cycle or
    reconciliation pass always looks up by this ID before ever retrying a
    submission."""
    key = f"{symbol}|{level_id}|{pd.Timestamp(confirmation_time).isoformat()}|{leg}"
    return f"slc-{hashlib.sha256(key.encode()).hexdigest()[:20]}-{leg}"


@dataclass(frozen=True)
class ShortabilityResult:
    ok: bool
    reason: Optional[str]


def check_shortability(get_asset_raw: Callable[[str], dict], symbol: str, *, today: Optional[date] = None) -> ShortabilityResult:
    """`get_asset_raw(symbol)` returns the RAW GET /v2/assets/{symbol}
    response as a dict - the installed alpaca-py (0.43.5, verified) has no
    `borrow_status` field on its typed Asset model, only `easy_to_borrow`,
    but the raw API response may add the field before the SDK model
    catches up, so the raw response is inspected directly rather than
    relying solely on the SDK's typed model."""
    today = today or date.today()
    try:
        asset = get_asset_raw(symbol)
    except Exception as exc:  # noqa: BLE001 - any lookup failure fails closed
        return ShortabilityResult(False, f"asset lookup failed: {exc}")
    if not asset.get("tradable", False):
        return ShortabilityResult(False, "not tradable")
    if not asset.get("shortable", False):
        return ShortabilityResult(False, "not shortable")
    borrow_status = asset.get("borrow_status")
    if borrow_status is not None:
        if borrow_status == "easy_to_borrow":
            return ShortabilityResult(True, None)
        return ShortabilityResult(False, f"borrow_status={borrow_status!r}")
    # borrow_status absent: fall back to easy_to_borrow, but only before the
    # field's announced removal date - reconfirm this exact date against
    # Alpaca's current notice before this ships, per rev. 6's own caveat.
    if today >= EASY_TO_BORROW_FALLBACK_EXPIRES:
        return ShortabilityResult(False, "borrow_status absent and easy_to_borrow fallback has expired")
    if asset.get("easy_to_borrow", False):
        return ShortabilityResult(True, None)
    return ShortabilityResult(False, "easy_to_borrow is False")


def _validate_long_bracket(entry: float, stop: float, target: float) -> None:
    if not (stop < entry < target):
        raise ValueError(f"invalid long bracket: stop={stop} entry={entry} target={target}")


def _validate_short_bracket(entry: float, stop: float, target: float) -> None:
    if not (target < entry < stop):
        raise ValueError(f"invalid short bracket: target={target} entry={entry} stop={stop}")


def retarget_after_fill(direction: str, actual_fill_price: float, level_stop: float) -> float:
    """Exact 2R from the actual fill; the stop stays anchored at the
    level's frozen price throughout, never recomputed here."""
    risk = abs(actual_fill_price - level_stop)
    return actual_fill_price + 2.0 * risk if direction == "long" else actual_fill_price - 2.0 * risk


@dataclass(frozen=True)
class DirectionalValidation:
    ok: bool
    reason: Optional[str]


def post_fill_directional_validation(direction: str, actual_fill: float, stop: float, recalculated_target: float) -> DirectionalValidation:
    """stop < fill < target for a long, target < fill < stop for a short -
    if the fill crossed the stop or the recalculated target isn't properly
    ordered, the bracket is genuinely invalid; use emergency
    cancel-and-flatten, never attempt the target-replacement PATCH."""
    if direction == "long":
        ok = stop < actual_fill < recalculated_target
    else:
        ok = recalculated_target < actual_fill < stop
    return DirectionalValidation(ok, None if ok else "post_fill_directional_validation_failed")


def post_fill_monetary_risk_check(qty: float, actual_fill: float, stop: float, equity: float, *, risk_pct: float) -> bool:
    """True if within budget. Caller flattens immediately on False."""
    return qty * abs(actual_fill - stop) <= risk_pct * equity


@dataclass(frozen=True)
class BracketSubmission:
    client_order_id: str
    alpaca_order_id: Optional[str]
    accepted: bool
    error: Optional[str] = None
    # True unless the failure is a DEFINITIVE rejection (a real API
    # response proving the order was never created) - a network/timeout
    # failure or an unclear server error means we genuinely don't know
    # whether the broker accepted the order before the connection
    # dropped, and must never be treated as "confirmed nothing happened"
    # (found via review: the original code treated every non-duplicate
    # exception as confirmed_rejected, which could hide a real accepted -
    # possibly filled - order behind a status that nothing ever
    # reconciles). Only meaningful when accepted=False.
    ambiguous: bool = False


# Alpaca validates a new-order request synchronously before ever creating
# the order - a 400/403/422 response is raised BEFORE any order exists,
# so it's the only class of failure treated as a definitive rejection.
# Anything else (a network failure with no response at all, or a
# server-side 5xx/429 whose relationship to order creation is unclear)
# stays ambiguous - mirrors reconcile_by_client_id()'s "only a genuine
# 404 counts as confirmed" conservatism, applied to the opposite side of
# the same idempotency problem.
_DEFINITIVE_REJECTION_STATUS_CODES = (400, 403, 422)


def _is_definitive_rejection(exc: Exception) -> bool:
    try:
        from alpaca.common.exceptions import APIError
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(exc, APIError):
        return False
    return getattr(exc, "status_code", None) in _DEFINITIVE_REJECTION_STATUS_CODES


def submit_bracket_entry(
    client, *, symbol: str, direction: str, qty: int, stop: float,
    provisional_target: float, order_client_id: str,
) -> BracketSubmission:
    """One atomic order: MARKET entry + fixed stop_loss + provisional
    take_profit, for either direction. Long uses BUY/BUY_TO_OPEN with the
    protective leg selling; short uses SELL/SELL_TO_OPEN with the
    protective leg buying - the mirror image, both submitted atomically."""
    from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest

    if direction == "long":
        side, intent = OrderSide.BUY, PositionIntent.BUY_TO_OPEN
    else:
        side, intent = OrderSide.SELL, PositionIntent.SELL_TO_OPEN

    # stop/target arrive as amendment_005 tick-rounded Decimal values (or a
    # plain float/int, converted the same way) - float() conversion happens
    # only here, at the literal SDK request-object construction boundary.
    request = MarketOrderRequest(
        symbol=symbol, qty=qty, side=side, position_intent=intent,
        time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
        stop_loss=StopLossRequest(stop_price=float(stop)),
        take_profit=TakeProfitRequest(limit_price=float(provisional_target)),
        client_order_id=order_client_id,
    )
    try:
        order = client.submit_order(request)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller, never silently swallowed
        if _is_duplicate_client_order_id_error(exc):
            # Evidence an order likely already exists under this id - the
            # caller must reconcile, never treat this as a confirmed
            # rejection (rev. 11 point 3).
            return BracketSubmission(order_client_id, None, False, f"duplicate_client_order_id: {exc}", ambiguous=True)
        if _is_definitive_rejection(exc):
            return BracketSubmission(order_client_id, None, False, str(exc), ambiguous=False)
        # No confirmed response reached us (network/timeout failure, or a
        # server-side error that doesn't prove the order was never
        # created) - genuinely don't know if the broker accepted this
        # order. Never confirmed_rejected.
        return BracketSubmission(order_client_id, None, False, str(exc), ambiguous=True)
    return BracketSubmission(order_client_id, str(order.id), True)


def _is_duplicate_client_order_id_error(exc: Exception) -> bool:
    """Best-effort detection on the exception text - not verified against
    a live account (no order may be submitted during this work), so this
    heuristic should be reconfirmed against Alpaca's actual error shape at
    the first real dry-run/paper session."""
    message = str(exc).lower()
    return "client_order_id" in message and ("already" in message or "duplicate" in message)


class ReconcileOutcome(Enum):
    CONFIRMED_ABSENT = "confirmed_absent"
    CONFIRMED_PRESENT = "confirmed_present"
    AMBIGUOUS_UNREACHABLE = "ambiguous_unreachable"


@dataclass(frozen=True)
class ReconcileResult:
    outcome: ReconcileOutcome
    order: Optional[object] = None
    error: Optional[str] = None


def reconcile_by_client_id(client, order_client_id: str) -> ReconcileResult:
    """Always called before any retry - an existing order under this ID
    means the prior attempt already reached the broker. Three explicit
    outcomes (rev. 11 point 3, fixes a real fail-open bug): a broker
    timeout must never look like "no order exists" - only a genuine 404
    (order provably never created) is CONFIRMED_ABSENT. Any other failure,
    including a raw network/timeout error raised before a response ever
    arrives (verified: alpaca-py's rest.py only wraps HTTPError as
    APIError *after* a response is received - a pre-response failure is a
    bare exception, not an APIError), is AMBIGUOUS_UNREACHABLE. Only
    CONFIRMED_ABSENT permits a new submission."""
    try:
        order = client.get_order_by_client_id(order_client_id)
        return ReconcileResult(ReconcileOutcome.CONFIRMED_PRESENT, order=order)
    except Exception as exc:  # noqa: BLE001
        if is_confirmed_not_found(exc):
            return ReconcileResult(ReconcileOutcome.CONFIRMED_ABSENT)
        return ReconcileResult(ReconcileOutcome.AMBIGUOUS_UNREACHABLE, error=str(exc))


def _fetch_order_with_legs(client, order_id: str):
    """get_order_by_id() omits nested bracket legs unless explicitly
    requested (found via review: Alpaca's GetOrderByIdRequest.nested
    defaults to False server-side when the filter is omitted entirely,
    which every call site here previously did) - callers that need
    .legs (identifying the stop/take-profit children) must use this, not
    a bare get_order_by_id(order_id)."""
    from alpaca.trading.requests import GetOrderByIdRequest
    return client.get_order_by_id(order_id, filter=GetOrderByIdRequest(nested=True))


def poll_fill_status(client, alpaca_order_id: str, *, deadline_seconds: float = PARTIAL_FILL_DEADLINE_SECONDS,
                      sleep_fn: Callable[[float], None] = time.sleep, now_fn: Callable[[], float] = time.monotonic):
    """Polls the fixed 15-second partial-fill deadline. Returns the final
    order snapshot observed (caller decides accept/cancel-and-flatten) -
    with nested bracket legs populated, since the entry-order caller
    needs .legs to identify the stop/take-profit children."""
    deadline = now_fn() + deadline_seconds
    order = _fetch_order_with_legs(client, alpaca_order_id)
    while order_status_value(order) not in ("filled", "canceled", "expired", "rejected") and now_fn() < deadline:
        sleep_fn(min(1.0, max(0.0, deadline - now_fn())))
        order = _fetch_order_with_legs(client, alpaca_order_id)
    return order


class CancelUnfilledOutcome(Enum):
    CONFIRMED_CANCELED_ZERO_FILL = "confirmed_canceled_zero_fill"
    FILLED_DURING_CANCEL = "filled_during_cancel"  # raced with a real fill - never assumed canceled
    AMBIGUOUS = "ambiguous"                         # cancellation never confirmed


@dataclass(frozen=True)
class CancelUnfilledResult:
    outcome: CancelUnfilledOutcome
    order: Optional[object] = None  # the final observed order snapshot, whatever the outcome


def cancel_and_confirm_unfilled_entry(
    client, order_id: str, *, deadline_seconds: float = PARTIAL_FILL_DEADLINE_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep, now_fn: Callable[[], float] = time.monotonic,
) -> CancelUnfilledResult:
    """A bracket entry order still resting (new/accepted/pending_new/...)
    with zero fill at poll_fill_status()'s deadline must be ACTIVELY
    canceled here, never just labeled canceled without ever calling
    cancel_order_by_id (found via review: the original code treated a
    still-live order the same as one already canceled by the broker,
    leaving a real resting order unmanaged - it could still fill later,
    invisible to SLC's own state). Confirms via re-fetch, never trusts
    cancel_order_by_id's own response alone, and re-checks filled
    quantity on every observation, since the order can race-fill in the
    window between the last poll and this cancel request."""
    try:
        client.cancel_order_by_id(order_id)
    except Exception:  # noqa: BLE001
        pass  # confirmed via re-fetch below regardless of what this raised

    def _resolve(order) -> Optional[CancelUnfilledResult]:
        status = order_status_value(order)
        filled_qty = float(getattr(order, "filled_qty", 0) or 0)
        if filled_qty > 0:
            return CancelUnfilledResult(CancelUnfilledOutcome.FILLED_DURING_CANCEL, order)
        if status in ("canceled", "expired", "rejected"):
            return CancelUnfilledResult(CancelUnfilledOutcome.CONFIRMED_CANCELED_ZERO_FILL, order)
        return None

    deadline = now_fn() + deadline_seconds
    order = _fetch_order_with_legs(client, order_id)
    result = _resolve(order)
    while result is None and now_fn() < deadline:
        sleep_fn(min(1.0, max(0.0, deadline - now_fn())))
        order = _fetch_order_with_legs(client, order_id)
        result = _resolve(order)
    return result if result is not None else CancelUnfilledResult(CancelUnfilledOutcome.AMBIGUOUS, order)


class CloseOutcome(Enum):
    CONFIRMED_FLAT = "confirmed_flat"
    AMBIGUOUS_CANCEL = "ambiguous_cancel"        # cancellation never confirmed - never proceeded to close
    AMBIGUOUS_SUBMISSION = "ambiguous_submission"  # the close order's own submit was ambiguous
    FAILED = "failed"                              # close order reached a terminal non-filled state


@dataclass(frozen=True)
class CloseResult:
    outcome: CloseOutcome
    fill_price: Optional[float] = None   # the close order's ACTUAL fill price - never the
                                          # position's theoretical target (found via review:
                                          # a closeout flatten exits at whatever the market
                                          # is, not the frozen strategy's target)
    close_order_id: Optional[str] = None


def cancel_then_confirm_then_close(
    client, symbol: str, protective_order_id: Optional[str], qty: float, side,
    *, close_client_order_id: Optional[str] = None,
) -> CloseResult:
    """The corrected recovery ordering (rev. 6/11): confirm cancellation of
    any resting protective order FIRST, then submit the close (under its
    own deterministic client_order_id, reconciled before submission - a
    crash-and-retry of this path must be idempotent, never a second close
    order), then POLL for the close order's own fill status (never infer
    flat from a successful submission response alone - rev. 11 point 7),
    then verify flat via a broker read-back. Returns the actual close
    fill price whenever one is available, so the caller can record real
    P&L instead of a fabricated figure."""
    from alpaca.trading.enums import TimeInForce
    from alpaca.trading.requests import MarketOrderRequest

    if protective_order_id:
        client.cancel_order_by_id(protective_order_id)
        deadline = time.monotonic() + PARTIAL_FILL_DEADLINE_SECONDS
        confirmed = False
        while time.monotonic() < deadline:
            order = client.get_order_by_id(protective_order_id)
            if order_status_value(order) == "canceled":
                confirmed = True
                break
            time.sleep(1.0)
        if not confirmed:
            # An unconfirmed cancellation is ambiguous, not "canceled" -
            # never proceed to submit a close against a reservation that
            # might still be live (rev. 11 point 4).
            return CloseResult(CloseOutcome.AMBIGUOUS_CANCEL)

    oc_id = close_client_order_id or client_order_id(symbol, "exit", pd.Timestamp.now(tz="UTC"), "exit")
    reconciled = reconcile_by_client_id(client, oc_id)
    if reconciled.outcome == ReconcileOutcome.AMBIGUOUS_UNREACHABLE:
        return CloseResult(CloseOutcome.AMBIGUOUS_SUBMISSION)
    close_order = reconciled.order
    if reconciled.outcome == ReconcileOutcome.CONFIRMED_ABSENT:
        try:
            close_order = client.submit_order(MarketOrderRequest(
                symbol=symbol, qty=abs(qty), side=side, time_in_force=TimeInForce.DAY,
                client_order_id=oc_id,
            ))
        except Exception as exc:  # noqa: BLE001
            if _is_duplicate_client_order_id_error(exc):
                re_reconciled = reconcile_by_client_id(client, oc_id)
                if re_reconciled.outcome != ReconcileOutcome.CONFIRMED_PRESENT:
                    return CloseResult(CloseOutcome.AMBIGUOUS_SUBMISSION)
                close_order = re_reconciled.order
            else:
                return CloseResult(CloseOutcome.AMBIGUOUS_SUBMISSION)
        if not str(getattr(close_order, "id", "") or ""):
            # A successful submit normally returns the created order and
            # its id. If it does not, reconcile once by deterministic ID;
            # a fresh 404 here is NOT permission to submit again because
            # the preceding submit call already returned successfully.
            post_submit = reconcile_by_client_id(client, oc_id)
            if post_submit.outcome != ReconcileOutcome.CONFIRMED_PRESENT:
                return CloseResult(CloseOutcome.AMBIGUOUS_SUBMISSION)
            close_order = post_submit.order

    close_order_id = str(getattr(close_order, "id", "")) if close_order is not None else None
    fill_price = None
    if close_order_id:
        final = poll_fill_status(client, close_order_id)
        status = order_status_value(final)
        if status not in ("filled",):
            outcome = CloseOutcome.FAILED if status in ("canceled", "expired", "rejected") else CloseOutcome.AMBIGUOUS_SUBMISSION
            return CloseResult(outcome, close_order_id=close_order_id)
        fill_price = float(getattr(final, "filled_avg_price", 0) or 0) or None

    try:
        position = client.get_open_position(symbol) if hasattr(client, "get_open_position") else None
    except Exception as exc:  # noqa: BLE001
        if is_confirmed_not_found(exc):
            # Alpaca returns HTTP 404 when the symbol is genuinely flat;
            # this is positive broker evidence, not an execution error.
            position = None
        else:
            # The close order filled, but the independent position
            # read-back failed ambiguously.  Preserve the fill details and
            # leave reconciliation for a later cycle/guardian invocation.
            return CloseResult(
                CloseOutcome.AMBIGUOUS_SUBMISSION,
                fill_price=fill_price,
                close_order_id=close_order_id,
            )
    is_flat = position is None or float(getattr(position, "qty", 0)) == 0.0
    outcome = CloseOutcome.CONFIRMED_FLAT if is_flat else CloseOutcome.FAILED
    return CloseResult(outcome, fill_price=fill_price, close_order_id=close_order_id)


def take_profit_leg(legs) -> Optional[object]:
    """The bracket's take-profit child leg specifically - a LIMIT order,
    distinct from the stop-loss child leg (STOP/STOP_LIMIT) - identified
    by its order type, never by array position (Alpaca's `.legs` ordering
    is not documented or guaranteed). Shared by verify_target_replacement()
    below and by run_slc_live.py's replace_order_by_id call site, so
    there is exactly one place that decides which leg is "the target" -
    two independent, potentially-diverging lookups is exactly how a
    replace could silently land on the stop leg instead."""
    return next((leg for leg in (legs or []) if getattr(leg, "type", "") in ("limit",)), None)


def protective_leg(legs) -> Optional[object]:
    """The bracket's stop-loss child leg specifically - STOP or
    STOP_LIMIT, distinct from the take-profit leg - identified by order
    type, never by array position, mirroring take_profit_leg()'s same
    reasoning."""
    return next((leg for leg in (legs or []) if getattr(leg, "type", "") in ("stop", "stop_limit")), None)


@dataclass(frozen=True)
class TargetVerificationResult:
    confirmed: bool
    target_order_id: Optional[str] = None  # the leg's order id - per the SDK's
                                            # replaces/replaced_by semantics, a successful
                                            # replace_order_by_id issues a NEW id, which the
                                            # caller must save onto the position record
                                            # (found via review: this was never persisted,
                                            # leaving closeout with no way to find it later)


def verify_target_replacement(
    client, entry_order_id: str, expected_target, direction: str, *, expected_qty: Optional[float] = None,
) -> TargetVerificationResult:
    """Re-fetches the bracket parent WITH nested legs (found via review:
    the original bare get_order_by_id() call never requested them, so
    .legs was always empty and this function could never see a
    replacement even on success - see _fetch_order_with_legs) and
    confirms the replacement leg's actual price/qty/status reflect the
    intended exact-2R value - a successful PATCH call alone does not
    prove this (rev. 11: a successful replace_order_by_id issues a NEW
    order id, per the SDK's replaces/replaced_by fields)."""
    parent = _fetch_order_with_legs(client, entry_order_id)
    target_leg = take_profit_leg(getattr(parent, "legs", None))
    if target_leg is None:
        return TargetVerificationResult(False)
    if order_status_value(target_leg) not in ("new", "accepted", "held", "open"):
        return TargetVerificationResult(False)
    leg_price = getattr(target_leg, "limit_price", None)
    if leg_price is None:
        return TargetVerificationResult(False)
    if to_decimal(leg_price) != to_decimal(expected_target):
        return TargetVerificationResult(False)
    if expected_qty is not None:
        leg_qty = getattr(target_leg, "qty", None)
        if leg_qty is None or float(leg_qty) != float(expected_qty):
            return TargetVerificationResult(False)
    target_order_id = str(getattr(target_leg, "id", "")) or None
    return TargetVerificationResult(True, target_order_id=target_order_id)


@dataclass(frozen=True)
class MarketableExitResult:
    filled: bool
    fill_price: Optional[float] = None


def exit_via_marketable_replacement(
    client, symbol: str, target_order_id: str, direction: str,
    *, deadline_seconds: float = PARTIAL_FILL_DEADLINE_SECONDS,
) -> MarketableExitResult:
    """protected_degraded recovery (rev. 8/11): replaces (never cancels)
    the take-profit leg with a marketable limit price, so the STOP LEG
    STAYS LINKED AND ACTIVE for the entire attempt - Alpaca cancels an OCO
    sibling automatically on a CANCEL of one leg, but not on a REPLACE, so
    the position is never unprotected during this exit attempt. Not
    empirically verified against a live paper account (no order may be
    submitted during this work) - documented as a limitation. Callable
    from run_slc_live.py's protected_degraded resolution step (found via
    review: this function existed but had no caller anywhere - every
    degraded position sat unresolved until end-of-day closeout)."""
    from alpaca.trading.requests import ReplaceOrderRequest

    try:
        quote = client.get_latest_quote(symbol)
        bid, ask = float(quote.bid_price), float(quote.ask_price)
    except Exception:  # noqa: BLE001
        return MarketableExitResult(False)
    # Aggressively marketable: sell (long exit) below the bid, buy (short
    # exit) above the ask.
    marketable_price = round(bid * 0.99, 2) if direction == "long" else round(ask * 1.01, 2)
    try:
        replaced = client.replace_order_by_id(target_order_id, ReplaceOrderRequest(limit_price=marketable_price))
    except Exception:  # noqa: BLE001
        return MarketableExitResult(False)
    # A successful replace issues a NEW order id (per the SDK's
    # replaces/replaced_by semantics - see execution.verify_target_replacement's
    # own docstring) - found via review: polling the OLD target_order_id
    # here would just see it sitting in status="replaced" forever (the
    # OLD order object itself never fills; the NEW one does), so this
    # recovery path could never actually confirm success.
    new_order_id = str(getattr(replaced, "id", "")) or target_order_id
    final = poll_fill_status(client, new_order_id, deadline_seconds=deadline_seconds)
    if order_status_value(final) != "filled":
        return MarketableExitResult(False)
    fill_price = float(getattr(final, "filled_avg_price", 0) or 0) or None
    return MarketableExitResult(True, fill_price=fill_price)
