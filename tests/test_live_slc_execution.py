from datetime import date

import pytest

from live_slc.execution import (
    EASY_TO_BORROW_FALLBACK_EXPIRES,
    QUOTE_FRESHNESS_DEADLINE_SECONDS,
    PARTIAL_FILL_DEADLINE_SECONDS,
    CancelUnfilledOutcome,
    _validate_long_bracket,
    _validate_short_bracket,
    cancel_and_confirm_unfilled_entry,
    check_shortability,
    client_order_id,
    order_status_value,
    poll_fill_status,
    exit_via_marketable_replacement,
    post_fill_directional_validation,
    post_fill_monetary_risk_check,
    retarget_after_fill,
    submit_bracket_entry,
    take_profit_leg,
    verify_target_replacement,
)


def test_order_status_value_unwraps_real_alpaca_enum_and_string_mock():
    from alpaca.trading.enums import OrderStatus

    assert order_status_value(type("O", (), {"status": OrderStatus.FILLED})()) == "filled"
    assert order_status_value(type("O", (), {"status": "filled"})()) == "filled"
    assert str(OrderStatus.FILLED) == "OrderStatus.FILLED"  # proves why bare str(...) is unsafe


def test_poll_fill_status_stops_on_real_alpaca_filled_enum():
    from alpaca.trading.enums import OrderStatus

    calls = []

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            calls.append(order_id)
            return type("O", (), {"status": OrderStatus.FILLED})()

    result = poll_fill_status(_Client(), "entry-1", sleep_fn=lambda _: pytest.fail("must not sleep"))
    assert result.status is OrderStatus.FILLED
    assert calls == ["entry-1"]


def test_partial_fill_deadline_is_fixed_at_15_seconds_not_tunable():
    assert PARTIAL_FILL_DEADLINE_SECONDS == 15.0


def test_quote_freshness_deadline_is_5_seconds():
    assert QUOTE_FRESHNESS_DEADLINE_SECONDS == 5.0


def test_client_order_id_deterministic_and_leg_scoped():
    a = client_order_id("AAPL", "demand:x", "2026-08-13 10:00", "entry")
    b = client_order_id("AAPL", "demand:x", "2026-08-13 10:00", "entry")
    c = client_order_id("AAPL", "demand:x", "2026-08-13 10:00", "protect")
    assert a == b
    assert a != c
    assert a.startswith("slc-")


@pytest.mark.parametrize("direction,fill,stop,expected", [("long", 100.0, 98.0, 104.0), ("short", 100.0, 102.0, 96.0)])
def test_retarget_after_fill_exact_2r(direction, fill, stop, expected):
    assert retarget_after_fill(direction, fill, stop) == expected


def test_post_fill_directional_validation_long():
    assert post_fill_directional_validation("long", 100.0, 98.0, 104.0).ok
    assert not post_fill_directional_validation("long", 97.0, 98.0, 104.0).ok  # fill crossed stop


def test_post_fill_directional_validation_short():
    assert post_fill_directional_validation("short", 100.0, 102.0, 96.0).ok
    assert not post_fill_directional_validation("short", 103.0, 102.0, 96.0).ok  # fill crossed stop


def test_post_fill_monetary_risk_check():
    assert post_fill_monetary_risk_check(qty=100, actual_fill=100.0, stop=98.0, equity=100000.0, risk_pct=0.0025)
    assert not post_fill_monetary_risk_check(qty=200, actual_fill=100.0, stop=98.0, equity=100000.0, risk_pct=0.0025)


def test_bracket_validation_both_directions():
    _validate_long_bracket(100, 98, 104)
    _validate_short_bracket(100, 102, 96)
    with pytest.raises(ValueError):
        _validate_long_bracket(100, 101, 104)
    with pytest.raises(ValueError):
        _validate_short_bracket(100, 99, 96)


def test_shortability_easy_to_borrow_fallback_expires_exactly_on_the_announced_date():
    def asset(symbol):
        return {"tradable": True, "shortable": True, "easy_to_borrow": True}
    before = check_shortability(asset, "AAPL", today=date(2026, 9, 21))
    on_date = check_shortability(asset, "AAPL", today=EASY_TO_BORROW_FALLBACK_EXPIRES)
    after = check_shortability(asset, "AAPL", today=date(2026, 9, 23))
    assert before.ok
    assert not on_date.ok
    assert not after.ok


def test_shortability_prefers_borrow_status_when_present():
    def easy(symbol):
        return {"tradable": True, "shortable": True, "borrow_status": "easy_to_borrow", "easy_to_borrow": False}
    def hard(symbol):
        return {"tradable": True, "shortable": True, "borrow_status": "hard_to_borrow", "easy_to_borrow": True}
    assert check_shortability(easy, "AAPL").ok
    assert not check_shortability(hard, "AAPL").ok  # borrow_status wins over the (misleading) easy_to_borrow=True


def test_shortability_fails_closed_on_lookup_exception():
    def failing(symbol):
        raise RuntimeError("network error")
    result = check_shortability(failing, "AAPL")
    assert not result.ok


def test_shortability_fails_closed_on_not_shortable():
    def not_shortable(symbol):
        return {"tradable": True, "shortable": False}
    assert not check_shortability(not_shortable, "AAPL").ok


class _FakeOrder:
    id = "order-1"


class _FakeClient:
    def __init__(self):
        self.last_request = None

    def submit_order(self, request):
        self.last_request = request
        return _FakeOrder()

    def get_order_by_client_id(self, oc_id):
        if oc_id == "known":
            return _FakeOrder()
        raise RuntimeError("not found")


def test_submit_bracket_entry_long_and_short_directions():
    client = _FakeClient()
    result = submit_bracket_entry(client, symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-1-entry")
    assert result.accepted and result.alpaca_order_id == "order-1"
    assert str(client.last_request.side) == "OrderSide.BUY"
    assert str(client.last_request.position_intent) == "PositionIntent.BUY_TO_OPEN"

    submit_bracket_entry(client, symbol="TSLA", direction="short", qty=5, stop=102.0,
                          provisional_target=96.0, order_client_id="slc-2-entry")
    assert str(client.last_request.side) == "OrderSide.SELL"
    assert str(client.last_request.position_intent) == "PositionIntent.SELL_TO_OPEN"


def test_submit_bracket_entry_surfaces_broker_failure():
    class _FailingClient:
        def submit_order(self, request):
            raise RuntimeError("rejected")
    result = submit_bracket_entry(_FailingClient(), symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-3-entry")
    assert not result.accepted
    assert "rejected" in result.error


def test_submit_bracket_entry_bare_network_failure_is_ambiguous_not_confirmed_rejected():
    """The exact bug found via review: a network/timeout failure during
    submit_order() was previously always treated as confirmed_rejected,
    which could hide a real accepted (possibly filled) order behind a
    status nothing ever reconciles."""
    class _TimeoutClient:
        def submit_order(self, request):
            raise TimeoutError("connection timed out")
    result = submit_bracket_entry(_TimeoutClient(), symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-4-entry")
    assert not result.accepted
    assert result.ambiguous is True


def test_submit_bracket_entry_definitive_validation_rejection_is_not_ambiguous():
    from alpaca.common.exceptions import APIError

    class _RejectingClient:
        def submit_order(self, request):
            resp = type("R", (), {"status_code": 422})()
            err = type("E", (), {"response": resp})()
            raise APIError("insufficient buying power", http_error=err)
    result = submit_bracket_entry(_RejectingClient(), symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-5-entry")
    assert not result.accepted
    assert result.ambiguous is False


def test_submit_bracket_entry_unclear_server_error_is_ambiguous():
    """A 5xx doesn't prove the order was never created - conservatively
    ambiguous, mirroring reconcile_by_client_id's "only 404 is confirmed"
    rule applied to the opposite side of the same idempotency problem."""
    from alpaca.common.exceptions import APIError

    class _ServerErrorClient:
        def submit_order(self, request):
            resp = type("R", (), {"status_code": 503})()
            err = type("E", (), {"response": resp})()
            raise APIError("service unavailable", http_error=err)
    result = submit_bracket_entry(_ServerErrorClient(), symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-6-entry")
    assert not result.accepted
    assert result.ambiguous is True


def test_submit_bracket_entry_duplicate_client_order_id_is_ambiguous():
    class _DuplicateClient:
        def submit_order(self, request):
            raise RuntimeError("client_order_id already exists for this account")
    result = submit_bracket_entry(_DuplicateClient(), symbol="AAPL", direction="long", qty=10, stop=98.0,
                                   provisional_target=104.0, order_client_id="slc-7-entry")
    assert not result.accepted
    assert result.ambiguous is True


# -- cancel_and_confirm_unfilled_entry (rev. 11 review point 1) -------------

class _FakeStillLiveOrder:
    def __init__(self, *, status, filled_qty=0):
        self.id = "entry-1"
        self.status = status
        self.filled_qty = filled_qty
        self.legs = []


def test_cancel_and_confirm_unfilled_entry_actively_cancels_a_resting_order():
    """The exact bug found via review: a still-live zero-fill order at
    the poll deadline was previously just labeled canceled without ever
    calling cancel_order_by_id, leaving a real resting order unmanaged."""
    calls = []

    class _Client:
        def __init__(self):
            self._status = "new"

        def cancel_order_by_id(self, order_id):
            calls.append(("cancel", order_id))
            self._status = "canceled"

        def get_order_by_id(self, order_id, filter=None):
            calls.append(("get", order_id))
            return _FakeStillLiveOrder(status=self._status)

    result = cancel_and_confirm_unfilled_entry(_Client(), "entry-1", sleep_fn=lambda s: None)
    assert result.outcome == CancelUnfilledOutcome.CONFIRMED_CANCELED_ZERO_FILL
    assert ("cancel", "entry-1") in calls  # cancel_order_by_id was actually called


def test_cancel_and_confirm_unfilled_entry_detects_a_race_into_a_real_fill():
    """The order actually filled in the window between the last poll and
    this cancel attempt - must never be assumed canceled just because
    cancel_order_by_id was called."""
    class _Client:
        def cancel_order_by_id(self, order_id):
            pass  # too late - it already filled

        def get_order_by_id(self, order_id, filter=None):
            return _FakeStillLiveOrder(status="filled", filled_qty=10)

    result = cancel_and_confirm_unfilled_entry(_Client(), "entry-1", sleep_fn=lambda s: None)
    assert result.outcome == CancelUnfilledOutcome.FILLED_DURING_CANCEL
    assert result.order.filled_qty == 10


def test_cancel_and_confirm_unfilled_entry_accepts_real_canceled_enum():
    from alpaca.trading.enums import OrderStatus

    class _Client:
        def cancel_order_by_id(self, order_id):
            pass

        def get_order_by_id(self, order_id, filter=None):
            return _FakeStillLiveOrder(status=OrderStatus.CANCELED, filled_qty=0)

    result = cancel_and_confirm_unfilled_entry(_Client(), "entry-1", sleep_fn=lambda _: None)
    assert result.outcome == CancelUnfilledOutcome.CONFIRMED_CANCELED_ZERO_FILL


def test_cancel_and_confirm_unfilled_entry_ambiguous_when_never_confirmed():
    class _Client:
        def cancel_order_by_id(self, order_id):
            pass

        def get_order_by_id(self, order_id, filter=None):
            return _FakeStillLiveOrder(status="pending_cancel")  # never resolves

    result = cancel_and_confirm_unfilled_entry(
        _Client(), "entry-1", deadline_seconds=0.05, sleep_fn=lambda s: None,
    )
    assert result.outcome == CancelUnfilledOutcome.AMBIGUOUS


class _FakeLeg:
    def __init__(self, *, id, type, status="new", limit_price=None, qty=None):
        self.id = id
        self.type = type
        self.status = status
        self.limit_price = limit_price
        self.qty = qty


def test_take_profit_leg_identified_by_type_not_array_position():
    """The exact bug found via review (never triggered live - no order
    submitted during this work): picking legs[0] unconditionally would
    silently select the STOP leg whenever Alpaca happens to return it
    first, since .legs ordering is not documented or guaranteed."""
    stop_first = [_FakeLeg(id="stop-1", type="stop"), _FakeLeg(id="target-1", type="limit", limit_price=104.0)]
    assert take_profit_leg(stop_first).id == "target-1"

    target_first = [_FakeLeg(id="target-1", type="limit", limit_price=104.0), _FakeLeg(id="stop-1", type="stop")]
    assert take_profit_leg(target_first).id == "target-1"


def test_take_profit_leg_none_when_no_limit_leg_present():
    assert take_profit_leg([_FakeLeg(id="stop-1", type="stop")]) is None
    assert take_profit_leg([]) is None
    assert take_profit_leg(None) is None


def test_verify_target_replacement_uses_type_based_lookup_even_with_stop_leg_first():
    class _ParentOrder:
        legs = [_FakeLeg(id="stop-1", type="stop"), _FakeLeg(id="target-1", type="limit", limit_price=104.0, qty=125)]

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            return _ParentOrder()

    result = verify_target_replacement(_Client(), "entry-1", 104.0, "long")
    assert result.confirmed is True
    # the (possibly new, per replaces/replaced_by semantics) leg order id
    # is returned so the caller can persist it (rev. 11 review point 5).
    assert result.target_order_id == "target-1"
    assert verify_target_replacement(_Client(), "entry-1", 105.0, "long").confirmed is False  # wrong expected price


def test_verify_target_replacement_false_when_leg_status_not_live():
    class _ParentOrder:
        legs = [_FakeLeg(id="target-1", type="limit", limit_price=104.0, status="canceled")]

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            return _ParentOrder()

    assert verify_target_replacement(_Client(), "entry-1", 104.0, "long").confirmed is False


def test_verify_target_replacement_accepts_real_live_status_enum():
    from alpaca.trading.enums import OrderStatus

    class _ParentOrder:
        legs = [_FakeLeg(
            id="target-1", type="limit", limit_price=104.0,
            status=OrderStatus.ACCEPTED, qty=125,
        )]

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            return _ParentOrder()

    result = verify_target_replacement(_Client(), "entry-1", 104.0, "long", expected_qty=125)
    assert result.confirmed is True


def test_verify_target_replacement_requests_nested_legs_explicitly():
    """The exact bug found via review: get_order_by_id() omits bracket
    legs entirely unless nested=True is explicitly requested - the
    original bare call could never see a replacement even on success."""
    from alpaca.trading.requests import GetOrderByIdRequest

    class _ParentOrder:
        legs = [_FakeLeg(id="target-1", type="limit", limit_price=104.0, qty=125)]

    seen = {}

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            seen["filter"] = filter
            return _ParentOrder()

    verify_target_replacement(_Client(), "entry-1", 104.0, "long")
    assert isinstance(seen["filter"], GetOrderByIdRequest)
    assert seen["filter"].nested is True


def test_verify_target_replacement_false_on_quantity_mismatch():
    class _ParentOrder:
        legs = [_FakeLeg(id="target-1", type="limit", limit_price=104.0, qty=100)]

    class _Client:
        def get_order_by_id(self, order_id, filter=None):
            return _ParentOrder()

    result = verify_target_replacement(_Client(), "entry-1", 104.0, "long", expected_qty=125)
    assert result.confirmed is False


# -- exit_via_marketable_replacement polls the NEW (post-replace) order id --

def test_exit_via_marketable_replacement_polls_the_new_order_id_not_the_old_one():
    """The exact bug found via review: a successful replace_order_by_id
    issues a NEW order id (the old one transitions to status="replaced"
    and never fills itself) - polling the OLD id would see "replaced"
    forever and this recovery path could never confirm success."""
    calls = []

    class _Client:
        def get_latest_quote(self, symbol):
            return type("Q", (), {"bid_price": 100.0, "ask_price": 100.1})()

        def replace_order_by_id(self, order_id, request):
            calls.append(("replace", order_id))
            return type("O", (), {"id": "new-target-leg-id"})()

        def get_order_by_id(self, order_id, filter=None):
            calls.append(("get", order_id))
            return type("O", (), {"status": "filled", "filled_avg_price": 99.0})()

    result = exit_via_marketable_replacement(_Client(), "AAPL", "old-target-leg-id", "long")
    assert result.filled is True
    assert result.fill_price == 99.0
    assert ("replace", "old-target-leg-id") in calls
    assert ("get", "new-target-leg-id") in calls
    assert ("get", "old-target-leg-id") not in calls


def test_exit_via_marketable_replacement_false_on_quote_failure():
    class _Client:
        def get_latest_quote(self, symbol):
            raise TimeoutError("timed out")

    result = exit_via_marketable_replacement(_Client(), "AAPL", "target-1", "long")
    assert result.filled is False


def test_exit_via_marketable_replacement_falls_back_to_old_id_if_response_has_no_id():
    """Defensive fallback only - the normal path always uses the new id."""
    class _Client:
        def get_latest_quote(self, symbol):
            return type("Q", (), {"bid_price": 100.0, "ask_price": 100.1})()

        def replace_order_by_id(self, order_id, request):
            return type("O", (), {})()  # no .id at all

        def get_order_by_id(self, order_id, filter=None):
            assert order_id == "old-target-leg-id"
            return type("O", (), {"status": "filled", "filled_avg_price": 99.0})()

    result = exit_via_marketable_replacement(_Client(), "AAPL", "old-target-leg-id", "long")
    assert result.filled is True


def test_exit_via_marketable_replacement_accepts_real_filled_enum():
    from alpaca.trading.enums import OrderStatus

    class _Client:
        def get_latest_quote(self, symbol):
            return type("Q", (), {"bid_price": 100.0, "ask_price": 100.1})()

        def replace_order_by_id(self, order_id, request):
            return type("O", (), {"id": "replacement-1"})()

        def get_order_by_id(self, order_id, filter=None):
            return type("O", (), {
                "status": OrderStatus.FILLED, "filled_avg_price": 99.0,
            })()

    result = exit_via_marketable_replacement(_Client(), "AAPL", "target-1", "long")
    assert result.filled is True
    assert result.fill_price == 99.0


# reconcile_by_client_id's 3-outcome model has its own dedicated test file
# (test_live_slc_execution_reconcile.py) - it needs a real alpaca.common.
# exceptions.APIError with a mocked http_error to properly distinguish a
# confirmed 404 from a bare/ambiguous exception, which _FakeClient here
# (a plain get_order_by_client_id returning None/raising bare RuntimeError)
# isn't shaped to exercise correctly.
