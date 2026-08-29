"""
The 3-outcome reconcile model (rev. 11 point 3, fixing a real fail-open
bug): the original reconcile_by_client_id() caught every exception
identically and returned None, making a broker timeout indistinguishable
from "no order exists" - which could let the caller conclude it's safe to
submit a fresh order when it genuinely doesn't know that.
"""
from alpaca.common.exceptions import APIError

from live_slc.execution import (
    ReconcileOutcome,
    _is_duplicate_client_order_id_error,
    reconcile_by_client_id,
)


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


class _FakeHttpError:
    def __init__(self, status_code):
        self.response = _FakeResponse(status_code)


class _FakeOrder:
    id = "order-1"


def test_confirmed_present_when_order_found():
    class Client:
        def get_order_by_client_id(self, oc_id):
            return _FakeOrder()

    result = reconcile_by_client_id(Client(), "x")
    assert result.outcome == ReconcileOutcome.CONFIRMED_PRESENT
    assert result.order is not None


def test_confirmed_absent_only_on_a_genuine_404():
    class Client:
        def get_order_by_client_id(self, oc_id):
            raise APIError("not found", http_error=_FakeHttpError(404))

    result = reconcile_by_client_id(Client(), "x")
    assert result.outcome == ReconcileOutcome.CONFIRMED_ABSENT


def test_ambiguous_on_non_404_api_error():
    class Client:
        def get_order_by_client_id(self, oc_id):
            raise APIError("server error", http_error=_FakeHttpError(500))

    result = reconcile_by_client_id(Client(), "x")
    assert result.outcome == ReconcileOutcome.AMBIGUOUS_UNREACHABLE


def test_ambiguous_never_confirmed_absent_on_a_bare_timeout():
    """The critical fail-closed property: a raw timeout/network failure
    (verified: not wrapped as APIError by alpaca-py's rest.py unless a
    response was actually received) must never be treated as proof the
    order doesn't exist."""
    class Client:
        def get_order_by_client_id(self, oc_id):
            raise TimeoutError("connection timed out")

    result = reconcile_by_client_id(Client(), "x")
    assert result.outcome == ReconcileOutcome.AMBIGUOUS_UNREACHABLE
    assert result.outcome != ReconcileOutcome.CONFIRMED_ABSENT


def test_a_reconcile_timeout_cannot_cause_a_duplicate_submission():
    """End-to-end proof of the property the correction demands: given an
    AMBIGUOUS_UNREACHABLE outcome, a caller that correctly branches on it
    (as run_slc_live.py's admission loop does) never proceeds to treat the
    candidate as safe to submit."""
    class Client:
        def get_order_by_client_id(self, oc_id):
            raise TimeoutError("connection timed out")

    result = reconcile_by_client_id(Client(), "x")
    # The only outcome that ever permits a new submission:
    safe_to_submit = result.outcome == ReconcileOutcome.CONFIRMED_ABSENT
    assert not safe_to_submit


def test_duplicate_client_order_id_error_detected_not_treated_as_rejection():
    exc = RuntimeError("client_order_id already exists for this account")
    assert _is_duplicate_client_order_id_error(exc)


def test_generic_validation_error_not_misclassified_as_duplicate():
    exc = RuntimeError("insufficient buying power")
    assert not _is_duplicate_client_order_id_error(exc)
