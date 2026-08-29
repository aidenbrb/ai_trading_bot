import pandas as pd
import pytest

from live_slc.closeout import CLOSEOUT_LEAD_MINUTES, run_closeout, should_begin_closeout
from utils.market_calendar import session_for


class _Pos:
    def __init__(self, symbol, direction, qty, status, protective_order_id=None):
        self.symbol = symbol
        self.level_id = f"demand:{symbol}"
        self.confirmation_time = pd.Timestamp("2026-08-13 14:00:00")
        self.opened_at = pd.Timestamp("2026-08-13 14:05:00")
        self.direction = direction
        self.qty = qty
        self.status = status
        self.protective_order_id = protective_order_id


class _FakeOrder:
    def __init__(self, order_id, status, filled_avg_price=100.0):
        self.id = order_id
        self.status = status
        self.filled_avg_price = filled_avg_price
        self.legs = []


class _FakeClient:
    def __init__(self, cancel_confirms=True, close_leaves_flat=True, close_fills=True):
        self.cancel_confirms = cancel_confirms
        self.close_leaves_flat = close_leaves_flat
        self.close_fills = close_fills
        self.canceled_ids = []
        self.submitted = []
        self._known_close_orders = {}

    def cancel_order_by_id(self, oid):
        self.canceled_ids.append(oid)

    def get_order_by_id(self, oid, filter=None):
        if oid in self._known_close_orders:
            return self._known_close_orders[oid]
        return _FakeOrder(oid, "canceled" if self.cancel_confirms else "pending_cancel")

    def get_order_by_client_id(self, oc_id):
        from alpaca.common.exceptions import APIError

        class _Resp:
            status_code = 404

        class _HttpErr:
            response = _Resp()

        if oc_id in self._known_close_orders:
            return self._known_close_orders[oc_id]
        raise APIError("not found", http_error=_HttpErr())

    def submit_order(self, request):
        self.submitted.append(request)
        order_id = f"close-order-{len(self.submitted)}"
        status = "filled" if self.close_fills else "canceled"
        order = _FakeOrder(order_id, status)
        self._known_close_orders[order_id] = order
        self._known_close_orders[request.client_order_id] = order
        return order

    def get_open_position(self, symbol):
        if self.close_leaves_flat:
            return None

        class P:
            qty = 5
        return P()


def test_closeout_lead_is_5_minutes():
    assert CLOSEOUT_LEAD_MINUTES == 5


def test_should_begin_closeout_regular_session():
    day = pd.Timestamp("2024-10-02").date()
    close = pd.Timestamp(session_for(day)["close"])
    assert should_begin_closeout(close - pd.Timedelta(minutes=6), day) is False
    assert should_begin_closeout(close - pd.Timedelta(minutes=5), day) is True


def test_should_begin_closeout_early_close_session():
    day = pd.Timestamp("2024-11-29").date()  # day after Thanksgiving
    close = pd.Timestamp(session_for(day)["close"])
    assert close.hour < 19 or close < pd.Timestamp(f"{day} 19:00:00")
    assert should_begin_closeout(close - pd.Timedelta(minutes=6), day) is False
    assert should_begin_closeout(close - pd.Timedelta(minutes=5), day) is True


def test_should_begin_closeout_false_on_non_trading_day():
    saturday = pd.Timestamp("2024-10-05").date()
    assert should_begin_closeout(pd.Timestamp("2024-10-05 20:00:00"), saturday) is False


def test_ambiguous_positions_are_never_touched():
    positions = [
        _Pos("AAPL", "long", 10, "open", "prot-1"),
        _Pos("TSLA", "short", 5, "ambiguous"),
    ]
    client = _FakeClient()
    result = run_closeout(client, positions, side_for=lambda s: s)
    assert [p.symbol for p, _ in result.flattened] == ["AAPL"]
    assert [p.symbol for p in result.skipped_ambiguous] == ["TSLA"]
    assert client.canceled_ids == ["prot-1"]  # TSLA's protective order was never touched


def test_cancel_then_confirm_then_close_ordering_on_success():
    client = _FakeClient(cancel_confirms=True, close_leaves_flat=True, close_fills=True)
    positions = [_Pos("AAPL", "long", 10, "open", "prot-1")]
    result = run_closeout(client, positions, side_for=lambda s: s)
    assert len(result.flattened) == 1
    assert client.canceled_ids == ["prot-1"]
    assert len(client.submitted) == 1  # close order submitted only after cancel confirmed
    # deterministic close client_order_id was actually used
    assert client.submitted[0].client_order_id.startswith("slc-")
    # the ACTUAL close fill price is carried through (rev. 11 review
    # point 9) - never the position's theoretical target price, which
    # this fixture doesn't even define a matching value for.
    position, fill_price = result.flattened[0]
    assert fill_price == 100.0


def test_alpaca_position_404_after_filled_close_is_confirmed_flat():
    """Exact 2026-08-19 TGT regression: Alpaca represents a flat symbol
    by raising APIError(404) from get_open_position().  A filled close
    followed by that response is confirmed flat, not a cycle crash."""
    from alpaca.common.exceptions import APIError

    class _HttpErr:
        response = type("R", (), {"status_code": 404})()

    client = _FakeClient(cancel_confirms=True, close_fills=True)

    def _confirmed_absent(symbol):
        raise APIError("position does not exist", http_error=_HttpErr())

    client.get_open_position = _confirmed_absent
    result = run_closeout(
        client, [_Pos("TGT", "long", 55, "protected_degraded", "prot-1")],
        side_for=lambda s: s,
    )
    assert len(result.flattened) == 1
    assert result.flattened[0][1] == 100.0
    assert result.ambiguous_close == []


def test_non_404_position_read_failure_after_filled_close_stays_ambiguous():
    from alpaca.common.exceptions import APIError

    class _HttpErr:
        response = type("R", (), {"status_code": 500})()

    client = _FakeClient(cancel_confirms=True, close_fills=True)

    def _unreachable(symbol):
        raise APIError("server error", http_error=_HttpErr())

    client.get_open_position = _unreachable
    result = run_closeout(
        client, [_Pos("TGT", "long", 55, "protected_degraded", "prot-1")],
        side_for=lambda s: s,
    )
    assert result.flattened == []
    assert len(result.ambiguous_close) == 1


def test_closeout_accepts_real_alpaca_canceled_and_filled_enums():
    from alpaca.trading.enums import OrderStatus

    client = _FakeClient(cancel_confirms=True, close_leaves_flat=True, close_fills=True)

    def _get_order_by_id(oid, filter=None):
        if oid in client._known_close_orders:
            order = client._known_close_orders[oid]
            order.status = OrderStatus.FILLED
            return order
        return _FakeOrder(oid, OrderStatus.CANCELED)

    client.get_order_by_id = _get_order_by_id
    result = run_closeout(
        client, [_Pos("AAPL", "long", 10, "open", "prot-1")],
        side_for=lambda s: s,
    )
    assert len(result.flattened) == 1
    assert result.flattened[0][1] == 100.0


def test_failure_to_confirm_flat_is_reported_not_silently_accepted():
    client = _FakeClient(cancel_confirms=True, close_leaves_flat=False, close_fills=True)
    positions = [_Pos("AAPL", "long", 10, "open", "prot-1")]
    result = run_closeout(client, positions, side_for=lambda s: s)
    assert len(result.failed) == 1
    assert len(result.flattened) == 0


def test_unconfirmed_cancellation_is_ambiguous_not_silently_treated_as_canceled():
    client = _FakeClient(cancel_confirms=False)
    positions = [_Pos("AAPL", "long", 10, "open", "prot-1")]
    result = run_closeout(client, positions, side_for=lambda s: s)
    assert len(result.ambiguous_close) == 1
    assert client.submitted == []  # never proceeded to submit a close against an unconfirmed cancel


def test_close_order_that_never_fills_is_not_assumed_flat():
    client = _FakeClient(cancel_confirms=True, close_leaves_flat=False, close_fills=False)
    positions = [_Pos("AAPL", "long", 10, "open", "prot-1")]
    result = run_closeout(client, positions, side_for=lambda s: s)
    assert len(result.flattened) == 0
    assert len(result.failed) == 1
