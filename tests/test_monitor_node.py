"""Tests for nodes/monitor_node.py - _evaluate_position() rule logic and the
broker-unreachable regression fix (a failed connection must never be treated
as "zero open positions")."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nodes.monitor_node import _evaluate_position


def _make_pos(entry_price=100.0, stop_price=95.0, target_price=110.0):
    p = MagicMock()
    p.entry_price = entry_price
    p.stop_price = stop_price
    p.target_price = target_price
    return p


def _make_ind(trend="UPTREND", rsi_14=60.0, macd_hist=0.5, atr_14=2.0):
    i = MagicMock()
    i.trend = trend
    i.rsi_14 = rsi_14
    i.macd_hist = macd_hist
    i.atr_14 = atr_14
    return i


class TestEvaluatePositionClose:
    def test_downtrend_closes(self):
        result = _evaluate_position("AAPL", _make_pos(), current_price=101.0,
                                    ind=_make_ind(trend="DOWNTREND"))
        assert result["action"] == "CLOSE"

    def test_macd_negative_and_rsi_breakdown_closes(self):
        ind = _make_ind(trend="UPTREND", rsi_14=40.0, macd_hist=-0.2)
        result = _evaluate_position("AAPL", _make_pos(), current_price=101.0, ind=ind)
        assert result["action"] == "CLOSE"

    def test_macd_negative_but_rsi_above_breakdown_does_not_close(self):
        ind = _make_ind(trend="UPTREND", rsi_14=50.0, macd_hist=-0.2)
        result = _evaluate_position("AAPL", _make_pos(), current_price=100.5, ind=ind)
        assert result["action"] != "CLOSE"


class TestEvaluatePositionTightenStop:
    def test_favorable_move_tightens_stop(self):
        pos = _make_pos(entry_price=100.0, stop_price=95.0)
        ind = _make_ind(atr_14=2.0)
        # price moved +3 (1.5x ATR of 2.0) - above the 1x ATR threshold
        result = _evaluate_position("AAPL", pos, current_price=103.0, ind=ind)
        assert result["action"] == "TIGHTEN_STOP"
        assert result["new_stop"] > pos.stop_price
        assert result["new_stop"] < 103.0

    def test_small_favorable_move_holds(self):
        pos = _make_pos(entry_price=100.0, stop_price=95.0)
        ind = _make_ind(atr_14=2.0)
        # price moved +0.5 - below the 1x ATR (2.0) threshold
        result = _evaluate_position("AAPL", pos, current_price=100.5, ind=ind)
        assert result["action"] == "HOLD"


class TestEvaluatePositionHold:
    def test_neutral_holds(self):
        pos = _make_pos(entry_price=100.0, stop_price=95.0)
        ind = _make_ind(trend="SIDEWAYS", rsi_14=55.0, macd_hist=0.1, atr_14=2.0)
        result = _evaluate_position("AAPL", pos, current_price=100.2, ind=ind)
        assert result["action"] == "HOLD"

    def test_no_indicator_no_price_holds(self):
        result = _evaluate_position("AAPL", _make_pos(), current_price=None, ind=None)
        assert result["action"] == "HOLD"


class TestNoAnthropicDependency:
    def test_module_does_not_import_anthropic(self):
        import nodes.monitor_node as mn
        assert not hasattr(mn, "anthropic")


class TestBrokerUnreachableSkipsReconciliation:
    def test_broker_unreachable_does_not_reconcile_or_mutate_positions(self):
        import nodes.monitor_node as mn
        with patch.object(mn, "_get_alpaca_positions", return_value=({}, False)), \
             patch.object(mn, "_reconcile") as mock_reconcile, \
             patch.object(mn, "_load_open_positions") as mock_load_open, \
             patch.object(mn, "init_db"), \
             patch.object(mn, "_write_log"), \
             patch.object(mn.settings, "ROBINHOOD_ENABLED", False):
            result = mn.run(tickers=["AAPL"])

        mock_reconcile.assert_not_called()
        mock_load_open.assert_not_called()
        assert result["broker_connected"] is False
        assert result["held"] == []
        assert result["closed"] == []
        assert result["reconciled"] == []


def test_filled_bracket_exit_identifies_confirmed_stop_fill():
    import nodes.monitor_node as mn

    order = SimpleNamespace(legs=[
        SimpleNamespace(
            side="OrderSide.SELL", type="OrderType.LIMIT",
            status="OrderStatus.CANCELED", filled_avg_price=None,
            filled_at=None,
        ),
        SimpleNamespace(
            side="OrderSide.SELL", type="OrderType.STOP",
            status="OrderStatus.FILLED", filled_avg_price="185.10",
            filled_at=datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc),
        ),
    ])

    assert mn._filled_bracket_exit(order) == {
        "price": 185.10,
        "filled_at": datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc),
        "reason": "broker_stop",
    }


def test_monitor_reports_pending_position_reconciled_between_runs():
    import nodes.monitor_node as mn

    pending_result = {
        "reconciled": ["CRM"],
        "canceled": [],
        "failed": [],
    }
    with patch.object(mn, "init_db"), \
         patch.object(mn.settings, "ROBINHOOD_ENABLED", False), \
         patch.object(mn, "_get_alpaca_positions", return_value=({}, True)), \
         patch.object(mn, "_reconcile_pending_positions", return_value=pending_result), \
         patch.object(mn, "_promote_pending_positions", return_value=[]), \
         patch.object(mn, "_load_open_positions", return_value=[]), \
         patch.object(mn, "_write_log"):
        result = mn.run()

    assert result["reconciled"] == ["CRM"]
    assert result["failed"] == []
