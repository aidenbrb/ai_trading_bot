"""Tests for nodes/intraday_shadow_node.py - post-close outcome reconstruction."""
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import nodes.intraday_shadow_node as shadow_node


_SESSION = {"open": datetime(2025, 6, 2, 13, 30), "close": datetime(2025, 6, 2, 20, 0)}


def _signal_row(**overrides):
    row = MagicMock()
    row.id = "signal-1"
    row.ticker_id = "ticker-1"
    row.opening_bar_time = datetime(2025, 6, 2, 13, 30)
    row.entry_trigger_price = 10.5
    row.stop_price = 10.0
    row.direction = "long"
    row.simulated_quantity = 100.0
    row.simulated_risk_amount = 50.0
    row.cost_model_version = "baseline_v1"
    for k, v in overrides.items():
        setattr(row, k, v)
    return row


def _one_min_bars(rows):
    """rows: list of (timestamp_str, open, high, low, close)."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    return pd.DataFrame(
        {"open": [r[1] for r in rows], "high": [r[2] for r in rows],
         "low": [r[3] for r in rows], "close": [r[4] for r in rows]},
        index=idx,
    )


class TestReconstructOutcome:
    def test_never_triggered_returns_sim_without_pnl_fields(self):
        row = _signal_row()
        bars = _one_min_bars([("2025-06-02 13:35", 9.9, 10.0, 9.8, 9.9)])
        with patch.object(shadow_node, "fetch_intraday_bars", return_value=bars):
            outcome = shadow_node._reconstruct_outcome(row, _SESSION)
        assert outcome["breakout_triggered"] is False
        assert "gross_pnl" not in outcome

    def test_no_bars_returns_none(self):
        row = _signal_row()
        with patch.object(shadow_node, "fetch_intraday_bars", return_value=pd.DataFrame()):
            assert shadow_node._reconstruct_outcome(row, _SESSION) is None

    def test_triggered_and_stopped_computes_pnl_in_r(self):
        row = _signal_row()
        bars = _one_min_bars([
            ("2025-06-02 13:35", 10.2, 10.6, 10.2, 10.5),   # non-gap trigger (open 10.2 < 10.5)
            ("2025-06-02 13:36", 10.1, 10.4, 9.9, 10.0),    # stop hit intrabar, open 10.1 > stop 10.0 (no gap)
        ])
        with patch.object(shadow_node, "fetch_intraday_bars", return_value=bars):
            outcome = shadow_node._reconstruct_outcome(row, _SESSION)
        assert outcome["exit_reason"] == "stop"
        # entry ~10.5, exit ~10.0, qty=100 -> gross pnl approx (10.0-10.5)*100 = -50
        assert outcome["gross_pnl"] == pytest.approx(-50.0)
        assert outcome["cost_adjusted_pnl"] < outcome["gross_pnl"]  # costs make a losing long exit worse
        assert outcome["pnl_r"] == pytest.approx(outcome["cost_adjusted_pnl"] / 50.0)


class TestApplyOutcomeIsUpsertOnly:
    def test_updates_existing_row_never_inserts(self):
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False
        existing = MagicMock()
        mock_session.exec.return_value.first.return_value = existing

        with patch.object(shadow_node, "get_session", return_value=mock_cm):
            shadow_node._apply_outcome("signal-1", {
                "breakout_triggered": True, "trigger_time": datetime(2025, 6, 2, 13, 35),
                "simulated_entry_price": 10.5, "stop_hit": True,
                "exit_time": datetime(2025, 6, 2, 13, 36), "exit_price": 10.0,
                "exit_reason": "stop", "outcome_ambiguous": False,
                "gross_pnl": -50.0, "cost_adjusted_pnl": -55.0, "pnl_r": -1.1,
            })

        mock_session.add.assert_called_once_with(existing)
        assert existing.exit_reason == "stop"
        assert existing.gross_pnl == -50.0

    def test_missing_row_is_a_noop(self):
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False
        mock_session.exec.return_value.first.return_value = None

        with patch.object(shadow_node, "get_session", return_value=mock_cm):
            shadow_node._apply_outcome("missing-id", {"breakout_triggered": False})

        mock_session.add.assert_not_called()


class TestRunOnlyProcessesSelectedSignals:
    def test_skips_when_not_a_trading_day(self):
        with patch.object(shadow_node, "init_db"), \
             patch.object(shadow_node, "is_trading_day", return_value=False):
            result = shadow_node.run(as_of=date(2025, 6, 7))
        assert result == {"reconstructed": [], "skipped": []}
