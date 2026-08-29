"""
Tests for nodes/day_strategy_node.py - the 5-min ORB signal-only node.

The most important property tested here is structural isolation: this node
must NEVER write to Strategy/RiskApproval and must NEVER return a
strategies_run_id, since that's what makes it impossible (not just unlikely)
for a day-mode run to be picked up by risk_node/execution_node.
"""
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import nodes.day_strategy_node as day_node
from db.models import RiskApproval, Strategy


_SESSION = {
    "open": datetime(2025, 6, 2, 13, 30, 0),
    "close": datetime(2025, 6, 2, 20, 0, 0),
}


def _stats(avg_vol=2_000_000.0, atr=1.0, avg_open_vol=10_000.0):
    stats = MagicMock()
    stats.avg_daily_volume_14d = avg_vol
    stats.daily_atr_14 = atr
    stats.avg_opening_volume_14d = avg_open_vol
    return stats


def _bar_df(open_=10.0, high=10.5, low=9.9, close=10.3, volume=20_000.0):
    idx = pd.DatetimeIndex([pd.Timestamp("2025-06-02 13:30:00")])
    return pd.DataFrame(
        {"open": [open_], "high": [high], "low": [low], "close": [close], "volume": [volume]},
        index=idx,
    )


class TestNeverTouchesStrategyOrRiskApproval:
    def test_run_return_value_has_no_strategies_run_id_key(self):
        with patch.object(day_node, "init_db"), \
             patch.object(day_node, "is_trading_day", return_value=False):
            result = day_node.run(tickers=["AAPL"], as_of=date(2025, 6, 7))
        assert "strategies_run_id" not in result

    def test_never_constructs_a_strategy_or_riskapproval_row(self):
        """
        Every session.add(...) call across a full run must never be passed a
        Strategy or RiskApproval instance - locks in the structural isolation
        as an actual regression test, not just a design intent.
        """
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        with patch.object(day_node, "init_db"), \
             patch.object(day_node, "is_trading_day", return_value=True), \
             patch.object(day_node, "news_gate_status", return_value=(None, False, "")), \
             patch.object(day_node, "session_for", return_value=_SESSION), \
             patch.object(day_node, "get_session", return_value=mock_cm), \
             patch.object(day_node, "_get_ticker_id", return_value="ticker-1"), \
             patch.object(day_node, "_get_daily_stats", return_value=_stats()), \
             patch.object(day_node, "fetch_intraday_bars", return_value=_bar_df()):
            day_node.run(tickers=["AAPL"], as_of=date(2025, 6, 2))

        for call in mock_session.add.call_args_list:
            written = call.args[0]
            assert not isinstance(written, (Strategy, RiskApproval))


class TestNewsGateFailsClosedForWholeSession:
    def test_blocked_news_gate_excludes_every_symbol(self):
        with patch.object(day_node, "init_db"), \
             patch.object(day_node, "is_trading_day", return_value=True), \
             patch.object(day_node, "news_gate_status",
                          return_value=(None, True, "morning report is missing")):
            result = day_node.run(tickers=["AAPL", "MSFT"], as_of=date(2025, 6, 2))

        assert result["generated"] == []
        assert set(result["excluded"]) == {"AAPL", "MSFT"}
        assert "morning report is missing" in result["blocked"]


class TestPerSymbolMissingBarExclusionIsIsolated:
    def test_one_symbols_missing_bar_does_not_block_others(self):
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        def fake_fetch(symbol, start, end, minutes=5):
            if symbol == "MISSING":
                return pd.DataFrame()
            return _bar_df()

        with patch.object(day_node, "init_db"), \
             patch.object(day_node, "is_trading_day", return_value=True), \
             patch.object(day_node, "news_gate_status", return_value=(None, False, "")), \
             patch.object(day_node, "session_for", return_value=_SESSION), \
             patch.object(day_node, "get_session", return_value=mock_cm), \
             patch.object(day_node, "_get_ticker_id", return_value="ticker-1"), \
             patch.object(day_node, "_get_daily_stats", return_value=_stats()), \
             patch.object(day_node, "fetch_intraday_bars", side_effect=fake_fetch):
            result = day_node.run(tickers=["AAPL", "MISSING"], as_of=date(2025, 6, 2))

        excluded_symbols = {e["symbol"] for e in result["excluded"]}
        assert excluded_symbols == {"MISSING"}
        assert "AAPL" in result["generated"]


class TestCandidateConstruction:
    def _build(self, **stats_kwargs):
        with patch.object(day_node, "_get_ticker_id", return_value="ticker-1"), \
             patch.object(day_node, "_get_daily_stats", return_value=_stats(**stats_kwargs)):
            return day_node._build_candidate("AAPL", date(2025, 6, 2), _SESSION)

    def test_bullish_breakout_gets_long_direction_and_sizing(self):
        with patch.object(day_node, "fetch_intraday_bars",
                           return_value=_bar_df(open_=10.0, close=10.5, high=10.6, low=9.8, volume=50_000)):
            candidate = self._build(avg_vol=2_000_000, atr=1.0, avg_open_vol=10_000)
        assert candidate["candle_type"] == "bullish"
        assert candidate["direction"] == "long"
        assert candidate["passed_filters"] is True
        assert candidate["entry_trigger_price"] == 10.6  # opening high
        assert candidate["stop_price"] == pytest.approx(10.6 - 0.10 * 1.0)
        assert candidate["simulated_quantity"] > 0
        assert candidate["account_equity_used"] == day_node._SIZING_EQUITY

    def test_doji_never_trades_and_has_no_sizing(self):
        with patch.object(day_node, "fetch_intraday_bars",
                           return_value=_bar_df(open_=10.0, close=10.0, volume=50_000)):
            candidate = self._build(avg_vol=2_000_000, atr=1.0, avg_open_vol=10_000)
        assert candidate["candle_type"] == "doji"
        assert candidate["direction"] is None
        assert candidate["passed_filters"] is False
        assert candidate["entry_trigger_price"] is None
        assert "simulated_quantity" not in candidate
        assert candidate["rejection_reason"] == "doji - no trade"

    def test_failing_filter_records_rejection_reason(self):
        with patch.object(day_node, "fetch_intraday_bars",
                           return_value=_bar_df(open_=10.0, close=10.5, volume=50_000)):
            candidate = self._build(avg_vol=2_000_000, atr=0.30, avg_open_vol=10_000)  # ATR below $0.50
        assert candidate["passed_filters"] is False
        assert "ATR" in candidate["rejection_reason"]

    def test_missing_ticker_returns_none(self):
        with patch.object(day_node, "_get_ticker_id", return_value=None):
            assert day_node._build_candidate("UNKNOWN", date(2025, 6, 2), _SESSION) is None

    def test_missing_daily_stats_returns_none(self):
        with patch.object(day_node, "_get_ticker_id", return_value="ticker-1"), \
             patch.object(day_node, "_get_daily_stats", return_value=None):
            assert day_node._build_candidate("AAPL", date(2025, 6, 2), _SESSION) is None


class TestUpsertIdempotency:
    def test_rerun_updates_existing_row_instead_of_duplicating(self):
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False
        existing = MagicMock()
        mock_session.exec.return_value.first.return_value = existing

        candidate = {
            "ticker_id": "ticker-1", "opening_bar_time": datetime(2025, 6, 2, 13, 30),
            "opening_open": 10.0, "opening_high": 10.5, "opening_low": 9.9, "opening_close": 10.3,
            "opening_volume": 20_000.0, "candle_type": "bullish", "direction": "long",
            "opening_price": 10.3, "avg_daily_volume_14d": 2_000_000.0, "daily_atr_14": 1.0,
            "avg_opening_volume_14d": 10_000.0, "opening_rel_volume": 2.0,
            "passed_filters": True, "rank": 1, "selected": True,
            "entry_trigger_price": 10.5, "stop_price": 10.4, "rejection_reason": None,
        }

        with patch.object(day_node, "get_session", return_value=mock_cm):
            day_node._upsert_signal("run-1", date(2025, 6, 2), candidate)

        mock_session.add.assert_called_once_with(existing)
        assert existing.opening_rel_volume == 2.0
