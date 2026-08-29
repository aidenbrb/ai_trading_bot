"""Tests for nodes/intraday_reference_node.py - prior-session-only ORB stats."""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import nodes.intraday_reference_node as ref_node


def _daily_df(n=20, start_volume=1_000_000):
    idx = pd.date_range("2025-05-01", periods=n, freq="D")
    return pd.DataFrame({
        "open":   [100.0] * n,
        "high":   [101.0] * n,
        "low":    [99.0] * n,
        "close":  [100.5] * n,
        "volume": [float(start_volume)] * n,
    }, index=idx)


def _opening_bar_df(volume=50_000.0, open_time=None):
    idx = pd.DatetimeIndex([open_time or pd.Timestamp("2025-05-20 13:30:00")])
    return pd.DataFrame({
        "open": [10.0], "high": [10.5], "low": [9.9], "close": [10.2], "volume": [volume],
    }, index=idx)


class TestComputeStats:
    def test_returns_stats_when_data_available(self):
        prior_sessions = [date(2025, 5, d) for d in range(1, 15)]
        with patch.object(ref_node, "fetch_daily_bars", return_value=_daily_df(20)), \
             patch.object(ref_node, "fetch_intraday_bars", return_value=_opening_bar_df()), \
             patch.object(ref_node, "session_for", return_value={
                 "open": pd.Timestamp("2025-05-20 13:30:00").to_pydatetime(),
                 "close": pd.Timestamp("2025-05-20 20:00:00").to_pydatetime(),
             }):
            stats = ref_node._compute_stats("AAPL", prior_sessions, date(2025, 5, 1), date(2025, 5, 20))
        assert stats is not None
        assert stats["avg_daily_volume_14d"] == pytest.approx(1_000_000)
        assert stats["daily_atr_14"] is not None
        assert stats["avg_opening_volume_14d"] == pytest.approx(50_000)

    def test_no_prior_sessions_returns_none(self):
        assert ref_node._compute_stats("AAPL", [], date(2025, 5, 1), date(2025, 5, 20)) is None

    def test_empty_daily_bars_returns_none(self):
        with patch.object(ref_node, "fetch_daily_bars", return_value=pd.DataFrame()):
            result = ref_node._compute_stats(
                "AAPL", [date(2025, 5, 1)], date(2025, 5, 1), date(2025, 5, 20)
            )
        assert result is None

    def test_missing_opening_bars_returns_none(self):
        prior_sessions = [date(2025, 5, d) for d in range(1, 5)]
        with patch.object(ref_node, "fetch_daily_bars", return_value=_daily_df(20)), \
             patch.object(ref_node, "fetch_intraday_bars", return_value=pd.DataFrame()), \
             patch.object(ref_node, "session_for", return_value={
                 "open": pd.Timestamp("2025-05-20 13:30:00").to_pydatetime(),
                 "close": pd.Timestamp("2025-05-20 20:00:00").to_pydatetime(),
             }):
            stats = ref_node._compute_stats("AAPL", prior_sessions, date(2025, 5, 1), date(2025, 5, 20))
        assert stats is None

    def test_no_look_ahead_daily_fetch_never_includes_as_of_session(self):
        """
        Regression test for the round-1 look-ahead bug: the daily reference
        fetch must request data strictly through the prior session, never
        including (or extending into) the session these stats are FOR.
        """
        as_of = date(2025, 5, 20)
        prior_sessions = [date(2025, 5, d) for d in range(6, 20)]  # 14 sessions, all < as_of
        captured = {}

        def fake_fetch_daily_bars(symbol, start, end):
            captured["start"] = start
            captured["end"] = end
            return _daily_df(20)

        with patch.object(ref_node, "fetch_daily_bars", side_effect=fake_fetch_daily_bars), \
             patch.object(ref_node, "fetch_intraday_bars", return_value=_opening_bar_df()), \
             patch.object(ref_node, "session_for", return_value={
                 "open": pd.Timestamp("2025-05-20 13:30:00").to_pydatetime(),
                 "close": pd.Timestamp("2025-05-20 20:00:00").to_pydatetime(),
             }):
            ref_node._compute_stats("AAPL", prior_sessions, prior_sessions[0], as_of)

        assert captured["end"] <= as_of
        assert all(s < as_of for s in prior_sessions)


class TestRunSkipsNonTradingDays:
    def test_run_skips_computation_on_non_trading_day(self):
        saturday = date(2025, 5, 24)
        with patch.object(ref_node, "is_trading_day", return_value=False), \
             patch.object(ref_node, "init_db"):
            result = ref_node.run(tickers=["AAPL"], as_of=saturday)
        assert result["computed"] == []
        assert result["skipped"] == ["AAPL"]


class TestUpsert:
    def test_upsert_updates_existing_row_instead_of_duplicating(self):
        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        existing = MagicMock()
        mock_session.exec.return_value.first.return_value = existing

        with patch.object(ref_node, "get_session", return_value=mock_cm), \
             patch.object(ref_node, "_get_ticker_id", return_value="ticker-1"):
            ref_node._upsert("AAPL", date(2025, 5, 20), {
                "avg_daily_volume_14d": 1_000_000.0,
                "daily_atr_14": 1.5,
                "avg_opening_volume_14d": 50_000.0,
            })

        assert existing.avg_daily_volume_14d == 1_000_000.0
        assert existing.daily_atr_14 == 1.5
        mock_session.add.assert_called_once_with(existing)
