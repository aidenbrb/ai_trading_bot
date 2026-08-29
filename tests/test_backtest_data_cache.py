"""
Tests for backtest/data_cache.py - cache-first, multi-symbol batched,
transactional Alpaca bar fetching.
"""
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

import backtest.data_cache as cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "_CACHE_PATH", tmp_path / "bars_cache.db")
    yield


def _df(rows):
    """rows: list of (timestamp_str, price, volume) -> a single-symbol DataFrame."""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0]) for r in rows])
    prices = [r[1] for r in rows]
    return pd.DataFrame(
        {"open": prices, "high": prices, "low": prices, "close": prices,
         "volume": [r[2] for r in rows]},
        index=idx,
    )


class TestCacheFirstBehavior:
    """get_intraday_bars/get_daily_bars are single-symbol wrappers over the
    multi-symbol batch functions - fetch_bars is mocked with the real
    dict[symbol, DataFrame] contract those batch functions actually use."""

    def test_first_fetch_hits_the_network(self):
        remote = {"AAPL": _df([("2025-06-02 13:30", 10.0, 1000)])}
        with patch.object(cache, "fetch_bars", return_value=remote) as mock_fetch:
            result = cache.get_intraday_bars(
                "AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5
            )
        assert mock_fetch.called
        assert len(result) == 1

    def test_second_fetch_of_same_range_does_not_hit_the_network(self):
        remote = {"AAPL": _df([("2025-06-02 13:30", 10.0, 1000)])}
        with patch.object(cache, "fetch_bars", return_value=remote) as mock_fetch:
            cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
            assert mock_fetch.call_count == 1
            cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
            assert mock_fetch.call_count == 1  # still 1 - served from cache

    def test_wider_range_triggers_a_refetch(self):
        remote1 = {"AAPL": _df([("2025-06-02 13:30", 10.0, 1000)])}
        remote2 = {"AAPL": _df([("2025-06-02 13:30", 10.0, 1000), ("2025-06-02 13:35", 10.1, 1100)])}
        with patch.object(cache, "fetch_bars", side_effect=[remote1, remote2]) as mock_fetch:
            cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
            result = cache.get_intraday_bars(
                "AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 40), minutes=5
            )
        assert mock_fetch.call_count == 2
        assert len(result) == 2

    def test_different_symbols_cached_independently(self):
        def fake_fetch(symbols, *a, **kw):
            return {s: _df([("2025-06-02 13:30", 10.0, 1000)]) for s in symbols}

        with patch.object(cache, "fetch_bars", side_effect=fake_fetch) as mock_fetch:
            cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
            cache.get_intraday_bars("MSFT", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
        assert mock_fetch.call_count == 2

    def test_empty_result_from_source_returns_empty_frame(self):
        with patch.object(cache, "fetch_bars", return_value={"HALTED": pd.DataFrame()}):
            result = cache.get_intraday_bars("HALTED", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35))
        assert result.empty


class TestMultiSymbolBatching:
    def test_one_batch_request_fetches_multiple_uncovered_symbols(self):
        def fake_fetch(symbols, *a, **kw):
            return {s: _df([("2025-06-02 13:30", 10.0, 1000)]) for s in symbols}

        with patch.object(cache, "fetch_bars", side_effect=fake_fetch) as mock_fetch:
            result = cache.get_intraday_bars_multi(
                ["AAPL", "MSFT", "TSLA"], datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5
            )
        assert mock_fetch.call_count == 1  # one request for all three, not three
        assert set(mock_fetch.call_args.args[0]) == {"AAPL", "MSFT", "TSLA"}
        assert set(result.keys()) == {"AAPL", "MSFT", "TSLA"}

    def test_already_covered_symbols_excluded_from_the_batch_request(self):
        def fake_fetch(symbols, *a, **kw):
            return {s: _df([("2025-06-02 13:30", 10.0, 1000)]) for s in symbols}

        with patch.object(cache, "fetch_bars", side_effect=fake_fetch) as mock_fetch:
            cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
            cache.get_intraday_bars_multi(
                ["AAPL", "MSFT"], datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5
            )
        # Second call: AAPL already covered, only MSFT should be requested.
        assert mock_fetch.call_args.args[0] == ["MSFT"]

    def test_successful_response_missing_one_symbol_marks_it_covered_not_retried(self):
        """A successful batch response with no bar for one symbol (halted,
        no trades) is legitimate missing data - not a failure - and must not
        be retried on a later call."""
        call_count = {"n": 0}

        def fake_fetch(symbols, *a, **kw):
            call_count["n"] += 1
            return {s: (_df([("2025-06-02 13:30", 10.0, 1000)]) if s != "HALTED" else pd.DataFrame()) for s in symbols}

        with patch.object(cache, "fetch_bars", side_effect=fake_fetch):
            cache.get_intraday_bars_multi(
                ["AAPL", "HALTED"], datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5
            )
            result = cache.get_intraday_bars_multi(
                ["AAPL", "HALTED"], datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5
            )
        assert call_count["n"] == 1  # second call hit cache for BOTH symbols, including the empty one
        assert result["HALTED"].empty


class TestTransactionalSafety:
    def test_raising_fetcher_leaves_no_rows_and_no_coverage(self):
        with patch.object(cache, "fetch_bars", side_effect=RuntimeError("network error")):
            with pytest.raises(RuntimeError):
                cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)

        conn = cache._connect()
        try:
            bar_count = conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
            range_count = conn.execute("SELECT COUNT(*) FROM fetched_ranges").fetchone()[0]
        finally:
            conn.close()
        assert bar_count == 0
        assert range_count == 0

    def test_resumed_run_retries_after_a_failed_fetch(self):
        with patch.object(cache, "fetch_bars", side_effect=RuntimeError("network error")):
            with pytest.raises(RuntimeError):
                cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)

        with patch.object(cache, "fetch_bars", return_value={"AAPL": _df([("2025-06-02 13:30", 10.0, 1000)])}) as mock_fetch:
            result = cache.get_intraday_bars("AAPL", datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 35), minutes=5)
        assert mock_fetch.called  # retried, not silently skipped as "already covered"
        assert len(result) == 1


class TestNormalizeAndMergeCoverage:
    def test_sparse_ranges_do_not_falsely_cover_a_wider_window(self):
        """
        The unsafe design this replaces: 'any overlap = covered'. Two sparse,
        narrow, non-adjacent ranges inside a wider window must NOT be
        treated as covering that wider window.
        """
        conn = cache._connect()
        try:
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-02 13:30:00", "2025-06-02 13:35:00"),
            )
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-03 13:30:00", "2025-06-03 13:35:00"),
            )
            conn.commit()
            merged = cache._merged_ranges(conn, "AAPL", "5Min")
            covered = cache._fully_covered(
                merged, datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 3, 13, 35)
            )
        finally:
            conn.close()
        assert covered is False

    def test_adjacent_ranges_merge_into_one_covering_interval(self):
        conn = cache._connect()
        try:
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-02 13:30:00", "2025-06-02 13:35:00"),
            )
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-02 13:35:00", "2025-06-02 13:40:00"),
            )
            conn.commit()
            merged = cache._merged_ranges(conn, "AAPL", "5Min")
            covered = cache._fully_covered(
                merged, datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 40)
            )
        finally:
            conn.close()
        assert len(merged) == 1
        assert covered is True

    def test_old_tz_aware_format_normalizes_and_still_merges_correctly(self):
        """Old rows may exist as '...+00:00' (the pre-fix format) - must
        normalize and merge correctly alongside new naive-UTC rows."""
        conn = cache._connect()
        try:
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-02 13:30:00+00:00", "2025-06-02 13:35:00+00:00"),
            )
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?,?,?,?)",
                ("AAPL", "5Min", "2025-06-02 13:35:00", "2025-06-02 13:40:00"),
            )
            conn.commit()
            merged = cache._merged_ranges(conn, "AAPL", "5Min")
            covered = cache._fully_covered(
                merged, datetime(2025, 6, 2, 13, 30), datetime(2025, 6, 2, 13, 40)
            )
        finally:
            conn.close()
        assert len(merged) == 1
        assert covered is True
