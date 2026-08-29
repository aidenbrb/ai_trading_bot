"""Tests for backtest/v2_readonly_adapters.py.

read_cached_bars_or_none has three possible results and this module must
keep them distinct: None (not fully cached - unknown) must become
CacheCoverageError, never a silently-substituted empty frame; an empty
DataFrame (fully cached, legitimately no bars) and a nonempty DataFrame
must both pass through unchanged.
"""
import sqlite3
from datetime import datetime

import pandas as pd
import pytest

import backtest.readonly_bar_cache as readonly_bar_cache
from backtest.v2_readonly_adapters import (
    CacheCoverageError,
    _read_only_symbol,
    make_hourly_fetcher,
    make_minute_fetcher,
)

INTERVAL = "research-stock-sip-1Minute"


def _make_cache_db(path, rows, ranges):
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE bars (
            symbol TEXT, interval TEXT, bar_time TEXT,
            open REAL, high REAL, low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, interval, bar_time)
        )
    """)
    conn.execute("""
        CREATE TABLE fetched_ranges (
            symbol TEXT, interval TEXT, start TEXT, end TEXT
        )
    """)
    conn.executemany(
        "INSERT INTO bars VALUES (?,?,?,?,?,?,?,?)",
        [(s, i, str(t), o, h, l, c, v) for s, i, t, o, h, l, c, v in rows],
    )
    conn.executemany(
        "INSERT INTO fetched_ranges VALUES (?,?,?,?)",
        [(s, i, str(a), str(b)) for s, i, a, b in ranges],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def cache_db(tmp_path, monkeypatch):
    db_path = tmp_path / "bars_cache.db"
    rows = [
        ("AAPL", INTERVAL, datetime(2026, 6, 1, 13, 30), 100, 101, 99, 100.5, 1000),
        ("AAPL", INTERVAL, datetime(2026, 6, 1, 13, 31), 100.5, 102, 100, 101.5, 1000),
    ]
    ranges = [
        ("AAPL", INTERVAL, datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32)),
        # MSFT is fully covered but genuinely has no bars in this window -
        # a known non-event, distinct from AAPL's coverage over the same
        # window and from an uncovered/unknown symbol like SPY below.
        ("MSFT", INTERVAL, datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32)),
    ]
    _make_cache_db(db_path, rows, ranges)
    monkeypatch.setattr(readonly_bar_cache, "_CACHE_PATH", db_path)
    return db_path


WINDOW = (datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32))


def test_covered_with_bars_passes_through_unchanged(cache_db):
    frame = _read_only_symbol("AAPL", INTERVAL, *WINDOW)
    assert len(frame) == 2


def test_covered_with_no_bars_is_empty_not_an_error(cache_db):
    frame = _read_only_symbol("MSFT", INTERVAL, *WINDOW)
    assert frame is not None
    assert frame.empty


def test_uncovered_raises_cache_coverage_error(cache_db):
    with pytest.raises(CacheCoverageError):
        _read_only_symbol("SPY", INTERVAL, *WINDOW)


def test_minute_fetcher_propagates_coverage_error_and_logs_miss(cache_db):
    misses = []
    fetch = make_minute_fetcher("stock", on_miss=misses.append)
    with pytest.raises(CacheCoverageError):
        fetch(["SPY"], WINDOW[0], WINDOW[1], amount=1, unit="Minute")
    assert len(misses) == 1
    assert misses[0]["symbol"] == "SPY"
    assert misses[0]["market"] == "stock"


def test_minute_fetcher_returns_covered_bars_without_logging_a_miss(cache_db):
    misses = []
    fetch = make_minute_fetcher("stock", on_miss=misses.append)
    result = fetch(["AAPL"], WINDOW[0], WINDOW[1], amount=1, unit="Minute")
    assert len(result["AAPL"]) == 2
    assert misses == []


def test_minute_fetcher_returns_covered_empty_without_raising_or_logging(cache_db):
    misses = []
    fetch = make_minute_fetcher("stock", on_miss=misses.append)
    result = fetch(["MSFT"], WINDOW[0], WINDOW[1], amount=1, unit="Minute")
    assert result["MSFT"].empty
    assert misses == []


def test_hourly_fetcher_catches_coverage_error_returns_empty_and_logs_miss(cache_db):
    misses = []
    fetch = make_hourly_fetcher("stock", on_miss=misses.append)
    result = fetch(["AAPL", "SPY"], WINDOW[0], WINDOW[1], amount=1, unit="Minute")
    assert len(result["AAPL"]) == 2
    assert result["SPY"].empty
    assert len(misses) == 1
    assert misses[0]["symbol"] == "SPY"


def test_never_imports_network_capable_fetchers():
    import backtest.v2_readonly_adapters as adapters
    module_names = vars(adapters)
    assert "fetch_bars" not in module_names
    assert "fetch_crypto_bars" not in module_names
    assert "get_stock_research_bars_multi" not in module_names
    assert "get_crypto_research_bars_multi" not in module_names
    for name, value in module_names.items():
        if name.startswith("__"):
            continue
        assert getattr(value, "__module__", "") != "utils.alpaca_bars", name
