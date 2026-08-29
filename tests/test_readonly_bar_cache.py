import sqlite3
from datetime import datetime

import pytest

import backtest.readonly_bar_cache as readonly_bar_cache
from backtest.readonly_bar_cache import read_cached_bars_or_none


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
        ("AAPL", "research-stock-sip-1Minute", datetime(2026, 6, 1, 13, 30), 100, 101, 99, 100.5, 1000),
        ("AAPL", "research-stock-sip-1Minute", datetime(2026, 6, 1, 13, 31), 100.5, 102, 100, 101.5, 1000),
    ]
    ranges = [
        ("AAPL", "research-stock-sip-1Minute", datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32)),
    ]
    _make_cache_db(db_path, rows, ranges)
    monkeypatch.setattr(readonly_bar_cache, "_CACHE_PATH", db_path)
    return db_path


def test_fully_covered_range_returns_cached_bars(cache_db):
    result = read_cached_bars_or_none(
        "AAPL", "research-stock-sip-1Minute",
        datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32),
    )
    assert result is not None
    assert len(result) == 2


def test_partially_covered_range_returns_none(cache_db):
    result = read_cached_bars_or_none(
        "AAPL", "research-stock-sip-1Minute",
        datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 14, 0),
    )
    assert result is None


def test_unknown_symbol_returns_none(cache_db):
    result = read_cached_bars_or_none(
        "MSFT", "research-stock-sip-1Minute",
        datetime(2026, 6, 1, 13, 30), datetime(2026, 6, 1, 13, 32),
    )
    assert result is None


def test_connection_is_read_only(cache_db):
    conn = readonly_bar_cache._readonly_connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO bars VALUES ('X','x','2020-01-01',1,1,1,1,1)"
            )
    finally:
        conn.close()


def test_never_imports_network_capable_fetchers():
    module_names = vars(readonly_bar_cache)
    assert "fetch_bars" not in module_names
    assert "fetch_crypto_bars" not in module_names
    for name, value in module_names.items():
        if name.startswith("__"):
            continue
        assert getattr(value, "__module__", "") != "utils.alpaca_bars", name
