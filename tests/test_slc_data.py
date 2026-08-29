"""Tests that SLC data uses its own transactional cache."""
from datetime import datetime

import pandas as pd
import pytest

import backtest.slc_data as data


def _frame(start):
    index = pd.date_range(start, periods=2, freq="5min")
    return pd.DataFrame({
        "open": [10.0, 10.1], "high": [10.2, 10.3], "low": [9.9, 10.0],
        "close": [10.1, 10.2], "volume": [100.0, 200.0],
    }, index=index)


def test_fetch_then_cache_only_reuses_isolated_cache(tmp_path, monkeypatch):
    cache = tmp_path / "slc.db"
    monkeypatch.setattr(data, "SLC_CACHE_PATH", cache)
    calls = []
    def fetch(symbols, start, end, amount, unit, feed):
        calls.append((symbols, amount, unit, feed))
        return {symbol: _frame(start) for symbol in symbols}
    monkeypatch.setattr(data, "fetch_bars", fetch)
    start, end = datetime(2025, 6, 2), datetime(2025, 6, 3)
    first, misses = data.load_stock_bars(
        ["AAPL"], start, end, minutes=5, mode="fetch"
    )
    assert cache.exists()
    assert len(first["AAPL"]) == 2
    assert misses == []
    assert calls == [(["AAPL"], 5, "Minute", "sip")]

    monkeypatch.setattr(data, "fetch_bars", lambda *a, **k: pytest.fail("network called"))
    second, misses = data.load_stock_bars(
        ["AAPL"], start, end, minutes=5, mode="cache-only"
    )
    assert len(second["AAPL"]) == 2
    assert misses == []


def test_cache_only_missing_range_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SLC_CACHE_PATH", tmp_path / "absent.db")
    frames, misses = data.load_stock_bars(
        ["AAPL"], datetime(2025, 6, 2), datetime(2025, 6, 3),
        minutes=5, mode="cache-only",
    )
    assert frames["AAPL"].empty
    assert len(misses) == 1


def test_failed_fetch_writes_no_false_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "SLC_CACHE_PATH", tmp_path / "slc.db")
    def fail(*args, **kwargs):
        raise RuntimeError("source unavailable")
    monkeypatch.setattr(data, "fetch_bars", fail)
    start, end = datetime(2025, 6, 2), datetime(2025, 6, 3)
    with pytest.raises(RuntimeError):
        data.load_stock_bars(["AAPL"], start, end, minutes=5, mode="fetch")
    _, misses = data.load_stock_bars(
        ["AAPL"], start, end, minutes=5, mode="cache-only"
    )
    assert len(misses) == 1
