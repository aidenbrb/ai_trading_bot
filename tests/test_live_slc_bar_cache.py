import pandas as pd
import pytest

import live_slc.bar_cache as bar_cache
import live_slc.models as models
from live_slc.bar_cache import (
    BAR_POLL_DEADLINE_SECONDS,
    _default_fetch,
    _is_regular_trading_hours,
    backfill_gaps,
    fetch_expected_bar_batch,
    last_cached_bar_time,
    persist_bars,
)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def test_bar_poll_deadline_is_15_seconds():
    assert BAR_POLL_DEADLINE_SECONDS == 15.0


def test_fetch_expected_bar_batch_is_one_call_across_all_symbols_not_per_symbol():
    call_count = {"n": 0}
    expected = pd.Timestamp("2026-08-13 13:35:00")

    def fetch(symbols, start, end):
        call_count["n"] += 1
        assert len(symbols) > 1  # a single batched call, not one per symbol
        return {s: pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}, index=[expected]) for s in symbols}

    found, missing = fetch_expected_bar_batch(["AAPL", "MSFT", "NVDA"], expected, fetch_fn=fetch, deadline_seconds=1, retry_interval_seconds=0.1)
    assert call_count["n"] == 1
    assert set(found.keys()) == {"AAPL", "MSFT", "NVDA"}
    assert missing == []


def test_fetch_expected_bar_batch_only_accepts_exact_expected_timestamp():
    expected = pd.Timestamp("2026-08-13 13:35:00")
    wrong_time = pd.Timestamp("2026-08-13 13:30:00")

    def fetch(symbols, start, end):
        return {s: pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [1.0]}, index=[wrong_time]) for s in symbols}

    found, missing = fetch_expected_bar_batch(["AAPL"], expected, fetch_fn=fetch, deadline_seconds=0.3, retry_interval_seconds=0.1)
    assert found == {}
    assert missing == ["AAPL"]


def test_fetch_expected_bar_batch_records_missing_at_deadline():
    expected = pd.Timestamp("2026-08-13 13:35:00")

    def fetch(symbols, start, end):
        return {s: pd.DataFrame(columns=["open", "high", "low", "close", "volume"]) for s in symbols}

    found, missing = fetch_expected_bar_batch(["AAPL", "MSFT"], expected, fetch_fn=fetch, deadline_seconds=0.3, retry_interval_seconds=0.1)
    assert found == {}
    assert set(missing) == {"AAPL", "MSFT"}


def test_persist_bars_never_creates_duplicates():
    ts = pd.Timestamp("2026-08-13 13:35:00")
    row = pd.Series({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert persist_bars("AAPL", {ts: row}) == 1
    assert persist_bars("AAPL", {ts: row}) == 0
    assert last_cached_bar_time("AAPL") == ts


def test_backfill_gaps_fetches_everything_since_last_cached_bar():
    first = pd.Timestamp("2026-08-13 13:30:00")
    row = pd.Series({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    persist_bars("AAPL", {first: row})

    def fetch(symbols, start, end):
        idx = pd.date_range(start, end, freq="5min")
        return {s: pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx) for s in symbols}

    through = first + pd.Timedelta(minutes=15)
    result = backfill_gaps(["AAPL"], through, fetch_fn=fetch)
    assert result["AAPL"] == 3  # +5, +10, +15 minutes - not re-fetching `first`
    assert last_cached_bar_time("AAPL") == through


def test_is_regular_trading_hours_accepts_session_open_rejects_premarket_and_weekend():
    assert _is_regular_trading_hours(pd.Timestamp("2026-08-13 13:30:00")) is True  # 9:30am ET open
    assert _is_regular_trading_hours(pd.Timestamp("2026-08-13 19:59:00")) is True  # last bar before 4pm ET close
    assert _is_regular_trading_hours(pd.Timestamp("2026-08-13 13:25:00")) is False  # pre-market
    assert _is_regular_trading_hours(pd.Timestamp("2026-08-13 20:00:00")) is False  # at/after close
    assert _is_regular_trading_hours(pd.Timestamp("2026-08-15 14:00:00")) is False  # Saturday


def test_default_fetch_drops_extended_hours_rows_from_the_broker_response(monkeypatch):
    premarket = pd.Timestamp("2026-08-13 09:00:00")
    regular = pd.Timestamp("2026-08-13 13:30:00")
    afterhours = pd.Timestamp("2026-08-13 21:00:00")

    def fake_fetch_bars(symbols, start, end, amount, unit, feed):
        idx = [premarket, regular, afterhours]
        frame = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)
        return {s: frame for s in symbols}

    monkeypatch.setattr(bar_cache, "fetch_bars", fake_fetch_bars)
    result = _default_fetch(["AAPL"], pd.Timestamp("2026-08-13 08:00:00"), pd.Timestamp("2026-08-13 22:00:00"))
    assert list(result["AAPL"].index) == [regular]


def test_persist_bars_defense_in_depth_drops_extended_hours_bar_even_without_default_fetch():
    """A custom fetch_fn (as callers may supply) that itself forgot to
    filter must still be blocked at the DB-write boundary."""
    premarket = pd.Timestamp("2026-08-13 09:00:00")
    row = pd.Series({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert persist_bars("AAPL", {premarket: row}) == 0
    assert last_cached_bar_time("AAPL") is None
