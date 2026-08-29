"""Tests for utils/market_calendar.py against known NYSE calendar dates."""
from datetime import date

from utils.market_calendar import is_early_close, is_trading_day, session_for, trading_days_between


def test_ordinary_weekday_is_trading_day_not_early_close():
    day = date(2025, 6, 2)  # ordinary Monday
    assert is_trading_day(day)
    assert not is_early_close(day)


def test_day_after_thanksgiving_is_early_close():
    day = date(2025, 11, 28)
    assert is_trading_day(day)
    assert is_early_close(day)


def test_christmas_eve_is_early_close():
    day = date(2025, 12, 24)
    assert is_trading_day(day)
    assert is_early_close(day)


def test_july_third_is_early_close():
    day = date(2025, 7, 3)
    assert is_trading_day(day)
    assert is_early_close(day)


def test_christmas_day_is_full_holiday():
    day = date(2025, 12, 25)
    assert not is_trading_day(day)
    assert session_for(day) is None
    assert not is_early_close(day)


def test_saturday_is_not_a_trading_day():
    assert not is_trading_day(date(2025, 6, 7))


def test_session_for_returns_open_and_close():
    session = session_for(date(2025, 6, 2))
    assert session["open"] < session["close"]
    assert session["open"].hour == 13  # 9:30am ET = 13:30 UTC in EDT


def test_trading_days_between_excludes_weekends_and_holidays():
    days = trading_days_between(date(2025, 6, 2), date(2025, 6, 8))  # Mon-Sun
    assert date(2025, 6, 2) in days
    assert date(2025, 6, 6) in days  # Friday
    assert date(2025, 6, 7) not in days  # Saturday
    assert date(2025, 6, 8) not in days  # Sunday
    assert len(days) == 5


def test_session_for_returns_naive_datetimes():
    """
    Regression test: session_for() must return naive (no tzinfo) datetimes,
    matching this codebase's naive-UTC convention everywhere else. A
    tz-aware return value here previously caused SQLite's text-based date
    comparisons in backtest/data_cache.py to silently exclude the 9:30
    opening bar (see utils/market_calendar.py::session_for's docstring).
    """
    session = session_for(date(2025, 6, 2))
    assert session["open"].tzinfo is None
    assert session["close"].tzinfo is None
