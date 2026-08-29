"""
NYSE trading-session calendar - session open/close times, including early
closes and holidays.

Used by the day-trading (ORB) backtester and intraday nodes for correctly
identifying the 9:30 ET session open and the actual close time (which is not
always 4:00pm ET - half-days like the day after Thanksgiving close at 1:00pm).

Deliberately separate from utils/market_session.py, which only does the
weekday(stocks)/weekend(crypto) routing switch for the existing swing
pipeline - unrelated concern, not touched here.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Optional

import pandas_market_calendars as mcal

_NYSE = mcal.get_calendar("XNYS")


@lru_cache(maxsize=8)
def _schedule_for_year(year: int):
    return _NYSE.schedule(start_date=f"{year}-01-01", end_date=f"{year}-12-31")


def session_for(day: date) -> Optional[dict]:
    """
    Return {"open": datetime, "close": datetime} in naive UTC for a trading
    day, or None if the market is closed that day (weekend or holiday).

    Naive (tz stripped), matching this codebase's naive-UTC convention
    everywhere else (see db/models.py's own "UTC, naive" comments) -
    pandas_market_calendars returns tz-aware UTC Timestamps, and mixing
    those with naive datetimes elsewhere caused a real bug: SQLite's
    text-based date comparisons in backtest/data_cache.py silently excluded
    the 9:30 opening bar because "...13:30:00" (naive) sorts lexicographically
    before "...13:30:00+00:00" (aware).
    """
    schedule = _schedule_for_year(day.year)
    ts = str(day)
    if ts not in schedule.index.strftime("%Y-%m-%d"):
        return None
    row = schedule.loc[schedule.index.strftime("%Y-%m-%d") == ts].iloc[0]
    return {
        "open":  row["market_open"].to_pydatetime().replace(tzinfo=None),
        "close": row["market_close"].to_pydatetime().replace(tzinfo=None),
    }


def is_trading_day(day: date) -> bool:
    return session_for(day) is not None


def is_early_close(day: date) -> bool:
    """
    True if the session closes well before the normal 4:00pm ET close.
    Normal close is 20:00 UTC (EDT) or 21:00 UTC (EST); NYSE half-days
    (day after Thanksgiving, Christmas/July 3rd eve) close at ~1:00pm ET,
    i.e. 17:00-18:00 UTC - comfortably below this threshold either way.
    """
    session = session_for(day)
    if session is None:
        return False
    return session["close"].hour < 19


def prior_trading_days(before: date, count: int) -> list[date]:
    """The `count` most recent NYSE trading days strictly before `before`."""
    from datetime import timedelta

    schedule = _NYSE.schedule(
        start_date=str(before - timedelta(days=count * 3 + 20)),
        end_date=str(before - timedelta(days=1)),
    )
    sessions = [ts.date() for ts in schedule.index]
    return sessions[-count:] if sessions else []


def trading_days_between(start: date, end: date) -> list[date]:
    """All NYSE trading days in [start, end], inclusive of both ends."""
    schedule = _NYSE.schedule(start_date=str(start), end_date=str(end))
    return [ts.date() for ts in schedule.index]


def next_trading_day(after: date) -> date:
    schedule = _NYSE.schedule(
        start_date=str(after), end_date=f"{after.year + 1}-12-31"
    )
    for ts in schedule.index:
        d = ts.date()
        if d > after:
            return d
    raise ValueError(f"no trading day found after {after}")
