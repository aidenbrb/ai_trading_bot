"""
Shared Alpaca historical-bars fetch helper for the day-trading (ORB) feature -
used by both the live intraday nodes (nodes/intraday_reference_node.py,
nodes/day_strategy_node.py) and the backtest/ package, so they can never
diverge on feed/adjustment choice.

Feed/adjustment rules - verified against the bot's own live Alpaca credentials,
not just documentation (see the day-trading-mode plan for the direct test):
  feed=delayed_sip -> ERROR {"message":"invalid feed: delayed_sip"}
  feed=sip         -> OK
  feed=iex         -> OK

  - Daily reference bars (14-day avg volume, daily ATR-14) always cover only
    PRIOR, fully-closed sessions -> feed=DataFeed.SIP (free-tier accessible
    since the data requested is always well past the 15-minute real-time
    restriction; DELAYED_SIP is not a valid request parameter at all despite
    existing in the SDK enum).
  - Today's live opening 5-min bar, and the 14-session opening-volume
    average used for the relative-volume ratio -> feed=DataFeed.IEX
    (real-time, free tier), kept IEX-vs-IEX for that specific ratio so a
    stable IEX opening-minutes market-share bias approximately cancels out.
  - adjustment=Adjustment.SPLIT for ALL historical bars (both SIP daily and
    IEX intraday) - today's live opening bar is unavoidably on the stock's
    CURRENT share basis, so only split-adjusted historical data lines up
    with it correctly. Raw/unadjusted data does not solve the split problem
    even if used consistently across both series.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

import pandas as pd

from config.settings import settings


def _client():
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
    )


def _clean_single(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "bar_time"
    return df[["open", "high", "low", "close", "volume"]].copy()


def fetch_bars(
    symbols,             # str for a single symbol, or a list[str] for a batch
    start: datetime,
    end: datetime,
    amount: int,
    unit: str,          # "Minute" | "Day"
    feed: str,           # "sip" | "iex"
):
    """
    Fetch split-adjusted historical bars in ONE Alpaca request, single- or
    multi-symbol. `symbols` as a list issues a single multi-symbol request
    (the SDK's `symbol_or_symbols` accepts a list) and returns
    `dict[symbol, DataFrame]` - this is what makes the day-trading backtest
    tractable (one request per session date across the whole universe,
    instead of one request per symbol per session - see backtest/data_cache.py
    and backtest/engine.py). `symbols` as a single string returns a plain
    DataFrame, unchanged from the original single-symbol API. Returns an
    empty DataFrame (or a dict of empty DataFrames) on no data - never raises
    for "no data," only for genuine request failures.
    """
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment, DataFeed

    is_batch = not isinstance(symbols, str)
    symbol_list = list(symbols) if is_batch else [symbols]

    request = StockBarsRequest(
        symbol_or_symbols=symbol_list,
        timeframe=TimeFrame(amount, getattr(TimeFrameUnit, unit)),
        start=start,
        end=end,
        feed=DataFeed(feed),
        adjustment=Adjustment.SPLIT,
    )
    bars = _client().get_stock_bars(request)
    df = bars.df

    if not is_batch:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol_list[0], level=0)
        return _clean_single(df)

    result = {s: pd.DataFrame() for s in symbol_list}
    if df is None or df.empty:
        return result
    if isinstance(df.index, pd.MultiIndex):
        present = df.index.get_level_values(0).unique()
        for s in present:
            if s in result:
                result[s] = _clean_single(df.xs(s, level=0))
    return result


def fetch_crypto_bars(
    symbols,
    start: datetime,
    end: datetime,
    amount: int,
    unit: str,  # "Minute" | "Hour" | "Day"
):
    """Fetch Alpaca-US crypto bars without any Yahoo/Coinbase fallback.

    Symbols use the bot's ``BTC-USD`` spelling at this boundary and are
    translated to Alpaca's ``BTC/USD`` spelling only for the request.
    """
    from alpaca.data.enums import CryptoFeed
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    is_batch = not isinstance(symbols, str)
    requested = list(symbols) if is_batch else [symbols]
    broker_symbols = [s.upper().replace("-USD", "/USD") for s in requested]
    reverse = {b: s for b, s in zip(broker_symbols, requested)}
    client = CryptoHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY or None,
        secret_key=settings.ALPACA_SECRET_KEY or None,
    )
    request = CryptoBarsRequest(
        symbol_or_symbols=broker_symbols,
        timeframe=TimeFrame(amount, getattr(TimeFrameUnit, unit)),
        start=start,
        end=end,
    )
    bars = client.get_crypto_bars(request, feed=CryptoFeed.US)
    df = bars.df

    result = {s: pd.DataFrame() for s in requested}
    if df is not None and not df.empty:
        if isinstance(df.index, pd.MultiIndex):
            present = df.index.get_level_values(0).unique()
            for broker_symbol in present:
                local_symbol = reverse.get(str(broker_symbol))
                if local_symbol:
                    result[local_symbol] = _clean_single(df.xs(broker_symbol, level=0))
        elif len(requested) == 1:
            result[requested[0]] = _clean_single(df)
    return result if is_batch else result[requested[0]]


def fetch_daily_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Prior-session daily bars via SIP, split-adjusted."""
    return fetch_bars(
        symbol,
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        amount=1, unit="Day", feed="sip",
    )


def fetch_intraday_bars(
    symbol: str, start: datetime, end: datetime, minutes: int = 5, feed: str = "iex"
) -> pd.DataFrame:
    """
    Intraday bars, split-adjusted. Defaults to IEX (real-time, free tier) -
    the live day-trading nodes always use the default. `feed="sip"` is only
    ever valid for historical (>15-min-old) requests, used by the
    backtester's IEX-vs-SIP ranking-overlap analysis (§7 decision gate) -
    never by anything running against today's live opening bar.
    """
    return fetch_bars(symbol, start=start, end=end, amount=minutes, unit="Minute", feed=feed)


def first_regular_session_bar(df: pd.DataFrame, session_open_utc: datetime) -> Optional[pd.Series]:
    """
    The first bar at or after the session's true regular-hours open (per
    utils/market_calendar.session_for) - guards against a feed returning
    pre-market activity as the "first" bar. Returns None if no bar exists at
    or after the open (missing/incomplete data for that symbol/session).
    """
    if df is None or df.empty:
        return None
    regular = df[df.index >= session_open_utc.replace(tzinfo=None)]
    if regular.empty:
        return None
    return regular.iloc[0]
