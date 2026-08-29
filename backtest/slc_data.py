"""Market-data adapters for the research-only SLC backtest.

``cache-only`` opens the dedicated research cache read-only and reports a
coverage miss without fetching. ``fetch`` may call Alpaca's historical-data
API and populate that cache. Neither path imports an order/trading client.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from utils.alpaca_bars import fetch_bars

SLC_CACHE_PATH = Path(__file__).parent / "cache" / "slc_bars_cache.db"


def interval_name(minutes: int) -> str:
    return f"research-stock-sip-{minutes}Minute"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex([], name="bar_time"),
    )


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    present = [part for part in parts if part is not None and not part.empty]
    if not present:
        return _empty()
    frame = pd.concat(present).sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def _normalise_boundary(value: datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.to_pydatetime()


def _connect(*, readonly: bool) -> sqlite3.Connection | None:
    if readonly:
        if not SLC_CACHE_PATH.exists():
            return None
        return sqlite3.connect(f"file:/{SLC_CACHE_PATH.resolve().as_posix()}?mode=ro", uri=True)
    SLC_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(SLC_CACHE_PATH)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol TEXT NOT NULL, interval TEXT NOT NULL, bar_time TEXT NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            PRIMARY KEY (symbol, interval, bar_time)
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS fetched_ranges (
            symbol TEXT NOT NULL, interval TEXT NOT NULL,
            start TEXT NOT NULL, end TEXT NOT NULL
        )
    """)
    return connection


def _covered(connection: sqlite3.Connection, symbol: str, interval: str,
             start: datetime, end: datetime) -> bool:
    rows = connection.execute(
        "SELECT start,end FROM fetched_ranges WHERE symbol=? AND interval=?",
        (symbol, interval),
    ).fetchall()
    ranges = sorted(
        (_normalise_boundary(pd.Timestamp(a)), _normalise_boundary(pd.Timestamp(b)))
        for a, b in rows
    )
    merged: list[list[datetime]] = []
    for left, right in ranges:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return any(left <= start and right >= end for left, right in merged)


def _read(connection: sqlite3.Connection, symbol: str, interval: str,
          start: datetime, end: datetime) -> pd.DataFrame:
    frame = pd.read_sql_query(
        "SELECT bar_time,open,high,low,close,volume FROM bars "
        "WHERE symbol=? AND interval=? AND bar_time>=? AND bar_time<? ORDER BY bar_time",
        connection, params=(symbol, interval, str(start), str(end)),
    )
    if frame.empty:
        return _empty()
    frame["bar_time"] = pd.to_datetime(frame["bar_time"], format="mixed")
    return frame.set_index("bar_time")


def _read_or_none(symbol: str, interval: str, start: datetime,
                  end: datetime) -> pd.DataFrame | None:
    connection = _connect(readonly=True)
    if connection is None:
        return None
    try:
        if not _covered(connection, symbol, interval, start, end):
            return None
        return _read(connection, symbol, interval, start, end)
    finally:
        connection.close()


def _fetch_cached(symbols: list[str], interval: str, start: datetime,
                  end: datetime, *, minutes: int) -> dict[str, pd.DataFrame]:
    connection = _connect(readonly=False)
    try:
        needed = [symbol for symbol in symbols if not _covered(
            connection, symbol, interval, start, end
        )]
        if needed:
            fresh = fetch_bars(
                needed, start, end, amount=minutes, unit="Minute", feed="sip"
            )
            with connection:
                for symbol in needed:
                    frame = fresh.get(symbol, _empty())
                    if frame is not None and not frame.empty:
                        connection.executemany(
                            "INSERT OR REPLACE INTO bars "
                            "(symbol,interval,bar_time,open,high,low,close,volume) "
                            "VALUES (?,?,?,?,?,?,?,?)",
                            [
                                (symbol, interval, str(timestamp), row.open, row.high,
                                 row.low, row.close, row.volume)
                                for timestamp, row in frame.iterrows()
                            ],
                        )
                    connection.execute(
                        "INSERT INTO fetched_ranges (symbol,interval,start,end) VALUES (?,?,?,?)",
                        (symbol, interval, str(start), str(end)),
                    )
        return {symbol: _read(connection, symbol, interval, start, end) for symbol in symbols}
    finally:
        connection.close()


def _windows(start: datetime, end: datetime, days: int = 28):
    cursor = start
    while cursor < end:
        boundary = min(cursor + timedelta(days=days), end)
        yield cursor, boundary
        cursor = boundary


def load_stock_bars(
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    minutes: int,
    mode: str,
    batch_size: int = 20,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Load split-adjusted SIP bars and preserve cache misses explicitly."""
    if mode not in {"cache-only", "fetch"}:
        raise ValueError("mode must be 'cache-only' or 'fetch'")
    unique = sorted(set(symbol.upper() for symbol in symbols))
    output: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in unique}
    misses: list[dict] = []
    interval = interval_name(minutes)
    windows = list(_windows(start, end))
    total = len(windows) * max(1, (len(unique) + batch_size - 1) // batch_size)
    done = 0
    for window_start, window_end in windows:
        for offset in range(0, len(unique), batch_size):
            batch = unique[offset : offset + batch_size]
            done += 1
            if progress:
                progress(
                    f"data {done}/{total}: {window_start.date()}..{window_end.date()} "
                    f"({len(batch)} symbols, {minutes}m, {mode})"
                )
            if mode == "fetch":
                loaded = _fetch_cached(
                    batch, interval, window_start, window_end, minutes=minutes
                )
            else:
                loaded = {}
                for symbol in batch:
                    frame = _read_or_none(symbol, interval, window_start, window_end)
                    if frame is None:
                        misses.append({
                            "symbol": symbol,
                            "interval": interval,
                            "start": window_start,
                            "end": window_end,
                        })
                        frame = _empty()
                    loaded[symbol] = frame
            for symbol in batch:
                frame = loaded.get(symbol)
                if frame is not None and not frame.empty:
                    output[symbol].append(frame)
    return {symbol: _concat(parts) for symbol, parts in output.items()}, misses


def load_session_minutes(
    symbols: list[str],
    start: datetime,
    end: datetime,
    *,
    mode: str,
) -> tuple[dict[str, pd.DataFrame | None], list[dict]]:
    """Load a single outcome window, keeping unknown distinct from empty."""
    unique = sorted(set(symbols))
    interval = interval_name(1)
    misses: list[dict] = []
    if mode == "fetch":
        frames = _fetch_cached(unique, interval, start, end, minutes=1)
        return frames, misses
    frames: dict[str, pd.DataFrame | None] = {}
    for symbol in unique:
        frame = _read_or_none(symbol, interval, start, end)
        if frame is None:
            misses.append({"symbol": symbol, "interval": interval, "start": start, "end": end})
        frames[symbol] = frame
    return frames, misses
