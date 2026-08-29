"""
Local bar cache for the backtester - a dedicated SQLite file
(backtest/cache/bars_cache.db), completely separate from trading.db, so a
broken backtest run can never touch the live pipeline's database.

Coverage tracking: `fetched_ranges(symbol, interval, start, end)` records
which [start, end) windows have been fetched. A candidate window counts as
"already fetched" only when it is FULLY CONTAINED in the MERGED union of a
symbol's recorded ranges - not merely overlapping one. The cache's actual
bar data is sparse (early fetches only ever pulled narrow 5-minute opening
windows), so "any overlap" would falsely mark large uncovered spans as
complete; only exact containment in the merged coverage is safe. Existing
rows may be in an old, pre-fix tz-aware string format
("...13:30:00+00:00") - `_parse_and_normalize` handles both.

Batching: `get_daily_bars_multi`/`get_intraday_bars_multi` issue ONE Alpaca
request (via utils/alpaca_bars.py::fetch_bars's multi-symbol support) for
every symbol not yet covered, rather than one request per symbol - this is
the fix for the fetch-batching defect (see the day-trading-mode plan).

Transactional safety: a batch's bar rows and coverage records are written
together in one SQLite transaction, committed only after the fetch request
itself has succeeded. A request-level failure (exception/timeout) leaves
nothing written, so a resumed run correctly retries. A SUCCESSFUL response
that simply has no bar for one symbol (halted, no trades that session) is
NOT a failure - that symbol's coverage is still marked complete (as
"checked, no data") in the same transaction, so it is never endlessly
retried.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from utils.alpaca_bars import fetch_bars, fetch_crypto_bars

_CACHE_PATH = Path(__file__).parent / "cache" / "bars_cache.db"


def _strip_tz(dt: datetime) -> datetime:
    """Defensive normalization - belt-and-suspenders on top of session_for()'s
    own fix, so a future tz-aware caller can't silently reintroduce the
    lexicographic-string-comparison bug this cache layer once had."""
    return dt.replace(tzinfo=None) if getattr(dt, "tzinfo", None) is not None else dt


def _parse_and_normalize(s: str) -> datetime:
    """Parse a stored range boundary, handling both the current naive-UTC
    format and the old pre-fix tz-aware format, always returning naive."""
    ts = pd.Timestamp(s)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.to_pydatetime()


def _connect() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bars (
            symbol   TEXT NOT NULL,
            interval TEXT NOT NULL,
            bar_time TEXT NOT NULL,
            open     REAL NOT NULL,
            high     REAL NOT NULL,
            low      REAL NOT NULL,
            close    REAL NOT NULL,
            volume   REAL NOT NULL,
            PRIMARY KEY (symbol, interval, bar_time)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetched_ranges (
            symbol   TEXT NOT NULL,
            interval TEXT NOT NULL,
            start    TEXT NOT NULL,
            end      TEXT NOT NULL
        )
    """)
    return conn


def _merged_ranges(conn: sqlite3.Connection, symbol: str, interval: str) -> list[tuple[datetime, datetime]]:
    rows = conn.execute(
        "SELECT start, end FROM fetched_ranges WHERE symbol=? AND interval=?", (symbol, interval)
    ).fetchall()
    parsed = sorted((_parse_and_normalize(s), _parse_and_normalize(e)) for s, e in rows)
    merged: list[list[datetime]] = []
    for s, e in parsed:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _fully_covered(merged: list[tuple[datetime, datetime]], start: datetime, end: datetime) -> bool:
    return any(s <= start and e >= end for s, e in merged)


def _read(conn: sqlite3.Connection, symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT bar_time, open, high, low, close, volume FROM bars "
        "WHERE symbol=? AND interval=? AND bar_time>=? AND bar_time<? ORDER BY bar_time",
        conn, params=(symbol, interval, str(start), str(end)),
    )
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"])
    return df.set_index("bar_time")


def _write_batch_transactional(
    conn: sqlite3.Connection,
    interval: str,
    windows: dict[str, tuple[datetime, datetime]],
    per_symbol_df: dict[str, pd.DataFrame],
) -> None:
    """
    All symbols in `windows` are written together: bar rows (if any) plus a
    coverage record for EVERY symbol, including ones with an empty (but
    successfully-checked) DataFrame. Runs as one transaction - if anything
    raises partway through, nothing commits (`with conn:` semantics).
    """
    with conn:
        for symbol, (start, end) in windows.items():
            df = per_symbol_df.get(symbol, pd.DataFrame())
            if df is not None and not df.empty:
                rows = [
                    (symbol, interval, str(idx), row["open"], row["high"], row["low"], row["close"], row["volume"])
                    for idx, row in df.iterrows()
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO bars (symbol, interval, bar_time, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            conn.execute(
                "INSERT INTO fetched_ranges (symbol, interval, start, end) VALUES (?, ?, ?, ?)",
                (symbol, interval, str(start), str(end)),
            )


def _cached_fetch_multi(
    symbols: list[str],
    interval: str,
    start: datetime,
    end: datetime,
    fetcher: Callable[[list[str]], dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    start, end = _strip_tz(start), _strip_tz(end)
    conn = _connect()
    try:
        need_fetch = [s for s in symbols if not _fully_covered(_merged_ranges(conn, s, interval), start, end)]
        if need_fetch:
            # Fetch happens OUTSIDE any transaction/write - if it raises, the
            # exception propagates and nothing below is ever written or marked.
            fresh = fetcher(need_fetch)
            windows = {s: (start, end) for s in need_fetch}
            _write_batch_transactional(conn, interval, windows, fresh)
        return {s: _read(conn, s, interval, start, end) for s in symbols}
    finally:
        conn.close()


def get_daily_bars_multi(symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """Cache-first split-adjusted SIP daily bars, one batch request for every uncovered symbol."""
    dt_start = datetime.combine(start, datetime.min.time())
    dt_end = datetime.combine(end, datetime.min.time())
    return _cached_fetch_multi(
        symbols, "1Day", dt_start, dt_end,
        lambda syms: fetch_bars(syms, dt_start, dt_end, amount=1, unit="Day", feed="sip"),
    )


def get_intraday_bars_multi(
    symbols: list[str], start: datetime, end: datetime, minutes: int = 5, feed: str = "iex"
) -> dict[str, pd.DataFrame]:
    """Cache-first split-adjusted intraday bars, one batch request for every uncovered symbol."""
    interval = f"{minutes}Min" if feed == "iex" else f"{minutes}Min-{feed}"
    return _cached_fetch_multi(
        symbols, interval, start, end,
        lambda syms: fetch_bars(syms, start, end, amount=minutes, unit="Minute", feed=feed),
    )


def get_daily_bars(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Single-symbol convenience wrapper over get_daily_bars_multi."""
    return get_daily_bars_multi([symbol], start, end)[symbol]


def get_intraday_bars(
    symbol: str, start: datetime, end: datetime, minutes: int = 5, feed: str = "iex"
) -> pd.DataFrame:
    """Single-symbol convenience wrapper over get_intraday_bars_multi."""
    return get_intraday_bars_multi([symbol], start, end, minutes=minutes, feed=feed)[symbol]


# -- Whole-bot evidence cache -------------------------------------------------

def get_stock_research_bars_multi(
    symbols: list[str], start: datetime, end: datetime, *, amount: int, unit: str
) -> dict[str, pd.DataFrame]:
    """Split-adjusted SIP stock bars for the evidence-first simulator."""
    interval = f"research-stock-sip-{amount}{unit}"
    return _cached_fetch_multi(
        symbols,
        interval,
        start,
        end,
        lambda syms: fetch_bars(
            syms, start, end, amount=amount, unit=unit, feed="sip"
        ),
    )


def get_crypto_research_bars_multi(
    symbols: list[str], start: datetime, end: datetime, *, amount: int, unit: str
) -> dict[str, pd.DataFrame]:
    """Alpaca-US crypto bars; no cross-provider fallback is permitted."""
    interval = f"research-crypto-us-{amount}{unit}"
    return _cached_fetch_multi(
        symbols,
        interval,
        start,
        end,
        lambda syms: fetch_crypto_bars(
            syms, start, end, amount=amount, unit=unit
        ),
    )
