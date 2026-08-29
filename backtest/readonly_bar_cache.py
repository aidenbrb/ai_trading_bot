"""Read-only, cache-only bar access for post-hoc analysis scripts.

Opens backtest/cache/bars_cache.db in SQLite read-only mode. A requested
range that is not already fully cached returns None - it is never filled
by a live fetch. This module must never import fetch_bars, fetch_crypto_bars,
or anything else from utils.alpaca_bars.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd

from backtest.data_cache import _CACHE_PATH, _fully_covered, _merged_ranges, _read


def _readonly_connect() -> sqlite3.Connection:
    return sqlite3.connect(f"file:/{_CACHE_PATH.resolve().as_posix()}?mode=ro", uri=True)


def read_cached_bars_or_none(
    symbol: str, interval: str, start: datetime, end: datetime
) -> pd.DataFrame | None:
    """Return cached bars for [start, end) only if that range is fully
    covered by recorded fetched_ranges; otherwise None. There is no other
    way to read bars through this module, so a caller cannot bypass the
    coverage check.
    """
    conn = _readonly_connect()
    try:
        merged = _merged_ranges(conn, symbol, interval)
        if not _fully_covered(merged, start, end):
            return None
        return _read(conn, symbol, interval, start, end)
    finally:
        conn.close()
