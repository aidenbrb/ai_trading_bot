"""Read-only, cache-only bar adapters for the stock_trend_momentum_v2
ablation harness.

Reuses backtest/readonly_bar_cache.py::read_cached_bars_or_none (not
modified by this module) for the actual cache lookup, and
backtest/data_cache.py::_strip_tz (not modified either) for the same
start/end normalization _cached_fetch_multi already applies before its
own coverage check. This module must never import fetch_bars,
fetch_crypto_bars, or anything else that can reach a live market-data
client - a coverage gap always raises CacheCoverageError or resolves to
an empty frame, never a network fetch.

read_cached_bars_or_none has three possible results, and they must stay
distinct all the way through: None (range not fully cached - unknown),
an empty DataFrame (range fully cached, legitimately no bars - a known
non-event), or a nonempty DataFrame. Collapsing None into an empty frame
would let a genuine coverage gap resolve as a definitive outcome
downstream (e.g. Delta1's expired_unfilled) instead of
outcome_data_missing.
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from backtest.data_cache import _strip_tz
from backtest.readonly_bar_cache import read_cached_bars_or_none

CacheMissCallback = Callable[[dict], None]


class CacheCoverageError(RuntimeError):
    """A requested [start, end) range for one symbol is not fully covered
    by the frozen bars_cache.db. Never filled by a live fetch."""


def _read_only_symbol(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    frame = read_cached_bars_or_none(symbol, interval, _strip_tz(start), _strip_tz(end))
    if frame is None:
        raise CacheCoverageError(f"{symbol} {interval} {start}..{end} not fully cached")
    return frame  # empty DataFrame is a legitimate, covered "no bars" result


def _interval_prefix(market: str) -> str:
    if market not in ("stock", "crypto"):
        raise ValueError(f"unknown market: {market}")
    return "research-stock-sip" if market == "stock" else "research-crypto-us"


def _log_miss(
    on_miss: CacheMissCallback | None,
    *, symbol: str, interval: str, start: datetime, end: datetime, market: str,
) -> None:
    if on_miss is not None:
        on_miss({
            "symbol": symbol, "interval": interval,
            "window_start": start, "window_end": end, "market": market,
        })


def make_minute_fetcher(market: str, *, on_miss: CacheMissCallback | None = None) -> Callable:
    """Read-only fetch_stock/fetch_crypto replacement for
    simulate_order_outcome (via _minute_chunks), which only ever calls the
    fetcher with a single-symbol list. CacheCoverageError propagates - the
    engine's existing broad except-handler resolves it to
    outcome_data_missing, never expired_unfilled.
    """
    prefix = _interval_prefix(market)

    def fetch(
        symbols: list[str], start: datetime, end: datetime, *, amount: int, unit: str
    ) -> dict[str, pd.DataFrame]:
        interval = f"{prefix}-{amount}{unit}"
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                out[symbol] = _read_only_symbol(symbol, interval, start, end)
            except CacheCoverageError:
                _log_miss(on_miss, symbol=symbol, interval=interval, start=start, end=end, market=market)
                raise
        return out

    return fetch


def make_hourly_fetcher(market: str, *, on_miss: CacheMissCallback | None = None) -> Callable:
    """Read-only fetch_stock/fetch_crypto replacement for
    load_research_data (via _load_hourly_in_chunks), which calls the
    fetcher with multi-symbol batches. CacheCoverageError is caught per
    symbol and turned into an empty frame for that symbol, matching
    _load_hourly_in_chunks's existing handling of a missing/empty
    per-symbol frame - "nothing accumulates for this window," never a
    raised exception at that layer.
    """
    prefix = _interval_prefix(market)

    def fetch(
        symbols: list[str], start: datetime, end: datetime, *, amount: int, unit: str
    ) -> dict[str, pd.DataFrame]:
        interval = f"{prefix}-{amount}{unit}"
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                out[symbol] = _read_only_symbol(symbol, interval, start, end)
            except CacheCoverageError:
                _log_miss(on_miss, symbol=symbol, interval=interval, start=start, end=end, market=market)
                out[symbol] = pd.DataFrame()
        return out

    return fetch
