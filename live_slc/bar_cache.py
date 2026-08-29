"""
Batched multi-symbol live IEX bar ingestion for live_slc, with the exact
data-freshness limits from amendment_004: a single batched fetch across
all symbols (never one call per symbol - that would blow the cycle
budget), polled for at most 15 seconds, accepting only the exact expected
closed-bar timestamp. A miss is recorded as `stale_or_missing_data`, never
silently treated as "no signal."

Persists into live_slc's own SlcFiveMinBar table (never trading.db, never
the SIP research caches) and is gap-tolerant on backfill: always fetches
everything newer than the last cached bar per symbol, so a missed cycle
self-heals on the next run. Backfilled bars are fed through the reducer to
keep state current, but a confirmation revealed only in hindsight by a
backfill must never be turned into a live order once its entry window has
passed - that rule is enforced by the caller (run_slc_live.py's cycle
orchestration), not here; this module only ever answers "what bars exist,"
never "should we still act on this."

RTH-only, enforced at two points (rev. 11 Step 10, defense in depth):
`_default_fetch()` is the common fetch boundary every real path funnels
through (fetch_expected_bar_batch, backfill_gaps, and preflight's
bootstrap all take `fetch_fn` defaulted to this), so it drops any
extended-hours row before a caller ever sees it - Alpaca's IEX feed does
not restrict itself to regular hours on request, and reducer.process_new_bar()
has no RTH gate of its own for raw_bar_tail (only _update_4h_aggregate
bounds-checks against the session window), so an extended-hours bar
reaching the reducer would silently contaminate the ATR/stochastic tail
used for level detection. `persist_bars()` re-checks independently at the
DB-write boundary, so a future/alternative code path supplying its own
`fetch_fn` (as several tests already do) still cannot write a non-RTH bar
into SlcFiveMinBar.
"""
from __future__ import annotations

import time
from datetime import timedelta
from typing import Callable, Optional

import pandas as pd
from sqlmodel import select

from live_slc.models import SlcFiveMinBar, get_live_slc_session
from utils.alpaca_bars import fetch_bars
from utils.market_calendar import session_for

BAR_POLL_DEADLINE_SECONDS = 15.0
POLL_RETRY_INTERVAL_SECONDS = 2.0

FetchFn = Callable[[list, "pd.Timestamp", "pd.Timestamp"], dict]


def _is_regular_trading_hours(bar_time) -> bool:
    """True if `bar_time` (naive, this codebase's UTC convention) falls in
    [session open, session close) for its own calendar date - NYSE regular
    hours only, no pre/post market, no weekends/holidays."""
    ts = pd.Timestamp(bar_time)
    session = session_for(ts.date())
    if session is None:
        return False
    return session["open"] <= ts < session["close"]


def _filter_regular_trading_hours(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    mask = [_is_regular_trading_hours(ts) for ts in frame.index]
    return frame.loc[mask]


def _default_fetch(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict:
    raw = fetch_bars(symbols, start=start.to_pydatetime(), end=end.to_pydatetime(),
                      amount=5, unit="Minute", feed="iex")
    return {symbol: _filter_regular_trading_hours(frame) for symbol, frame in raw.items()}


def fetch_expected_bar_batch(
    symbols: list[str],
    expected_bar_time: pd.Timestamp,
    *,
    fetch_fn: FetchFn = _default_fetch,
    deadline_seconds: float = BAR_POLL_DEADLINE_SECONDS,
    retry_interval_seconds: float = POLL_RETRY_INTERVAL_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], float] = time.monotonic,
) -> tuple[dict[str, Optional[pd.Series]], list[str]]:
    """One batched fetch across every symbol, polled for up to
    `deadline_seconds`. Only a bar whose index is EXACTLY
    `expected_bar_time` is accepted for a symbol - anything else (missing,
    or a different timestamp) leaves that symbol in the miss list.
    Returns (symbol -> bar row or None, list of symbols still missing at
    the deadline)."""
    remaining = list(dict.fromkeys(symbols))
    found: dict[str, Optional[pd.Series]] = {}
    deadline = now_fn() + deadline_seconds
    start = expected_bar_time - timedelta(minutes=1)
    end = expected_bar_time + timedelta(minutes=5)
    while remaining and now_fn() < deadline:
        frames = fetch_fn(remaining, start, end)
        still_missing = []
        for symbol in remaining:
            frame = frames.get(symbol) if frames else None
            if frame is not None and not frame.empty and expected_bar_time in frame.index:
                found[symbol] = frame.loc[expected_bar_time]
            else:
                still_missing.append(symbol)
        remaining = still_missing
        if remaining and now_fn() < deadline:
            sleep_fn(min(retry_interval_seconds, max(0.0, deadline - now_fn())))
    return found, remaining


def persist_bars(symbol: str, bars: dict) -> int:
    """bars: {bar_time -> pd.Series(open,high,low,close,volume)}. Returns
    the number of newly-persisted rows (existing rows are left alone)."""
    if not bars:
        return 0
    written = 0
    with get_live_slc_session() as session:
        for bar_time, row in bars.items():
            if not _is_regular_trading_hours(bar_time):
                continue
            existing = session.exec(
                select(SlcFiveMinBar).where(
                    SlcFiveMinBar.symbol == symbol, SlcFiveMinBar.bar_time == bar_time
                )
            ).first()
            if existing is not None:
                continue
            session.add(SlcFiveMinBar(
                symbol=symbol, bar_time=pd.Timestamp(bar_time).to_pydatetime(),
                open=float(row["open"]), high=float(row["high"]),
                low=float(row["low"]), close=float(row["close"]),
                volume=float(row["volume"]),
            ))
            written += 1
    return written


def last_cached_bar_time(symbol: str) -> Optional[pd.Timestamp]:
    with get_live_slc_session() as session:
        row = session.exec(
            select(SlcFiveMinBar)
            .where(SlcFiveMinBar.symbol == symbol)
            .order_by(SlcFiveMinBar.bar_time.desc())
            .limit(1)
        ).first()
        return pd.Timestamp(row.bar_time) if row is not None else None


def backfill_gaps(
    symbols: list[str],
    through: pd.Timestamp,
    *,
    fetch_fn: FetchFn = _default_fetch,
) -> dict[str, int]:
    """Fetch everything newer than each symbol's last cached bar, up
    through `through` - always the full gap, never just the single latest
    bar, so a missed cycle self-heals. Returns symbol -> rows written."""
    written: dict[str, int] = {}
    by_start: dict[pd.Timestamp, list[str]] = {}
    for symbol in symbols:
        last = last_cached_bar_time(symbol)
        start = (last + timedelta(minutes=5)) if last is not None else (through - timedelta(days=1))
        if start > through:
            written[symbol] = 0
            continue
        by_start.setdefault(start, []).append(symbol)

    for start, batch in by_start.items():
        frames = fetch_fn(batch, start, through + timedelta(minutes=5))
        for symbol in batch:
            frame = frames.get(symbol) if frames else None
            if frame is None or frame.empty:
                written[symbol] = 0
                continue
            bars = {ts: frame.loc[ts] for ts in frame.index if start <= ts <= through}
            written[symbol] = persist_bars(symbol, bars)
    return written
