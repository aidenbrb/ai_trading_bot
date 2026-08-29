"""
Incremental SLC level/signal state machine.

Reuses the frozen utils/slc_signals.py building blocks (atr14,
stochastic_5_3_3, classify_structure, confirmed_pivots via
classify_structure, and the private-but-reusable _overlaps/_local_day/
_entry_not_late helpers) completely unmodified. Only detect_levels()'s and
generate_signals()'s state-machine LOOP is reimplemented here, bar by bar,
because that loop is (a) the O(bars x levels) Python-loop performance
bottleneck measured at ~35s/symbol/120 days, and (b) structurally one-bar-
late: generate_signals() cannot know about a confirmation until the bar
*after* it already exists in its input, which live operation can't wait
for. This reducer emits a confirmation the instant the confirming bar
closes - the actual Alpaca fill later becomes the live entry price
(amendment_004), never a value read from history.

Deliberately excludes entry price, target price, and initial risk from
Confirmation: none of those are knowable at confirmation time in live
operation (the frozen generate_signals() reads them from the NEXT
historical bar, which doesn't exist yet live). The frozen stop formula
*is* knowable (computed from the level + ATR, independent of entry) and is
included. A narrow, expected consequence: the frozen batch reference can
additionally reject a signal on offline entry/risk validity (entry <= stop
etc) using the hindsight entry price it has and this reducer doesn't -
that check is deliberately deferred to execution.py's post-fill
directional validation (rev. 6 Step 6), which is exactly why it exists as
a separate, later gate. Confirmation-output parity testing (Step 5) is
scoped only to fields both sides can produce - this divergence is
by design, not a bug.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from math import isclose, isfinite
from typing import Optional

import pandas as pd

from utils.market_calendar import session_for, trading_days_between
from utils.slc_signals import (
    EASTERN,
    VERSION,
    _entry_not_late,
    _frame,
    _local_day,
    _overlaps,
    atr14,
    classify_structure,
    generate_signals,
    stochastic_5_3_3,
)

MAX_LEVEL_SESSIONS = 20
# >= ATR14 warmup (14) + the 3-bar impulse window (4), with margin for
# Stochastic's 5+3+3 warmup too. Pruned back down to this size after every
# bar - raw bars are never retained beyond what these three computations need.
RAW_BAR_TAIL_SIZE = 30

# Comfortably beyond MAX_LEVEL_SESSIONS=20 trading sessions even across a
# holiday cluster - see process_new_bar()'s pruning of signaled_sessions.
SIGNALED_SESSION_RETENTION_DAYS = 45


@dataclass
class _ActiveLevel:
    level_id: str
    direction: str
    low: float
    high: float
    base_time: pd.Timestamp
    active_time: pd.Timestamp
    impulse_atr: float
    activation_session: date
    state: str = "fresh"
    eligible_after: Optional[pd.Timestamp] = None
    touch_started: bool = False
    armed: bool = False
    prior_k: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "level_id": self.level_id,
            "direction": self.direction,
            "low": self.low,
            "high": self.high,
            "base_time": self.base_time.isoformat(),
            "active_time": self.active_time.isoformat(),
            "impulse_atr": self.impulse_atr,
            "activation_session": self.activation_session.isoformat(),
            "state": self.state,
            "eligible_after": self.eligible_after.isoformat() if self.eligible_after is not None else None,
            "touch_started": self.touch_started,
            "armed": self.armed,
            "prior_k": self.prior_k,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_ActiveLevel":
        return cls(
            level_id=d["level_id"], direction=d["direction"], low=d["low"], high=d["high"],
            base_time=pd.Timestamp(d["base_time"]), active_time=pd.Timestamp(d["active_time"]),
            impulse_atr=d["impulse_atr"],
            activation_session=date.fromisoformat(d["activation_session"]),
            state=d["state"],
            eligible_after=pd.Timestamp(d["eligible_after"]) if d["eligible_after"] else None,
            touch_started=d["touch_started"], armed=d["armed"], prior_k=d["prior_k"],
        )


@dataclass(frozen=True)
class Confirmation:
    strategy_version: str
    symbol: str
    level_id: str
    direction: str
    level_state: str
    level_low: float
    level_high: float
    level_active_time: pd.Timestamp
    confirmation_time: pd.Timestamp
    entry_time: pd.Timestamp
    stop: float
    stochastic_k: float
    stochastic_d: Optional[float]
    atr14: float
    structure: str
    impulse_atr: float


@dataclass
class ReducerState:
    symbol: str
    bootstrap_completed: bool = False
    last_processed_bar_time: Optional[pd.Timestamp] = None
    # Plain list of {"bar_time": iso_str, open/high/low/close/volume}, oldest
    # first, capped at RAW_BAR_TAIL_SIZE - kept as a list, not a DataFrame,
    # because appending a row to a DataFrame every single bar (pd.concat in
    # a tight loop) is real per-call overhead that adds up across thousands
    # of bars; a DataFrame is built from this list once per bar, only when
    # atr14()/stochastic_5_3_3()/level detection actually need one.
    raw_bar_tail: list = field(default_factory=list)
    in_progress_4h: Optional[dict] = None
    completed_4h_bars: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    )
    active_levels: dict = field(default_factory=dict)  # level_id -> _ActiveLevel
    # Trading days that already produced this symbol's one signal for the
    # session (generate_signals()'s cross-level, per-day dedup) - a level
    # confirming on a day already in this set is discarded, never emitted.
    signaled_sessions: set = field(default_factory=set)

    def to_json(self) -> dict:
        bars4h = self.completed_4h_bars.reset_index().rename(columns={"index": "end"})
        bars4h["end"] = bars4h["end"].astype(str)
        ip = None
        if self.in_progress_4h is not None:
            ip = dict(self.in_progress_4h)
            ip["start"] = ip["start"].isoformat()
            ip["end"] = ip["end"].isoformat()
        return {
            "bootstrap_completed": self.bootstrap_completed,
            "last_processed_bar_time": (
                self.last_processed_bar_time.isoformat() if self.last_processed_bar_time is not None else None
            ),
            "raw_bar_tail": self.raw_bar_tail,
            "in_progress_4h": ip,
            "completed_4h_bars": bars4h.to_dict(orient="records"),
            "active_levels": [lvl.to_dict() for lvl in self.active_levels.values()],
            "signaled_sessions": sorted(d.isoformat() for d in self.signaled_sessions),
        }

    @classmethod
    def from_json(cls, symbol: str, data: dict) -> "ReducerState":
        tail = list(data.get("raw_bar_tail") or [])
        bars4h_rows = data.get("completed_4h_bars") or []
        if bars4h_rows:
            bars4h = pd.DataFrame(bars4h_rows).set_index("end")
            bars4h.index = pd.to_datetime(bars4h.index)
        else:
            bars4h = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        ip = data.get("in_progress_4h")
        if ip is not None:
            ip = dict(ip)
            ip["start"] = pd.Timestamp(ip["start"])
            ip["end"] = pd.Timestamp(ip["end"])
        levels = {d["level_id"]: _ActiveLevel.from_dict(d) for d in (data.get("active_levels") or [])}
        signaled = {date.fromisoformat(d) for d in (data.get("signaled_sessions") or [])}
        last_bar = data.get("last_processed_bar_time")
        return cls(
            symbol=symbol,
            bootstrap_completed=bool(data.get("bootstrap_completed", False)),
            last_processed_bar_time=pd.Timestamp(last_bar) if last_bar else None,
            raw_bar_tail=tail,
            in_progress_4h=ip,
            completed_4h_bars=bars4h,
            active_levels=levels,
            signaled_sessions=signaled,
        )


def _bucket_bounds_for_day(day: date) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    session = session_for(day)
    if session is None:
        return []
    session_open = pd.Timestamp(session["open"])
    session_close = pd.Timestamp(session["close"])
    split = min(session_open + timedelta(hours=4), session_close)
    buckets = []
    for start, end in ((session_open, split), (split, session_close)):
        if end <= start:
            continue
        expected = int((end - start).total_seconds() // 300)
        if expected > 0:
            buckets.append((start, end, expected))
    return buckets


def _update_4h_aggregate(state: ReducerState, row: pd.Series, ts: pd.Timestamp) -> Optional[dict]:
    """Feed one new closed 5-min bar into the in-progress 4h aggregate.

    Mirrors session_anchored_4h_bars()'s fail-closed contract exactly: a
    bucket is finalized only if every expected 5-min slot was seen, in
    order, with no gaps - any gap permanently discards that bucket
    attempt, matching "incomplete buckets fail closed and are omitted."
    """
    day = _local_day(ts)
    bucket = next(((s, e, exp) for s, e, exp in _bucket_bounds_for_day(day) if s <= ts < e), None)
    if bucket is None:
        return None  # outside any session-anchored 4h window (e.g. extended hours)
    start, end, expected = bucket
    ip = state.in_progress_4h
    if ip is not None and ip["start"] == start:
        next_expected_ts = ip["start"] + timedelta(minutes=5 * ip["bar_count"])
        if ts != next_expected_ts:
            state.in_progress_4h = None
            ip = None
    if ip is None or ip["start"] != start:
        if ts != start:
            return None
        ip = {"start": start, "end": end, "expected": expected, "bar_count": 0,
              "open": None, "high": None, "low": None, "close": None, "volume": 0.0}
    o, h, l, c, v = (float(row["open"]), float(row["high"]), float(row["low"]),
                     float(row["close"]), float(row["volume"]))
    if ip["bar_count"] == 0:
        ip["open"], ip["high"], ip["low"] = o, h, l
    else:
        ip["high"] = max(ip["high"], h)
        ip["low"] = min(ip["low"], l)
    ip["close"] = c
    ip["volume"] += v
    ip["bar_count"] += 1
    if ip["bar_count"] == ip["expected"]:
        finalized = {"end": end, "open": ip["open"], "high": ip["high"],
                     "low": ip["low"], "close": ip["close"], "volume": ip["volume"]}
        state.in_progress_4h = None
        return finalized
    state.in_progress_4h = ip
    return None


def _detect_new_levels(tail: pd.DataFrame, atr_series: pd.Series) -> list[dict]:
    """Mirrors detect_levels()'s per-i body exactly, evaluated only at the
    single newest possible base position (i = len(tail)-4)."""
    n = len(tail)
    if n < 18:
        return []
    i = n - 4
    base = tail.iloc[i]
    impulse = tail.iloc[i + 1 : i + 4]
    timestamps = tail.index[i : i + 4]
    if any((timestamps[j + 1] - timestamps[j]) != timedelta(minutes=5) for j in range(3)):
        return []
    local_days = timestamps.tz_localize("UTC").tz_convert(EASTERN).date
    if len(set(local_days)) != 1:
        return []
    value = atr_series.iloc[i]
    if not isfinite(float(value)) or value <= 0:
        return []
    base_low, base_high, base_close = map(float, (base["low"], base["high"], base["close"]))
    if base_high <= base_low:
        return []
    bullish = int((impulse["close"] > impulse["open"]).sum())
    bearish = int((impulse["close"] < impulse["open"]).sum())
    up_distance = float(impulse["high"].max()) - base_close
    down_distance = base_close - float(impulse["low"].min())
    active = pd.Timestamp(tail.index[i + 3]) + timedelta(minutes=5)
    session_day = active.tz_localize("UTC").tz_convert(EASTERN).date()
    base_time = pd.Timestamp(tail.index[i])
    found: list[dict] = []
    if bullish >= 2 and float(impulse.iloc[-1]["close"]) > base_high and up_distance >= value:
        found.append({
            "level_id": f"demand:{base_time.isoformat()}", "direction": "long",
            "low": base_low, "high": base_high, "base_time": base_time,
            "active_time": active, "impulse_atr": up_distance / float(value),
            "activation_session": session_day,
        })
    if bearish >= 2 and float(impulse.iloc[-1]["close"]) < base_low and down_distance >= value:
        found.append({
            "level_id": f"supply:{base_time.isoformat()}", "direction": "short",
            "low": base_low, "high": base_high, "base_time": base_time,
            "active_time": active, "impulse_atr": down_distance / float(value),
            "activation_session": session_day,
        })
    return found


@lru_cache(maxsize=4096)
def _sessions_elapsed(activation_session: date, day: date) -> int:
    """Memoized: (activation_session, day) pairs repeat heavily across bars
    and across levels sharing an activation day - trading_days_between()
    hits pandas_market_calendars' schedule computation on every call, which
    is too slow to redo per-level-per-bar without caching."""
    if day == activation_session:
        return 0
    return len(trading_days_between(activation_session, day)) - 1


def _advance_level(
    level: _ActiveLevel, ts: pd.Timestamp, row: pd.Series, day: date,
    current_k: Optional[float], completed_4h_bars: pd.DataFrame,
) -> tuple[str, Optional[Confirmation]]:
    """Returns ("keep"|"invalidate"|"confirm", confirmation_or_None)."""
    if _sessions_elapsed(level.activation_session, day) > MAX_LEVEL_SESSIONS:
        return "invalidate", None

    close = float(row["close"])
    wrong_break = (level.direction == "long" and close < level.low) or (
        level.direction == "short" and close > level.high
    )
    reclaim = (level.direction == "long" and close > level.high) or (
        level.direction == "short" and close < level.low
    )

    if level.state == "fresh" and wrong_break:
        level.state = "waiting_reclaim"
        level.touch_started = level.armed = False
        level.prior_k = None
        return "keep", None
    if level.state == "waiting_reclaim":
        if reclaim:
            level.state = "reclaimed"
            level.eligible_after = ts
            level.touch_started = level.armed = False
            level.prior_k = None
        return "keep", None
    if level.state == "reclaimed" and wrong_break:
        return "invalidate", None
    if level.eligible_after is not None and ts < level.eligible_after:
        return "keep", None

    overlaps = _overlaps(row, level)
    current_k_value = float(current_k) if current_k is not None and pd.notna(current_k) else None
    if not overlaps:
        if level.touch_started:
            return "invalidate", None
        level.prior_k = current_k_value
        return "keep", None

    level.touch_started = True
    if current_k_value is None:
        level.prior_k = None
        return "keep", None
    if level.direction == "short" and current_k_value > 80.0:
        level.armed = True
    elif level.direction == "long" and current_k_value < 20.0:
        level.armed = True

    crossed = False
    if level.armed and level.prior_k is not None:
        crossed = (
            (level.direction == "short" and level.prior_k >= 80.0 and current_k_value < 80.0)
            or (level.direction == "long" and level.prior_k <= 20.0 and current_k_value > 20.0)
        )
    prior_k_for_output = level.prior_k
    level.prior_k = current_k_value
    if not crossed:
        return "keep", None

    # No "does the next bar exist" gate here - this is the rev. 2 fix.
    # Confirmation is knowable the instant this bar closes.
    confirmation_end = ts + timedelta(minutes=5)
    structure = classify_structure(completed_4h_bars, confirmation_end)
    required_structure = "uptrend" if level.direction == "long" else "downtrend"
    if structure != required_structure:
        return "keep", None  # still touched/armed - may confirm on a later bar

    entry_time = confirmation_end
    if _local_day(entry_time) != day:
        return "invalidate", None
    if not _entry_not_late(entry_time):
        return "invalidate", None

    return "confirm_pending", None  # caller fills in ATR (needs the fresh tail-based series)


def process_new_bar(
    state: ReducerState, row: pd.Series, ts: pd.Timestamp,
) -> tuple[ReducerState, list[Confirmation]]:
    """The core incremental step. Returns (updated_state, confirmations)."""
    day = _local_day(ts)

    finalized_4h = _update_4h_aggregate(state, row, ts)
    if finalized_4h is not None:
        state.completed_4h_bars = pd.concat(
            [state.completed_4h_bars, pd.DataFrame([finalized_4h]).set_index("end")]
        )

    state.raw_bar_tail.append({
        "bar_time": ts.isoformat(),
        "open": float(row["open"]), "high": float(row["high"]),
        "low": float(row["low"]), "close": float(row["close"]),
        "volume": float(row["volume"]),
    })
    if len(state.raw_bar_tail) > RAW_BAR_TAIL_SIZE:
        del state.raw_bar_tail[: len(state.raw_bar_tail) - RAW_BAR_TAIL_SIZE]
    tail = pd.DataFrame(state.raw_bar_tail).set_index("bar_time")
    tail.index = pd.to_datetime(tail.index)

    atr_series = atr14(tail)
    stoch = stochastic_5_3_3(tail)
    current_atr = atr_series.iloc[-1] if len(atr_series) else float("nan")
    current_k = stoch.iloc[-1]["k"] if len(stoch) else None
    current_d = stoch.iloc[-1]["d"] if len(stoch) else None

    for new_level in _detect_new_levels(tail, atr_series):
        if new_level["level_id"] not in state.active_levels:
            state.active_levels[new_level["level_id"]] = _ActiveLevel(**new_level)

    # generate_signals() collapses to AT MOST ONE signal per symbol per
    # session across ALL levels ("Generate at most one fully formed SLC
    # signal per symbol/session") - the earliest confirmation in the day
    # wins outright; a genuine tie (multiple levels confirming on the same
    # bar) breaks via (level_active_time desc, impulse_atr desc), exactly
    # matching generate_signals()'s own tie-break. Processing bars strictly
    # in chronological order means the first bar of the day that produces
    # any confirm_pending candidate(s) IS the day's earliest by
    # construction - no candidate seen on a later bar this same day can
    # ever be earlier, so this reproduces the batch "earliest wins" rule
    # causally, one bar at a time, without ever needing future data.
    to_remove: list[str] = []
    confirm_candidates: list[_ActiveLevel] = []
    for level_id, level in list(state.active_levels.items()):
        if ts < level.active_time:
            continue
        result, _ = _advance_level(level, ts, row, day, current_k, state.completed_4h_bars)
        if result == "invalidate":
            to_remove.append(level_id)
        elif result == "confirm_pending":
            confirm_candidates.append(level)
            to_remove.append(level_id)  # one-shot per level, regardless of which candidate wins

    confirmations: list[Confirmation] = []
    if confirm_candidates and day not in state.signaled_sessions:
        if isfinite(float(current_atr)) and current_atr > 0:
            winner = max(confirm_candidates, key=lambda lvl: (lvl.active_time, lvl.impulse_atr))
            buffer = max(0.01, 0.10 * float(current_atr))
            stop = winner.low - buffer if winner.direction == "long" else winner.high + buffer
            confirmations.append(Confirmation(
                strategy_version=VERSION, symbol=state.symbol, level_id=winner.level_id,
                direction=winner.direction, level_state=winner.state,
                level_low=winner.low, level_high=winner.high, level_active_time=winner.active_time,
                confirmation_time=ts + timedelta(minutes=5), entry_time=ts + timedelta(minutes=5),
                stop=stop, stochastic_k=float(current_k), stochastic_d=(
                    float(current_d) if current_d is not None and pd.notna(current_d) else None
                ),
                atr14=float(current_atr), structure=(
                    "uptrend" if winner.direction == "long" else "downtrend"
                ), impulse_atr=winner.impulse_atr,
            ))
            state.signaled_sessions.add(day)
            # Pruned relative to `day` (the bar's own session date), never
            # wall-clock date.today() - a backfill replaying old bars can't
            # miscompute a "future" cutoff this way. Always runs AFTER the
            # insertion above, and only removes entries strictly older than
            # day - 45, so the entry just added can never be the thing
            # pruned. 45 calendar days comfortably exceeds
            # MAX_LEVEL_SESSIONS=20 trading sessions even across a holiday
            # cluster, bounding growth without risking re-allowing a
            # duplicate signal.
            cutoff = day - timedelta(days=SIGNALED_SESSION_RETENTION_DAYS)
            state.signaled_sessions = {d for d in state.signaled_sessions if d >= cutoff}
        # else: bar-level ATR invalid - none of this bar's candidates confirm,
        # matching the frozen per-level `break` on an invalid atr_value.

    for level_id in to_remove:
        state.active_levels.pop(level_id, None)

    state.last_processed_bar_time = ts
    return state, confirmations


def bootstrap(symbol: str, bars: pd.DataFrame) -> ReducerState:
    """Replay `bars` (>=120 calendar days of RTH 5-min bars, per
    amendment_004) bar by bar through process_new_bar to build real
    starting state. Confirmations during bootstrap are discarded - this is
    for state reconstruction only, never live order construction."""
    state = ReducerState(symbol=symbol)
    frame = _frame(bars)
    for ts, row in frame.iterrows():
        state, _ = process_new_bar(state, row, ts)
    state.bootstrap_completed = True
    return state


# -- Engine-parity comparison (rev. 11 Steps 9/13) ---------------------------
# Empirically verified (against the frozen AAPL/AMD fixture corpus): every
# field the reducer's Confirmation and the batch engine's SlcSignal have in
# common matches bit-exactly EXCEPT stochastic_k/stochastic_d, which show
# ~1e-10 relative float noise. Both sides call the identical
# stochastic_5_3_3() function, but the reducer recomputes it over a
# truncated raw_bar_tail window each bar while the batch engine computes it
# once over the full frame - pandas' rolling-window arithmetic is not
# bit-associative across different window lengths, even for a
# mathematically identical result. This is real, benign floating-point
# noise, not a logic divergence - every other field (including atr14's own
# rolling computation) needs no tolerance at all, and none is given.
PARITY_FIELDS = (
    "strategy_version", "symbol", "level_id", "direction", "level_state",
    "level_low", "level_high", "level_active_time", "confirmation_time",
    "entry_time", "stop", "stochastic_k", "stochastic_d", "atr14",
    "structure", "impulse_atr",
)
_PARITY_FLOAT_TOLERANCE_FIELDS = ("stochastic_k", "stochastic_d")
_PARITY_REL_TOL = 1e-9
_PARITY_ABS_TOL = 1e-9


def _parity_field_matches(field_name: str, batch_value, reducer_value) -> bool:
    if field_name in _PARITY_FLOAT_TOLERANCE_FIELDS:
        if batch_value is None or reducer_value is None:
            return batch_value is None and reducer_value is None
        return isclose(float(batch_value), float(reducer_value),
                        rel_tol=_PARITY_REL_TOL, abs_tol=_PARITY_ABS_TOL)
    if isinstance(batch_value, pd.Timestamp) or isinstance(reducer_value, pd.Timestamp):
        return pd.Timestamp(batch_value) == pd.Timestamp(reducer_value)
    return batch_value == reducer_value


def confirmation_matches_signal(confirmation: "Confirmation", signal) -> bool:
    """Exact-equality parity (rev. 11 Step 13) across every field the
    reducer's Confirmation and the frozen batch engine's SlcSignal have in
    common - no blanket round(x, 6). Used by both the offline
    fixture-corpus parity test and check_engine_parity() below, so
    "parity" means exactly the same thing in both places."""
    return all(
        _parity_field_matches(name, getattr(signal, name), getattr(confirmation, name))
        for name in PARITY_FIELDS
    )


def _parity_sort_key(obj) -> tuple:
    return (str(obj.confirmation_time), obj.level_id, obj.direction)


@dataclass(frozen=True)
class EngineParityResult:
    matched: bool
    signal_count: int  # how many confirmations were actually compared -
                        # lets the caller distinguish a genuine pass from
                        # a vacuous "both sides produced zero signals" one


def check_engine_parity(
    symbol: str, bars: pd.DataFrame, *, window_end: Optional[pd.Timestamp] = None,
) -> EngineParityResult:
    """Read-only, no I/O: replay `bars` through a scratch reducer and
    compare its confirmations against the frozen batch engine's output
    over the identical bars. matched=True only if both sides produce the
    same number of signals and every one matches
    confirmation_matches_signal() after pairing by _parity_sort_key() -
    vacuously True (with signal_count=0) if both sides produce zero
    signals. The caller decides whether that counts as a meaningful check
    (see run_slc_live.run_preflight()'s engine-parity self-check, which
    requires at least one symbol to report signal_count > 0 before
    treating the session-level check as passed, so an all-empty universe
    or an all-quiet sample window can never trivially pass).

    `window_end`, if given, restricts the comparison to confirmations
    whose confirmation_time falls strictly before it - used when `bars`
    carries extra trailing bars beyond the window that matters (e.g. the
    frozen validation corpus includes one extra real bar past its
    declared evaluation window solely because generate_signals()
    structurally requires a trailing bar; that extra bar's output must
    never itself be evaluated). None (the default) compares the full
    output of both engines with no truncation - unchanged behavior for
    every existing caller."""
    frame = _frame(bars)
    batch_signals = generate_signals(symbol, frame)
    state = ReducerState(symbol=symbol)
    reducer_confirmations: list[Confirmation] = []
    for ts, row in frame.iterrows():
        state, confs = process_new_bar(state, row, ts)
        reducer_confirmations.extend(confs)
    if window_end is not None:
        batch_signals = [s for s in batch_signals if s.confirmation_time < window_end]
        reducer_confirmations = [c for c in reducer_confirmations if c.confirmation_time < window_end]
    if len(batch_signals) != len(reducer_confirmations):
        return EngineParityResult(matched=False, signal_count=len(reducer_confirmations))
    batch_sorted = sorted(batch_signals, key=_parity_sort_key)
    reducer_sorted = sorted(reducer_confirmations, key=_parity_sort_key)
    matched = all(
        confirmation_matches_signal(r, b) for r, b in zip(reducer_sorted, batch_sorted)
    )
    return EngineParityResult(matched=matched, signal_count=len(reducer_sorted))
