"""Pure, deterministic SLC (Structure, Level, Confirmation) research rules.

This module implements the frozen ``slc_4h_5m_stock_v1`` preregistration.
It contains no database, network, broker, or order-submission code.  All
timestamps returned by the public functions are naive UTC, matching the rest
of this codebase's historical-data convention.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import isfinite
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.market_calendar import session_for, trading_days_between

VERSION = "slc_4h_5m_stock_v1"
EASTERN = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class SlcLevel:
    level_id: str
    direction: str
    low: float
    high: float
    base_time: pd.Timestamp
    active_time: pd.Timestamp
    impulse_atr: float
    activation_session: date


@dataclass(frozen=True)
class SlcSignal:
    strategy_version: str
    symbol: str
    direction: str
    level_id: str
    level_state: str
    level_low: float
    level_high: float
    level_active_time: pd.Timestamp
    confirmation_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    initial_risk: float
    reward_risk: float
    stochastic_k: float
    stochastic_d: float | None
    atr14: float
    structure: str
    impulse_atr: float

    def to_dict(self) -> dict:
        return asdict(self)


def _frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, finite OHLCV data with a naive-UTC DatetimeIndex."""
    if bars is None or bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    required = {"open", "high", "low", "close"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    out = bars.copy()
    idx = pd.to_datetime(out.index, errors="coerce", utc=True).tz_localize(None)
    out.index = idx
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for column in required | ({"volume"} if "volume" in out else set()):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    finite = np.isfinite(out[list(required)].to_numpy(dtype=float)).all(axis=1)
    out = out.loc[finite]
    if "volume" not in out:
        out["volume"] = 0.0
    return out[["open", "high", "low", "close", "volume"]]


def atr14(bars: pd.DataFrame) -> pd.Series:
    """Frozen simple-mean ATR(14); incomplete windows return NaN."""
    frame = _frame(bars)
    previous = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous).abs(),
            (frame["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    return tr.rolling(14, min_periods=14).mean()


def stochastic_5_3_3(bars: pd.DataFrame) -> pd.DataFrame:
    """TradingView-style Stochastic 5/3/3 using completed bars only."""
    frame = _frame(bars)
    lowest = frame["low"].rolling(5, min_periods=5).min()
    highest = frame["high"].rolling(5, min_periods=5).max()
    spread = highest - lowest
    raw_k = 100.0 * (frame["close"] - lowest) / spread.where(spread > 0)
    k = raw_k.rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()
    return pd.DataFrame({"raw_k": raw_k, "k": k, "d": d}, index=frame.index)


def session_anchored_4h_bars(five_minute_bars: pd.DataFrame) -> pd.DataFrame:
    """Build complete 09:30-13:30 and 13:30-close NYSE bars.

    The output index is each aggregate bar's end timestamp, not its start.
    Incomplete buckets fail closed and are omitted.
    """
    frame = _frame(five_minute_bars)
    rows: list[dict] = []
    if frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    local_dates = sorted(
        set(frame.index.tz_localize("UTC").tz_convert(EASTERN).date)
    )
    for day in local_dates:
        session = session_for(day)
        if not session:
            continue
        session_open = pd.Timestamp(session["open"])
        session_close = pd.Timestamp(session["close"])
        split = min(session_open + timedelta(hours=4), session_close)
        for start, end in ((session_open, split), (split, session_close)):
            if end <= start:
                continue
            bucket = frame.loc[(frame.index >= start) & (frame.index < end)]
            expected = int((end - start).total_seconds() // 300)
            if expected <= 0 or len(bucket) != expected:
                continue
            expected_index = pd.date_range(start, periods=expected, freq="5min")
            if not bucket.index.equals(expected_index):
                continue
            rows.append(
                {
                    "end": end,
                    "open": float(bucket.iloc[0]["open"]),
                    "high": float(bucket["high"].max()),
                    "low": float(bucket["low"].min()),
                    "close": float(bucket.iloc[-1]["close"]),
                    "volume": float(bucket["volume"].sum()),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.DataFrame(rows).set_index("end")


# Performance only (Phase 6 - profiled: classify_structure() is called
# once per bar reaching the confirmation check in generate_signals()'s
# inner loop, always with the SAME trend_bars frame object for the whole
# call - previously this recomputed confirmed_pivots() over that entire,
# unchanging frame from scratch every single time (1,171 full rescans for
# just a 5-month/14.5k-bar AAPL window in profiling, ~45% of total
# runtime). Cached by object IDENTITY (`is`, not equality) with a strong
# reference held to the cached frame itself - this can never return a
# stale result for a different frame no matter how Python reuses freed
# objects' ids, because the cached frame object is kept alive and
# compared directly. side_bars is part of the key too, even though every
# real call site uses the default - defensive, not exercised today.
# Zero behavior change: same inputs always produce the same output as an
# uncached call: confirmed_pivots()'s own logic is untouched below.
_confirmed_pivots_cache: dict = {"frame": None, "side_bars": None, "result": None}


def confirmed_pivots(four_hour_bars: pd.DataFrame, *, side_bars: int = 2) -> list[dict]:
    """Return strict pivots and the timestamp each one became knowable."""
    cache = _confirmed_pivots_cache
    if cache["frame"] is four_hour_bars and cache["side_bars"] == side_bars:
        return cache["result"]
    result = _confirmed_pivots_uncached(four_hour_bars, side_bars=side_bars)
    cache["frame"] = four_hour_bars
    cache["side_bars"] = side_bars
    cache["result"] = result
    return result


def _confirmed_pivots_uncached(four_hour_bars: pd.DataFrame, *, side_bars: int = 2) -> list[dict]:
    frame = _frame(four_hour_bars)
    if side_bars < 1:
        raise ValueError("side_bars must be positive")
    pivots: list[dict] = []
    for i in range(side_bars, len(frame) - side_bars):
        high = float(frame.iloc[i]["high"])
        low = float(frame.iloc[i]["low"])
        neighbours = frame.iloc[i - side_bars : i + side_bars + 1]
        other_highs = np.delete(neighbours["high"].to_numpy(dtype=float), side_bars)
        other_lows = np.delete(neighbours["low"].to_numpy(dtype=float), side_bars)
        confirmed_at = pd.Timestamp(frame.index[i + side_bars])
        pivot_at = pd.Timestamp(frame.index[i])
        if np.all(high > other_highs):
            pivots.append({"kind": "high", "price": high, "pivot_at": pivot_at,
                           "confirmed_at": confirmed_at})
        if np.all(low < other_lows):
            pivots.append({"kind": "low", "price": low, "pivot_at": pivot_at,
                           "confirmed_at": confirmed_at})
    return sorted(pivots, key=lambda p: (p["confirmed_at"], p["pivot_at"], p["kind"]))


def classify_structure(four_hour_bars: pd.DataFrame, asof: pd.Timestamp) -> str:
    """Classify uptrend/downtrend/consolidation without future pivots."""
    cutoff = pd.Timestamp(asof)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    known = [p for p in confirmed_pivots(four_hour_bars) if p["confirmed_at"] <= cutoff]
    highs = [p for p in known if p["kind"] == "high"]
    lows = [p for p in known if p["kind"] == "low"]
    if len(highs) < 2 or len(lows) < 2:
        return "consolidation"
    high_up = highs[-1]["price"] > highs[-2]["price"]
    high_down = highs[-1]["price"] < highs[-2]["price"]
    low_up = lows[-1]["price"] > lows[-2]["price"]
    low_down = lows[-1]["price"] < lows[-2]["price"]
    if high_up and low_up:
        return "uptrend"
    if high_down and low_down:
        return "downtrend"
    return "consolidation"


def detect_levels(five_minute_bars: pd.DataFrame) -> list[SlcLevel]:
    """Detect frozen three-bar impulse levels using only completed bars."""
    frame = _frame(five_minute_bars)
    atr = atr14(frame)
    levels: list[SlcLevel] = []
    for i in range(0, len(frame) - 3):
        base = frame.iloc[i]
        impulse = frame.iloc[i + 1 : i + 4]
        timestamps = frame.index[i : i + 4]
        if any((timestamps[j + 1] - timestamps[j]) != timedelta(minutes=5) for j in range(3)):
            continue
        local_days = timestamps.tz_localize("UTC").tz_convert(EASTERN).date
        if len(set(local_days)) != 1:
            continue
        value = atr.iloc[i]
        if not isfinite(float(value)) or value <= 0:
            continue
        base_low, base_high, base_close = map(float, (base["low"], base["high"], base["close"]))
        if base_high <= base_low:
            continue
        bullish = int((impulse["close"] > impulse["open"]).sum())
        bearish = int((impulse["close"] < impulse["open"]).sum())
        up_distance = float(impulse["high"].max()) - base_close
        down_distance = base_close - float(impulse["low"].min())
        active = pd.Timestamp(frame.index[i + 3]) + timedelta(minutes=5)
        session_day = active.tz_localize("UTC").tz_convert(EASTERN).date()
        if bullish >= 2 and float(impulse.iloc[-1]["close"]) > base_high and up_distance >= value:
            strength = up_distance / float(value)
            levels.append(SlcLevel(
                f"demand:{pd.Timestamp(frame.index[i]).isoformat()}", "long",
                base_low, base_high, pd.Timestamp(frame.index[i]), active,
                strength, session_day,
            ))
        if bearish >= 2 and float(impulse.iloc[-1]["close"]) < base_low and down_distance >= value:
            strength = down_distance / float(value)
            levels.append(SlcLevel(
                f"supply:{pd.Timestamp(frame.index[i]).isoformat()}", "short",
                base_low, base_high, pd.Timestamp(frame.index[i]), active,
                strength, session_day,
            ))
    return sorted(levels, key=lambda level: (level.active_time, level.level_id))


def _overlaps(row: pd.Series, level: SlcLevel) -> bool:
    return float(row["low"]) <= level.high and float(row["high"]) >= level.low


def _local_day(timestamp: pd.Timestamp) -> date:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_convert(EASTERN).date()


def _entry_not_late(timestamp: pd.Timestamp) -> bool:
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.tz_convert(EASTERN)
    return (local.hour, local.minute) <= (15, 30)


def generate_signals(
    symbol: str,
    five_minute_bars: pd.DataFrame,
    four_hour_bars: pd.DataFrame | None = None,
    *,
    max_level_sessions: int = 20,
) -> list[SlcSignal]:
    """Generate at most one fully formed SLC signal per symbol/session."""
    frame = _frame(five_minute_bars)
    if frame.empty:
        return []
    trend_bars = session_anchored_4h_bars(frame) if four_hour_bars is None else _frame(four_hour_bars)
    levels = detect_levels(frame)
    stochastic = stochastic_5_3_3(frame)
    atr = atr14(frame)
    observed_sessions = sorted(set(_local_day(ts) for ts in frame.index))
    sessions = trading_days_between(observed_sessions[0], observed_sessions[-1])
    ordinal = {day: i for i, day in enumerate(sessions)}
    candidates: list[SlcSignal] = []

    # Performance only (Phase 6 - profiled: this loop's body ran millions
    # of individual .iloc[]/DatetimeIndex[] scalar lookups, each carrying
    # real pandas per-call overhead - see confirmed_pivots()'s cache note
    # above for the profiling numbers). Every array below is derived
    # ONCE, up front, from the exact same frame/stochastic/atr objects the
    # loop already used - positional access into a plain numpy array
    # (index i -> value) is mathematically identical to the .iloc[i]/
    # DataFrame[i] access it replaces, just without pandas' per-call
    # dispatch cost. No control flow, threshold, or ordering below
    # changed - only how each already-computed value is fetched.
    index_arr = frame.index.to_numpy()
    close_arr = frame["close"].to_numpy(dtype=float)
    open_arr = frame["open"].to_numpy(dtype=float)
    low_arr = frame["low"].to_numpy(dtype=float)
    high_arr = frame["high"].to_numpy(dtype=float)
    k_arr = stochastic["k"].to_numpy(dtype=float)
    d_arr = stochastic["d"].to_numpy(dtype=float)
    atr_num_arr = atr.to_numpy(dtype=float)
    # frame.index is always tz-naive UTC (see _frame()'s docstring) -
    # _local_day() on a tz-naive input always takes its tz_localize("UTC")
    # branch, so this vectorized form is exactly equivalent to calling
    # _local_day() on each index position individually.
    local_day_arr = frame.index.tz_localize("UTC").tz_convert(EASTERN).date

    for level in levels:
        state = "fresh"
        eligible_after: pd.Timestamp | None = level.active_time
        touch_started = False
        armed = False
        prior_k: float | None = None
        indices = np.flatnonzero(frame.index >= level.active_time)
        for position in indices:
            ts = pd.Timestamp(index_arr[position])
            day = local_day_arr[position]
            if ordinal.get(day, 0) - ordinal.get(level.activation_session, 0) > max_level_sessions:
                break
            close = float(close_arr[position])

            wrong_break = (
                (level.direction == "long" and close < level.low)
                or (level.direction == "short" and close > level.high)
            )
            reclaim = (
                (level.direction == "long" and close > level.high)
                or (level.direction == "short" and close < level.low)
            )
            if state == "fresh" and wrong_break:
                state = "waiting_reclaim"
                touch_started = armed = False
                prior_k = None
                continue
            if state == "waiting_reclaim":
                if reclaim:
                    state = "reclaimed"
                    eligible_after = ts
                    touch_started = armed = False
                    prior_k = None
                continue
            if state == "reclaimed" and wrong_break:
                break
            if eligible_after is not None and ts < eligible_after:
                continue

            overlaps = low_arr[position] <= level.high and high_arr[position] >= level.low
            current_k = k_arr[position]
            current_d = d_arr[position]
            current_k_value = float(current_k) if pd.notna(current_k) else None
            if not overlaps:
                if touch_started:
                    break
                prior_k = current_k_value
                continue

            touch_started = True
            if current_k_value is None:
                prior_k = None
                continue
            if level.direction == "short" and current_k_value > 80.0:
                armed = True
            elif level.direction == "long" and current_k_value < 20.0:
                armed = True

            crossed = False
            if armed and prior_k is not None:
                crossed = (
                    level.direction == "short" and prior_k >= 80.0 and current_k_value < 80.0
                ) or (
                    level.direction == "long" and prior_k <= 20.0 and current_k_value > 20.0
                )
            prior_k = current_k_value
            if not crossed or position + 1 >= len(frame):
                continue

            confirmation_end = ts + timedelta(minutes=5)
            structure = classify_structure(trend_bars, confirmation_end)
            required_structure = "uptrend" if level.direction == "long" else "downtrend"
            if structure != required_structure:
                continue
            entry_time = pd.Timestamp(index_arr[position + 1])
            if local_day_arr[position + 1] != day or entry_time - ts != timedelta(minutes=5):
                break
            if not _entry_not_late(entry_time):
                break
            entry = float(open_arr[position + 1])
            atr_value = float(atr_num_arr[position]) if pd.notna(atr_num_arr[position]) else float("nan")
            if not isfinite(atr_value) or atr_value <= 0:
                break
            buffer = max(0.01, 0.10 * atr_value)
            stop = level.low - buffer if level.direction == "long" else level.high + buffer
            risk = entry - stop if level.direction == "long" else stop - entry
            if not isfinite(risk) or risk <= 0:
                break
            target = entry + 2.0 * risk if level.direction == "long" else entry - 2.0 * risk
            candidates.append(SlcSignal(
                VERSION, symbol.upper(), level.direction, level.level_id, state,
                level.low, level.high, level.active_time, confirmation_end,
                entry_time, entry, stop, target, risk, 2.0, current_k_value,
                float(current_d) if pd.notna(current_d) else None, atr_value,
                structure, level.impulse_atr,
            ))
            break

    # Earliest entry wins for a symbol/session. Simultaneous alternatives use
    # the frozen most-recent-level, strongest-impulse tie-break.
    selected: list[SlcSignal] = []
    for day in sorted(set(_local_day(signal.entry_time) for signal in candidates)):
        same_day = [s for s in candidates if _local_day(s.entry_time) == day]
        earliest = min(s.entry_time for s in same_day)
        simultaneous = [s for s in same_day if s.entry_time == earliest]
        simultaneous.sort(key=lambda s: (s.level_active_time, s.impulse_atr), reverse=True)
        selected.append(simultaneous[0])
    return sorted(selected, key=lambda signal: signal.entry_time)


def simulate_intraday_outcome(
    signal: SlcSignal,
    minute_bars: pd.DataFrame | None,
    *,
    session_close: pd.Timestamp,
) -> dict:
    """Simulate stop/2R/EOD behavior; ties resolve to the stop."""
    if minute_bars is None:
        return {"status": "outcome_data_missing", "reason": "coverage_missing"}
    frame = _frame(minute_bars)
    bars = frame.loc[(frame.index >= signal.entry_time) & (frame.index < pd.Timestamp(session_close))]
    if bars.empty:
        return {"status": "outcome_data_missing", "reason": "no_bars_after_entry"}

    for ts, row in bars.iterrows():
        open_, high, low = map(float, (row["open"], row["high"], row["low"]))
        if signal.direction == "long":
            if open_ <= signal.stop:
                return _outcome(signal, ts, open_, "stop_gap", False)
            stop_hit, target_hit = low <= signal.stop, high >= signal.target
        else:
            if open_ >= signal.stop:
                return _outcome(signal, ts, open_, "stop_gap", False)
            stop_hit, target_hit = high >= signal.stop, low <= signal.target
        if stop_hit:
            return _outcome(signal, ts, signal.stop, "stop", bool(target_hit))
        if target_hit:
            return _outcome(signal, ts, signal.target, "target", False)

    last_ts = pd.Timestamp(bars.index[-1])
    return _outcome(signal, last_ts + timedelta(minutes=1), float(bars.iloc[-1]["close"]), "eod", False)


def _outcome(signal: SlcSignal, exit_time: pd.Timestamp, exit_price: float,
             reason: str, ambiguous: bool) -> dict:
    gross = (
        exit_price - signal.entry
        if signal.direction == "long"
        else signal.entry - exit_price
    )
    return {
        "status": "closed",
        "exit_time": pd.Timestamp(exit_time),
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "ambiguous": bool(ambiguous),
        "gross_per_share": float(gross),
        "gross_r": float(gross / signal.initial_risk),
    }
