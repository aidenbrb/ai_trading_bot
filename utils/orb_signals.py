"""
Pure Opening Range Breakout (ORB) signal logic - shared, byte-for-byte, by
both the day-trading backtester (backtest/) and the live signal-only pipeline
(nodes/day_strategy_node.py), so backtested and live rules can never drift.

No DB, no I/O, no broker calls - every function here takes plain values and
returns plain values so it can be unit tested in isolation.

Faithful first reproduction of the researched spec - no invented parameters:
  - Doji is EXACTLY close == open (no body-ratio threshold). Alternate doji
    definitions are explicitly out of scope for the first faithful backtest
    and would be separate, later variants.
  - Filter boundaries use the spec's exact comparison operators: price and
    daily ATR are strict '>', average volume and opening relative volume are
    '>='.
  - Stop distance is exactly 10% of daily ATR-14 from the entry trigger.
"""
from __future__ import annotations

from typing import Optional


def classify_opening_candle(open_: float, close: float) -> str:
    """'bullish' | 'bearish' | 'doji'. Doji is exactly close == open."""
    if close > open_:
        return "bullish"
    if close < open_:
        return "bearish"
    return "doji"


def orb_direction(candle_type: str) -> str:
    """'long' | 'short' | 'none' - doji (or any unrecognized type) trades nothing."""
    return {"bullish": "long", "bearish": "short"}.get(candle_type, "none")


def same_time_opening_volume_avg(
    prior_opening_volumes: list[float],
    lookback: int = 14,
) -> Optional[float]:
    """
    Mean of the same symbol's opening-bar volume over the previous `lookback`
    sessions. Requires ALL `lookback` sessions to be present - returns None
    if fewer are available, rather than silently averaging an incomplete
    sample (which could be biased by whichever specific days happened to be
    missing, with no indication the baseline was computed on a short window).
    """
    window = prior_opening_volumes[-lookback:]
    if len(window) < lookback:
        return None
    return sum(window) / len(window)


def opening_relative_volume(
    today_open_volume: float,
    avg_open_volume: Optional[float],
) -> Optional[float]:
    """today's opening volume / the same-time-of-day historical average."""
    if not avg_open_volume:
        return None
    return today_open_volume / avg_open_volume


def passes_orb_filters(
    *,
    price: float,
    avg_daily_volume_14d: Optional[float],
    daily_atr_14: Optional[float],
    opening_rel_volume: Optional[float],
    min_price: float = 5.0,
    min_avg_volume: float = 1_000_000,
    min_atr: float = 0.50,
    min_opening_rel_volume: float = 1.0,
) -> tuple[bool, str]:
    """Returns (passed, reason). Exact spec operators - see module docstring."""
    if price <= min_price:
        return False, f"price ${price:.2f} not above ${min_price:.2f} minimum"
    if avg_daily_volume_14d is None or avg_daily_volume_14d < min_avg_volume:
        val = f"{avg_daily_volume_14d:,.0f}" if avg_daily_volume_14d is not None else "N/A"
        return False, f"14-day avg daily volume {val} below {min_avg_volume:,.0f} minimum"
    if daily_atr_14 is None or daily_atr_14 <= min_atr:
        val = f"${daily_atr_14:.2f}" if daily_atr_14 is not None else "N/A"
        return False, f"daily ATR-14 {val} not above ${min_atr:.2f} minimum"
    if opening_rel_volume is None or opening_rel_volume < min_opening_rel_volume:
        val = f"{opening_rel_volume:.2f}x" if opening_rel_volume is not None else "N/A"
        return False, f"opening relative volume {val} below {min_opening_rel_volume:.2f}x minimum"
    return True, "all filters passed"


def rank_and_select(candidates: list[dict], top_n: int = 20) -> list[dict]:
    """
    Rank candidates (each a dict with an 'opening_rel_volume' key) by
    descending relative volume, assigning 1-indexed 'rank' and 'selected'
    (True for the top `top_n`) to each. Candidates with no relative-volume
    value sort last and are never selected. Returns a new list; does not
    mutate the input dicts.
    """
    def _key(c: dict) -> float:
        rv = c.get("opening_rel_volume")
        return rv if rv is not None else float("-inf")

    ranked = sorted(candidates, key=_key, reverse=True)
    out = []
    for i, c in enumerate(ranked, start=1):
        c = dict(c)
        c["rank"] = i
        c["selected"] = i <= top_n and c.get("opening_rel_volume") is not None
        out.append(c)
    return out


def compute_stop_price(
    entry_trigger: float,
    daily_atr_14: float,
    direction: str,
    stop_atr_fraction: float = 0.10,
) -> float:
    """Stop is exactly stop_atr_fraction * daily ATR-14 from the entry trigger."""
    offset = stop_atr_fraction * daily_atr_14
    if direction == "long":
        return entry_trigger - offset
    if direction == "short":
        return entry_trigger + offset
    raise ValueError(f"invalid direction: {direction!r} (expected 'long' or 'short')")


def compute_daily_reference_stats(
    daily_bars,
    opening_volumes: list[float],
    lookback: int = 14,
) -> Optional[dict]:
    """
    Pure computation of the IntradayDailyStats-equivalent stats from an
    already-fetched daily OHLCV frame (high/low/close/volume columns,
    chronological) and already-fetched prior-session opening-bar volumes.
    Returns None if there isn't enough data for a valid ATR-14.

    Does NOT do any date filtering itself - the caller is responsible for
    only ever passing PRIOR-session data (this is what prevents look-ahead;
    see nodes/intraday_reference_node.py and backtest/engine.py, both of
    which call this with exactly the same shape of already-restricted input).
    """
    if daily_bars is None or daily_bars.empty or len(daily_bars) < 2:
        return None

    from utils.indicators import atr

    avg_daily_volume_14d = float(daily_bars["volume"].tail(lookback).mean())
    atr_series = atr(daily_bars["high"], daily_bars["low"], daily_bars["close"], period=14)
    if atr_series.empty or atr_series.iloc[-1] != atr_series.iloc[-1]:  # NaN-safe (NaN != NaN)
        return None
    daily_atr_14 = float(atr_series.iloc[-1])

    avg_opening_volume_14d = same_time_opening_volume_avg(opening_volumes, lookback=lookback)
    if avg_opening_volume_14d is None:
        return None

    return {
        "avg_daily_volume_14d": avg_daily_volume_14d,
        "daily_atr_14": daily_atr_14,
        "avg_opening_volume_14d": avg_opening_volume_14d,
    }


def build_candidate_fields(
    *,
    opening_open: float,
    opening_high: float,
    opening_low: float,
    opening_close: float,
    opening_volume: float,
    avg_daily_volume_14d: Optional[float],
    daily_atr_14: Optional[float],
    avg_opening_volume_14d: Optional[float],
) -> dict:
    """
    The shared, I/O-free core of ORB candidate construction: classify the
    opening candle, determine direction, apply the exact-spec filters, and
    (if tradable) compute the entry trigger and stop. Used identically by
    nodes/day_strategy_node.py (live) and backtest/engine.py (historical) so
    the two can never diverge on the signal logic itself - only on how the
    inputs were fetched.
    """
    candle_type = classify_opening_candle(opening_open, opening_close)
    direction = orb_direction(candle_type)
    rel_vol = opening_relative_volume(opening_volume, avg_opening_volume_14d)

    passed, reason = passes_orb_filters(
        price=opening_open,
        avg_daily_volume_14d=avg_daily_volume_14d,
        daily_atr_14=daily_atr_14,
        opening_rel_volume=rel_vol,
    )
    tradable = passed and direction != "none"

    entry_trigger = stop_price = None
    if tradable:
        entry_trigger = opening_high if direction == "long" else opening_low
        stop_price = compute_stop_price(entry_trigger, daily_atr_14, direction)

    rejection_reason = None
    if not tradable:
        rejection_reason = reason if not passed else "doji - no trade"

    return {
        "candle_type": candle_type,
        "direction": direction if direction != "none" else None,
        "opening_rel_volume": rel_vol,
        "passed_filters": tradable,
        "entry_trigger_price": entry_trigger,
        "stop_price": stop_price,
        "rejection_reason": rejection_reason,
    }


def simulate_intraday_outcome(bars, entry_trigger: float, stop_price: float, direction: str) -> dict:
    """
    Walk 1-minute bars (a DataFrame indexed by bar time, with open/high/low/
    close columns, in chronological order) to reconstruct whether/when the
    breakout triggered, whether the stop was hit, and the exit - applying a
    conservative gap-through-fill rule to BOTH the entry and the exit.

    Entry gap: if the triggering bar's `open` has already gapped through the
    entry trigger, the fill is priced at that `open` (worse than the nominal
    trigger), not the trigger itself - `entry_gapped=True`.

    Exit gap: when the stop condition is satisfied on a bar, filling at that
    bar's `open` (instead of the nominal `stop_price`) is only valid when a
    position already existed BEFORE that bar opened - i.e. the bar is
    strictly after the trigger bar, or it IS the trigger bar but entry
    itself also gapped at that same open. On the trigger bar, if entry
    happened via a non-gap intrabar touch, the position did not exist at
    that bar's open, so that bar's open must never be used to price an exit
    on the same bar even if it numerically looks like a stop-side gap - this
    case is instead flagged `outcome_ambiguous=True` and filled adversely at
    the plain `stop_price`. This is also the ordinary "can't tell if the
    intrabar path hit the trigger or the stop first" ambiguity: a single
    5-min (or even 1-min) OHLC bar cannot reveal order-of-touches within
    itself, so this same condition covers both explanations at once.
    """
    result = {
        "breakout_triggered": False, "trigger_time": None, "simulated_entry_price": None,
        "entry_gapped": False,
        "stop_hit": None, "exit_time": None, "exit_price": None, "exit_gapped": False,
        "exit_reason": "no_trigger", "outcome_ambiguous": False,
    }
    triggered = False
    trigger_ts = None
    for ts, bar in bars.iterrows():
        if not triggered:
            hit_trigger = (bar["high"] >= entry_trigger) if direction == "long" else (bar["low"] <= entry_trigger)
            if not hit_trigger:
                continue
            entry_gapped = bool((bar["open"] >= entry_trigger) if direction == "long" else (bar["open"] <= entry_trigger))
            triggered = True
            trigger_ts = ts
            result["breakout_triggered"] = True
            result["trigger_time"] = ts
            result["simulated_entry_price"] = float(bar["open"]) if entry_gapped else entry_trigger
            result["entry_gapped"] = entry_gapped

        hit_stop = bool((bar["low"] <= stop_price) if direction == "long" else (bar["high"] >= stop_price))
        if hit_stop:
            same_bar_as_trigger = (ts == trigger_ts)
            if same_bar_as_trigger and not result["entry_gapped"]:
                # Position did not exist at this bar's open (entry was an
                # intrabar touch, same bar) - genuinely unknowable order,
                # never price the exit off this bar's open.
                result["outcome_ambiguous"] = True
                exit_price, exit_gapped = stop_price, False
            else:
                gapped_through_stop = bool((bar["open"] <= stop_price) if direction == "long" else (bar["open"] >= stop_price))
                exit_price = float(bar["open"]) if gapped_through_stop else stop_price
                exit_gapped = gapped_through_stop
            result["stop_hit"] = True
            result["exit_time"] = ts
            result["exit_price"] = exit_price
            result["exit_gapped"] = exit_gapped
            result["exit_reason"] = "stop"
            return result

    if triggered:
        result["stop_hit"] = False
        result["exit_time"] = bars.index[-1]
        result["exit_price"] = float(bars.iloc[-1]["close"])
        result["exit_reason"] = "eod_flatten"
    return result
