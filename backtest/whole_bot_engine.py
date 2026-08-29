"""Deterministic, research-only simulator for the stock and crypto v1 rules.

Only Alpaca market-data clients are used.  This module must never import
``alpaca.trading`` or any execution node.
"""
from __future__ import annotations

import math
import time as wall_time
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from types import SimpleNamespace
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.data_cache import (
    get_crypto_research_bars_multi,
    get_stock_research_bars_multi,
)
from config.universe import CRYPTO_SET
from utils.indicators import adx, atr, macd, relative_volume, rsi, sma, trend_label
from utils.market_calendar import session_for, trading_days_between
from utils.strategy_signals import (
    CRYPTO_DAILY_DONCHIAN_LOOKBACK,
    CRYPTO_DAILY_SMA50_RISING_LOOKBACK,
    CRYPTO_STRATEGY_VERSION,
    STOCK_STRATEGY_VERSION,
    crypto_trend_momentum_v1,
    stock_trend_momentum_v1,
)

EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc
DECISION_TIME_ET = time(11, 16)
TECHNICAL_ONLY_DISCLOSURE = (
    "Historical results are the technical strategy before the RSS news and "
    "earnings vetoes; those point-in-time vetoes are validated only in forward paper trading."
)


@dataclass(frozen=True)
class ResearchCost:
    name: str
    stock_bps_per_leg: float
    crypto_bps_per_leg: float


ZERO_COST = ResearchCost("zero", 0.0, 0.0)
BASELINE_COST = ResearchCost("baseline", 5.0, 35.0)
STRESSED_COST = ResearchCost("stressed", 13.0, 50.0)
COSTS = {c.name: c for c in (ZERO_COST, BASELINE_COST, STRESSED_COST)}


@dataclass(frozen=True)
class ResearchPortfolio:
    name: str
    starting_equity: float = 100_000.0
    risk_per_trade: float = 0.01
    max_positions: int = 5
    max_new_trades_per_day: int = 2
    max_position_pct: float = 0.20
    daily_loss_limit: float = 0.02


CURRENT_RISK = ResearchPortfolio("current_1pct", risk_per_trade=0.01)
SAFE_RISK = ResearchPortfolio("safe_0_25pct", risk_per_trade=0.0025)
PORTFOLIOS = {p.name: p for p in (CURRENT_RISK, SAFE_RISK)}


@dataclass
class Candidate:
    symbol: str
    market: str
    strategy_version: str
    decision_time: datetime
    signal_bar_end: datetime
    entry: float
    stop: float
    target: float
    conviction: int
    atr: float
    # "hourly" (default, preserves every existing call site's output
    # unchanged) or "daily" - lets exit simulation later dispatch on the
    # timeframe the entry strategy actually used, instead of always
    # consulting the hourly indicator frame (see daily crypto additions
    # below and monitor_node's own live-side equivalent).
    timeframe: str = "hourly"


@dataclass
class ActiveOrder:
    candidate: Candidate
    quantity: float
    reserved_notional: float
    risk_amount: float
    outcome: dict


def decision_time_utc(day: date) -> datetime:
    local = datetime.combine(day, DECISION_TIME_ET, tzinfo=EASTERN)
    return local.astimezone(UTC).replace(tzinfo=None)


def completed_bar_cutoff(day: date, market: str) -> datetime:
    """Latest hourly bar start allowed at the 11:16 ET decision."""
    delay = timedelta(minutes=15) if market == "stock" else timedelta(0)
    return decision_time_utc(day) - timedelta(hours=1) - delay


def _normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = df.copy()
    idx = pd.to_datetime(out.index, utc=True).tz_convert(None)
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[["open", "high", "low", "close", "volume"]].dropna()


def indicator_frame(df: pd.DataFrame, *, include_adx: bool = False) -> pd.DataFrame:
    """Vectorized equivalent of utils.indicators.compute_all for every bar."""
    source = _normalise_frame(df)
    if source.empty:
        return source
    out = source.copy()
    out["sma_20"] = sma(out["close"], 20)
    out["sma_50"] = sma(out["close"], 50)
    out["sma_200"] = sma(out["close"], 200)
    out["rsi_14"] = rsi(out["close"], 14)
    _, _, hist = macd(out["close"])
    out["macd_hist"] = hist
    out["atr_14"] = atr(out["high"], out["low"], out["close"], 14)
    avg_volume = out["volume"].rolling(20, min_periods=20).mean()
    # Frozen v1 behavior: relative volume comes from the second-to-last bar.
    out["rel_volume"] = relative_volume(out["volume"], avg_volume).shift(1)
    out["trend"] = [
        trend_label(c, s20, s50, s200)
        for c, s20, s50, s200 in zip(
            out["close"], out["sma_20"], out["sma_50"], out["sma_200"]
        )
    ]
    if include_adx:
        out["adx_14"] = adx(out["high"], out["low"], out["close"], 14)
    return out


def _snapshot(frame: pd.DataFrame, day: date, market: str):
    if frame is None or frame.empty:
        return None
    eligible = frame[frame.index <= completed_bar_cutoff(day, market)]
    if eligible.empty:
        return None
    row = eligible.iloc[-1]
    required = ("sma_20", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume")
    if any(pd.isna(row[name]) for name in required):
        return None
    return row


def _indicator_object(row: pd.Series, *, include_adx: bool = False, bars_observed: int = 0):
    obj = SimpleNamespace(
        trend=str(row["trend"]),
        rsi_14=float(row["rsi_14"]),
        macd_hist=float(row["macd_hist"]),
        rel_volume=float(row["rel_volume"]),
        atr_14=float(row["atr_14"]),
        sma_20=float(row["sma_20"]),
        sma_50=float(row["sma_50"]),
    )
    if include_adx:
        obj.adx_14 = float(row["adx_14"]) if pd.notna(row.get("adx_14")) else None
        obj.bars_observed = bars_observed
    return obj


def _btc_macro_ok(crypto_frames: dict[str, pd.DataFrame], day: date) -> bool | None:
    btc = crypto_frames.get("BTC-USD")
    if btc is None or btc.empty:
        return None
    # Strictly completed UTC days only.  This prevents the current partial
    # daily candle from leaking into the historical decision.
    completed = btc[btc.index.date < day]
    if completed.empty:
        return None
    daily = completed["close"].resample("1D").last().dropna()
    if len(daily) < 20:
        return None
    return float(daily.iloc[-1]) > float(daily.tail(20).mean())


def build_signal_calendar(
    stock_frames: dict[str, pd.DataFrame],
    crypto_frames: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    stock_signal_fn: Callable = stock_trend_momentum_v1,
) -> tuple[dict[date, dict[str, list[Candidate]]], dict]:
    """Precompute all technical candidates and auditable coverage counts."""
    stock_ind = {symbol: indicator_frame(df, include_adx=True) for symbol, df in stock_frames.items()}
    crypto_ind = {symbol: indicator_frame(df) for symbol, df in crypto_frames.items()}
    required = ["sma_20", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume"]
    first_usable = {}
    for market, frames in (("stock", stock_ind), ("crypto", crypto_ind)):
        for symbol, frame in frames.items():
            valid = frame.dropna(subset=required) if not frame.empty else frame
            first_usable[(market, symbol)] = valid.index[0] if not valid.empty else None
    calendar: dict[date, dict[str, list[Candidate]]] = {}
    attempted = {"stock": 0, "crypto": 0}
    usable = {"stock": 0, "crypto": 0}
    exclusions: list[dict] = []

    day = start_date
    while day <= end_date:
        daily = {"stock": [], "crypto": []}
        if session_for(day) is not None:
            for symbol, frame in stock_ind.items():
                first = first_usable[("stock", symbol)]
                cutoff = completed_bar_cutoff(day, "stock")
                if first is not None and cutoff < first:
                    exclusions.append({"date": day, "symbol": symbol, "market": "stock",
                                       "reason": "pre-inception or indicator warmup"})
                    continue
                attempted["stock"] += 1
                row = _snapshot(frame, day, "stock")
                if row is None or row.name.date() != day:
                    exclusions.append({"date": day, "symbol": symbol, "market": "stock",
                                       "reason": "insufficient or missing completed hourly data"})
                    continue
                usable["stock"] += 1
                bars_observed = len(frame[frame.index <= cutoff])
                decision = stock_signal_fn(
                    symbol, _indicator_object(row, include_adx=True, bars_observed=bars_observed),
                    float(row["close"]), min_price=10.0, min_rr=2.0,
                )
                if decision.passed and (decision.conviction_score or 0) >= 70:
                    daily["stock"].append(_candidate(symbol, "stock", row, day, decision))

        macro = _btc_macro_ok(crypto_frames, day)
        for symbol, frame in crypto_ind.items():
            first = first_usable[("crypto", symbol)]
            cutoff = completed_bar_cutoff(day, "crypto")
            if first is not None and cutoff < first:
                exclusions.append({"date": day, "symbol": symbol, "market": "crypto",
                                   "reason": "pre-inception or indicator warmup"})
                continue
            attempted["crypto"] += 1
            row = _snapshot(frame, day, "crypto")
            stale = row is not None and cutoff - row.name.to_pydatetime() > timedelta(hours=2)
            if row is None or stale or macro is None:
                exclusions.append({"date": day, "symbol": symbol, "market": "crypto",
                                   "reason": "insufficient hourly data or BTC macro history"})
                continue
            usable["crypto"] += 1
            if symbol != "BTC-USD" and not macro:
                continue
            decision = crypto_trend_momentum_v1(
                symbol, _indicator_object(row), float(row["close"]), min_rr=2.0
            )
            if decision.passed and (decision.conviction_score or 0) >= 70:
                daily["crypto"].append(_candidate(symbol, "crypto", row, day, decision))

        daily["stock"].sort(key=lambda c: (-c.conviction, c.symbol))
        daily["crypto"].sort(key=lambda c: (-c.conviction, c.symbol))
        calendar[day] = daily
        day += timedelta(days=1)

    coverage = {
        market: {
            "attempted": attempted[market],
            "usable": usable[market],
            "coverage_rate": usable[market] / attempted[market] if attempted[market] else None,
        }
        for market in attempted
    }
    return calendar, {"coverage": coverage, "exclusions": exclusions,
                      "stock_indicator_frames": stock_ind, "crypto_indicator_frames": crypto_ind}


def _candidate(symbol: str, market: str, row: pd.Series, day: date, decision) -> Candidate:
    bar_start = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
    return Candidate(
        symbol=symbol,
        market=market,
        strategy_version=decision.strategy_version,
        decision_time=decision_time_utc(day),
        signal_bar_end=bar_start + timedelta(hours=1),
        entry=float(decision.entry),
        stop=float(decision.stop),
        target=float(decision.target),
        conviction=int(decision.conviction_score),
        atr=float(row["atr_14"]),
    )


# -- Daily-bar crypto timeframe -----------------------------------------------
#
# Added alongside the existing hourly path (which is left completely
# unmodified above) after a Phase 1 review found every "SMA20/50/200" trend
# calculation in the hourly path actually runs on 20/50/200 HOURLY bars
# (~0.8/2.1/8.3 calendar days), not the daily periods the names imply. These
# functions give crypto_trend_daily_v1/crypto_xsec_momentum_v1 a genuine
# daily-bar indicator frame - fetched natively as "Day" bars (Alpaca does not
# build these by resampling its own hourly bars, so this avoids inheriting
# the hourly path's ~51% coverage gaps), not resampled from the gappy hourly
# frame. Daily boundary is UTC midnight throughout, confirmed empirically:
# Alpaca's native crypto "Day" bars are indexed exactly at UTC 00:00 per
# calendar date (verified 2026-08-28 against real BTC-USD data).

DAILY_CRYPTO_DECISION_TIME_UTC = time(0, 5)  # just after the UTC daily close
DAILY_WARMUP_DAYS = 220  # >= 200 (SMA200) plus a buffer, in CALENDAR days


def daily_decision_time_utc(day: date) -> datetime:
    """The decision instant for a daily-timeframe crypto strategy: shortly
    after `day`'s own UTC close, so the bar dated `day` (see
    daily_completed_bar_cutoff) is the most recent complete one available."""
    return datetime.combine(day, DAILY_CRYPTO_DECISION_TIME_UTC)


def daily_completed_bar_cutoff(day: date) -> datetime:
    """The most recent COMPLETE daily bar at a decision made just after
    `day`'s own UTC close is the bar dated `day - 1` (it covers
    [day-1 00:00, day 00:00) UTC, which just finished)."""
    return datetime.combine(day - timedelta(days=1), time.min)


def _daily_snapshot(frame: pd.DataFrame, day: date):
    """Daily equivalent of _snapshot(): 'usable' means a bar dated exactly
    day-1 exists with every required indicator populated. No staleness
    window is needed - unlike the hourly path's 2-hour gap check, a
    completed daily bar either exists at its own native resolution or it
    doesn't."""
    if frame is None or frame.empty:
        return None
    cutoff = daily_completed_bar_cutoff(day)
    if cutoff not in frame.index:
        return None
    row = frame.loc[cutoff]
    required = ("sma_20", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume")
    if any(pd.isna(row[name]) for name in required):
        return None
    return row


def _daily_candidate(symbol: str, row: pd.Series, day: date, decision) -> Candidate:
    bar_start = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
    return Candidate(
        symbol=symbol,
        market="crypto",  # unchanged - existing sizing/cost-model/mode-dispatch
                           # logic already branches on market=="stock" vs else,
                           # so daily crypto candidates flow through it correctly
                           # without any of that logic needing to change.
        strategy_version=decision.strategy_version,
        decision_time=daily_decision_time_utc(day),
        signal_bar_end=bar_start + timedelta(days=1),
        entry=float(decision.entry),
        stop=float(decision.stop),
        target=float(decision.target),
        conviction=int(decision.conviction_score),
        atr=float(row["atr_14"]),
        timeframe="daily",
    )


def fetch_daily_crypto_frames(
    symbols: list[str],
    start_date: date,
    end_date: date,
    fetch_crypto: Callable = get_crypto_research_bars_multi,
) -> dict[str, pd.DataFrame]:
    """Native daily crypto bars (not resampled from hourly), with a
    DAILY_WARMUP_DAYS calendar-day warmup so SMA200 is populated from the
    first day of the actual test range."""
    start = datetime.combine(start_date - timedelta(days=DAILY_WARMUP_DAYS), time.min)
    end = datetime.combine(end_date + timedelta(days=1), time.min)
    raw = fetch_crypto(symbols, start, end, amount=1, unit="Day")
    return {symbol: _normalise_frame(df) for symbol, df in raw.items()}


def build_daily_crypto_indicator_frames(
    daily_bars: dict[str, pd.DataFrame], *, include_adx: bool = False
) -> dict[str, pd.DataFrame]:
    """Same vectorized indicator computation as indicator_frame() (reused
    unchanged - it was already timeframe-agnostic), just fed daily bars."""
    return {symbol: indicator_frame(df, include_adx=include_adx) for symbol, df in daily_bars.items()}


def _daily_indicator_object(frame: pd.DataFrame, row: pd.Series):
    """_indicator_object(row) plus the two extra fields crypto_trend_daily_v1
    consults only for its non-default entry_mode values (see that function's
    own docstring for their exact definitions). Populated unconditionally -
    cheap to compute, and harmless for entry_mode="strict_stack" (the
    default), which never reads them."""
    obj = _indicator_object(row)
    pos = frame.index.get_loc(row.name)

    sma_50_prior = None
    prior_pos = pos - CRYPTO_DAILY_SMA50_RISING_LOOKBACK
    if prior_pos >= 0:
        prior_val = frame.iloc[prior_pos]["sma_50"]
        if pd.notna(prior_val):
            sma_50_prior = float(prior_val)
    obj.sma_50_prior = sma_50_prior

    high_20 = None
    window_start = pos - CRYPTO_DAILY_DONCHIAN_LOOKBACK
    if window_start >= 0:
        window = frame.iloc[window_start:pos]["high"]
        if not window.empty and window.notna().any():
            high_20 = float(window.max())
    obj.high_20 = high_20

    return obj


def build_daily_crypto_calendar(
    daily_ind: dict[str, pd.DataFrame],
    hourly_crypto_frames: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    crypto_signal_fn: Callable,
    *,
    min_rr: float = 2.0,
) -> tuple[dict[date, list[Candidate]], dict]:
    """Daily-timeframe equivalent of build_signal_calendar(), kept fully
    separate (not merged into it) so the existing hourly calendar and its
    output are completely unaffected. `hourly_crypto_frames` is reused only
    for the BTC macro gate (_btc_macro_ok already computes a genuine daily
    resample from it - that check was never the hourly-mislabeled-as-daily
    problem this addition exists to fix, so it's reused as-is rather than
    rebuilt against the new native daily frame)."""
    required = ("sma_20", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume")
    first_usable = {}
    for symbol, frame in daily_ind.items():
        valid = frame.dropna(subset=required) if not frame.empty else frame
        first_usable[symbol] = valid.index[0] if not valid.empty else None

    calendar: dict[date, list[Candidate]] = {}
    attempted = 0
    usable = 0
    exclusions: list[dict] = []

    day = start_date
    while day <= end_date:
        candidates: list[Candidate] = []
        macro = _btc_macro_ok(hourly_crypto_frames, day)
        for symbol, frame in daily_ind.items():
            first = first_usable[symbol]
            cutoff = daily_completed_bar_cutoff(day)
            if first is not None and cutoff < first:
                exclusions.append({"date": day, "symbol": symbol, "market": "crypto",
                                   "reason": "pre-inception or indicator warmup"})
                continue
            attempted += 1
            row = _daily_snapshot(frame, day)
            if row is None or macro is None:
                exclusions.append({"date": day, "symbol": symbol, "market": "crypto",
                                   "reason": "no completed prior-day bar or BTC macro history"})
                continue
            usable += 1
            if symbol != "BTC-USD" and not macro:
                continue
            decision = crypto_signal_fn(
                symbol, _daily_indicator_object(frame, row), float(row["close"]), min_rr=min_rr
            )
            if decision.passed and (decision.conviction_score or 0) >= 70:
                candidates.append(_daily_candidate(symbol, row, day, decision))

        candidates.sort(key=lambda c: (-c.conviction, c.symbol))
        calendar[day] = candidates
        day += timedelta(days=1)

    coverage = {
        "attempted": attempted,
        "usable": usable,
        "coverage_rate": usable / attempted if attempted else None,
    }
    return calendar, {"coverage": coverage, "exclusions": exclusions}


def load_research_data(
    stock_symbols: list[str],
    crypto_symbols: list[str],
    start_date: date,
    end_date: date,
    fetch_stock: Callable = get_stock_research_bars_multi,
    fetch_crypto: Callable = get_crypto_research_bars_multi,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Fetch/cache all hourly signal data, with a 90-day warmup."""
    start = datetime.combine(start_date - timedelta(days=90), time.min)
    end = datetime.combine(end_date + timedelta(days=2), time.min)
    stock = _load_hourly_in_chunks(
        stock_symbols, start, end, fetch_stock,
        market="stock", symbol_batch_size=50,
    )
    crypto = _load_hourly_in_chunks(
        crypto_symbols, start, end, fetch_crypto,
        market="crypto", symbol_batch_size=25,
    )
    return (stock, crypto)


def _load_hourly_in_chunks(
    symbols: list[str],
    start: datetime,
    end: datetime,
    fetcher: Callable,
    *,
    market: str,
    symbol_batch_size: int,
) -> dict[str, pd.DataFrame]:
    """Bound requests by year and symbol count so every cache commit is resumable."""
    collected: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    if not symbols:
        return {}
    windows = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=366), end)
        windows.append((cursor, window_end))
        cursor = window_end
    batches = [symbols[i:i + symbol_batch_size]
               for i in range(0, len(symbols), symbol_batch_size)]
    total = len(windows) * len(batches)
    completed = 0
    for window_start, window_end in windows:
        for batch in batches:
            completed += 1
            print(
                f"  Loading {market} hourly data chunk {completed}/{total}: "
                f"{window_start.date()} -> {window_end.date()} symbols={len(batch)}",
                flush=True,
            )
            frames = fetcher(
                batch, window_start, window_end, amount=1, unit="Hour"
            )
            for symbol in batch:
                frame = frames.get(symbol, pd.DataFrame())
                if frame is not None and not frame.empty:
                    collected[symbol].append(frame)
    output = {}
    for symbol, pieces in collected.items():
        if not pieces:
            output[symbol] = _normalise_frame(pd.DataFrame())
            continue
        output[symbol] = _normalise_frame(pd.concat(pieces).sort_index())
    return output


def _minute_chunks(
    candidate: Candidate,
    end_date: date,
    fetch_stock: Callable = get_stock_research_bars_multi,
    fetch_crypto: Callable = get_crypto_research_bars_multi,
):
    cursor = candidate.decision_time
    final = datetime.combine(end_date + timedelta(days=1), time.min)
    while cursor < final:
        chunk_end = min(cursor + timedelta(days=31), final)
        fetcher = fetch_stock if candidate.market == "stock" else fetch_crypto
        frames = fetcher([candidate.symbol], cursor, chunk_end, amount=1, unit="Minute")
        bars = _normalise_frame(frames.get(candidate.symbol, pd.DataFrame()))
        yield bars, chunk_end
        cursor = chunk_end


def _stock_regular_minute(ts: datetime) -> bool:
    session = session_for(ts.date())
    return bool(session and session["open"] <= ts < session["close"])


def _stock_regular_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter a minute frame to NYSE regular sessions with one calendar lookup/day."""
    if bars.empty:
        return bars
    pieces = []
    for day_value, day_bars in bars.groupby(bars.index.date, sort=False):
        session = session_for(day_value)
        if session is None:
            continue
        regular = day_bars.loc[
            (day_bars.index >= session["open"])
            & (day_bars.index < session["close"])
        ]
        if not regular.empty:
            pieces.append(regular)
    if not pieces:
        return bars.iloc[0:0]
    return pd.concat(pieces)


def simulate_order_outcome(
    candidate: Candidate,
    end_date: date,
    indicator_frames: dict[str, pd.DataFrame],
    *,
    fetch_stock: Callable = get_stock_research_bars_multi,
    fetch_crypto: Callable = get_crypto_research_bars_multi,
    expiration_sessions: int | None = None,
) -> dict:
    """Simulate the broker order plus the bot's 30-minute monitor."""
    filled_at = None
    fill_price = None
    current_stop = candidate.stop
    stop_limit_triggered = False
    last_row = None
    missing = False
    saw_relevant_bar = False

    try:
        expires_at = None
        if expiration_sessions is not None and candidate.market == "stock":
            expiration_days = trading_days_between(
                candidate.decision_time.date(),
                candidate.decision_time.date() + timedelta(days=30),
            )
            expires_at = session_for(expiration_days[expiration_sessions - 1])["close"]

        chunks = _minute_chunks(candidate, end_date, fetch_stock, fetch_crypto)
        for bars, chunk_end in chunks:
            if bars.empty:
                if filled_at is None and expires_at is not None and chunk_end >= expires_at:
                    return {"status": "expired_unfilled", "outcome_data_missing": False,
                            "filled_at": None, "fill_price": None, "expires_at": expires_at}
                continue
            if candidate.market == "stock":
                bars = _stock_regular_bars(bars)
                if bars.empty:
                    if filled_at is None and expires_at is not None and chunk_end >= expires_at:
                        return {"status": "expired_unfilled", "outcome_data_missing": False,
                                "filled_at": None, "fill_price": None, "expires_at": expires_at}
                    continue
            saw_relevant_bar = True

            # A GTC stock limit may remain untouched for years.  Locate its
            # first possible fill with a vectorized low-price comparison, then
            # use the exact same chronological per-minute logic from that bar
            # onward.  This changes no execution rule; it only avoids millions
            # of Python iterations over bars that provably cannot fill.
            if filled_at is None and candidate.market == "stock":
                # expires_at bounds only the entry search. Once a fill is
                # found, processing resumes on the original, untruncated
                # `bars` below, so post-fill bars after expires_at stay
                # visible to the stop/target/monitor/end-of-test logic.
                fill_eligible = bars if expires_at is None else bars[bars.index < expires_at]
                hit_positions = (
                    np.flatnonzero(fill_eligible["low"].to_numpy(dtype=float) <= candidate.entry)
                    if not fill_eligible.empty else np.array([], dtype=int)
                )
                if len(hit_positions) == 0:
                    last_row = fill_eligible.iloc[-1] if not fill_eligible.empty else bars.iloc[-1]
                    if expires_at is not None and chunk_end >= expires_at:
                        return {"status": "expired_unfilled", "outcome_data_missing": False,
                                "filled_at": None, "fill_price": None, "expires_at": expires_at}
                    continue
                bars = bars.iloc[int(hit_positions[0]):]

            for ts_value, row in bars.iterrows():
                ts = ts_value.to_pydatetime() if hasattr(ts_value, "to_pydatetime") else ts_value
                last_row = row

                if filled_at is None:
                    if candidate.market == "crypto":
                        filled_at = ts
                        fill_price = float(row["open"])
                    elif float(row["low"]) <= candidate.entry:
                        filled_at = ts
                        fill_price = min(candidate.entry, float(row["open"]))
                    else:
                        continue

                low, high = float(row["low"]), float(row["high"])
                # Broker-side downside protection is checked before upside in
                # the same minute, the required adverse ambiguity assumption.
                if candidate.market == "stock":
                    if low <= current_stop:
                        return _outcome(candidate, filled_at, fill_price, ts, current_stop,
                                        "stop", current_stop, ambiguous=high >= candidate.target)
                    if high >= candidate.target:
                        return _outcome(candidate, filled_at, fill_price, ts, candidate.target,
                                        "target", current_stop)
                else:
                    stop_limit = current_stop * 0.98
                    if not stop_limit_triggered and low <= current_stop:
                        stop_limit_triggered = True
                    if stop_limit_triggered and high >= stop_limit:
                        exit_price = min(current_stop, max(stop_limit, float(row["open"])))
                        return _outcome(candidate, filled_at, fill_price, ts, exit_price,
                                        "stop_limit", current_stop)

                if ts.minute not in {0, 30}:
                    continue
                snapshot_day = ts.date()
                if ts < decision_time_utc(snapshot_day):
                    snapshot_day -= timedelta(days=1)
                snapshot = _snapshot(
                    indicator_frames[candidate.symbol], snapshot_day, candidate.market
                )
                if candidate.market == "crypto" and float(row["close"]) >= candidate.target:
                    return _outcome(candidate, filled_at, fill_price, ts, float(row["close"]),
                                    "monitor_target", current_stop)
                if snapshot is not None:
                    close_now = float(row["close"])
                    if str(snapshot["trend"]) == "DOWNTREND" or (
                        float(snapshot["macd_hist"]) < 0 and float(snapshot["rsi_14"]) < 45
                    ):
                        return _outcome(candidate, filled_at, fill_price, ts, close_now,
                                        "monitor_reversal", current_stop)
                    current_atr = float(snapshot["atr_14"])
                    if close_now - float(fill_price) >= current_atr:
                        proposed = close_now - 1.5 * current_atr
                        if proposed > current_stop and proposed < close_now:
                            current_stop = round(proposed, 8 if candidate.market == "crypto" else 2)
    except Exception as exc:
        missing = True
        error = str(exc)

    if missing:
        return {
            "status": "outcome_data_missing", "outcome_data_missing": True,
            "reason": error, "filled_at": filled_at, "fill_price": fill_price,
        }
    if not saw_relevant_bar:
        return {"status": "outcome_data_missing", "outcome_data_missing": True,
                "reason": "no relevant minute bars returned", "filled_at": None,
                "fill_price": None}
    if filled_at is None:
        return {"status": "unfilled_end", "outcome_data_missing": False,
                "filled_at": None, "fill_price": None}
    if last_row is None:
        return {"status": "outcome_data_missing", "outcome_data_missing": True,
                "reason": "no minute bars after fill", "filled_at": filled_at,
                "fill_price": fill_price}
    final_ts = datetime.combine(end_date, time.max)
    return _outcome(candidate, filled_at, fill_price, final_ts, float(last_row["close"]),
                    "end_of_test", current_stop)


def _outcome(candidate, filled_at, fill_price, exit_time, exit_price, reason,
             current_stop, ambiguous=False) -> dict:
    return {
        "status": "closed",
        "outcome_data_missing": False,
        "filled_at": filled_at,
        "fill_price": float(fill_price),
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "final_stop": float(current_stop),
        "ambiguous": bool(ambiguous),
    }


def _size(candidate: Candidate, portfolio: ResearchPortfolio, equity: float, cash: float) -> dict:
    risk_per_unit = candidate.entry - candidate.stop
    if risk_per_unit <= 0:
        return {"quantity": 0.0, "risk": 0.0, "notional": 0.0}
    quantity = equity * portfolio.risk_per_trade / risk_per_unit
    max_value = min(equity * portfolio.max_position_pct, cash)
    quantity = min(quantity, max_value / candidate.entry)
    if candidate.market == "stock":
        quantity = math.floor(quantity)
    else:
        quantity = math.floor(quantity * 1_000_000_000) / 1_000_000_000
    notional = quantity * candidate.entry
    return {"quantity": quantity, "risk": quantity * risk_per_unit, "notional": notional}


def _transaction_cost(active: ActiveOrder, cost: ResearchCost) -> float:
    outcome = active.outcome
    if outcome.get("status") != "closed":
        return 0.0
    bps = cost.stock_bps_per_leg if active.candidate.market == "stock" else cost.crypto_bps_per_leg
    entry_notional = active.quantity * float(outcome["fill_price"])
    exit_notional = active.quantity * float(outcome["exit_price"])
    return (entry_notional + exit_notional) * bps / 10_000.0


def _occupies_slot(item: ActiveOrder, decision: datetime, today: date) -> bool:
    """Whether an active order counts against max_positions as of `decision`.

    Orders approved earlier THIS SAME decision day count immediately,
    regardless of fill status yet - matches risk_node.py incrementing
    current_positions right after approval, before any fill. Orders
    approved on a PRIOR day only count once they have actually filled and
    are still open as of today's decision point - a never-filled (or
    not-yet-filled) resting order is not a real position and must not
    permanently occupy capacity, matching live where each new risk_node run
    re-reads the real, broker-confirmed position count fresh.
    """
    if item.candidate.decision_time.date() == today:
        exit_time = item.outcome.get("exit_time")
        return not (exit_time and exit_time <= decision)
    filled_at = item.outcome.get("filled_at")
    if not filled_at or filled_at > decision:
        return False
    exit_time = item.outcome.get("exit_time")
    return not (exit_time and exit_time <= decision)


def simulate_portfolio(
    signal_calendar: dict[date, dict[str, list[Candidate]]],
    indicator_frames: dict[str, dict[str, pd.DataFrame]],
    start_date: date,
    end_date: date,
    portfolio: ResearchPortfolio,
    cost: ResearchCost,
    mode: str,
    outcome_cache: dict | None = None,
    outcome_simulator: Callable = simulate_order_outcome,
    expiration_sessions: int | None = None,
) -> dict:
    """Chronological portfolio accounting for one risk/cost/mode slice."""
    if mode not in {"stock_only", "crypto_weekends", "crypto_7day", "combined_primary"}:
        raise ValueError(f"unknown simulation mode: {mode}")
    outcome_cache = outcome_cache if outcome_cache is not None else {}
    cash = portfolio.starting_equity
    realized_pnl = 0.0
    active: list[ActiveOrder] = []
    trades: list[dict] = []
    missing_outcomes: list[dict] = []
    rejected: list[dict] = []
    daily_equity: list[dict] = []
    previous_decision_equity = portfolio.starting_equity
    total_days = (end_date - start_date).days + 1
    started = wall_time.monotonic()
    processed = 0

    day = start_date
    while day <= end_date:
        processed += 1
        if processed == 1 or processed % 100 == 0 or processed == total_days:
            elapsed = wall_time.monotonic() - started
            eta = (elapsed / processed) * max(total_days - processed, 0)
            print(
                f"    [{mode}/{portfolio.name}/{cost.name}] "
                f"day {processed}/{total_days} elapsed={elapsed:.0f}s eta={eta:.0f}s",
                flush=True,
            )
        decision = decision_time_utc(day)

        # Realize positions/orders whose modeled lifecycle ended before this decision.
        remaining: list[ActiveOrder] = []
        day_realized = 0.0
        for item in active:
            exit_time = item.outcome.get("exit_time")
            if exit_time and exit_time <= decision:
                gross = item.quantity * (
                    float(item.outcome["exit_price"]) - float(item.outcome["fill_price"])
                )
                costs = _transaction_cost(item, cost)
                net = gross - costs
                realized_pnl += net
                day_realized += net
                cash += item.reserved_notional + net
            elif (
                item.outcome.get("status") == "expired_unfilled"
                and item.outcome.get("expires_at")
                and item.outcome["expires_at"] <= decision
            ):
                cash += item.reserved_notional   # full return - no net P&L, no cost
            elif item.outcome.get("status") == "unfilled_end":
                remaining.append(item)
            else:
                remaining.append(item)
        active = remaining

        unrealized = 0.0
        for item in active:
            filled_at = item.outcome.get("filled_at")
            exit_time = item.outcome.get("exit_time")
            if not filled_at or filled_at > decision or (exit_time and exit_time <= decision):
                continue
            frame = indicator_frames[item.candidate.market].get(item.candidate.symbol)
            row = _snapshot(frame, day, item.candidate.market) if frame is not None else None
            if row is None:
                continue
            bps = (cost.stock_bps_per_leg if item.candidate.market == "stock"
                   else cost.crypto_bps_per_leg)
            entry_cost = item.quantity * float(item.outcome["fill_price"]) * bps / 10_000.0
            unrealized += item.quantity * (
                float(row["close"]) - float(item.outcome["fill_price"])
            ) - entry_cost
        equity = portfolio.starting_equity + realized_pnl + unrealized
        daily_equity.append({"date": day, "equity": equity})
        daily_return = (
            equity / previous_decision_equity - 1.0
            if previous_decision_equity > 0 else -1.0
        )
        loss_blocked = daily_return <= -portfolio.daily_loss_limit
        previous_decision_equity = equity

        candidates: list[Candidate] = []
        is_stock_session = session_for(day) is not None
        if mode in {"stock_only", "combined_primary"} and is_stock_session:
            candidates.extend(signal_calendar[day]["stock"])
        if mode in {"crypto_weekends", "combined_primary"} and day.weekday() >= 5:
            candidates.extend(signal_calendar[day]["crypto"])
        if mode == "crypto_7day":
            candidates.extend(signal_calendar[day]["crypto"])
        candidates.sort(key=lambda c: (-c.conviction, c.symbol))

        held = {item.candidate.symbol for item in active}
        approved_today = 0
        for candidate in candidates:
            if loss_blocked:
                rejected.append({"date": day, "symbol": candidate.symbol,
                                 "reason": "daily_loss_limit"})
                continue
            if candidate.symbol in held:
                rejected.append({"date": day, "symbol": candidate.symbol,
                                 "reason": "already_held_or_pending"})
                continue
            occupied_slots = sum(1 for item in active if _occupies_slot(item, decision, day))
            if occupied_slots >= portfolio.max_positions:
                rejected.append({"date": day, "symbol": candidate.symbol,
                                 "reason": "max_positions"})
                continue
            if approved_today >= portfolio.max_new_trades_per_day:
                rejected.append({"date": day, "symbol": candidate.symbol,
                                 "reason": "max_new_trades_per_day"})
                continue

            sizing = _size(candidate, portfolio, equity, cash)
            if sizing["quantity"] <= 0 or sizing["notional"] < 1.0:
                rejected.append({"date": day, "symbol": candidate.symbol,
                                 "reason": "insufficient_cash_or_quantity"})
                continue

            cache_key = (
                candidate.strategy_version, candidate.symbol,
                candidate.decision_time.isoformat(), candidate.entry,
                candidate.stop, candidate.target, expiration_sessions,
            )
            if cache_key not in outcome_cache:
                outcome_cache[cache_key] = outcome_simulator(
                    candidate,
                    end_date,
                    indicator_frames[candidate.market],
                    expiration_sessions=expiration_sessions,
                )
            outcome = outcome_cache[cache_key]
            if outcome.get("outcome_data_missing"):
                missing_outcomes.append({"date": day, "symbol": candidate.symbol,
                                         "reason": outcome.get("reason", "unknown")})

            item = ActiveOrder(candidate, sizing["quantity"], sizing["notional"],
                               sizing["risk"], outcome)
            active.append(item)
            held.add(candidate.symbol)
            approved_today += 1
            cash -= sizing["notional"]

            gross = net = costs = pnl_r = None
            if outcome.get("status") == "closed":
                gross = sizing["quantity"] * (
                    float(outcome["exit_price"]) - float(outcome["fill_price"])
                )
                costs = _transaction_cost(item, cost)
                net = gross - costs
                pnl_r = net / sizing["risk"] if sizing["risk"] else None
            trades.append({
                **asdict(candidate),
                "portfolio": portfolio.name,
                "cost_model": cost.name,
                "mode": mode,
                "quantity": sizing["quantity"],
                "risk_amount": sizing["risk"],
                "reserved_notional": sizing["notional"],
                **outcome,
                "gross_pnl": gross,
                "transaction_cost": costs,
                "net_pnl": net,
                "pnl_r": pnl_r,
            })
        day += timedelta(days=1)

    # Realize all modeled end-of-test exits for final equity.
    for item in active:
        if item.outcome.get("status") == "closed":
            gross = item.quantity * (
                float(item.outcome["exit_price"]) - float(item.outcome["fill_price"])
            )
            realized_pnl += gross - _transaction_cost(item, cost)
    final_equity = portfolio.starting_equity + realized_pnl
    if daily_equity:
        daily_equity[-1]["equity"] = final_equity
    return {
        "mode": mode,
        "portfolio": portfolio.name,
        "cost_model": cost.name,
        "trades": trades,
        "rejected": rejected,
        "missing_outcomes": missing_outcomes,
        "daily_equity": daily_equity,
        "final_equity": final_equity,
    }


# == crypto_xsec_momentum_v1 ===================================================
#
# Cross-sectional momentum, added alongside crypto_trend_momentum_v1 and
# crypto_trend_daily_v1 (both untouched). Structurally different from every
# strategy above: it ranks the WHOLE universe together and holds a fixed
# number of positions between weekly rebalances, rather than admitting
# candidates one at a time off an independent per-symbol gate. Forcing that
# shape through simulate_portfolio()/simulate_order_outcome() (built for
# independent per-symbol admission against a shared capacity limit) would
# have needed sizing/exit tricks layered on top of machinery that doesn't
# fit - a dedicated, purpose-built simulator below is simpler and less
# error-prone than that. Reuses the daily-bar frame and helpers from the
# crypto_trend_daily_v1 work (fetch_daily_crypto_frames,
# build_daily_crypto_indicator_frames, daily_completed_bar_cutoff,
# _btc_macro_ok, ResearchCost) rather than duplicating them.

XSEC_REBALANCE_WEEKDAY_DEFAULT = 0  # Monday. Config, not hardcoded - see XsecMomentumConfig.


@dataclass(frozen=True)
class XsecMomentumConfig:
    lookback_days: int = 90          # trailing-return ranking window
    top_n: int = 3
    rebalance_weekday: int = XSEC_REBALANCE_WEEKDAY_DEFAULT
    vol_lookback_days: int = 30      # realized-vol window for position sizing
    target_daily_vol_pct: float = 0.005   # 0.5% of equity, per position, per day
    catastrophic_atr_mult: float = 3.0    # fixed stop, set at entry, never trailed
    target_atr_mult: float = 10.0         # wide - see module note below
    btc_macro_gate: bool = True
    max_position_pct: float = 0.20


def _trailing_return(frame: pd.DataFrame, day: date, lookback_days: int):
    """(close at day's completed prior-day bar) / (close lookback_days
    bars earlier) - 1. None if either bar is unavailable."""
    cutoff = daily_completed_bar_cutoff(day)
    if cutoff not in frame.index:
        return None
    pos = frame.index.get_loc(cutoff)
    lookback_pos = pos - lookback_days
    if lookback_pos < 0:
        return None
    recent = frame.iloc[pos]["close"]
    past = frame.iloc[lookback_pos]["close"]
    if pd.isna(recent) or pd.isna(past) or past <= 0:
        return None
    return float(recent / past - 1.0)


def _realized_vol(frame: pd.DataFrame, day: date, vol_lookback_days: int):
    """Standard deviation of daily pct-change returns over the trailing
    vol_lookback_days bars ending at day's completed prior-day bar."""
    cutoff = daily_completed_bar_cutoff(day)
    if cutoff not in frame.index:
        return None
    pos = frame.index.get_loc(cutoff)
    start_pos = pos - vol_lookback_days + 1
    if start_pos < 0:
        return None
    closes = frame.iloc[start_pos:pos + 1]["close"]
    if len(closes) < vol_lookback_days or closes.isna().any():
        return None
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    vol = float(returns.std())
    return vol if vol > 0 else None


def _xsec_bar(frame: pd.DataFrame, day: date):
    """The completed prior-day bar itself (close/low/atr_14), for entry
    pricing, stop-hit checks, and sizing - None if unavailable."""
    cutoff = daily_completed_bar_cutoff(day)
    if cutoff not in frame.index:
        return None
    row = frame.loc[cutoff]
    if pd.isna(row.get("close")) or pd.isna(row.get("low")) or pd.isna(row.get("atr_14")):
        return None
    return row


@dataclass
class _XsecPosition:
    symbol: str
    entry_date: date
    entry_price: float
    quantity: float
    stop: float          # fixed at entry, never trailed
    target: float
    entry_notional: float


def simulate_xsec_momentum_portfolio(
    daily_ind: dict[str, pd.DataFrame],
    hourly_crypto_frames: dict[str, pd.DataFrame],
    start_date: date,
    end_date: date,
    portfolio: ResearchPortfolio,
    cost: ResearchCost,
    config: XsecMomentumConfig = XsecMomentumConfig(),
) -> dict:
    """Weekly cross-sectional rebalance, daily stop monitoring in between.
    Reuses portfolio.starting_equity/max_position_pct (portfolio.risk_per_trade
    and max_new_trades_per_day are NOT used here - sizing is vol-targeted,
    not risk-per-trade, and admission isn't a per-day-count-limited gate;
    both fields are simply irrelevant to this strategy's shape)."""
    equity = portfolio.starting_equity
    cash = portfolio.starting_equity
    open_positions: dict[str, _XsecPosition] = {}
    trades: list[dict] = []
    daily_equity: list[dict] = []
    rejected: list[dict] = []

    day = start_date
    while day <= end_date:
        # -- 1. Daily catastrophic-stop check on every open position ------
        for symbol in list(open_positions.keys()):
            frame = daily_ind.get(symbol)
            bar = _xsec_bar(frame, day) if frame is not None else None
            if bar is None:
                continue
            pos = open_positions[symbol]
            if float(bar["low"]) <= pos.stop:
                exit_price = pos.stop
                trades.append(_xsec_close_trade(pos, day, exit_price, "catastrophic_stop", cost))
                net = pos.quantity * (exit_price - pos.entry_price) - _xsec_transaction_cost(pos, exit_price, cost)
                cash += pos.entry_notional + net
                del open_positions[symbol]

        # -- 2. Weekly rebalance --------------------------------------------
        if day.weekday() == config.rebalance_weekday:
            macro = _btc_macro_ok(hourly_crypto_frames, day) if config.btc_macro_gate else True
            ranked: list[tuple[str, float]] = []
            if not config.btc_macro_gate or macro:
                for symbol, frame in daily_ind.items():
                    ret = _trailing_return(frame, day, config.lookback_days)
                    if ret is None:
                        continue
                    ranked.append((symbol, ret))
            ranked.sort(key=lambda t: (-t[1], t[0]))
            target_symbols = {s for s, _ in ranked[:config.top_n]}

            # Exit anything that dropped out of the top N.
            for symbol in list(open_positions.keys()):
                if symbol in target_symbols:
                    continue
                frame = daily_ind.get(symbol)
                bar = _xsec_bar(frame, day) if frame is not None else None
                if bar is None:
                    rejected.append({"date": day, "symbol": symbol, "reason": "no_bar_for_rebalance_exit"})
                    continue
                pos = open_positions[symbol]
                exit_price = float(bar["close"])
                trades.append(_xsec_close_trade(pos, day, exit_price, "dropped_from_top_n", cost))
                gross = pos.quantity * (exit_price - pos.entry_price)
                net = gross - _xsec_transaction_cost(pos, exit_price, cost)
                cash += pos.entry_notional + net
                del open_positions[symbol]

            # Enter anything newly in the top N that isn't already held.
            for symbol in target_symbols:
                if symbol in open_positions:
                    continue
                frame = daily_ind[symbol]
                bar = _xsec_bar(frame, day)
                vol = _realized_vol(frame, day, config.vol_lookback_days)
                if bar is None or vol is None:
                    rejected.append({"date": day, "symbol": symbol, "reason": "insufficient_data_for_entry"})
                    continue
                price = float(bar["close"])
                atr_14 = float(bar["atr_14"])
                target_dollar_vol = equity * config.target_daily_vol_pct
                notional = target_dollar_vol / vol
                notional = min(notional, equity * config.max_position_pct, cash)
                quantity = math.floor(notional / price * 1_000_000_000) / 1_000_000_000
                if quantity <= 0:
                    rejected.append({"date": day, "symbol": symbol, "reason": "insufficient_cash_or_quantity"})
                    continue
                entry_notional = quantity * price
                stop = price - config.catastrophic_atr_mult * atr_14
                target = price + config.target_atr_mult * atr_14
                # No separate entry-side cost deduction here - the full
                # round-trip fee (entry leg + exit leg) is applied exactly
                # once, at close time, via _xsec_transaction_cost() below
                # (matching _transaction_cost()'s existing pattern in
                # simulate_portfolio() above). Deducting it here too would
                # double-count the entry leg.
                cash -= entry_notional
                open_positions[symbol] = _XsecPosition(
                    symbol=symbol, entry_date=day, entry_price=price,
                    quantity=quantity, stop=stop, target=target, entry_notional=entry_notional,
                )

        # -- 3. Mark-to-market equity for the day ----------------------------
        unrealized = 0.0
        for symbol, pos in open_positions.items():
            frame = daily_ind.get(symbol)
            bar = _xsec_bar(frame, day) if frame is not None else None
            if bar is None:
                continue
            unrealized += pos.quantity * (float(bar["close"]) - pos.entry_price)
        equity = cash + sum(p.entry_notional for p in open_positions.values()) + unrealized
        daily_equity.append({"date": day, "equity": equity})

        day += timedelta(days=1)

    # Realize all still-open positions at end of test for final equity.
    final_equity = cash
    for symbol, pos in open_positions.items():
        frame = daily_ind.get(symbol)
        bar = _xsec_bar(frame, end_date) if frame is not None else None
        exit_price = float(bar["close"]) if bar is not None else pos.entry_price
        trades.append(_xsec_close_trade(pos, end_date, exit_price, "end_of_test", cost))
        gross = pos.quantity * (exit_price - pos.entry_price)
        net = gross - _xsec_transaction_cost(pos, exit_price, cost)
        final_equity += pos.entry_notional + net
    if daily_equity:
        daily_equity[-1]["equity"] = final_equity

    return {
        "mode": "crypto_xsec_momentum",
        "portfolio": portfolio.name,
        "cost_model": cost.name,
        "config": asdict(config),
        "trades": trades,
        "rejected": rejected,
        "missing_outcomes": [],
        "daily_equity": daily_equity,
        "final_equity": final_equity,
    }


def _xsec_transaction_cost(pos: _XsecPosition, exit_price: float, cost: ResearchCost) -> float:
    entry_notional = pos.quantity * pos.entry_price
    exit_notional = pos.quantity * exit_price
    return (entry_notional + exit_notional) * cost.crypto_bps_per_leg / 10_000.0


def _xsec_close_trade(pos: _XsecPosition, exit_date: date, exit_price: float, reason: str, cost: ResearchCost) -> dict:
    gross = pos.quantity * (exit_price - pos.entry_price)
    transaction_cost = _xsec_transaction_cost(pos, exit_price, cost)
    net = gross - transaction_cost
    risk_amount = pos.quantity * (pos.entry_price - pos.stop)
    return {
        "symbol": pos.symbol,
        "market": "crypto",
        "strategy_version": "crypto_xsec_momentum_v1",
        "timeframe": "daily",
        "entry_date": pos.entry_date,
        "exit_date": exit_date,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "quantity": pos.quantity,
        "stop": pos.stop,
        "target": pos.target,
        "exit_reason": reason,
        "gross_pnl": gross,
        "transaction_cost": transaction_cost,
        "net_pnl": net,
        "pnl_r": net / risk_amount if risk_amount else None,
    }
