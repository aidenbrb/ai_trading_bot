"""
Phase 3 Step 6: in-sample-only sensitivity grid for crypto_trend_daily_v1's
sma50_rising entry mode - SMA length x rising-lookback.

crypto_trend_daily_v1 itself (utils/strategy_signals.py) is NOT modified:
its sma50_rising branch only ever reads indicator.sma_50 and
indicator.sma_50_prior by attribute name - it has no idea those happen to
be a 50-period SMA under the hood. This harness exploits that: it computes
a CUSTOM-length SMA and a CUSTOM-lookback "prior" value directly from each
symbol's raw daily closes, attaches them as .sma_50/.sma_50_prior on an
otherwise-normal daily indicator object (same rsi_14/macd_hist/atr_14/
rel_volume/close as the real daily frame), and calls the real, unmodified
crypto_trend_daily_v1(entry_mode="sma50_rising") - so every other gate
(RSI/MACD/relvol/ATR bracket) stays byte-identical to the real strategy.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from backtest.whole_bot_engine import (
    Candidate,
    _btc_macro_ok,
    _indicator_object,
    daily_completed_bar_cutoff,
    daily_decision_time_utc,
)
from utils.strategy_signals import crypto_trend_daily_v1


def _custom_sma_series(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(window=length, min_periods=length).mean()


def build_sma_sensitivity_calendar(
    daily_ind: dict[str, pd.DataFrame],
    hourly_crypto_frames: dict[str, pd.DataFrame],
    start: date,
    end: date,
    *,
    sma_length: int,
    rising_lookback: int,
    min_rr: float = 2.0,
) -> tuple[dict[date, list[Candidate]], dict]:
    """sma50_rising with a configurable SMA length and rising-lookback,
    using the real, unmodified crypto_trend_daily_v1 for every other gate.
    Same calendar/coverage shape as build_daily_crypto_calendar()."""
    custom_sma = {sym: _custom_sma_series(frame["close"], sma_length) for sym, frame in daily_ind.items()}

    # Matches build_daily_crypto_calendar()/_daily_snapshot()'s required set
    # exactly (sma_20/sma_200 included even though sma50_rising's own gate
    # never reads them) - not doing so let this harness treat a symbol as
    # "usable" up to ~150 days before the real strategy would (sma_200's
    # 200-bar warmup is stricter than sma50_rising's own requirements),
    # silently comparing against a larger/earlier candidate universe than
    # cell #1 was ever actually evaluated against. Caught by cross-checking
    # this harness reproduces the real path's exact candidate set at
    # matching (sma_length=50, rising_lookback=10) parameters before
    # trusting it for the sensitivity grid.
    required = ("close", "sma_20", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume")
    first_usable = {}
    for symbol, frame in daily_ind.items():
        sma_col = custom_sma[symbol]
        if frame.empty:
            # indicator_frame() returns early (raw OHLCV columns only, no
            # sma_20/rsi_14/etc.) on empty input - e.g. a symbol with zero
            # rows within just this narrower in-sample window even with
            # warmup padding. reindex(columns=...) below would otherwise
            # KeyError on frame[list(required)] for exactly this case.
            first_usable[symbol] = None
            continue
        valid_mask = frame.reindex(columns=required).notna().all(axis=1) & sma_col.reindex(frame.index).notna()
        valid_idx = frame.index[valid_mask]
        first_usable[symbol] = valid_idx[0] if len(valid_idx) else None

    calendar: dict[date, list[Candidate]] = {}
    attempted = 0
    usable = 0
    exclusions: list[dict] = []

    day = start
    while day <= end:
        candidates: list[Candidate] = []
        macro = _btc_macro_ok(hourly_crypto_frames, day)
        cutoff = daily_completed_bar_cutoff(day)
        for symbol, frame in daily_ind.items():
            first = first_usable[symbol]
            if first is not None and cutoff < first:
                exclusions.append({"date": day, "symbol": symbol, "reason": "pre-inception or indicator warmup"})
                continue
            attempted += 1
            if cutoff not in frame.index:
                exclusions.append({"date": day, "symbol": symbol, "reason": "no completed prior-day bar"})
                continue
            row = frame.loc[cutoff]
            sma_series = custom_sma[symbol]
            sma_now = sma_series.get(cutoff)
            if any(pd.isna(row.get(col)) for col in required if col != "close") or pd.isna(row.get("close")) or pd.isna(sma_now):
                exclusions.append({"date": day, "symbol": symbol, "reason": "missing indicator or custom SMA"})
                continue
            pos = sma_series.index.get_loc(cutoff)
            prior_pos = pos - rising_lookback
            sma_prior = float(sma_series.iloc[prior_pos]) if prior_pos >= 0 and pd.notna(sma_series.iloc[prior_pos]) else None
            usable += 1
            if symbol != "BTC-USD" and not macro:
                continue

            indicator = _indicator_object(row)
            indicator.sma_50 = float(sma_now)
            indicator.sma_50_prior = sma_prior
            decision = crypto_trend_daily_v1(
                symbol, indicator, float(row["close"]), min_rr=min_rr, entry_mode="sma50_rising",
            )
            if decision.passed and (decision.conviction_score or 0) >= 70:
                bar_start = row.name.to_pydatetime() if hasattr(row.name, "to_pydatetime") else row.name
                candidates.append(Candidate(
                    symbol=symbol, market="crypto", strategy_version=decision.strategy_version,
                    decision_time=daily_decision_time_utc(day),
                    signal_bar_end=bar_start + timedelta(days=1),
                    entry=float(decision.entry), stop=float(decision.stop), target=float(decision.target),
                    conviction=int(decision.conviction_score), atr=float(row["atr_14"]),
                    timeframe="daily",
                ))

        candidates.sort(key=lambda c: (-c.conviction, c.symbol))
        calendar[day] = candidates
        day += timedelta(days=1)

    coverage = {
        "attempted": attempted, "usable": usable,
        "coverage_rate": usable / attempted if attempted else None,
    }
    return calendar, {"coverage": coverage, "exclusions": exclusions}
