"""
Tests for backtest/crypto_sensitivity.py (Phase 3 Step 6) - the
sma50_rising sensitivity harness. Its core claim is that at matching
parameters (sma_length=50, rising_lookback=10) it reproduces the real,
unmodified crypto_trend_daily_v1(entry_mode="sma50_rising") exactly - a
real bug (missing sma_20/sma_200 in the "usable" bar, letting the harness
treat symbols as usable up to ~150 days earlier than the real strategy
ever would) was caught by this exact comparison before trusting the grid.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import backtest.whole_bot_engine as engine
from backtest.crypto_sensitivity import build_sma_sensitivity_calendar
from utils.strategy_signals import crypto_trend_daily_v1
from functools import partial


def _noisy_trend(daily_return, days, seed, start_price=100.0, noise_std=0.01):
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_return) * (1 + rng.normal(0, noise_std)))
    return prices


def _multi_symbol_frames(days=260, start="2022-01-01"):
    out = {}
    for i, sym in enumerate(["BTC-USD", "ETH-USD", "SOL-USD"]):
        closes = _noisy_trend(0.002, days, seed=i, noise_std=0.02)
        idx = pd.date_range(start, periods=days, freq="D")
        df = pd.DataFrame({
            "open": closes, "high": [c * 1.02 for c in closes], "low": [c * 0.98 for c in closes],
            "close": closes, "volume": 1000.0,
        }, index=idx)
        out[sym] = df
    return out


def test_matches_the_real_strategy_exactly_at_matching_parameters():
    raw = _multi_symbol_frames()
    daily_ind = engine.build_daily_crypto_indicator_frames(raw)
    hourly_idx = pd.date_range("2021-10-01", "2022-09-18", freq="h")
    hourly_btc = pd.DataFrame({
        "open": np.linspace(80.0, 120.0, len(hourly_idx)),
        "high": np.linspace(81.0, 121.0, len(hourly_idx)),
        "low": np.linspace(79.0, 119.0, len(hourly_idx)),
        "close": np.linspace(80.0, 120.0, len(hourly_idx)),
        "volume": 1000.0,
    }, index=hourly_idx)
    hourly = {"BTC-USD": hourly_btc}

    start, end = date(2022, 1, 1), date(2022, 9, 17)
    fn = partial(crypto_trend_daily_v1, entry_mode="sma50_rising")
    real_cal, real_meta = engine.build_daily_crypto_calendar(daily_ind, hourly, start, end, fn)
    sens_cal, sens_meta = build_sma_sensitivity_calendar(
        daily_ind, hourly, start, end, sma_length=50, rising_lookback=10,
    )

    assert sens_meta["coverage"] == real_meta["coverage"]
    real_keys = {(d, c.symbol) for d, cands in real_cal.items() for c in cands}
    sens_keys = {(d, c.symbol) for d, cands in sens_cal.items() for c in cands}
    assert real_keys == sens_keys


def test_empty_symbol_frame_does_not_crash():
    """The exact scenario that surfaced the required-columns bug: a
    symbol with zero rows within the test window (even with warmup
    padding) must be excluded cleanly, not raise a KeyError."""
    raw = _multi_symbol_frames()
    raw["DEAD-USD"] = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    daily_ind = engine.build_daily_crypto_indicator_frames(raw)
    hourly = {"BTC-USD": pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
    }, index=pd.date_range("2021-10-01", "2022-09-18", freq="h"))}
    start, end = date(2022, 1, 1), date(2022, 3, 1)
    calendar, meta = build_sma_sensitivity_calendar(daily_ind, hourly, start, end, sma_length=50, rising_lookback=10)
    assert meta["coverage"]["attempted"] > 0  # ran to completion, no crash
