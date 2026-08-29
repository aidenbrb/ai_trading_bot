"""
Tests for backtest/crypto_regime_benchmarks.py (Phase 3 Step 3 addendum
item 1) - control benchmarks isolating the BTC 20-day-SMA regime filter
alone, with no other entry logic.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import backtest.whole_bot_engine as engine
from backtest.crypto_regime_benchmarks import _regime_on, simulate_regime_benchmark


def _noisy_trend(daily_return, days, seed, start_price=100.0, noise_std=0.01):
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_return) * (1 + rng.normal(0, noise_std)))
    return prices


def _frame(closes, start):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    df = pd.DataFrame({
        "open": closes, "high": [c * 1.01 for c in closes], "low": [c * 0.99 for c in closes],
        "close": closes, "volume": 1000.0,
    }, index=idx)
    return engine.build_daily_crypto_indicator_frames({"X": df})["X"]


WARMUP = 25  # >= 20 (SMA20) with a small buffer
VOL_WARMUP = 35  # >= 30 (default vol_lookback_days for inverse_vol) with a small buffer


def test_regime_on_true_above_sma20_false_below():
    # Rising trend -> close should sit above its own trailing SMA20 late in the series.
    closes = _noisy_trend(0.02, 60, seed=1, noise_std=0.002)
    frame = _frame(closes, "2026-01-01")
    day = date(2026, 1, 1) + timedelta(days=55)
    assert _regime_on({"BTC-USD": frame}, day) is True

    # Flat-then-crash -> close should sit below its own trailing SMA20 right after the crash.
    closes2 = _noisy_trend(0.0, 40, seed=2, noise_std=0.002)
    closes2 += [closes2[-1] * 0.5] * 5
    frame2 = _frame(closes2, "2026-01-01")
    day2 = date(2026, 1, 1) + timedelta(days=42)
    assert _regime_on({"BTC-USD": frame2}, day2) is False


def test_regime_on_none_when_data_missing():
    assert _regime_on({}, date(2026, 1, 1)) is None
    assert _regime_on({"BTC-USD": pd.DataFrame()}, date(2026, 1, 1)) is None


def test_btc_only_benchmark_opens_and_closes_on_regime_flip():
    # Warmup flat, then a clean rally (regime turns ON), then a crash (regime turns OFF).
    warmup = _noisy_trend(0.0, WARMUP, seed=3, noise_std=0.001)
    rally = _noisy_trend(0.02, 30, seed=4, start_price=warmup[-1], noise_std=0.002)
    crash = [rally[-1] * 0.5] * 20
    closes = warmup + rally[1:] + crash
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}

    start = date(2026, 1, 1)
    end = start + timedelta(days=len(closes) - 1)
    result = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"])

    assert len(result["trades"]) >= 1
    t = result["trades"][0]
    assert t["symbol"] == "BTC-USD"
    # Entered during the rally phase, not during the flat warmup.
    assert t["entry_date"] > start + timedelta(days=WARMUP - 5)


def test_flat_the_whole_time_produces_no_trades():
    closes = _noisy_trend(0.0, 60, seed=5, noise_std=0.001)
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}
    start = date(2026, 1, 1)
    end = start + timedelta(days=59)
    result = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"])
    assert result["trades"] == []
    assert result["final_equity"] == pytest.approx(100_000.0)


def test_inverse_vol_basket_weights_the_lower_vol_symbol_more():
    warmup_a = _noisy_trend(0.0, VOL_WARMUP, seed=6, noise_std=0.001)   # low vol
    rally_a = _noisy_trend(0.01, 10, seed=6, start_price=warmup_a[-1], noise_std=0.001)
    a_closes = warmup_a + rally_a[1:]

    warmup_b = _noisy_trend(0.0, VOL_WARMUP, seed=7, noise_std=0.03)    # high vol
    rally_b = _noisy_trend(0.01, 10, seed=7, start_price=warmup_b[-1], noise_std=0.03)
    b_closes = warmup_b + rally_b[1:]

    frame_a = _frame(a_closes, "2026-01-01")
    frame_b = _frame(b_closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame_a, "ETH-USD": frame_b}

    start = date(2026, 1, 1)
    end = start + timedelta(days=len(a_closes) - 1)
    result = simulate_regime_benchmark(
        daily_ind, ["BTC-USD", "ETH-USD"], start, end, engine.COSTS["zero"], inverse_vol=True,
    )
    entries = [t for t in result["trades"]]
    btc_notional = next(t["quantity"] * t["fill_price"] for t in entries if t["symbol"] == "BTC-USD")
    eth_notional = next(t["quantity"] * t["fill_price"] for t in entries if t["symbol"] == "ETH-USD")
    # Lower-vol BTC must receive MORE notional than higher-vol ETH.
    assert btc_notional > eth_notional


def test_baseline_cost_reduces_final_equity_versus_zero_cost():
    warmup = _noisy_trend(0.0, WARMUP, seed=8, noise_std=0.001)
    rally = _noisy_trend(0.02, 20, seed=8, start_price=warmup[-1], noise_std=0.002)
    crash = [rally[-1] * 0.9] * 10
    closes = warmup + rally[1:] + crash
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}
    start = date(2026, 1, 1)
    end = start + timedelta(days=len(closes) - 1)

    zero = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"])
    baseline = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["baseline"])
    assert baseline["final_equity"] < zero["final_equity"]


# -- entry_days="weekend" / min_hold_days (Phase 3 addendum item 1) -----------

def test_weekend_entry_only_ever_lands_on_saturday_or_sunday():
    # Multiple regime on/off cycles across a long enough window that entries
    # land on a variety of weekdays under entry_days="daily" (control) - then
    # confirm entry_days="weekend" forces every single one onto a Sat/Sun,
    # regardless of which weekday the regime itself actually flipped on.
    warmup = _noisy_trend(0.0, WARMUP, seed=12, noise_std=0.02)
    cycles = []
    price = warmup[-1]
    for c in range(6):
        up = _noisy_trend(0.03, 8, seed=100 + c, start_price=price, noise_std=0.01)
        price = up[-1]
        down = _noisy_trend(-0.03, 8, seed=200 + c, start_price=price, noise_std=0.01)
        price = down[-1]
        cycles += up[1:] + down[1:]
    closes = warmup + cycles
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}
    start = date(2026, 1, 1)
    end = start + timedelta(days=len(closes) - 1)

    daily_mode = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"], entry_days="daily")
    weekend_mode = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"], entry_days="weekend")

    assert daily_mode["trades"], "sanity: control run must have actually traded"
    assert weekend_mode["trades"], "sanity: weekend-entry run must have actually traded"
    assert any(t["entry_date"].weekday() < 5 for t in daily_mode["trades"]), (
        "sanity: the daily-entry control should include at least one weekday entry, "
        "otherwise this scenario can't actually distinguish the two modes"
    )
    assert all(t["entry_date"].weekday() >= 5 for t in weekend_mode["trades"])


def test_min_hold_ignores_an_early_regime_off_signal():
    # Regime on, then off after just 2 days (well under a 7-day min hold) -
    # position must stay open at least 7 days regardless.
    warmup = _noisy_trend(0.0, WARMUP, seed=10, noise_std=0.001)
    rally = _noisy_trend(0.03, 5, seed=10, start_price=warmup[-1], noise_std=0.001)
    crash = [rally[-1] * 0.6] * 20  # regime flips off almost immediately after entry
    closes = warmup + rally[1:] + crash
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}
    start = date(2026, 1, 1)
    end = start + timedelta(days=len(closes) - 1)

    no_hold = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"], min_hold_days=0)
    with_hold = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"], min_hold_days=7)

    assert no_hold["trades"], "sanity: the no-hold run must have actually traded"
    assert with_hold["trades"], "sanity: the min-hold run must have actually traded"
    no_hold_hold_days = (no_hold["trades"][0]["exit_date"] - no_hold["trades"][0]["entry_date"]).days
    with_hold_hold_days = (with_hold["trades"][0]["exit_date"] - with_hold["trades"][0]["entry_date"]).days
    assert with_hold_hold_days >= 7
    assert with_hold_hold_days > no_hold_hold_days


def test_min_hold_zero_is_equivalent_to_the_default():
    warmup = _noisy_trend(0.0, WARMUP, seed=11, noise_std=0.001)
    rally = _noisy_trend(0.02, 20, seed=11, start_price=warmup[-1], noise_std=0.002)
    closes = warmup + rally[1:]
    frame = _frame(closes, "2026-01-01")
    daily_ind = {"BTC-USD": frame}
    start = date(2026, 1, 1)
    end = start + timedelta(days=len(closes) - 1)

    default = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"])
    explicit_zero_hold = simulate_regime_benchmark(daily_ind, ["BTC-USD"], start, end, engine.COSTS["zero"], min_hold_days=0)
    assert default["final_equity"] == pytest.approx(explicit_zero_hold["final_equity"])
