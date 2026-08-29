"""
Tests for backtest/whole_bot_engine.py's crypto_xsec_momentum_v1 simulator
(simulate_xsec_momentum_portfolio and its helpers) - a dedicated simulator,
not a reuse of simulate_portfolio()/simulate_order_outcome(), because
cross-sectional weekly rebalancing has a fundamentally different shape than
independent per-symbol daily admission.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import backtest.whole_bot_engine as engine
from backtest.whole_bot_engine import (
    XsecMomentumConfig,
    _realized_vol,
    _trailing_return,
    _xsec_bar,
    simulate_xsec_momentum_portfolio,
)


def _synthetic_daily_frame(closes, start="2025-01-01"):
    idx = pd.date_range(start, periods=len(closes), freq="D")
    close = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=idx)
    return engine.build_daily_crypto_indicator_frames({"X": df})["X"]


def _noisy_trend(daily_return, days, seed, start_price=100.0, noise_std=0.01):
    """A perfectly smooth compounding series has ~zero realized vol by
    construction, which breaks vol-targeted sizing (division by ~zero).
    Real daily bars are never this smooth - add small, seeded (deterministic)
    noise on top of the trend so realized vol is realistic and nonzero,
    while the trend still dominates the cumulative trailing return."""
    rng = np.random.default_rng(seed)
    prices = [start_price]
    for _ in range(days - 1):
        prices.append(prices[-1] * (1 + daily_return) * (1 + rng.normal(0, noise_std)))
    return prices


def _bullish_hourly_btc(start, end):
    idx = pd.date_range(start, end, freq="h")
    close = np.linspace(80.0, 120.0, len(idx))
    return {"BTC-USD": pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=idx)}


def _bearish_hourly_btc(start, end):
    idx = pd.date_range(start, end, freq="h")
    close = np.linspace(120.0, 80.0, len(idx))
    return {"BTC-USD": pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=idx)}


# -- Pure helpers -------------------------------------------------------------

def test_trailing_return_basic():
    frame = _synthetic_daily_frame([100.0 * (1.01 ** i) for i in range(30)])
    ret = _trailing_return(frame, date(2025, 1, 30), lookback_days=10)
    # close at day 29 (index) vs close 10 bars earlier
    expected = frame.iloc[-1]["close"] / frame.iloc[-11]["close"] - 1.0
    assert ret == pytest.approx(expected)


def test_trailing_return_none_when_insufficient_history():
    frame = _synthetic_daily_frame([100.0] * 5)
    assert _trailing_return(frame, date(2025, 1, 5), lookback_days=10) is None


def test_realized_vol_matches_manual_stdev():
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
    frame = _synthetic_daily_frame(closes)
    vol = _realized_vol(frame, date(2025, 1, 6), vol_lookback_days=5)
    # The window for a decision made "just after" 2025-01-06's UTC close is
    # the 5 bars [01-01..01-05] (cutoff = day-1) - 01-06's own bar is
    # correctly excluded, matching daily_completed_bar_cutoff().
    manual = pd.Series(closes[:5]).pct_change().dropna().std()
    assert vol == pytest.approx(manual)


def test_realized_vol_none_when_insufficient_history():
    frame = _synthetic_daily_frame([100.0, 101.0])
    assert _realized_vol(frame, date(2025, 1, 2), vol_lookback_days=30) is None


def test_xsec_bar_none_on_missing_day():
    frame = _synthetic_daily_frame([100.0] * 5)
    assert _xsec_bar(frame, date(2030, 1, 1)) is None


# -- Full simulator: ranking, sizing, exits -----------------------------------

ATR_WARMUP_DAYS = 20  # atr_14 needs 14 bars; pad before the intended start date


def _three_symbol_frames(days=40, start="2025-01-06"):
    """A: strong uptrend, B: mild uptrend, C: flat (all with small seeded
    noise so realized vol is realistic/nonzero - see _noisy_trend). start
    defaults to a real Monday so rebalance_weekday=0 lines up predictably.
    Generates ATR_WARMUP_DAYS of history before `start` so atr_14 (needed
    by _xsec_bar for every entry) is already warm by the first real
    rebalance - a bare `start` with zero pre-history left atr_14 NaN for
    the first ~2 weeks, silently suppressing every entry."""
    frame_start = pd.Timestamp(start) - timedelta(days=ATR_WARMUP_DAYS)
    total_days = days + ATR_WARMUP_DAYS
    a = _synthetic_daily_frame(_noisy_trend(0.03, total_days, seed=1), start=frame_start)
    b = _synthetic_daily_frame(_noisy_trend(0.005, total_days, seed=2), start=frame_start)
    c = _synthetic_daily_frame(_noisy_trend(0.0, total_days, seed=3), start=frame_start)
    return {"A-USD": a, "B-USD": b, "C-USD": c}


def test_start_date_is_actually_a_monday_sanity_check():
    assert date(2025, 1, 6).weekday() == 0


def test_top_n_one_selects_the_strongest_trailing_return():
    frames = _three_symbol_frames()
    start, end = date(2025, 1, 6), date(2025, 2, 3)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=10, top_n=1, vol_lookback_days=5, rebalance_weekday=0)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    symbols_traded = {t["symbol"] for t in result["trades"]}
    assert symbols_traded == {"A-USD"}  # never B or C - always outranked


def test_position_sizing_targets_the_configured_daily_dollar_vol():
    frames = _three_symbol_frames()
    start, end = date(2025, 1, 6), date(2025, 1, 20)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(
        lookback_days=10, top_n=1, vol_lookback_days=5, rebalance_weekday=0,
        target_daily_vol_pct=0.005, max_position_pct=1.0,  # cap disabled for this test
    )
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    first_trade = result["trades"][0]
    frame = frames["A-USD"]
    entry_vol = _realized_vol(frame, first_trade["entry_date"], config.vol_lookback_days)
    expected_notional = (100_000.0 * config.target_daily_vol_pct) / entry_vol
    actual_notional = first_trade["quantity"] * first_trade["entry_price"]
    assert actual_notional == pytest.approx(expected_notional, rel=1e-6)


def test_max_position_pct_caps_notional_when_vol_is_very_low():
    # A very-low-but-nonzero-vol symbol would otherwise size to a huge notional.
    days = 40 + ATR_WARMUP_DAYS
    closes = _noisy_trend(0.0002, days, seed=4, noise_std=0.0005)  # tiny drift, tiny noise
    frame_start = pd.Timestamp("2025-01-06") - timedelta(days=ATR_WARMUP_DAYS)
    frame = engine.build_daily_crypto_indicator_frames({"X": pd.DataFrame({
        "open": closes, "high": [c * 1.0001 for c in closes], "low": [c * 0.9999 for c in closes],
        "close": closes, "volume": 1000.0,
    }, index=pd.date_range(frame_start, periods=days, freq="D"))})["X"]
    frames = {"A-USD": frame}
    start, end = date(2025, 1, 6), date(2025, 1, 13)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(
        lookback_days=5, top_n=1, vol_lookback_days=5, rebalance_weekday=0,
        target_daily_vol_pct=0.005, max_position_pct=0.20,
    )
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    assert len(result["trades"]) >= 1
    trade = result["trades"][0]
    notional = trade["quantity"] * trade["entry_price"]
    assert notional == pytest.approx(100_000.0 * 0.20, rel=1e-6)


def test_catastrophic_stop_exits_at_the_stop_price():
    days = 40 + ATR_WARMUP_DAYS
    idx = pd.date_range(pd.Timestamp("2025-01-06") - timedelta(days=ATR_WARMUP_DAYS), periods=days, freq="D")
    closes = _noisy_trend(0.03, 20 + ATR_WARMUP_DAYS, seed=5)
    crash_len = days - len(closes)
    # A realistic ~1% daily range throughout (including the uptrend phase -
    # an inflated low there would inflate ATR14 enough to push the 3xATR
    # stop below zero, so the crash could never actually breach it) - then
    # a single-bar 50% crash, an order of magnitude beyond any reasonable
    # 3xATR stop on a ~1%-range asset.
    lows = [c * 0.99 for c in closes] + [closes[-1] * 0.5 * 0.99] + [closes[-1] * 0.5] * (crash_len - 1)
    closes = closes + [closes[-1] * 0.5] * crash_len
    frame = engine.build_daily_crypto_indicator_frames({"X": pd.DataFrame({
        "open": closes, "high": [c * 1.01 for c in closes], "low": lows,
        "close": closes, "volume": 1000.0,
    }, index=idx)})["X"]
    frames = {"A-USD": frame}
    start, end = date(2025, 1, 6), date(2025, 2, 10)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=5, top_n=1, vol_lookback_days=5, rebalance_weekday=0)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    stop_exits = [t for t in result["trades"] if t["exit_reason"] == "catastrophic_stop"]
    assert stop_exits
    t = stop_exits[0]
    assert t["exit_price"] == pytest.approx(t["stop"])


def test_dropped_from_top_n_closes_the_position():
    # A wins week 1's ranking, then decelerates while B accelerates so B
    # wins week 2 - A must be closed with exit_reason dropped_from_top_n.
    days = 21
    idx = pd.date_range(pd.Timestamp("2025-01-06") - timedelta(days=ATR_WARMUP_DAYS), periods=days + ATR_WARMUP_DAYS, freq="D")
    a_warmup = _noisy_trend(0.01, ATR_WARMUP_DAYS, seed=10)
    a_early = _noisy_trend(0.05, 7, seed=6, start_price=a_warmup[-1])
    a_closes = a_warmup + a_early + list(_noisy_trend(0.0, days - 7 + 1, seed=7, start_price=a_early[-1]))[1:]
    b_warmup = _noisy_trend(0.01, ATR_WARMUP_DAYS, seed=11)
    b_flat = _noisy_trend(0.0, 7, seed=8, start_price=b_warmup[-1])
    b_closes = b_warmup + b_flat + list(_noisy_trend(0.08, days - 7 + 1, seed=9, start_price=b_flat[-1]))[1:]
    frame_a = engine.build_daily_crypto_indicator_frames({"X": pd.DataFrame({
        "open": a_closes, "high": [c * 1.01 for c in a_closes], "low": [c * 0.99 for c in a_closes],
        "close": a_closes, "volume": 1000.0,
    }, index=idx)})["X"]
    frame_b = engine.build_daily_crypto_indicator_frames({"X": pd.DataFrame({
        "open": b_closes, "high": [c * 1.01 for c in b_closes], "low": [c * 0.99 for c in b_closes],
        "close": b_closes, "volume": 1000.0,
    }, index=idx)})["X"]
    frames = {"A-USD": frame_a, "B-USD": frame_b}
    start, end = date(2025, 1, 6), date(2025, 1, 27)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=5, top_n=1, vol_lookback_days=5, rebalance_weekday=0)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    dropped = [t for t in result["trades"] if t["exit_reason"] == "dropped_from_top_n" and t["symbol"] == "A-USD"]
    assert dropped


# -- BTC macro gate -------------------------------------------------------

def test_btc_macro_gate_on_blocks_entries_when_btc_bearish():
    frames = _three_symbol_frames()
    start, end = date(2025, 1, 6), date(2025, 1, 13)
    hourly = _bearish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=5, top_n=1, vol_lookback_days=5, rebalance_weekday=0, btc_macro_gate=True)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    assert result["trades"] == []


def test_btc_macro_gate_off_ignores_btc_trend():
    frames = _three_symbol_frames()
    start, end = date(2025, 1, 6), date(2025, 1, 13)
    hourly = _bearish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=5, top_n=1, vol_lookback_days=5, rebalance_weekday=0, btc_macro_gate=False)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(
        frames, hourly, start, end, portfolio, engine.COSTS["zero"], config,
    )
    assert len(result["trades"]) >= 1


# -- Fees -------------------------------------------------------------------

def test_baseline_cost_reduces_net_pnl_versus_zero_cost():
    frames = _three_symbol_frames()
    start, end = date(2025, 1, 6), date(2025, 2, 3)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=10, top_n=1, vol_lookback_days=5, rebalance_weekday=0)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    zero = simulate_xsec_momentum_portfolio(frames, hourly, start, end, portfolio, engine.COSTS["zero"], config)
    baseline = simulate_xsec_momentum_portfolio(frames, hourly, start, end, portfolio, engine.COSTS["baseline"], config)
    assert baseline["final_equity"] < zero["final_equity"]


def test_rejected_when_no_bar_available_for_entry():
    """A symbol that ranks in the top N but has no valid bar at the
    rebalance decision (e.g. NaN atr_14) must be rejected, not guessed."""
    days = 10
    idx = pd.date_range("2025-01-06", periods=days, freq="D")
    closes = [100.0] * days
    frame = pd.DataFrame({
        "close": closes, "sma_20": 95.0, "sma_50": 90.0, "sma_200": np.nan,
        "rsi_14": 60.0, "macd_hist": 1.0, "atr_14": np.nan, "rel_volume": 1.5,
        "high": closes, "low": closes, "trend": "UPTREND",
    }, index=idx)
    frames = {"A-USD": frame}
    start, end = date(2025, 1, 6), date(2025, 1, 6)
    hourly = _bullish_hourly_btc(start - timedelta(days=60), end)
    config = XsecMomentumConfig(lookback_days=1, top_n=1, vol_lookback_days=1, rebalance_weekday=0)
    portfolio = engine.ResearchPortfolio("test", starting_equity=100_000.0)
    result = simulate_xsec_momentum_portfolio(frames, hourly, start, end, portfolio, engine.COSTS["zero"], config)
    assert result["trades"] == []
