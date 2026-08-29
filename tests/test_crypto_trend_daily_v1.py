"""
Tests for utils/strategy_signals.py::crypto_trend_daily_v1 - added alongside
crypto_trend_momentum_v1 (completely unmodified) after a Phase 1 review
found v1's SMA20/50/200 trend classification runs on hourly bars, not the
daily periods the names imply.
"""
from types import SimpleNamespace

import pytest

from utils.strategy_signals import (
    CRYPTO_DAILY_STRATEGY_VERSION,
    CRYPTO_STRATEGY_VERSION,
    crypto_trend_daily_v1,
    crypto_trend_momentum_v1,
)


def _indicator(**overrides):
    values = dict(
        trend="UPTREND", rsi_14=60.0, macd_hist=0.1, rel_volume=1.5,
        atr_14=2.0, sma_20=105.0, sma_50=100.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# -- strict_stack: must genuinely replicate v1's gate logic on daily inputs -

def test_strict_stack_matches_v1_pass_and_reject_on_identical_inputs():
    indicator = _indicator(rsi_14=63.0, rel_volume=1.7, atr_14=4.0)
    v1 = crypto_trend_momentum_v1("BTC-USD", indicator, 120.0, min_rr=2.0)
    daily = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="strict_stack")
    assert v1.passed and daily.passed
    assert (v1.entry, v1.stop, v1.target, v1.rr) == (daily.entry, daily.stop, daily.target, daily.rr)
    assert daily.strategy_version == CRYPTO_DAILY_STRATEGY_VERSION
    assert v1.strategy_version == CRYPTO_STRATEGY_VERSION  # v1 itself untouched


def test_strict_stack_rejects_non_uptrend():
    indicator = _indicator(trend="SIDEWAYS")
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="strict_stack")
    assert not d.passed
    assert "UPTREND" in d.reason


def test_default_entry_mode_is_strict_stack():
    indicator = _indicator(trend="SIDEWAYS")
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0)
    assert not d.passed and "UPTREND" in d.reason


# -- sma50_rising -------------------------------------------------------------

def test_sma50_rising_passes_when_close_above_sma50_and_sma50_higher_than_prior():
    indicator = _indicator(trend="SIDEWAYS", sma_50=100.0)
    indicator.sma_50_prior = 90.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="sma50_rising")
    assert d.passed


def test_sma50_rising_rejects_when_close_not_above_sma50():
    indicator = _indicator(trend="SIDEWAYS", sma_50=130.0)
    indicator.sma_50_prior = 90.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="sma50_rising")
    assert not d.passed
    assert "<= SMA50" in d.reason


def test_sma50_rising_rejects_when_sma50_not_rising():
    indicator = _indicator(trend="SIDEWAYS", sma_50=100.0)
    indicator.sma_50_prior = 105.0  # SMA50 was HIGHER 10 bars ago - falling, not rising
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="sma50_rising")
    assert not d.passed
    assert "not above its value" in d.reason


def test_sma50_rising_rejects_when_prior_value_missing():
    """No fallback/guess when sma_50_prior wasn't supplied - must reject,
    not silently pass an unverifiable condition."""
    indicator = _indicator(trend="SIDEWAYS", sma_50=100.0)  # sma_50_prior absent entirely
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="sma50_rising")
    assert not d.passed
    assert "not above its value" in d.reason


# -- donchian -------------------------------------------------------------

def test_donchian_passes_on_genuine_breakout():
    indicator = _indicator(trend="SIDEWAYS")
    indicator.high_20 = 115.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="donchian")
    assert d.passed


def test_donchian_rejects_when_not_above_prior_high():
    indicator = _indicator(trend="SIDEWAYS")
    indicator.high_20 = 125.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="donchian")
    assert not d.passed
    assert "did not break above" in d.reason


def test_donchian_rejects_when_high_20_missing():
    indicator = _indicator(trend="SIDEWAYS")  # high_20 absent entirely
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="donchian")
    assert not d.passed
    assert "did not break above" in d.reason


# -- Shared gates 3-5 identical across all three modes -------------------

@pytest.mark.parametrize("entry_mode", ["strict_stack", "sma50_rising", "donchian"])
def test_rsi_gate_applies_identically_across_modes(entry_mode):
    indicator = _indicator(rsi_14=40.0)  # outside 50-75
    indicator.sma_50_prior = 90.0
    indicator.high_20 = 90.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode=entry_mode)
    assert not d.passed
    assert "RSI" in d.reason


@pytest.mark.parametrize("entry_mode", ["strict_stack", "sma50_rising", "donchian"])
def test_macd_gate_applies_identically_across_modes(entry_mode):
    indicator = _indicator(macd_hist=-0.1)
    indicator.sma_50_prior = 90.0
    indicator.high_20 = 90.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode=entry_mode)
    assert not d.passed
    assert "MACD" in d.reason


@pytest.mark.parametrize("entry_mode", ["strict_stack", "sma50_rising", "donchian"])
def test_relvol_gate_applies_identically_across_modes(entry_mode):
    indicator = _indicator(rel_volume=1.0)  # below 1.3x
    indicator.sma_50_prior = 90.0
    indicator.high_20 = 90.0
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode=entry_mode)
    assert not d.passed
    assert "volume" in d.reason.lower()


def test_unknown_entry_mode_raises():
    indicator = _indicator()
    with pytest.raises(ValueError, match="unknown entry_mode"):
        crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="bogus")


def test_bracket_uses_same_atr_multiples_as_v1():
    """2.5x stop / 5.0x target, same as v1 - reused constants, not
    reimplemented thresholds that could silently drift."""
    indicator = _indicator(rsi_14=63.0, rel_volume=1.7, atr_14=4.0)
    d = crypto_trend_daily_v1("BTC-USD", indicator, 120.0, min_rr=2.0, entry_mode="strict_stack")
    assert d.passed
    assert d.stop == pytest.approx(120.0 - 2.5 * 4.0)
    assert d.target == pytest.approx(120.0 + 5.0 * 4.0)
