import math
from types import SimpleNamespace

import nodes.crypto_strategy_node as crypto_node
import nodes.stock_strategy_node as stock_node
from utils.strategy_signals import (
    CRYPTO_STRATEGY_VERSION,
    STOCK_STRATEGY_V2_VERSION,
    STOCK_STRATEGY_VERSION,
    crypto_trend_momentum_v1,
    stock_trend_momentum_v1,
    stock_trend_momentum_v2,
)


def _indicator(**overrides):
    values = dict(
        trend="UPTREND", rsi_14=60.0, macd_hist=0.1, rel_volume=1.5,
        atr_14=2.0, sma_20=105.0, sma_50=100.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_stock_golden_master_live_wrapper_equals_shared_rule():
    indicator = _indicator()
    shared = stock_trend_momentum_v1(
        "AAPL", indicator, 100.0, min_price=10.0, min_rr=2.0
    )
    assert stock_node._evaluate("AAPL", indicator, 100.0) == shared.legacy_tuple()
    assert shared.strategy_version == STOCK_STRATEGY_VERSION
    assert shared.entry == 100.0
    assert shared.stop == 97.0
    assert shared.target == 106.0
    assert shared.conviction_score == 92


def test_crypto_golden_master_live_wrapper_equals_shared_rule():
    indicator = _indicator(rsi_14=63.0, rel_volume=1.7, atr_14=4.0)
    shared = crypto_trend_momentum_v1(
        "BTC-USD", indicator, 100.0, min_rr=2.0
    )
    assert crypto_node._evaluate("BTC-USD", indicator, 100.0) == shared.legacy_tuple()
    assert shared.strategy_version == CRYPTO_STRATEGY_VERSION
    assert shared.entry == 100.0
    assert shared.stop == 90.0
    assert shared.target == 120.0


def test_shared_rules_remain_side_effect_free_rejections():
    stock = stock_trend_momentum_v1(
        "AAPL", _indicator(rel_volume=1.19), 100.0,
        min_price=10.0, min_rr=2.0,
    )
    crypto = crypto_trend_momentum_v1(
        "BTC-USD", _indicator(rsi_14=76.0), 100.0, min_rr=2.0
    )
    assert stock.passed is False and "Relative volume" in stock.reason
    assert crypto.passed is False and "RSI" in crypto.reason


def _v2(indicator, close=100.0, *, enable_adx_filter, enable_cost_filter):
    return stock_trend_momentum_v2(
        "AAPL", indicator, close, min_price=10.0, min_rr=2.0,
        enable_adx_filter=enable_adx_filter, enable_cost_filter=enable_cost_filter,
    )


def test_v2_both_off_reproduces_v1_exactly():
    indicator = _indicator()
    v1 = stock_trend_momentum_v1("AAPL", indicator, 100.0, min_price=10.0, min_rr=2.0)
    v2 = _v2(indicator, enable_adx_filter=False, enable_cost_filter=False)
    assert v2.strategy_version == STOCK_STRATEGY_V2_VERSION
    assert (v2.passed, v2.reason, v2.entry, v2.stop, v2.target, v2.rr, v2.conviction_score) == (
        v1.passed, v1.reason, v1.entry, v1.stop, v1.target, v1.rr, v1.conviction_score,
    )


def test_v2_reproduces_v1_rejection_without_needing_adx_fields():
    """When v1 itself rejects, v2 must return the same rejection - and
    must never touch adx_14/bars_observed to get there, even with both
    Delta2/Delta3 flags on."""
    indicator = _indicator(rel_volume=1.19)  # v1 rejects: rel_volume < 1.2
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=True)
    assert v2.passed is False
    assert "Relative volume" in v2.reason
    assert v2.strategy_version == STOCK_STRATEGY_V2_VERSION


def test_v2_delta2_passes_with_sufficient_adx_and_bars():
    indicator = _indicator(adx_14=30.0, bars_observed=50)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=False)
    assert v2.passed is True
    assert v2.strategy_version == STOCK_STRATEGY_V2_VERSION


def test_v2_delta2_rejects_below_adx_threshold():
    indicator = _indicator(adx_14=24.9, bars_observed=50)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=False)
    assert v2.passed is False
    assert "Delta2" in v2.reason


def test_v2_delta2_rejects_below_bars_observed_floor():
    indicator = _indicator(adx_14=30.0, bars_observed=39)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=False)
    assert v2.passed is False
    assert "Delta2" in v2.reason


def test_v2_delta2_rejects_missing_adx():
    indicator = _indicator(adx_14=None, bars_observed=50)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=False)
    assert v2.passed is False
    assert "Delta2" in v2.reason


def test_v2_delta2_rejects_nonfinite_adx():
    indicator = _indicator(adx_14=math.nan, bars_observed=50)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=False)
    assert v2.passed is False
    assert "Delta2" in v2.reason


def test_v2_delta2_off_never_rejects_for_missing_adx():
    """Delta2-off variants must reproduce v1's eligibility exactly - they
    never read adx_14 at all, regardless of how much history exists."""
    indicator = _indicator(adx_14=None, bars_observed=0)
    v2 = _v2(indicator, enable_adx_filter=False, enable_cost_filter=False)
    assert v2.passed is True


def test_v2_delta3_rejects_below_50bps_move():
    # STOCK_ATR_TARGET_MULT=3.0 * atr=0.10 / entry=100.0 -> 30bps < 50bps.
    indicator = _indicator(atr_14=0.10)
    without_cost_filter = _v2(indicator, enable_adx_filter=False, enable_cost_filter=False)
    assert without_cost_filter.passed is True
    with_cost_filter = _v2(indicator, enable_adx_filter=False, enable_cost_filter=True)
    assert with_cost_filter.passed is False
    assert "Delta3" in with_cost_filter.reason


def test_v2_delta3_passes_at_or_above_50bps_move():
    indicator = _indicator()  # default atr_14=2.0 -> move far above 50bps
    v2 = _v2(indicator, enable_adx_filter=False, enable_cost_filter=True)
    assert v2.passed is True


def test_v2_full_combination_requires_both_filters_to_pass():
    indicator = _indicator(adx_14=30.0, bars_observed=50)
    v2 = _v2(indicator, enable_adx_filter=True, enable_cost_filter=True)
    assert v2.passed is True
    assert v2.strategy_version == STOCK_STRATEGY_V2_VERSION


def test_crypto_invalid_negative_atr_stop_fails_closed_instead_of_crashing():
    decision = crypto_trend_momentum_v1(
        "SHIB-USD",
        _indicator(
            rsi_14=60.0,
            macd_hist=0.0000001,
            rel_volume=1.5,
            atr_14=0.000009628,
            sma_20=0.0000095,
            sma_50=0.0000090,
        ),
        0.00000901,
        min_rr=2.0,
    )

    assert decision.passed is False
    assert decision.strategy_version == CRYPTO_STRATEGY_VERSION
    assert "Invalid ATR bracket" in decision.reason
