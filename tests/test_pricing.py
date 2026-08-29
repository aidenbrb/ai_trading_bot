"""Regression tests for broker-tick bracket price construction."""
from utils.pricing import rounded_long_bracket


def test_stock_rounding_preserves_minimum_rr():
    # This reproduces the PFE class of failure: independent cent rounding used
    # to turn an intended 2.0 setup into 1.96, which risk_node then rejected.
    result = rounded_long_bracket(
        close=25.30,
        atr=0.163333333,
        stop_atr=1.5,
        target_atr=3.0,
        min_rr=2.0,
        tick_size=0.01,
    )

    actual_rr = (
        (result["target"] - result["entry"])
        / (result["entry"] - result["stop"])
    )
    assert result["rr"] >= 2.0
    assert actual_rr >= 2.0
    assert result == {
        "entry": 25.30,
        "stop": 25.05,
        "target": 25.80,
        "rr": 2.0,
    }


def test_target_is_raised_when_stop_rounding_increases_risk():
    result = rounded_long_bracket(
        close=10.00,
        atr=0.101,
        stop_atr=1.5,
        target_atr=3.0,
        min_rr=2.0,
        tick_size=0.01,
    )
    assert result["entry"] == 10.00
    assert result["stop"] == 9.84
    assert result["target"] == 10.32
    assert result["rr"] == 2.0


def test_crypto_fractional_ticks_preserve_minimum_rr():
    result = rounded_long_bracket(
        close=0.00001234,
        atr=0.00000037,
        stop_atr=2.5,
        target_atr=5.0,
        min_rr=2.0,
        tick_size=0.00000001,
    )
    assert result["entry"] > result["stop"] > 0
    assert result["target"] > result["entry"]
    assert result["rr"] >= 2.0
