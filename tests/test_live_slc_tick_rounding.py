"""
Decimal-based broker-valid price precision (amendment_005, rev. 11 point
4). Every test here exists because the pre-rev.11 code used a blanket
`round(x, 2)` regardless of price or direction, which is wrong below
$1.00 and can silently shift the frozen stop or the exact-2R target.
"""
from decimal import Decimal

import pytest

from live_slc.execution import (
    effective_reward_risk,
    round_stop,
    round_target,
    tick_size,
    to_decimal,
)


def test_to_decimal_uses_str_conversion_never_binary_float_directly():
    """Decimal(0.1) != Decimal('0.1') - the whole point of this function."""
    assert to_decimal(0.1) == Decimal("0.1")
    assert str(to_decimal(0.1)) == "0.1"
    # Confirm the binary-float trap this guards against is real:
    assert Decimal(0.1) != Decimal("0.1")


def test_to_decimal_passthrough_for_existing_decimal():
    d = Decimal("1.23")
    assert to_decimal(d) is d


@pytest.mark.parametrize("price,expected", [
    (Decimal("0.99"), Decimal("0.0001")),
    (Decimal("1.00"), Decimal("0.01")),   # at $1.00 -> the coarser tick, per Alpaca's "at or above"
    (Decimal("1.01"), Decimal("0.01")),
    (Decimal("150.00"), Decimal("0.01")),
])
def test_tick_size_boundary(price, expected):
    assert tick_size(price) == expected


def test_long_stop_rounds_down_away_from_entry():
    assert round_stop(98.567, "long") == Decimal("98.56")


def test_short_stop_rounds_up_away_from_entry():
    assert round_stop(102.561, "short") == Decimal("102.57")


def test_long_target_rounds_up_to_guarantee_at_least_2r():
    assert round_target(104.001, "long") == Decimal("104.01")


def test_short_target_rounds_down_to_guarantee_at_least_2r():
    assert round_target(95.999, "short") == Decimal("95.99")


def test_sub_dollar_prices_use_four_decimal_tick():
    assert round_stop(0.98567, "long") == Decimal("0.9856")
    assert round_target(0.99012, "long") == Decimal("0.9902")


def test_effective_reward_risk_always_at_least_two_after_practical_rounding():
    rr_long = effective_reward_risk(100.0, Decimal("98.0"), round_target(104.0, "long"), "long")
    assert rr_long >= 2
    rr_short = effective_reward_risk(100.0, Decimal("102.0"), round_target(96.0, "short"), "short")
    assert rr_short >= 2
