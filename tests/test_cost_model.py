"""Tests for utils/cost_model.py."""
import pytest

from utils.cost_model import (
    ZERO_COST, BASELINE_COST, STRESSED_COST, by_version,
    apply_entry_cost, apply_exit_cost,
)


def test_by_version_returns_correct_model():
    assert by_version("zero") is ZERO_COST
    assert by_version("baseline_v1") is BASELINE_COST
    assert by_version("stressed") is STRESSED_COST


def test_zero_cost_is_a_no_op():
    assert apply_entry_cost(100.0, "long", ZERO_COST) == 100.0
    assert apply_exit_cost(100.0, "long", ZERO_COST) == 100.0


def test_long_entry_pays_up_long_exit_pays_down():
    entry = apply_entry_cost(100.0, "long", BASELINE_COST)
    exit_ = apply_exit_cost(100.0, "long", BASELINE_COST)
    assert entry > 100.0
    assert exit_ < 100.0


def test_short_entry_pays_down_short_exit_pays_up():
    entry = apply_entry_cost(100.0, "short", BASELINE_COST)
    exit_ = apply_exit_cost(100.0, "short", BASELINE_COST)
    assert entry < 100.0
    assert exit_ > 100.0


def test_stressed_cost_is_worse_than_baseline():
    baseline_entry = apply_entry_cost(100.0, "long", BASELINE_COST)
    stressed_entry = apply_entry_cost(100.0, "long", STRESSED_COST)
    assert stressed_entry > baseline_entry
