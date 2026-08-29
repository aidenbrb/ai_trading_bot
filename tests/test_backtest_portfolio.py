"""Tests for backtest/portfolio.py - position sizing and portfolio scenarios."""
import pytest

from backtest.portfolio import PortfolioConfig, size_trade, RESEARCH_FIDELITY, INTENDED_DEPLOYMENT


def _config(**overrides):
    base = dict(
        starting_equity=100_000.0, risk_per_trade_pct=0.01,
        max_gross_exposure_x=1.0, max_concurrent_positions=5,
        max_position_concentration_pct=None,
    )
    base.update(overrides)
    return PortfolioConfig(**base)


class TestSizeTrade:
    def test_uses_starting_equity_when_no_override_given(self):
        result = size_trade(_config(starting_equity=100_000.0, risk_per_trade_pct=0.01), entry=100.0, stop=95.0)
        assert result["risk_amount"] == pytest.approx(1_000.0)  # 1% of 100k
        assert result["quantity"] == pytest.approx(200.0)  # 1000 / 5

    def test_evolving_equity_overrides_starting_equity(self):
        """The core of fix #4: sizing must use the current, evolving balance,
        not always the original starting_equity."""
        config = _config(starting_equity=100_000.0, risk_per_trade_pct=0.01)
        result = size_trade(config, entry=100.0, stop=95.0, equity=50_000.0)
        assert result["risk_amount"] == pytest.approx(500.0)  # 1% of the PASSED equity, not starting_equity
        assert result["quantity"] == pytest.approx(100.0)

    def test_zero_risk_per_unit_returns_zero_size(self):
        result = size_trade(_config(), entry=100.0, stop=100.0)
        assert result == {"quantity": 0.0, "risk_amount": 0.0, "position_value": 0.0}

    def test_concentration_cap_uses_the_passed_equity(self):
        config = _config(starting_equity=100_000.0, max_position_concentration_pct=0.20)
        # Huge risk budget would want a huge position; concentration cap should
        # bind against the PASSED equity (50k), not starting_equity (100k).
        result = size_trade(config, entry=100.0, stop=99.0, equity=50_000.0)
        assert result["position_value"] <= 50_000.0 * 0.20 + 1e-6

    def test_gross_exposure_cap_uses_the_passed_equity(self):
        config = _config(starting_equity=100_000.0, max_gross_exposure_x=1.0, max_position_concentration_pct=None)
        result = size_trade(config, entry=100.0, stop=99.0, equity=10_000.0)
        assert result["position_value"] <= 10_000.0 + 1e-6


def test_research_fidelity_matches_locked_in_values():
    assert RESEARCH_FIDELITY.starting_equity == 25_000.0
    assert RESEARCH_FIDELITY.risk_per_trade_pct == 0.01
    assert RESEARCH_FIDELITY.max_concurrent_positions == 20


def test_intended_deployment_matches_locked_in_values():
    assert INTENDED_DEPLOYMENT.starting_equity == 100_000.0
    assert INTENDED_DEPLOYMENT.risk_per_trade_pct == 0.0025
    assert INTENDED_DEPLOYMENT.max_concurrent_positions == 5
    assert INTENDED_DEPLOYMENT.max_gross_exposure_x == 1.0
