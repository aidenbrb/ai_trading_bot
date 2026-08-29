"""
Named portfolio scenarios for the ORB backtest - locked in with the user,
see the day-trading-mode plan §6. Every backtest report runs BOTH scenarios
side by side, never just one: this separates "does the published edge
replicate at all" (RESEARCH_FIDELITY) from "would a safe version of this
actually be viable for this account" (INTENDED_DEPLOYMENT).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PortfolioConfig:
    starting_equity: float
    risk_per_trade_pct: float
    max_gross_exposure_x: float
    max_concurrent_positions: int
    max_position_concentration_pct: Optional[float] = None


RESEARCH_FIDELITY = PortfolioConfig(
    starting_equity=25_000.0,      # matches the original paper
    risk_per_trade_pct=0.01,       # 1%, matches the original paper
    max_gross_exposure_x=4.0,      # 4x gross exposure cap
    max_concurrent_positions=20,   # all qualifying top-20 entries, uncapped further
    max_position_concentration_pct=None,
)

INTENDED_DEPLOYMENT = PortfolioConfig(
    starting_equity=100_000.0,
    risk_per_trade_pct=0.0025,     # 0.25% - confirmed by the user, locked in
    max_gross_exposure_x=1.0,      # no leverage
    max_concurrent_positions=5,
    max_position_concentration_pct=0.20,
)

SCENARIOS = {"research_fidelity": RESEARCH_FIDELITY, "intended_deployment": INTENDED_DEPLOYMENT}


def size_trade(config: PortfolioConfig, entry: float, stop: float, equity: Optional[float] = None) -> dict:
    """
    Fixed-fractional position sizing, same shape as nodes/risk_node.py's
    formula: risk_budget = equity * risk_pct; qty = risk_budget / risk_per_unit.

    `equity` defaults to `config.starting_equity` only when omitted - the
    backtest engine passes the current, evolving (session-to-session)
    balance for a given (scenario, cost_model) so sizing compounds realized
    gains/losses instead of always sizing off the original starting capital.

    Also caps by max_position_concentration_pct and max_gross_exposure_x
    against a single trade's notional in isolation - full portfolio-level
    gross-exposure tracking across multiple concurrently-admitted trades is
    the calling engine's job (it knows the running total for the session).
    """
    equity = config.starting_equity if equity is None else equity
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return {"quantity": 0.0, "risk_amount": 0.0, "position_value": 0.0}

    risk_budget = equity * config.risk_per_trade_pct
    quantity = risk_budget / risk_per_unit
    position_value = quantity * entry

    max_position = equity * config.max_gross_exposure_x
    if config.max_position_concentration_pct is not None:
        max_position = min(max_position, equity * config.max_position_concentration_pct)
    if position_value > max_position and entry > 0:
        quantity = max_position / entry
        position_value = quantity * entry

    return {
        "quantity": quantity,
        "risk_amount": quantity * risk_per_unit,
        "position_value": position_value,
    }
