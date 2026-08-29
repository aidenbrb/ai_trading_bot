"""
Shared, reproducible cost model for the day-trading (ORB) feature - defined
directly in code (not just documentation) so backtest and shadow-outcome
results are reproducible. Used by nodes/intraday_shadow_node.py and the
backtest/ package.

Modeled in basis points of price (the ORB universe spans $5-$500+ names).
Alpaca's minute bars have no bid/ask, so real spread isn't directly
observable from the bars themselves - these are documented starting
assumptions, sensitivity-tested (zero vs. baseline vs. stressed) rather than
treated as verified numbers.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    entry_slippage_bps:   float = 8.0   # breakout entries chase price
    exit_slippage_bps:    float = 8.0   # EOD flatten is a market order into a possibly volatile close
    spread_bps:           float = 5.0   # half-spread charged per leg
    commission_per_share: float = 0.0   # Alpaca is commission-free


ZERO_COST     = CostModel(0.0, 0.0, 0.0, 0.0)
BASELINE_COST = CostModel()
STRESSED_COST = CostModel(16.0, 16.0, 10.0, 0.0)   # 2x baseline

_VERSIONS = {
    "zero":        ZERO_COST,
    "baseline_v1": BASELINE_COST,
    "stressed":    STRESSED_COST,
}


def by_version(version: str) -> CostModel:
    return _VERSIONS[version]


def apply_entry_cost(trigger_price: float, direction: str, model: CostModel) -> float:
    """Long entries (buys) pay UP; short entries (sells) pay DOWN."""
    bps = (model.entry_slippage_bps + model.spread_bps) / 10_000.0
    return trigger_price * (1 + bps) if direction == "long" else trigger_price * (1 - bps)


def apply_exit_cost(exit_price: float, direction: str, model: CostModel) -> float:
    """Long exits (sells) pay DOWN; short exits (buys to cover) pay UP."""
    bps = (model.exit_slippage_bps + model.spread_bps) / 10_000.0
    return exit_price * (1 - bps) if direction == "long" else exit_price * (1 + bps)
