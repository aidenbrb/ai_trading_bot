"""Pure helpers for conservative, long-call options research."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class OptionCandidate:
    symbol: str
    underlying: str
    expiration_date: date
    strike_price: float
    bid: float
    ask: float
    open_interest: int
    delta: Optional[float]
    implied_volatility: Optional[float]
    multiplier: int = 100
    contract_type: str = "call"

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.midpoint if self.midpoint > 0 else math.inf

    def dte(self, as_of: date) -> int:
        return (self.expiration_date - as_of).days


def candidate_rejection(
    candidate: OptionCandidate,
    *,
    as_of: date,
    min_dte: int,
    max_dte: int,
    min_delta: float,
    max_delta: float,
    min_open_interest: int,
    max_spread_pct: float,
) -> Optional[str]:
    """Return the first safety/liquidity rejection reason, or None."""
    if candidate.contract_type.lower() != "call":
        return "only long calls are supported in research v1"
    dte = candidate.dte(as_of)
    if not min_dte <= dte <= max_dte:
        return f"DTE {dte} outside {min_dte}-{max_dte}"
    if candidate.bid <= 0 or candidate.ask <= candidate.bid:
        return "invalid or non-marketable quote"
    if candidate.spread_pct > max_spread_pct:
        return f"spread {candidate.spread_pct:.1%} exceeds {max_spread_pct:.1%}"
    if candidate.open_interest < min_open_interest:
        return f"open interest {candidate.open_interest} below {min_open_interest}"
    if candidate.delta is None:
        return "delta unavailable"
    if not min_delta <= candidate.delta <= max_delta:
        return f"delta {candidate.delta:.2f} outside {min_delta:.2f}-{max_delta:.2f}"
    return None


def rank_candidate(candidate: OptionCandidate, underlying_price: float, as_of: date) -> tuple:
    """Lower tuple is better: delta≈0.625, tight spread, near-ATM, mid DTE."""
    return (
        abs((candidate.delta or 0) - 0.625),
        candidate.spread_pct,
        abs(candidate.strike_price - underlying_price) / underlying_price,
        abs(candidate.dte(as_of) - 45),
        -candidate.open_interest,
    )


def select_candidate(
    candidates: list[OptionCandidate],
    *,
    underlying_price: float,
    as_of: date,
    min_dte: int,
    max_dte: int,
    min_delta: float,
    max_delta: float,
    min_open_interest: int,
    max_spread_pct: float,
) -> Optional[OptionCandidate]:
    eligible = [
        c for c in candidates
        if candidate_rejection(
            c, as_of=as_of, min_dte=min_dte, max_dte=max_dte,
            min_delta=min_delta, max_delta=max_delta,
            min_open_interest=min_open_interest,
            max_spread_pct=max_spread_pct,
        ) is None
    ]
    return min(eligible, key=lambda c: rank_candidate(c, underlying_price, as_of)) if eligible else None


def size_long_option(
    midpoint: float,
    multiplier: int,
    equity: float,
    max_risk_pct: float,
    max_premium_pct: float,
) -> dict:
    """Size a long option whose maximum loss is the premium paid."""
    if midpoint <= 0 or multiplier <= 0 or equity <= 0:
        return {"contracts": 0, "premium_per_contract": 0.0,
                "total_premium": 0.0, "max_loss": 0.0}
    per_contract = midpoint * multiplier
    budget = min(equity * max_risk_pct, equity * max_premium_pct)
    contracts = math.floor(budget / per_contract)
    total = contracts * per_contract
    return {
        "contracts": contracts,
        "premium_per_contract": round(per_contract, 2),
        "total_premium": round(total, 2),
        "max_loss": round(total, 2),
    }
