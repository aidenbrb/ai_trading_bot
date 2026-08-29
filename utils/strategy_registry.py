"""Single source of truth for strategy identity and deployment eligibility."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from utils.strategy_signals import CRYPTO_STRATEGY_VERSION, STOCK_STRATEGY_VERSION


@dataclass(frozen=True)
class StrategyRegistration:
    version: str
    market: str
    mode: str
    execution_eligible: bool
    evidence_status: str
    reason: str


REGISTRY = {
    STOCK_STRATEGY_VERSION: StrategyRegistration(
        STOCK_STRATEGY_VERSION, "stock", "swing", False, "rejected",
        (
            "Comparison 20260811_102403 was invalidated by a confirmed "
            "position-capacity bug (never-filled resting limit orders "
            "permanently occupied portfolio slots, freezing 2024-2025 to "
            "zero trades) and superseded by corrected comparison "
            "20260811_142144, which fixed the accounting and reproduced "
            "deterministically (matching SHA256 hashes). The corrected run "
            "still fails qualification on its own merits: 1,227 closed "
            "trades (baseline) with 99.998% coverage, net P&L -$9,144.48 "
            "baseline / -$23,350.84 stressed, negative expectancy under "
            "both cost models, profit factor and Sharpe below threshold, "
            "Sharpe below the SPY benchmark, negative trailing-12-month "
            "P&L, and under 60% of quarters positive. 2024 (182 closed) and "
            "2025 (274 closed) confirm the freeze is gone. Separate finding "
            "(not a defect in this fix): from mid-2023 onward the "
            "simulation's own no-order-expiration design lets a handful of "
            "never-filled resting orders (LLY, NET, CRWD, QCOM, later "
            "MRVL) permanently reserve most of the $100k starting equity "
            "(~$90.8k reserved by 2026), making 'insufficient_cash_or_"
            "quantity' the dominant rejection reason for the back half of "
            "the run - a genuine capital-availability constraint, not a "
            "reintroduction of the fixed bug. Execution remains locked."
        ),
    ),
    CRYPTO_STRATEGY_VERSION: StrategyRegistration(
        CRYPTO_STRATEGY_VERSION, "crypto", "swing", False, "rejected",
        (
            "Locked comparison 20260811_102403 failed qualification: only "
            "71 weekend trades, 50.74% coverage, baseline net P&L "
            "-$13,206.07, and stressed net P&L -$16,710.98."
        ),
    ),
    "orb_v1": StrategyRegistration(
        "orb_v1", "stock", "day", False, "rejected",
        "Out-of-sample baseline/stressed performance failed and IEX/SIP overlap was 48.1%.",
    ),
}


def registration(version: str) -> StrategyRegistration:
    return REGISTRY[version]


def execution_block_reason(version: object) -> str | None:
    """Return a fail-closed reason when a strategy may not submit entries."""
    if not isinstance(version, str) or not version:
        return "strategy version is missing; unversioned strategies cannot execute"
    record = REGISTRY.get(version)
    if record is None:
        return f"strategy version {version!r} is not registered for execution"
    if not record.execution_eligible:
        return f"{record.version} is research-only: {record.reason}"
    return None


def registry_snapshot() -> dict[str, dict]:
    return {name: asdict(value) for name, value in REGISTRY.items()}
