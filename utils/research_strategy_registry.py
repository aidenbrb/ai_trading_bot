"""
Registry for research-only strategy variants that must NOT be added to
utils/strategy_registry.py's REGISTRY dict: that file is a Tier-1 guardrail
for the live SLC paper-trading system (live_slc/guardrails.py) and is
separately guardrail-hashed by backtest/run_slc_backtest.py - adding an
entry there broke BOTH systems' hash-based drift detection with zero actual
behavior change (found and fixed during the crypto strategy variant work,
2026-08-28; utils/strategy_registry.py itself was reverted byte-for-byte and
is untouched by this file).

Mirrors utils.strategy_registry.StrategyRegistration's shape for
consistency (imported, not duplicated), but this is a fully separate
registry: nothing in the live pipeline, live_slc, or either evidence gate
reads this file, and none of these strategies are wired into any live node.
"""
from __future__ import annotations

from dataclasses import asdict

from utils.strategy_registry import StrategyRegistration
from utils.strategy_signals import CRYPTO_DAILY_STRATEGY_VERSION

# crypto_xsec_momentum_v1 has no SignalDecision-based function in
# utils/strategy_signals.py (it's a portfolio simulator in
# backtest/whole_bot_engine.py, not a per-symbol gate), so there's no
# existing constant to import - this string must stay byte-identical to
# the literal "strategy_version" backtest/whole_bot_engine.py::
# _xsec_close_trade() writes onto every trade dict.
CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION = "crypto_xsec_momentum_v1"

RESEARCH_REGISTRY = {
    CRYPTO_DAILY_STRATEGY_VERSION: StrategyRegistration(
        CRYPTO_DAILY_STRATEGY_VERSION, "crypto", "swing", False, "research",
        (
            "New daily-bar variant added alongside crypto_trend_momentum_v1 "
            "(unmodified) after a review found v1's SMA20/50/200 trend "
            "classification runs on hourly bars, not the daily periods the "
            "names imply. Not yet evaluated through the evidence gate - "
            "execution_eligible stays False until a comparison run passes "
            "qualification on its own merits."
        ),
    ),
    CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION: StrategyRegistration(
        CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION, "crypto", "swing", False, "research",
        (
            "Cross-sectional momentum variant, evaluated alongside "
            "crypto_trend_daily_v1 through the same evidence-gate comparison "
            "(Phase 2 Step 5). Best cell (lookback_30, weekend_rebalance) "
            "reached only 4/13 qualification checks on the full universe. "
            "Not yet evaluated through the evidence gate for promotion - "
            "execution_eligible stays False until a comparison run passes "
            "qualification on its own merits."
        ),
    ),
}


def registration(version: str) -> StrategyRegistration:
    return RESEARCH_REGISTRY[version]


def execution_block_reason(version: object) -> str | None:
    """Same contract as utils.strategy_registry.execution_block_reason() -
    fail-closed reason when a research-registry strategy may not submit
    entries. Nothing in the live pipeline currently calls this (none of
    these strategies are wired into any live node), but it exists so a
    future live-wiring attempt has the same gate to check."""
    if not isinstance(version, str) or not version:
        return "strategy version is missing; unversioned strategies cannot execute"
    record = RESEARCH_REGISTRY.get(version)
    if record is None:
        return f"strategy version {version!r} is not registered for execution"
    if not record.execution_eligible:
        return f"{record.version} is research-only: {record.reason}"
    return None


def registry_snapshot() -> dict[str, dict]:
    return {name: asdict(value) for name, value in RESEARCH_REGISTRY.items()}
