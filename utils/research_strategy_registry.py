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
        CRYPTO_DAILY_STRATEGY_VERSION, "crypto", "swing", False, "archived",
        (
            "ARCHIVED 2026-08-29 (Phase 3 verdict) - no persistent edge. "
            "In-sample (2022-01-01 to 2024-06-30) looked promising for the "
            "sma50_rising entry mode (Sharpe 0.520, best of 3 entry modes), "
            "but out-of-sample (2024-07-01 to 2026-08-09) with those same "
            "frozen parameters: Sharpe -0.054 (vs. BTC buy-and-hold's 0.261 "
            "over the identical window), profit factor 0.94, and 4 of 5 OOS "
            "half-years negative - not one bad stretch, a persistent decay. "
            "A regime-filter-alone control benchmark (Phase 3 Step 3) further "
            "showed this family's raw return mostly traces to the shared "
            "BTC-SMA20 regime filter, not this strategy's own RSI/MACD/relvol "
            "gates. Full findings, tables, and the walk-forward methodology: "
            "the Crypto Evidence Ledger, PHASE 3 section "
            "(https://claude.ai/code/artifact/d034bcaa-80f8-4e79-a355-1bc62d6d5efc). "
            "Do not revive without reading that section first - the in-sample "
            "numbers alone are not sufficient justification; they're exactly "
            "what looked good before OOS testing was done."
        ),
    ),
    CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION: StrategyRegistration(
        CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION, "crypto", "swing", False, "archived",
        (
            "ARCHIVED 2026-08-29 (Phase 3 verdict) - no persistent edge. "
            "In-sample (2022-01-01 to 2024-06-30), lookback_30 (N=3, 10-symbol "
            "universe) reached Sharpe 0.212. Out-of-sample (2024-07-01 to "
            "2026-08-09) with those frozen parameters, the headline numbers "
            "look like a pass (Sharpe 0.564, profit factor 1.31) but both are "
            "a single-half-year artifact: 2024H2 alone contributes +$33,681 "
            "net P&L; every other OOS half-year, summed, is -$11,596. Max "
            "drawdown (-26.6%), positive-quarter rate (55.6%, just under the "
            "60% bar), bootstrap lower bound (-0.171), and trailing-12mo "
            "(-$23,639) all fail regardless. The sensitivity grid (Phase 3 "
            "Step 6) found no plateau at all - results across lookback x N "
            "were genuinely erratic, with several untested cells beating the "
            "selected one's in-sample Sharpe, a signal-vs-noise warning. Full "
            "findings: the Crypto Evidence Ledger, PHASE 3 section "
            "(https://claude.ai/code/artifact/d034bcaa-80f8-4e79-a355-1bc62d6d5efc). "
            "Do not revive without reading that section first."
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
