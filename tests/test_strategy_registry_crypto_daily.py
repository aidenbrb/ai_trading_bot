"""
Tests for utils/research_strategy_registry.py - deliberately SEPARATE from
utils/strategy_registry.py, which is a Tier-1 guardrail for the live SLC
paper-trading system and is also guardrail-hashed by
backtest/run_slc_backtest.py. Adding crypto_trend_daily_v1 there broke both
systems' hash-based drift detection with zero actual content change (found
2026-08-28); this file's own tests exist specifically to prove
utils/strategy_registry.py stays completely untouched by the new strategy
work.
"""
import hashlib

import utils.research_strategy_registry as research_registry
import utils.strategy_registry as registry
from utils.strategy_signals import CRYPTO_DAILY_STRATEGY_VERSION, CRYPTO_STRATEGY_VERSION


def test_crypto_trend_daily_v1_registered_and_not_execution_eligible():
    record = research_registry.registration(CRYPTO_DAILY_STRATEGY_VERSION)
    assert record.execution_eligible is False
    assert record.market == "crypto"


def test_crypto_trend_daily_v1_execution_blocked():
    reason = research_registry.execution_block_reason(CRYPTO_DAILY_STRATEGY_VERSION)
    assert reason is not None
    assert "research-only" in reason


def test_crypto_trend_daily_v1_not_in_the_guardrailed_registry():
    """The whole point of the separate file: crypto_trend_daily_v1 must
    never appear in utils.strategy_registry.REGISTRY."""
    assert CRYPTO_DAILY_STRATEGY_VERSION not in registry.REGISTRY


def test_v1_registration_unchanged():
    """v1's own registry entry in the real (Tier-1-guarded) registry must
    not have been touched by adding the new sibling entry elsewhere."""
    record = registry.registration(CRYPTO_STRATEGY_VERSION)
    assert record.execution_eligible is False
    assert record.evidence_status == "rejected"
    assert "71 weekend trades" in record.reason


def test_strategy_registry_file_is_byte_identical_to_the_verified_baseline():
    """Direct regression guard for the incident itself: utils/strategy_
    registry.py's on-disk bytes must match the hash live_slc's frozen
    Tier-1 baseline (and backtest/run_slc_backtest.py's own baseline)
    expect, independent of whichever guardrail test files happen to run."""
    import live_slc.guardrails as guardrails
    # Confirmed Tier-1 membership directly, rather than assuming the key name.
    assert any(
        str(path).replace("\\", "/").endswith("utils/strategy_registry.py")
        for path in guardrails.GUARDRAILS_TIER1.values()
    ), "utils/strategy_registry.py must stay a live_slc Tier-1 file for this test to be meaningful"
    path = next(
        p for p in guardrails.GUARDRAILS_TIER1.values()
        if str(p).replace("\\", "/").endswith("utils/strategy_registry.py")
    )
    with open(path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    import json
    baseline = json.loads(guardrails.DEPLOYMENT_BASELINE.read_text(encoding="utf-8"))
    key = next(k for k in baseline["guardrails"]["tier1"] if k.endswith("strategy_registry.py"))
    assert actual_hash == baseline["guardrails"]["tier1"][key]
