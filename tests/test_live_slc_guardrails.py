"""
Guardrail-hash tests for live_slc. Deliberately a separate file from
tests/test_slc_backtest.py (rev. 6 correction) - that file and its
guardrail tests remain fully untouched by this work.
"""
from pathlib import Path

import pytest

import live_slc.guardrails as guardrails
import live_slc.models as models
import backtest.run_slc_backtest as backtest_runner
from live_slc import reauth_signature
from live_slc.authorization import record_transition
from tests._slc_reauth_helpers import make_test_key, write_signed_baseline_event


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """Phase 6: verify_deployment_baseline()/resolve_active_baseline() now
    read SlcActivationEvent from the DB - this file never isolated its DB
    before (pure file-hash checks didn't need to), which would otherwise
    make these tests read/write the REAL production live_slc.db."""
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


@pytest.fixture
def signer(tmp_path, monkeypatch):
    key_path, allowed_signers_path, identity = make_test_key(tmp_path)
    monkeypatch.setattr(reauth_signature, "ALLOWED_SIGNERS_PATH", allowed_signers_path)
    return key_path, allowed_signers_path, identity


def test_scripts_slc_live_does_not_affect_backtest_runner_glob():
    """backtest/run_slc_backtest.py's GUARDRAILS dict globs scripts/*.bat
    non-recursively - proves scripts/slc_live/'s existence doesn't leak
    into it (the exact interaction found and fixed during review)."""
    hashes = backtest_runner._guardrail_hashes()
    assert set(hashes.keys()) == {
        "../run_bot.bat", "run_pipeline.py", "utils/strategy_registry.py",
        "nodes/execution_node.py", "scripts/run_daily.bat", "scripts/run_day_open.bat",
        "scripts/run_day_preflight.bat", "scripts/run_day_shadow.bat",
        "scripts/run_monitor_only.bat",
    }
    assert not any("slc_live" in name for name in hashes)


def test_tier1_diverges_from_the_backtest_runner_by_exactly_two_files():
    """Phase 6: live_slc/guardrails.py and live_slc/allowed_signers are
    now both Tier-1 guardrailed (the circularity fix, then the operator's
    own follow-up request once their key was in place - see the
    module-level notes in guardrails.py). Both are a deliberate divergence
    from backtest/run_slc_backtest.py's separate hash set - that module
    doesn't import or depend on live_slc.guardrails or the signature
    scheme at all, so it has no reason to protect either. Every OTHER
    Tier-1 file must still match exactly."""
    live_hashes = guardrails.guardrail_hashes()["tier1"]
    backtest_hashes = backtest_runner._guardrail_hashes()
    diverged = {"live_slc/guardrails.py", "live_slc/allowed_signers"}
    assert set(live_hashes) - set(backtest_hashes) == diverged
    shared = {name: value for name, value in live_hashes.items() if name not in diverged}
    assert shared == backtest_hashes


def test_guardrails_module_is_tier1_guardrailed_and_excluded_from_tier2():
    """Phase 6 circularity fix: guardrails.py no longer hardcodes a
    path/hash to "the current baseline" as literal module constants (that
    was the actual source of the old self-reference problem - see the
    module-level note above GUARDRAILS_TIER1) - its bytes are now stable
    across a re-baseline, so it can safely be included in its own
    GUARDRAILS_TIER1. Still excluded from Tier-2 (it was never there)."""
    assert "live_slc/guardrails.py" in guardrails.GUARDRAILS_TIER1
    assert not any("guardrails.py" in name for name in guardrails.GUARDRAILS_TIER2)


def test_allowed_signers_is_now_tier1_guardrailed():
    """Phase 6 (operator request): moved from "deliberately never
    guardrailed" to Tier-1, now that the bootstrap sequencing (operator
    adds their real key as one unprotected manual edit, THEN the next
    baseline protects the file) makes it safe - see the KNOWN GAP note
    above GUARDRAILS_TIER1 for the still-unsolved key-removal case."""
    assert "live_slc/allowed_signers" in guardrails.GUARDRAILS_TIER1
    assert not any("allowed_signers" in name for name in guardrails.GUARDRAILS_TIER2)


def test_reauth_signature_module_is_tier2_guardrailed():
    assert "live_slc/reauth_signature.py" in guardrails.GUARDRAILS_TIER2


def test_verify_deployment_baseline_passes_against_a_freshly_signed_baseline(tmp_path, signer):
    """Phase 6: verify_deployment_baseline() now requires resolving a
    signature-verified event first - can no longer be checked "against the
    real frozen state" directly (the real production live_slc.db has no
    signed event yet, by design, until the operator's first real signing).
    Builds its own fully self-contained signed scenario instead, matching
    the CURRENT real guardrail_hashes() so the drift comparison genuinely
    passes."""
    key_path, allowed_signers_path, identity = signer
    baseline_path = tmp_path / "baseline.json"
    current = guardrails.guardrail_hashes()
    write_signed_baseline_event(
        key_path=key_path, identity=identity, baseline_path=baseline_path,
        guardrails_dict=current,
    )
    result = guardrails.verify_deployment_baseline()
    assert result["guardrails"] == current
    assert len(result["guardrails"]["tier1"]) == 11  # Phase 6: +live_slc/guardrails.py, +live_slc/allowed_signers
    assert len(result["guardrails"]["tier2"]) == 28  # +migrations.py, +split_detection.py (rev. 11), +run_hidden.vbs (activation), +check_schedule_health.py + its .bat wrapper (sleep-fix), +reauth_signature.py + promotion.md + verify_tier1_independent.py (Phase 6)


def test_verify_tier1_independent_script_is_tier2_guardrailed():
    assert "scripts/slc_live/verify_tier1_independent.py" in guardrails.GUARDRAILS_TIER2


def test_run_hidden_vbs_present_in_tier2():
    assert "scripts/slc_live/run_hidden.vbs" in guardrails.GUARDRAILS_TIER2


def test_schedule_health_check_and_its_wrapper_present_in_tier2():
    """Unlike check_readiness.py/check_session_gate.py/check_paper_session_audit.py
    (exempt from Tier-2: a human exercises judgment before running them
    manually), check_schedule_health.py is invoked automatically by its
    own Scheduled Task, so that exemption doesn't apply here."""
    assert "live_slc/check_schedule_health.py" in guardrails.GUARDRAILS_TIER2
    assert "scripts/slc_live/run_slc_schedule_health.bat" in guardrails.GUARDRAILS_TIER2


def test_tier2_drift_detected_on_a_signal_side_file(tmp_path, signer):
    """Mutating a signal-side Tier-2 file must be detected as drift - the
    signed baseline claims a hash for live_slc/reducer.py that the actual
    current file no longer matches."""
    key_path, allowed_signers_path, identity = signer
    baseline_path = tmp_path / "baseline.json"
    hashes = guardrails.guardrail_hashes()
    write_signed_baseline_event(
        key_path=key_path, identity=identity, baseline_path=baseline_path,
        guardrails_dict={
            "tier1": hashes["tier1"],
            "tier2": {**hashes["tier2"], "live_slc/reducer.py": "0" * 64},
        },
    )
    with pytest.raises(RuntimeError, match="guardrail drift"):
        guardrails.verify_deployment_baseline()


def test_baseline_file_tampering_detected(tmp_path, signer):
    """The baseline file's actual on-disk content no longer matches the
    hash the signed event pinned (e.g. edited after signing) - must
    block, not silently trust the file."""
    key_path, allowed_signers_path, identity = signer
    baseline_path = tmp_path / "baseline.json"
    current = guardrails.guardrail_hashes()
    write_signed_baseline_event(
        key_path=key_path, identity=identity, baseline_path=baseline_path,
        guardrails_dict=current,
    )
    # Tamper with the file AFTER it was signed - its hash no longer
    # matches guardrail_baseline_sha256_at_transition.
    baseline_path.write_text('{"guardrails": {"tier1": {}, "tier2": {}}, "environment": {}}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="baseline file hash mismatch"):
        guardrails.verify_deployment_baseline()


def test_missing_guardrail_file_raises(tmp_path, monkeypatch):
    fake_tier2 = dict(guardrails.GUARDRAILS_TIER2)
    fake_tier2["does/not/exist.py"] = tmp_path / "nope.py"
    monkeypatch.setattr(guardrails, "GUARDRAILS_TIER2", fake_tier2)
    with pytest.raises(RuntimeError, match="guardrail file missing"):
        guardrails.guardrail_hashes()


# -- Closeout-survives-drift property, both directions (rev. 6 clarification) --

def test_closeout_gate_survives_signal_side_drift(tmp_path, monkeypatch):
    """A Tier-2 drift in a signal-side file (e.g. reducer.py) must NOT
    block assert_closeout_preconditions() - it doesn't depend on
    signal-generation fidelity at all."""
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")
    # No exception - closeout must not consult tier2 signal-fidelity files at all.
    guardrails.assert_closeout_preconditions(observed_account_id="acct-1")


# -- assert_submission_preconditions(): the activation-proposal hash check --

def _passing_status_record(**overrides):
    defaults = dict(
        status="paper_active", observed_account_id="acct-1",
        live_baseline_sha256=None, activation_proposal_sha256=None,
    )
    defaults.update(overrides)
    return type("StatusRecord", (), defaults)()


def _passing_env(monkeypatch):
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_PAPER_TRADE", True)
    monkeypatch.setattr(guardrails.live_slc_settings, "LIVE_TRADING", False)
    monkeypatch.setattr(guardrails.live_slc_settings, "ROBINHOOD_ENABLED", False)
    monkeypatch.setattr(guardrails.live_slc_settings, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_PAPER_EXECUTION_ENABLED", True)
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")


def test_submission_blocked_when_pinned_activation_proposal_file_is_missing(tmp_path, monkeypatch):
    """activation_proposal_sha256 pinned but the document doesn't exist on
    disk (missing/deleted) - must block, not skip the check."""
    _passing_env(monkeypatch)
    monkeypatch.setattr(guardrails, "REPO_ROOT", tmp_path)  # no research/ dir under here at all
    status_record = _passing_status_record(activation_proposal_sha256="a" * 64)
    operational = {"status": "paper_active", "status_record": status_record,
                   "baseline": {"baseline_sha256": "x", "environment": {}}}
    with pytest.raises(RuntimeError, match="activation proposal document is missing"):
        guardrails.assert_submission_preconditions(
            operational, observed_account_id="acct-1", daily_loss_breached=False,
        )


def test_submission_blocked_when_activation_proposal_hash_does_not_match(tmp_path, monkeypatch):
    """The document exists but its live hash no longer matches what was
    pinned at activation - e.g. edited after approval - must block."""
    _passing_env(monkeypatch)
    monkeypatch.setattr(guardrails, "REPO_ROOT", tmp_path)
    research_dir = tmp_path / "research"
    research_dir.mkdir()
    proposal_path = research_dir / "slc_4h_5m_stock_v1_paper_forward_activation_proposal.md"
    proposal_path.write_text("edited after approval", encoding="utf-8")
    status_record = _passing_status_record(activation_proposal_sha256="0" * 64)  # deliberately wrong
    operational = {"status": "paper_active", "status_record": status_record,
                   "baseline": {"baseline_sha256": "x", "environment": {}}}
    with pytest.raises(RuntimeError, match="does not match the version pinned"):
        guardrails.assert_submission_preconditions(
            operational, observed_account_id="acct-1", daily_loss_breached=False,
        )


def test_operational_preconditions_consults_the_baseline_resolution_gate(monkeypatch):
    """Phase 6: assert_operational_preconditions() must actually call
    verify_deployment_baseline() - which now embeds signature resolution
    via resolve_active_baseline() as its first step (no longer a separate
    verify_baseline_is_signed() call, now merged - see guardrails.py's
    module docstring on verify_deployment_baseline()). Proven by making
    verify_deployment_baseline() raise and confirming the error
    propagates from the real call site (the underlying resolution logic
    is already covered directly in test_live_slc_reauth_signature.py)."""
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")

    def _raise():
        raise RuntimeError("signature gate hit")

    monkeypatch.setattr(guardrails, "verify_deployment_baseline", _raise)
    with pytest.raises(RuntimeError, match="signature gate hit"):
        guardrails.assert_operational_preconditions(observed_account_id="acct-1")


def test_closeout_gate_blocks_on_execution_or_closeout_file_drift(tmp_path, signer, monkeypatch):
    """Drift in execution.py or closeout.py THEMSELVES must still block -
    those files' own hashes are explicitly part of the minimal closeout
    gate (the paired negative case to the test above)."""
    key_path, allowed_signers_path, identity = signer
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")

    hashes = guardrails.guardrail_hashes()
    baseline_path = tmp_path / "baseline.json"
    write_signed_baseline_event(
        key_path=key_path, identity=identity, baseline_path=baseline_path,
        guardrails_dict={
            "tier1": hashes["tier1"],
            "tier2": {**hashes["tier2"], "live_slc/execution.py": "0" * 64},
        },
    )
    with pytest.raises(RuntimeError, match="closeout blocked"):
        guardrails.assert_closeout_preconditions(observed_account_id="acct-1")
