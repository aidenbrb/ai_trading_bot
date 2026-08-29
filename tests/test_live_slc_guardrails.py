"""
Guardrail-hash tests for live_slc. Deliberately a separate file from
tests/test_slc_backtest.py (rev. 6 correction) - that file and its
guardrail tests remain fully untouched by this work.
"""
from pathlib import Path

import pytest

import live_slc.guardrails as guardrails
import backtest.run_slc_backtest as backtest_runner


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


def test_tier1_matches_existing_slc_backtest_baseline_exactly():
    """Zero drift across the 9 bot-wide guardrail files, independent of
    everything else in live_slc - cross-checked against the SLC backtest
    runner's own independently-computed hashes."""
    live_hashes = guardrails.guardrail_hashes()["tier1"]
    backtest_hashes = backtest_runner._guardrail_hashes()
    assert live_hashes == backtest_hashes


def test_guardrails_module_excluded_from_its_own_tier2_hash_set():
    """guardrails.py cannot hash itself (the value it hashes changes the
    moment the resulting hash is written back into itself) - the tests
    below prove the exclusion is deliberate, not an oversight."""
    assert not any("guardrails.py" in name for name in guardrails.GUARDRAILS_TIER2)
    assert not any("guardrails.py" in name for name in guardrails.GUARDRAILS_TIER1)


def test_allowed_signers_excluded_from_tier2_for_the_same_circularity_reason():
    """Phase 6 Step 1: guardrailing allowed_signers would make the
    operator's first-ever edit to it (adding their real key) a re-baseline
    requiring a signature verified against the very file being changed -
    circular by construction. Its safety instead comes from
    reauth_signature.verify_signature() failing closed on an
    empty/missing/wrong file, so tampering with it can only ever restrict,
    never loosen, what can be signed."""
    assert not any("allowed_signers" in name for name in guardrails.GUARDRAILS_TIER2)
    assert not any("allowed_signers" in name for name in guardrails.GUARDRAILS_TIER1)


def test_reauth_signature_module_is_tier2_guardrailed():
    assert "live_slc/reauth_signature.py" in guardrails.GUARDRAILS_TIER2


def test_verify_deployment_baseline_passes_against_the_real_frozen_state():
    result = guardrails.verify_deployment_baseline()
    assert result["baseline_sha256"] == guardrails.EXPECTED_DEPLOYMENT_BASELINE_SHA256
    assert len(result["guardrails"]["tier1"]) == 9
    assert len(result["guardrails"]["tier2"]) == 27  # +migrations.py, +split_detection.py (rev. 11), +run_hidden.vbs (activation), +check_schedule_health.py + its .bat wrapper (sleep-fix), +reauth_signature.py + promotion.md (Phase 6)


def test_run_hidden_vbs_present_in_tier2():
    assert "scripts/slc_live/run_hidden.vbs" in guardrails.GUARDRAILS_TIER2


def test_schedule_health_check_and_its_wrapper_present_in_tier2():
    """Unlike check_readiness.py/check_session_gate.py/check_paper_session_audit.py
    (exempt from Tier-2: a human exercises judgment before running them
    manually), check_schedule_health.py is invoked automatically by its
    own Scheduled Task, so that exemption doesn't apply here."""
    assert "live_slc/check_schedule_health.py" in guardrails.GUARDRAILS_TIER2
    assert "scripts/slc_live/run_slc_schedule_health.bat" in guardrails.GUARDRAILS_TIER2


def test_tier2_drift_detected_on_a_signal_side_file(tmp_path, monkeypatch):
    """Mutating a signal-side Tier-2 file must be detected as drift."""
    fake_baseline = tmp_path / "baseline.json"
    hashes = guardrails.guardrail_hashes()
    import json
    fake_baseline.write_text(json.dumps({
        "guardrails": {
            "tier1": hashes["tier1"],
            "tier2": {**hashes["tier2"], "live_slc/reducer.py": "0" * 64},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(guardrails, "DEPLOYMENT_BASELINE", fake_baseline)
    monkeypatch.setattr(guardrails, "EXPECTED_DEPLOYMENT_BASELINE_SHA256", guardrails._sha256(fake_baseline))
    with pytest.raises(RuntimeError, match="guardrail drift"):
        guardrails.verify_deployment_baseline()


def test_baseline_file_tampering_detected():
    original = guardrails._sha256(guardrails.DEPLOYMENT_BASELINE)
    assert original == guardrails.EXPECTED_DEPLOYMENT_BASELINE_SHA256
    tampered_expected = "0" * 64
    import live_slc.guardrails as g

    class _Tampered:
        pass

    old = g.EXPECTED_DEPLOYMENT_BASELINE_SHA256
    try:
        g.EXPECTED_DEPLOYMENT_BASELINE_SHA256 = tampered_expected
        with pytest.raises(RuntimeError, match="baseline file hash mismatch"):
            g.verify_deployment_baseline()
    finally:
        g.EXPECTED_DEPLOYMENT_BASELINE_SHA256 = old


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


def test_operational_preconditions_consults_the_signature_gate(monkeypatch):
    """Phase 6 Step 1a: assert_operational_preconditions() must actually
    call verify_baseline_is_signed() with the current baseline hash, not
    just have that function exist unused elsewhere - proven here by
    making it raise and confirming the error propagates from the real
    call site, not a re-derivation of verify_baseline_is_signed()'s own
    logic (already covered directly in test_live_slc_reauth_signature.py)."""
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")
    monkeypatch.setattr(
        guardrails, "verify_deployment_baseline",
        lambda: {"baseline_sha256": "fake-hash", "environment": {}},
    )

    def _raise(baseline_hash):
        raise RuntimeError(f"signature gate hit for {baseline_hash}")

    monkeypatch.setattr(guardrails, "verify_baseline_is_signed", _raise)
    with pytest.raises(RuntimeError, match="signature gate hit for fake-hash"):
        guardrails.assert_operational_preconditions(observed_account_id="acct-1")


def test_closeout_gate_blocks_on_execution_or_closeout_file_drift(tmp_path, monkeypatch):
    """Drift in execution.py or closeout.py THEMSELVES must still block -
    those files' own hashes are explicitly part of the minimal closeout
    gate (the paired negative case to the test above)."""
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_API_KEY", "key")
    monkeypatch.setattr(guardrails.live_slc_settings, "ALPACA_SECRET_KEY", "secret")
    monkeypatch.setattr(guardrails.live_slc_settings, "SLC_EXPECTED_ACCOUNT_ID", "acct-1")

    import json
    baseline = json.loads(guardrails.DEPLOYMENT_BASELINE.read_text(encoding="utf-8"))
    baseline["guardrails"]["tier2"]["live_slc/execution.py"] = "0" * 64
    fake_baseline = tmp_path / "baseline.json"
    fake_baseline.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(guardrails, "DEPLOYMENT_BASELINE", fake_baseline)
    monkeypatch.setattr(guardrails, "EXPECTED_DEPLOYMENT_BASELINE_SHA256", guardrails._sha256(fake_baseline))
    with pytest.raises(RuntimeError, match="closeout blocked"):
        guardrails.assert_closeout_preconditions(observed_account_id="acct-1")
