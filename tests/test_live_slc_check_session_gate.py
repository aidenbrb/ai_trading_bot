"""
Tests for live_slc.check_session_gate - the post-session dry-run verdict
that runs the actual authorization.evaluate_dry_run_session_gate()
against the frozen reducer validation corpus (for synthetic_fixtures_passed)
and a real SlcSessionStat row.
"""
import json
from datetime import date
from pathlib import Path

import pytest

import live_slc.check_session_gate as check_session_gate
import live_slc.models as models
from live_slc.authorization import EXPECTED_SCHEDULED_CYCLES_PER_SESSION, evaluate_dry_run_session_gate
from live_slc.models import SlcSessionStat, get_live_slc_session


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def _passing_session_stat(**overrides):
    defaults = dict(
        session_date=date(2026, 8, 17), valid_bar_coverage_pct=96.0, cycles_over_budget=0,
        duplicate_or_stale_signal_count=0, guardrail_check_passed=True,
        closeout_left_no_open_state=True,
        cycles_run=EXPECTED_SCHEDULED_CYCLES_PER_SESSION,
        expected_symbol_count=142, valid_symbol_count=140,
        failed_cycles=0, overlapping_cycles=0, reconciliation_discrepancy_count=0,
        unprotected_position_incident_count=0, engine_parity_check_passed=True,
        closeout_confirmed_flat_by_broker_readback=True, dry_run_proposal_count=2,
    )
    defaults.update(overrides)
    return SlcSessionStat(**defaults)


# -- verify_synthetic_fixtures(): the real corpus, no mocking ----------------

def test_verify_synthetic_fixtures_passes_against_the_real_corpus():
    """No mocking - proves the real, currently-installed reducer and
    frozen batch generator genuinely produce both long and short signals
    for both AAPL and AMD within the declared evaluation window."""
    passed, evidence = check_session_gate.verify_synthetic_fixtures()
    assert passed is True
    assert len(evidence) == 2
    for line in evidence:
        assert "matched=True" in line
        assert "-> OK" in line


def test_verify_synthetic_fixtures_fails_closed_on_tampered_fixture_hash(tmp_path, monkeypatch):
    """Same tampering pattern as run_slc_live's own corpus tests: a
    fixture whose bytes no longer match the frozen manifest hash must
    never be loaded and compared - fails closed with an evidence line,
    not a raised exception."""
    import live_slc.run_slc_live as run_slc_live

    real_manifest = run_slc_live._load_reducer_corpus_manifest()
    fixture = real_manifest["fixtures"]["AAPL"]
    original_path = run_slc_live.REPO_ROOT / fixture["path"]

    tampered_csv_dir = tmp_path / "slc_reducer_corpus"
    tampered_csv_dir.mkdir()
    tampered_csv_path = tampered_csv_dir / "AAPL_tampered.csv"
    tampered_csv_path.write_bytes(original_path.read_bytes() + b"\n# tampered")

    tampered_manifest = json.loads(json.dumps(real_manifest))
    # Normal Windows pytest temp directories live outside the repository;
    # an absolute fixture path is valid here and keeps this negative-path
    # test independent of the runner's temp-directory policy.
    tampered_manifest["fixtures"]["AAPL"]["path"] = str(tampered_csv_path)
    # sha256 left as the ORIGINAL (now-stale) hash - simulating drift/tampering.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(tampered_manifest), encoding="utf-8")
    monkeypatch.setattr(run_slc_live, "REDUCER_CORPUS_MANIFEST_PATH", manifest_path)

    passed, evidence = check_session_gate.verify_synthetic_fixtures()
    assert passed is False
    assert len(evidence) == 1
    assert "corpus invalid" in evidence[0]


def test_verify_synthetic_fixtures_fails_closed_on_missing_manifest(monkeypatch):
    import live_slc.run_slc_live as run_slc_live

    monkeypatch.setattr(run_slc_live, "REDUCER_CORPUS_MANIFEST_PATH", Path("/nonexistent/manifest.json"))
    passed, evidence = check_session_gate.verify_synthetic_fixtures()
    assert passed is False
    assert len(evidence) == 1
    assert "corpus invalid" in evidence[0]


# -- main(): script exit code / output matches the gate's direct return ------

def _run_main(monkeypatch, capsys, session_date: str):
    monkeypatch.setattr(check_session_gate.sys, "argv", ["check_session_gate.py", "--date", session_date])
    try:
        check_session_gate.main()
        exit_code = 0
    except SystemExit as exc:
        exit_code = exc.code
    return exit_code, capsys.readouterr().out


def test_main_passes_when_the_gate_and_fixtures_both_pass(monkeypatch, capsys):
    with get_live_slc_session() as session:
        session.add(_passing_session_stat())

    with get_live_slc_session() as session:
        stat = session.get(SlcSessionStat, date(2026, 8, 17))
        expected_failures = evaluate_dry_run_session_gate(
            stat, synthetic_fixtures_passed=True, session=session,
        )
    assert expected_failures == []

    exit_code, out = _run_main(monkeypatch, capsys, "2026-08-17")
    assert exit_code == 0
    assert "PASS - dry-run session gate satisfied." in out


def test_main_fails_when_the_session_stat_itself_fails_the_gate(monkeypatch, capsys):
    with get_live_slc_session() as session:
        session.add(_passing_session_stat(dry_run_proposal_count=5))

    with get_live_slc_session() as session:
        stat = session.get(SlcSessionStat, date(2026, 8, 17))
        expected_failures = evaluate_dry_run_session_gate(
            stat, synthetic_fixtures_passed=True, session=session,
        )
    assert expected_failures != []

    exit_code, out = _run_main(monkeypatch, capsys, "2026-08-17")
    assert exit_code == 1
    assert "FAIL" in out
    for failure in expected_failures:
        assert failure in out


def test_main_fails_when_no_session_stat_row_exists(monkeypatch, capsys):
    exit_code, out = _run_main(monkeypatch, capsys, "2099-01-01")
    assert exit_code == 1
    assert "No SlcSessionStat row" in out
