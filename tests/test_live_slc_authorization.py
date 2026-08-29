import pandas as pd
import pytest

import live_slc.models as models
from live_slc.authorization import (
    EXPECTED_SCHEDULED_CYCLES_PER_SESSION,
    MAX_CYCLE_SECONDS,
    MIN_SUCCESSFUL_CYCLES_PER_SESSION,
    MIN_VALID_BAR_COVERAGE_PCT,
    evaluate_dry_run_session_gate,
    get_current_deployment_record,
    record_transition,
)
from live_slc.models import SlcSessionStat, get_live_slc_session


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def _passing_session_stat(**overrides):
    defaults = dict(
        session_date="2026-08-13", valid_bar_coverage_pct=96.0, cycles_over_budget=0,
        duplicate_or_stale_signal_count=0, guardrail_check_passed=True,
        closeout_left_no_open_state=True,
        # rev. 11 Step 12's new required-to-pass fields:
        cycles_run=EXPECTED_SCHEDULED_CYCLES_PER_SESSION,
        expected_symbol_count=142, valid_symbol_count=140,
        failed_cycles=0, overlapping_cycles=0, reconciliation_discrepancy_count=0,
        unprotected_position_incident_count=0, engine_parity_check_passed=True,
        closeout_confirmed_flat_by_broker_readback=True,
    )
    defaults.update(overrides)
    return SlcSessionStat(**defaults)


def _evaluate(stat, *, synthetic_fixtures_passed=True):
    with get_live_slc_session() as session:
        return evaluate_dry_run_session_gate(
            stat, synthetic_fixtures_passed=synthetic_fixtures_passed, session=session,
        )


def test_initial_status_is_not_authorized():
    assert get_current_deployment_record().status == "not_authorized"


def test_full_lifecycle_not_authorized_to_dry_run_to_paper_active():
    record_transition("not_authorized", "dry_run", "starting dry run")
    assert get_current_deployment_record().status == "dry_run"

    record_transition(
        "dry_run", "paper_active", "one session passed",
        guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
        observed_account_id="acct-1",
    )
    record = get_current_deployment_record()
    assert record.status == "paper_active"
    assert record.live_baseline_sha256 == "a" * 64
    assert record.activation_proposal_sha256 == "b" * 64
    assert record.observed_account_id == "acct-1"


def test_transition_rejected_if_current_status_does_not_match_from_status():
    with pytest.raises(RuntimeError, match="current status is"):
        record_transition("dry_run", "paper_active", "skip ahead illegally")


def test_paper_active_requires_all_three_trust_chain_values():
    record_transition("not_authorized", "dry_run", "start")
    with pytest.raises(RuntimeError, match="guardrail_baseline_sha256"):
        record_transition("dry_run", "paper_active", "missing baseline hash",
                           activation_proposal_sha256="b" * 64, observed_account_id="acct-1")
    with pytest.raises(RuntimeError, match="activation_proposal_sha256"):
        record_transition("dry_run", "paper_active", "missing proposal hash",
                           guardrail_baseline_sha256="a" * 64, observed_account_id="acct-1")
    with pytest.raises(RuntimeError, match="observed_account_id"):
        record_transition("dry_run", "paper_active", "missing account id",
                           guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64)


def test_same_status_reauthorization_event_is_valid_for_tier2_refreeze():
    record_transition("not_authorized", "dry_run", "start")
    record_transition("dry_run", "paper_active", "activate",
                       guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
                       observed_account_id="acct-1")
    # A deliberate live_slc code change requires a new baseline + a new event,
    # even though status doesn't change.
    event = record_transition("paper_active", "paper_active", "tier2 re-freeze after a code change",
                               guardrail_baseline_sha256="c" * 64, activation_proposal_sha256="b" * 64,
                               observed_account_id="acct-1")
    assert event.from_status == event.to_status == "paper_active"
    assert get_current_deployment_record().live_baseline_sha256 == "c" * 64


def test_dry_run_gate_passes_a_fully_populated_session():
    assert _evaluate(_passing_session_stat()) == []


def test_dry_run_gate_fails_below_coverage_threshold():
    stat = _passing_session_stat(valid_bar_coverage_pct=MIN_VALID_BAR_COVERAGE_PCT - 1)
    assert any("coverage" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_on_cycle_overrun():
    stat = _passing_session_stat(cycles_over_budget=1)
    assert any(f"{MAX_CYCLE_SECONDS:.0f}s" in f for f in _evaluate(stat))


def test_dry_run_gate_requires_95pct_of_the_72_scheduled_cycles():
    below = _passing_session_stat(cycles_run=MIN_SUCCESSFUL_CYCLES_PER_SESSION - 1)
    assert any("scheduled-run success" in f for f in _evaluate(below))

    at_threshold = _passing_session_stat(cycles_run=MIN_SUCCESSFUL_CYCLES_PER_SESSION)
    assert not any("scheduled-run success" in f for f in _evaluate(at_threshold))


def test_dry_run_gate_fails_on_duplicate_signals():
    stat = _passing_session_stat(duplicate_or_stale_signal_count=1)
    assert any("duplicate signal" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_without_guardrail_check_passing():
    stat = _passing_session_stat(guardrail_check_passed=False)
    assert any("guardrail" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_if_closeout_left_open_state():
    stat = _passing_session_stat(closeout_left_no_open_state=False)
    assert any("closeout" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_without_synthetic_fixture_proof():
    assert any("synthetic" in f for f in _evaluate(_passing_session_stat(), synthetic_fixtures_passed=False))


def test_dry_run_gate_fails_when_proposals_exceed_the_daily_cap():
    """A future regression that let a session record 3+ simulated
    proposals must not pass the gate silently - dry_run_proposal_count
    defaults to 0 on the model, so this is purely additive and doesn't
    change any other passing test's outcome."""
    stat = _passing_session_stat(dry_run_proposal_count=3)
    assert any("exceeded" in f and "cap" in f for f in _evaluate(stat))


def test_dry_run_gate_passes_at_exactly_the_daily_cap():
    stat = _passing_session_stat(dry_run_proposal_count=2)
    assert _evaluate(stat) == []


# -- rev. 11 Step 12: the corrected criteria ---------------------------------

def test_dry_run_gate_fails_a_never_populated_all_default_session_first():
    """The exact degenerate case the correction targets: a brand-new
    SlcSessionStat row with every field left at its default must fail
    (cycles_run <= 0), not trivially pass because nothing was populated
    to fail against."""
    blank = SlcSessionStat(session_date="2026-08-13")
    failures = _evaluate(blank)
    assert any("cycles_run" in f for f in failures)
    assert len(failures) > 1  # every other unmet criterion is ALSO reported, not short-circuited


def test_dry_run_gate_fails_without_engine_parity_check():
    stat = _passing_session_stat(engine_parity_check_passed=False)
    assert any("engine-parity" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_on_zero_symbol_coverage_even_with_high_percentage():
    """A 0/0 degenerate case must never trivially satisfy a bare
    percentage check - expected/valid symbol counts must themselves be
    populated above zero."""
    stat = _passing_session_stat(expected_symbol_count=0, valid_symbol_count=0)
    assert any("symbol coverage" in f for f in _evaluate(stat))


def test_dry_run_gate_fails_on_failed_or_overlapping_cycles():
    assert any("failed" in f for f in _evaluate(_passing_session_stat(failed_cycles=1)))
    assert any("overlapped" in f for f in _evaluate(_passing_session_stat(overlapping_cycles=1)))


def test_dry_run_gate_fails_on_reconciliation_discrepancies_or_unprotected_incidents():
    assert any("reconciliation discrepancy" in f for f in _evaluate(_passing_session_stat(reconciliation_discrepancy_count=1)))
    assert any("unprotected-position" in f for f in _evaluate(_passing_session_stat(unprotected_position_incident_count=1)))


def test_dry_run_gate_fails_without_broker_readback_confirmation_distinct_from_local_view():
    """closeout_confirmed_flat_by_broker_readback is checked separately
    from closeout_left_no_open_state - a passing local view alone is not
    sufficient."""
    stat = _passing_session_stat(closeout_confirmed_flat_by_broker_readback=False)
    failures = _evaluate(stat)
    assert any("broker-side read-back" in f for f in failures)
    assert stat.closeout_left_no_open_state is True  # the local view alone was fine


def test_dry_run_gate_fails_on_live_unresolved_ambiguous_state_not_just_historical_stat():
    """Queried LIVE against current SlcPosition/SlcOrder state via
    risk.system_wide_entry_block_reasons() - a passing historical
    SlcSessionStat row must not paper over a real, currently-unresolved
    ambiguous position."""
    from live_slc.models import SlcPosition
    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol="AAPL", level_id="demand:x",
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-13").date(), status="protected_degraded",
            qty=10.0, entry_price=100.0, stop_price=98.0, target_price=104.0,
        ))
    failures = _evaluate(_passing_session_stat())
    assert any("unresolved ambiguous state" in f for f in failures)
