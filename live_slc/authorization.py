"""
SlcDeploymentStatus / SlcActivationEvent CRUD and the activation gate.

Deliberately has no dependency on guardrails.py (guardrails.py depends on
this module instead, to avoid a cycle) - this module is pure data access
plus the dry_run acceptance-criteria evaluator. A human always calls
record_transition() explicitly; nothing here auto-promotes. Importing
live_slc.risk here does NOT create that cycle: risk.py has no dependency
on this module, guardrails.py, or run_slc_live.py - only on models.py.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import datetime as dt
import secrets
import subprocess

from sqlmodel import select

from live_slc import reauth_signature, risk
from live_slc.models import (
    SlcActivationEvent,
    SlcDeploymentStatus,
    SlcReauthNonce,
    SlcSessionStat,
    get_live_slc_session,
)

STRATEGY_VERSION = "slc_4h_5m_stock_v1"
# "live" added Phase 6 Step 1: no code path can reach it yet (nothing sets
# EXECUTION_ENABLED for real-money SLC orders), but the signature gate's
# scope explicitly covers "any transition to paper_active or live" and a
# status the gate can't even name isn't a status it can meaningfully guard.
VALID_STATUSES = ("not_authorized", "dry_run", "paper_active", "suspended", "halted", "live")

# Restated in full (rev. 6 Step 9) rather than referencing the external
# Codex plan, so the acceptance bar is self-contained here.
MIN_VALID_BAR_COVERAGE_PCT = 95.0
MAX_CYCLE_SECONDS = 300.0
# The fixed Windows cycle trigger runs from 09:36 through 15:31 ET at
# five-minute intervals: 72 scheduled invocations on every trading day.
# The engineering gate promised at least 95% scheduled-run success; merely
# checking cycles_run > 0 allowed today's 54/72 session to look healthy.
EXPECTED_SCHEDULED_CYCLES_PER_SESSION = 72
MIN_SCHEDULED_RUN_SUCCESS_PCT = 95.0
MIN_SUCCESSFUL_CYCLES_PER_SESSION = math.ceil(
    EXPECTED_SCHEDULED_CYCLES_PER_SESSION * MIN_SCHEDULED_RUN_SUCCESS_PCT / 100.0
)


def get_current_deployment_record() -> SlcDeploymentStatus:
    with get_live_slc_session() as session:
        record = session.get(SlcDeploymentStatus, STRATEGY_VERSION)
        if record is None:
            record = SlcDeploymentStatus(strategy_version=STRATEGY_VERSION, status="not_authorized")
            session.add(record)
            session.flush()
            session.refresh(record)
        session.expunge(record)
        return record


def evaluate_dry_run_session_gate(
    session_stat: SlcSessionStat, *, synthetic_fixtures_passed: bool, session,
) -> list[str]:
    """The full corrected criteria (rev. 11 Step 12) a single dry_run
    session must meet before paper_active is even considered. Every
    boolean field on SlcSessionStat defaults False and every count
    defaults to zero, so a never-populated, all-default row fails by
    construction, not by accident - cycles_run <= 0 is checked first for
    exactly that reason, though (matching the pre-existing style) every
    other criterion is still evaluated and accumulated, never short-
    circuited, so a caller reviewing a failed session sees everything
    wrong with it at once.

    `session` is a LIVE DB session (not a historical count captured on
    the SlcSessionStat row) - the "zero unresolved ambiguous states"
    check reflects CURRENT SlcPosition/SlcOrder state at evaluation time
    via risk.system_wide_entry_block_reasons(), the same function the
    live admission loop itself checks before every candidate (rev. 11
    Step 3). Returns the list of failures - empty means the session
    passed."""
    failures: list[str] = []
    if session_stat.cycles_run <= 0:
        failures.append("no cycles were recorded for this session (cycles_run <= 0)")
    scheduled_success_pct = min(
        100.0,
        100.0 * session_stat.cycles_run / EXPECTED_SCHEDULED_CYCLES_PER_SESSION,
    )
    if session_stat.cycles_run < MIN_SUCCESSFUL_CYCLES_PER_SESSION:
        failures.append(
            f"scheduled-run success {scheduled_success_pct:.1f}% "
            f"({session_stat.cycles_run}/{EXPECTED_SCHEDULED_CYCLES_PER_SESSION}) < "
            f"{MIN_SCHEDULED_RUN_SUCCESS_PCT:.1f}% required"
        )
    if session_stat.valid_bar_coverage_pct < MIN_VALID_BAR_COVERAGE_PCT:
        failures.append(
            f"valid bar coverage {session_stat.valid_bar_coverage_pct:.1f}% < "
            f"{MIN_VALID_BAR_COVERAGE_PCT:.1f}% required"
        )
    if session_stat.expected_symbol_count <= 0 or session_stat.valid_symbol_count <= 0:
        failures.append(
            "symbol coverage not meaningfully populated (expected_symbol_count="
            f"{session_stat.expected_symbol_count}, valid_symbol_count={session_stat.valid_symbol_count})"
        )
    if session_stat.cycles_over_budget > 0:
        failures.append(f"{session_stat.cycles_over_budget} cycle(s) exceeded {MAX_CYCLE_SECONDS:.0f}s")
    if session_stat.failed_cycles > 0:
        failures.append(f"{session_stat.failed_cycles} cycle(s) failed")
    if session_stat.overlapping_cycles > 0:
        failures.append(f"{session_stat.overlapping_cycles} cycle(s) overlapped")
    if session_stat.duplicate_or_stale_signal_count > 0:
        failures.append(f"{session_stat.duplicate_or_stale_signal_count} duplicate signal(s) occurred")
    if session_stat.reconciliation_discrepancy_count > 0:
        failures.append(f"{session_stat.reconciliation_discrepancy_count} reconciliation discrepancy(ies) occurred")
    if session_stat.unprotected_position_incident_count > 0:
        failures.append(f"{session_stat.unprotected_position_incident_count} unprotected-position incident(s) occurred")
    if session_stat.dry_run_proposal_count > risk.SLC_MAX_NEW_TRADES_PER_DAY:
        failures.append(
            f"{session_stat.dry_run_proposal_count} dry-run proposal(s) exceeded "
            f"the {risk.SLC_MAX_NEW_TRADES_PER_DAY}/day cap"
        )
    if not session_stat.guardrail_check_passed:
        failures.append("guardrail check did not pass for the session")
    if not session_stat.engine_parity_check_passed:
        failures.append("engine-parity self-check did not pass for the session")
    if session_stat.closeout_left_no_open_state is not True:
        failures.append("closeout/reconciliation did not leave zero pending/open SLC state")
    if session_stat.closeout_confirmed_flat_by_broker_readback is not True:
        failures.append("closeout was not confirmed flat by an actual broker-side read-back")
    block_reasons = risk.system_wide_entry_block_reasons(session)
    if block_reasons:
        failures.append(f"unresolved ambiguous state(s) at evaluation time: {block_reasons}")
    if not synthetic_fixtures_passed:
        failures.append("synthetic long/short golden-master fixtures did not pass")
    return failures


def issue_reauth_nonce() -> str:
    """Server-generated, stored as pending, before any payload is ever
    shown to a human for signing (Phase 6 Step 1c). The nonce's own
    created_at, not any client-supplied timestamp, is what the 24h
    validity window in _verify_reauth_signature() is checked against."""
    nonce = secrets.token_hex(16)
    with get_live_slc_session() as session:
        session.add(SlcReauthNonce(nonce=nonce))
    return nonce


def _current_git_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=15, check=True,
    )
    return result.stdout.strip()


def _verify_reauth_signature(
    session, *, from_status: str, to_status: str, guardrail_baseline_sha256: Optional[str],
    activation_proposal_sha256: Optional[str], observed_account_id: Optional[str],
    signed_payload: Optional[str], signature_blob: Optional[str], signer_identity: Optional[str],
    nonce: Optional[str], git_commit_sha: Optional[str], changed_guardrail_paths: Optional[list[str]],
    baseline_file_relative_path: Optional[str],
) -> None:
    """Raises on any failure; returns (and marks the nonce consumed) only
    on a fully valid, fresh, matching, cryptographically-verified
    signature. Every field the payload claims is cross-checked against
    what's actually being recorded - a signature is only as trustworthy
    as the fields it was shown, and this refuses to trust the caller's
    separate arguments over the signed text on faith."""
    missing = [
        name for name, value in (
            ("signed_payload", signed_payload), ("signature_blob", signature_blob),
            ("signer_identity", signer_identity), ("nonce", nonce),
            ("git_commit_sha", git_commit_sha), ("guardrail_baseline_sha256", guardrail_baseline_sha256),
            ("baseline_file_relative_path", baseline_file_relative_path),
        ) if not value
    ]
    if missing:
        raise RuntimeError(f"signed re-authorization required but missing: {missing}")

    nonce_row = session.get(SlcReauthNonce, nonce)
    if nonce_row is None:
        raise RuntimeError(f"unknown nonce {nonce!r} - must be issued via issue_reauth_nonce() first")
    if nonce_row.consumed_at is not None:
        raise RuntimeError(f"nonce {nonce!r} was already consumed at {nonce_row.consumed_at} - replay rejected")
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    age = now - nonce_row.created_at
    if age > dt.timedelta(seconds=reauth_signature.NONCE_VALIDITY_SECONDS):
        raise RuntimeError(f"nonce {nonce!r} expired: issued {age} ago, validity window is 24h")

    fields = reauth_signature.parse_payload(signed_payload)
    expected = {
        "from_status": from_status,
        "to_status": to_status,
        "guardrail_baseline_sha256": guardrail_baseline_sha256,
        "activation_proposal_sha256": activation_proposal_sha256 or "",
        "observed_account_id": observed_account_id or "",
        "git_commit_sha": git_commit_sha,
        "changed_guardrail_paths": ",".join(sorted(changed_guardrail_paths or [])),
        "baseline_file_relative_path": baseline_file_relative_path,
        "nonce": nonce,
    }
    mismatched = [name for name, value in expected.items() if fields.get(name) != value]
    if mismatched:
        raise RuntimeError(
            f"signed payload does not match the transition being recorded: {mismatched} "
            f"(payload declared {[(name, fields.get(name)) for name in mismatched]}, "
            f"expected {[(name, expected[name]) for name in mismatched]})"
        )

    actual_head = _current_git_commit_sha()
    if git_commit_sha != actual_head:
        raise RuntimeError(
            f"signed payload's git_commit_sha {git_commit_sha!r} does not match "
            f"the actual current HEAD {actual_head!r}"
        )

    if not reauth_signature.verify_signature(
        signed_payload, signature_blob, signer_identity=signer_identity,
        allowed_signers_path=reauth_signature.ALLOWED_SIGNERS_PATH,
    ):
        raise RuntimeError("signature verification failed (invalid, tampered, or unknown signer)")

    nonce_row.consumed_at = now
    session.add(nonce_row)


def record_transition(
    from_status: str,
    to_status: str,
    reason: str,
    *,
    session_gate_metrics: Optional[dict] = None,
    guardrail_baseline_sha256: Optional[str] = None,
    activation_proposal_sha256: Optional[str] = None,
    observed_account_id: Optional[str] = None,
    operator_note: str = "",
    signed_payload: Optional[str] = None,
    signature_blob: Optional[str] = None,
    signer_identity: Optional[str] = None,
    nonce: Optional[str] = None,
    git_commit_sha: Optional[str] = None,
    changed_guardrail_paths: Optional[list[str]] = None,
    baseline_file_relative_path: Optional[str] = None,
) -> SlcActivationEvent:
    """The only way SlcDeploymentStatus changes. Always explicit, always
    audited. A same-status transition (e.g. paper_active -> paper_active)
    is valid and required whenever Tier 2 is deliberately re-frozen.

    Phase 6 Step 1: a human-only signature (signed_payload/signature_blob/
    signer_identity/nonce/git_commit_sha/changed_guardrail_paths) is
    required whenever this call pins a new guardrail_baseline_sha256 (any
    Tier-1/Tier-2 re-baseline) OR to_status is paper_active/live -
    regardless of whether guardrail_baseline_sha256 also changed. See
    live_slc/reauth_signature.py for the payload format and verification
    mechanism."""
    if to_status not in VALID_STATUSES:
        raise ValueError(f"invalid target status: {to_status!r}")
    requires_signature = guardrail_baseline_sha256 is not None or to_status in ("paper_active", "live")
    with get_live_slc_session() as session:
        record = session.get(SlcDeploymentStatus, STRATEGY_VERSION)
        if record is None:
            record = SlcDeploymentStatus(strategy_version=STRATEGY_VERSION, status="not_authorized")
            session.add(record)
            session.flush()
        if record.status != from_status:
            raise RuntimeError(
                f"cannot transition {from_status!r} -> {to_status!r}: "
                f"current status is {record.status!r}"
            )
        if to_status in ("paper_active", "live"):
            if not guardrail_baseline_sha256:
                raise RuntimeError(f"{to_status} requires a guardrail_baseline_sha256 to pin")
            if not activation_proposal_sha256:
                raise RuntimeError(f"{to_status} requires an activation_proposal_sha256 to pin")
            if not observed_account_id:
                raise RuntimeError(f"{to_status} requires an observed_account_id to pin")
        if requires_signature:
            _verify_reauth_signature(
                session, from_status=from_status, to_status=to_status,
                guardrail_baseline_sha256=guardrail_baseline_sha256,
                activation_proposal_sha256=activation_proposal_sha256,
                observed_account_id=observed_account_id,
                signed_payload=signed_payload, signature_blob=signature_blob,
                signer_identity=signer_identity, nonce=nonce, git_commit_sha=git_commit_sha,
                changed_guardrail_paths=changed_guardrail_paths,
                baseline_file_relative_path=baseline_file_relative_path,
            )
        record.status = to_status
        record.updated_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        record.updated_by = "record_transition"
        if guardrail_baseline_sha256 is not None:
            record.live_baseline_sha256 = guardrail_baseline_sha256
        if activation_proposal_sha256 is not None:
            record.activation_proposal_sha256 = activation_proposal_sha256
        if observed_account_id is not None:
            record.observed_account_id = observed_account_id
        session.add(record)

        event = SlcActivationEvent(
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            session_gate_metrics_json=json.dumps(session_gate_metrics or {}, sort_keys=True, default=str),
            guardrail_baseline_sha256_at_transition=guardrail_baseline_sha256,
            activation_proposal_sha256_at_transition=activation_proposal_sha256,
            observed_account_id=observed_account_id,
            operator_note=operator_note,
            signed_payload=signed_payload if requires_signature else None,
            signature_blob=signature_blob if requires_signature else None,
            git_commit_sha=git_commit_sha if requires_signature else None,
            changed_guardrail_paths_json=(
                json.dumps(sorted(changed_guardrail_paths)) if changed_guardrail_paths else None
            ),
            nonce=nonce if requires_signature else None,
            signer_identity=signer_identity if requires_signature else None,
            baseline_file_relative_path=baseline_file_relative_path if requires_signature else None,
        )
        session.add(event)
        session.flush()
        if requires_signature:
            nonce_row = session.get(SlcReauthNonce, nonce)
            nonce_row.consumed_by_event_id = event.id
            session.add(nonce_row)
        session.refresh(event)
        session.expunge(event)
        return event
