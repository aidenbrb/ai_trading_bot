"""Shared test-only helpers for producing a valid Phase 6 Step 1
re-authorization signature. Never imported by production code (the
leading underscore also keeps pytest from collecting this as a test
module). Generates a disposable SOFTWARE ed25519 test key per call - not
the hardware-backed ed25519-sk production keys use - but signs through
the exact same `ssh-keygen -Y sign` / `-Y verify` round trip real
signatures use, so what these tests exercise is the real verification
mechanism, not a stand-in for it.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

from live_slc import authorization, reauth_signature


def make_test_key(tmp_path: Path, identity: str = "test-signer") -> tuple[Path, Path, str]:
    key_path = tmp_path / f"{identity}_key"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", identity, "-q"],
        check=True, capture_output=True, timeout=15,
    )
    allowed_signers_path = tmp_path / "allowed_signers"
    pub = key_path.with_suffix(".pub").read_text(encoding="utf-8")
    allowed_signers_path.write_text(f"{identity} {pub}", encoding="utf-8")
    return key_path, allowed_signers_path, identity


def write_signed_baseline_event(
    *,
    key_path: Path,
    identity: str,
    baseline_path: Path,
    guardrails_dict: dict,
    environment: dict | None = None,
    from_status: str = "not_authorized",
    to_status: str = "paper_active",
    activation_proposal_sha256: str = "b" * 64,
    observed_account_id: str = "acct-1",
    changed_guardrail_paths: list[str] | None = None,
):
    """Writes a baseline JSON file with the given guardrails dict, signs a
    payload pinning it, and records the transition - the full Phase 6
    Step 1 (circularity-fix) scenario a test needs to exercise
    resolve_active_baseline()/verify_deployment_baseline() against a
    self-contained baseline rather than real repo/DB state. Returns the
    recorded SlcActivationEvent. Caller is responsible for getting
    `from_status` to match the DB's actual current status first (e.g. via
    a prior record_transition("not_authorized", "dry_run", ...) call) -
    this only performs the final signed transition.
    """
    import hashlib
    import json as _json

    from live_slc.authorization import record_transition

    baseline_path.write_text(_json.dumps({
        "guardrails": guardrails_dict,
        "environment": environment or {},
    }), encoding="utf-8")
    digest = hashlib.sha256()
    with baseline_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    baseline_hash = digest.hexdigest()

    # guardrails.py resolves this as REPO_ROOT / value - pathlib's own
    # semantics mean that when the right-hand side is already an absolute
    # path, the left side is discarded and the absolute path wins, so an
    # absolute tmp_path here resolves correctly without needing a real
    # repo-relative path in tests.
    baseline_rel_path = str(baseline_path)

    sig_kwargs = signed_reauth_kwargs(
        key_path=key_path, identity=identity, from_status=from_status, to_status=to_status,
        guardrail_baseline_sha256=baseline_hash, activation_proposal_sha256=activation_proposal_sha256,
        observed_account_id=observed_account_id, changed_guardrail_paths=changed_guardrail_paths,
        baseline_file_relative_path=baseline_rel_path,
    )
    return record_transition(
        from_status, to_status, "test signed baseline",
        guardrail_baseline_sha256=baseline_hash, activation_proposal_sha256=activation_proposal_sha256,
        observed_account_id=observed_account_id,
        **sig_kwargs,
    )


def signed_reauth_kwargs(
    *,
    key_path: Path,
    identity: str,
    from_status: str,
    to_status: str,
    guardrail_baseline_sha256: str,
    activation_proposal_sha256: str | None = None,
    observed_account_id: str | None = None,
    changed_guardrail_paths: list[str] | None = None,
    baseline_file_relative_path: str | None = None,
    git_commit_sha: str | None = None,
    nonce: str | None = None,
    timestamp_iso: str | None = None,
) -> dict:
    """Builds, signs, and returns the exact kwargs record_transition()
    needs for a valid signed transition. Uses a freshly-issued real nonce
    and the real current git HEAD by default, so what's signed matches
    what record_transition()'s own checks expect without any mocking of
    those two things. baseline_file_relative_path defaults to a
    nonce-unique placeholder when the test doesn't care about resolving a
    real baseline file afterward (e.g. tests only exercising the
    signature mechanics, not verify_deployment_baseline())."""
    if nonce is None:
        nonce = authorization.issue_reauth_nonce()
    if git_commit_sha is None:
        git_commit_sha = authorization._current_git_commit_sha()
    if timestamp_iso is None:
        timestamp_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    if baseline_file_relative_path is None:
        baseline_file_relative_path = f"research/test_baseline_{nonce}.json"
    payload = reauth_signature.build_payload(
        from_status=from_status, to_status=to_status,
        guardrail_baseline_sha256=guardrail_baseline_sha256,
        activation_proposal_sha256=activation_proposal_sha256,
        observed_account_id=observed_account_id,
        git_commit_sha=git_commit_sha,
        changed_guardrail_paths=changed_guardrail_paths or [],
        baseline_file_relative_path=baseline_file_relative_path,
        nonce=nonce, timestamp_iso=timestamp_iso,
    )
    signature_blob = reauth_signature.sign_for_test(payload, key_path)
    return dict(
        signed_payload=payload, signature_blob=signature_blob, signer_identity=identity,
        nonce=nonce, git_commit_sha=git_commit_sha, changed_guardrail_paths=changed_guardrail_paths,
        baseline_file_relative_path=baseline_file_relative_path,
    )
