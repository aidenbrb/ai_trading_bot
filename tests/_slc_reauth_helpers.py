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
    git_commit_sha: str | None = None,
    nonce: str | None = None,
    timestamp_iso: str | None = None,
) -> dict:
    """Builds, signs, and returns the exact kwargs record_transition()
    needs for a valid signed transition. Uses a freshly-issued real nonce
    and the real current git HEAD by default, so what's signed matches
    what record_transition()'s own checks expect without any mocking of
    those two things."""
    if nonce is None:
        nonce = authorization.issue_reauth_nonce()
    if git_commit_sha is None:
        git_commit_sha = authorization._current_git_commit_sha()
    if timestamp_iso is None:
        timestamp_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = reauth_signature.build_payload(
        from_status=from_status, to_status=to_status,
        guardrail_baseline_sha256=guardrail_baseline_sha256,
        activation_proposal_sha256=activation_proposal_sha256,
        observed_account_id=observed_account_id,
        git_commit_sha=git_commit_sha,
        changed_guardrail_paths=changed_guardrail_paths or [],
        nonce=nonce, timestamp_iso=timestamp_iso,
    )
    signature_blob = reauth_signature.sign_for_test(payload, key_path)
    return dict(
        signed_payload=payload, signature_blob=signature_blob, signer_identity=identity,
        nonce=nonce, git_commit_sha=git_commit_sha, changed_guardrail_paths=changed_guardrail_paths,
    )
