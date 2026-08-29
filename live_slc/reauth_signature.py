"""
Phase 6 Step 1: human-only re-authorization signatures.

Any Tier-1/Tier-2 guardrail re-baseline, and any transition to paper_active
or live, requires a valid signature over a canonical payload from a key
listed in live_slc/allowed_signers - verified via OpenSSH's own SSHSIG
format (`ssh-keygen -Y sign` / `-Y verify`), the same mechanism `git commit
-S`/`git tag -s` use for SSH-signed commits. Production keys are
ed25519-sk: hardware-backed (FIDO2, touch-required), generated on the
human's own device. The private key material never exists on this
machine, in this repo, or in any path an agent's tools can read - only the
resulting SSHSIG armored signature text is ever transcribed here, by hand,
by the human.

This module can VERIFY signatures (anyone can - that's the point of public-
key crypto) but never CREATES them in any code path reachable from
production. `sign_for_test()` below exists solely for this project's own
test suite, which generates a disposable software ed25519 key (not
ed25519-sk) to exercise the exact same ssh-keygen -Y sign/-Y verify
round-trip without needing real hardware - it is never imported by
anything under live_slc/ except its own tests.
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

SIGNATURE_NAMESPACE = "slc-reauth"
NONCE_VALIDITY_SECONDS = 24 * 60 * 60
ALLOWED_SIGNERS_PATH = Path(__file__).parent / "allowed_signers"


class SignatureVerificationError(RuntimeError):
    pass


def build_payload(
    *,
    from_status: str,
    to_status: str,
    guardrail_baseline_sha256: str,
    activation_proposal_sha256: str | None,
    observed_account_id: str,
    git_commit_sha: str,
    changed_guardrail_paths: list[str],
    baseline_file_relative_path: str,
    nonce: str,
    timestamp_iso: str,
) -> str:
    """Deterministic, human-readable, canonical - every field that matters
    to the transition is in the signed text itself, so a human reviewing
    what they're about to sign (or `git show <sha>`) never has to trust an
    out-of-band claim about what a hash means. changed_guardrail_paths is
    sorted here (not by the caller) so payload construction is itself
    deterministic regardless of caller iteration order. baseline_file_
    relative_path is the repo-relative path of the dated baseline JSON
    this signature pins - explicit in the signed text so a human signing
    can see exactly which file their signature will make authoritative."""
    lines = [
        f"namespace: {SIGNATURE_NAMESPACE}",
        f"from_status: {from_status}",
        f"to_status: {to_status}",
        f"guardrail_baseline_sha256: {guardrail_baseline_sha256}",
        f"activation_proposal_sha256: {activation_proposal_sha256 or ''}",
        f"observed_account_id: {observed_account_id}",
        f"git_commit_sha: {git_commit_sha}",
        f"changed_guardrail_paths: {','.join(sorted(changed_guardrail_paths))}",
        f"baseline_file_relative_path: {baseline_file_relative_path}",
        f"nonce: {nonce}",
        f"timestamp: {timestamp_iso}",
    ]
    return "\n".join(lines) + "\n"


def verify_signature(
    payload: str,
    signature_blob: str,
    *,
    signer_identity: str,
    allowed_signers_path: Path = ALLOWED_SIGNERS_PATH,
) -> bool:
    """Runs `ssh-keygen -Y verify` against the given allowed_signers file.
    Returns True only on a clean, unambiguous match for signer_identity;
    any nonzero exit, any stderr indicating a mismatch, or a missing
    allowed_signers file returns False - never raises for an ordinary
    verification failure (missing/invalid/tampered signature are all
    legitimate, expected outcomes this function must report as False, not
    crash on)."""
    if not allowed_signers_path.is_file():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.txt"
        sig_path = Path(tmp) / "payload.txt.sig"
        payload_path.write_text(payload, encoding="utf-8", newline="\n")
        sig_path.write_text(signature_blob, encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    "ssh-keygen", "-Y", "verify",
                    "-f", str(allowed_signers_path),
                    "-I", signer_identity,
                    "-n", SIGNATURE_NAMESPACE,
                    "-s", str(sig_path),
                ],
                input=payload.encode("utf-8"),
                capture_output=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0


def parse_payload(payload: str) -> dict[str, str]:
    """Inverse of build_payload() - splits the exact signed text back into
    its named fields so record_transition() can confirm what was ACTUALLY
    signed matches what's about to be recorded (the "wrong-baseline"/
    "wrong-commit" checks), rather than trusting the caller's separate
    arguments on faith."""
    fields: dict[str, str] = {}
    for line in payload.strip("\n").split("\n"):
        if ": " not in line:
            continue
        key, _, value = line.partition(": ")
        fields[key] = value
    return fields


def sign_for_test(payload: str, private_key_path: Path) -> str:
    """TEST-ONLY signer: shells out to `ssh-keygen -Y sign` with a
    disposable software ed25519 test key. Never called from any
    production code path - production signatures are always produced by a
    human, on their own hardware, outside this repository entirely."""
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.txt"
        payload_path.write_text(payload, encoding="utf-8", newline="\n")
        subprocess.run(
            [
                "ssh-keygen", "-Y", "sign",
                "-f", str(private_key_path),
                "-n", SIGNATURE_NAMESPACE,
                str(payload_path),
            ],
            check=True, capture_output=True, timeout=15,
        )
        return (Path(tmp) / "payload.txt.sig").read_text(encoding="utf-8")
