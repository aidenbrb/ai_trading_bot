"""
Independent Tier-1 guardrail verifier for live_slc (Phase 6 Step 2).

Deliberately has ZERO imports from the live_slc package - not
live_slc.guardrails, not live_slc.reauth_signature, not live_slc.models.
The entire point is a verification path that doesn't trust the same code
it's verifying to also tell it what "correct" looks like, or to correctly
report its own tampering. Everything here is independently
re-implemented: file hashing via hashlib directly, the Tier-1 file list
hardcoded in this file (not read from guardrails.py's GUARDRAILS_TIER1
dict - kept in sync by hand; that manual-sync cost is the deliberate
tradeoff for not trusting the file under test to define its own test),
the DB read via raw sqlite3 in read-only mode (not live_slc.models'
SQLModel session), and signature verification via its own subprocess call
to `ssh-keygen -Y verify` (not live_slc.reauth_signature.verify_signature,
even though that would do the identical thing - the redundancy IS the
protection).

Run by the scheduler (scripts/slc_live/run_slc_preflight.bat,
run_slc_cycle.bat, run_slc_closeout.bat) BEFORE launching the live
process. Fails closed: any error - missing file, hash mismatch, DB
unreadable, invalid/missing signature - is a nonzero exit, and the .bat
wrappers must not proceed to launch the live process past a nonzero exit
here.

Usage: python scripts/slc_live/verify_tier1_independent.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRADINGBOT_ROOT = REPO_ROOT.parent
LIVE_SLC_DB_PATH = REPO_ROOT / "live_slc" / "live_slc.db"
ALLOWED_SIGNERS_PATH = REPO_ROOT / "live_slc" / "allowed_signers"
SIGNATURE_NAMESPACE = "slc-reauth"

# Independently enumerated - NOT imported from live_slc/guardrails.py's
# GUARDRAILS_TIER1. Must be kept in sync by hand whenever that dict
# changes; see the module docstring for why that's the point, not an
# oversight.
TIER1_FILES: dict[str, Path] = {
    "../run_bot.bat": TRADINGBOT_ROOT / "run_bot.bat",
    "run_pipeline.py": REPO_ROOT / "run_pipeline.py",
    "utils/strategy_registry.py": REPO_ROOT / "utils" / "strategy_registry.py",
    "nodes/execution_node.py": REPO_ROOT / "nodes" / "execution_node.py",
    "live_slc/guardrails.py": REPO_ROOT / "live_slc" / "guardrails.py",
    **{
        f"scripts/{path.name}": path
        for path in sorted((REPO_ROOT / "scripts").glob("*.bat"))
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(reason: str) -> None:
    print(f"FAIL: {reason}", file=sys.stderr)
    sys.exit(1)


def _verify_signature(payload: str, signature_blob: str, signer_identity: str) -> bool:
    if not ALLOWED_SIGNERS_PATH.is_file():
        return False
    with tempfile.TemporaryDirectory() as tmp:
        sig_path = Path(tmp) / "payload.txt.sig"
        sig_path.write_text(signature_blob, encoding="utf-8")
        try:
            result = subprocess.run(
                [
                    "ssh-keygen", "-Y", "verify",
                    "-f", str(ALLOWED_SIGNERS_PATH),
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


def _most_recent_candidate_events(conn: sqlite3.Connection) -> list[tuple]:
    return conn.execute(
        "SELECT id, signed_payload, signature_blob, signer_identity, "
        "guardrail_baseline_sha256_at_transition, baseline_file_relative_path "
        "FROM slc_activation_events "
        "WHERE signed_payload IS NOT NULL AND signature_blob IS NOT NULL "
        "AND signer_identity IS NOT NULL AND baseline_file_relative_path IS NOT NULL "
        "AND guardrail_baseline_sha256_at_transition IS NOT NULL "
        "ORDER BY occurred_at DESC"
    ).fetchall()


def main() -> None:
    if not LIVE_SLC_DB_PATH.is_file():
        _fail(f"live_slc.db not found at {LIVE_SLC_DB_PATH}")

    try:
        conn = sqlite3.connect(f"file:{LIVE_SLC_DB_PATH.resolve().as_posix()}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        _fail(f"could not open live_slc.db read-only: {exc}")
        return

    try:
        rows = _most_recent_candidate_events(conn)
    except sqlite3.Error as exc:
        _fail(f"could not query slc_activation_events: {exc}")
        return
    finally:
        conn.close()

    active = None
    for row in rows:
        _id, signed_payload, signature_blob, signer_identity, baseline_hash, baseline_rel_path = row
        if _verify_signature(signed_payload, signature_blob, signer_identity):
            active = (baseline_hash, baseline_rel_path)
            break
    if active is None:
        _fail("no signature-verified activation event found - cannot resolve an active baseline")
        return
    baseline_hash, baseline_rel_path = active

    baseline_path = Path(baseline_rel_path)
    if not baseline_path.is_absolute():
        baseline_path = REPO_ROOT / baseline_rel_path
    if not baseline_path.is_file():
        _fail(f"baseline file {baseline_path} referenced by the signed event does not exist")
        return
    actual_baseline_hash = _sha256(baseline_path)
    if actual_baseline_hash != baseline_hash:
        _fail(f"baseline file hash mismatch: file={actual_baseline_hash} signed={baseline_hash}")
        return

    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"could not read/parse baseline file: {exc}")
        return
    expected_tier1 = baseline.get("guardrails", {}).get("tier1", {})

    missing = [name for name, path in TIER1_FILES.items() if not path.is_file()]
    if missing:
        _fail(f"Tier-1 file(s) missing: {missing}")
        return
    current_tier1 = {name: _sha256(path) for name, path in TIER1_FILES.items()}

    if set(current_tier1) != set(expected_tier1):
        _fail(
            "Tier-1 file SET differs from the signed baseline: "
            f"current={sorted(current_tier1)} expected={sorted(expected_tier1)}"
        )
        return
    changed = [name for name in current_tier1 if current_tier1[name] != expected_tier1.get(name)]
    if changed:
        _fail(f"Tier-1 drift detected: {changed}")
        return

    print(f"PASS - Tier-1 set ({len(current_tier1)} files) matches the signed baseline {baseline_hash}")
    sys.exit(0)


if __name__ == "__main__":
    main()
