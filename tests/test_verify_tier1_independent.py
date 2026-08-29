"""
Tests for scripts/slc_live/verify_tier1_independent.py (Phase 6 Step 2).

Loads the script as a module (importlib, not a package import - the
script itself has zero live_slc imports, and this test file's own
imports of it don't change that) so its module-level constants
(TIER1_FILES, LIVE_SLC_DB_PATH, ALLOWED_SIGNERS_PATH) can be monkeypatched
to point at an isolated tmp_path scenario per test, never the real repo
or real live_slc.db.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "slc_live" / "verify_tier1_independent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_tier1_independent", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_test_key(tmp_path: Path, identity: str = "test-signer"):
    key_path = tmp_path / f"{identity}_key"
    subprocess.run(
        ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", "", "-C", identity, "-q"],
        check=True, capture_output=True, timeout=15,
    )
    allowed_signers_path = tmp_path / "allowed_signers"
    pub = key_path.with_suffix(".pub").read_text(encoding="utf-8")
    allowed_signers_path.write_text(f"{identity} {pub}", encoding="utf-8")
    return key_path, allowed_signers_path, identity


def _sign(mod, payload: str, key_path: Path) -> str:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        payload_path = Path(tmp) / "payload.txt"
        payload_path.write_text(payload, encoding="utf-8", newline="\n")
        subprocess.run(
            ["ssh-keygen", "-Y", "sign", "-f", str(key_path), "-n", mod.SIGNATURE_NAMESPACE, str(payload_path)],
            check=True, capture_output=True, timeout=15,
        )
        return (Path(tmp) / "payload.txt.sig").read_text(encoding="utf-8")


def _tier1_scenario(tmp_path: Path) -> dict[str, Path]:
    """3 fake Tier-1 files, standing in for the real set - the script's
    own TIER1_FILES dict is monkeypatched to point at these."""
    files = {}
    for name in ("a.bat", "b.py", "c.py"):
        path = tmp_path / name
        path.write_text(f"content of {name}", encoding="utf-8")
        files[name] = path
    return files


def _seed_db(db_path: Path, *, rows: list[dict]):
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS slc_activation_events")
    conn.execute(
        "CREATE TABLE slc_activation_events ("
        "id TEXT PRIMARY KEY, occurred_at TEXT, signed_payload TEXT, signature_blob TEXT, "
        "signer_identity TEXT, guardrail_baseline_sha256_at_transition TEXT, "
        "baseline_file_relative_path TEXT)"
    )
    for i, row in enumerate(rows):
        conn.execute(
            "INSERT INTO slc_activation_events "
            "(id, occurred_at, signed_payload, signature_blob, signer_identity, "
            "guardrail_baseline_sha256_at_transition, baseline_file_relative_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(i), row.get("occurred_at", f"2026-08-3{i} 00:00:00"),
                row.get("signed_payload"), row.get("signature_blob"), row.get("signer_identity"),
                row.get("guardrail_baseline_sha256_at_transition"), row.get("baseline_file_relative_path"),
            ),
        )
    conn.commit()
    conn.close()


def _valid_scenario(mod, tmp_path):
    """Builds a fully valid end-to-end scenario: signed key, tier1 files,
    a baseline JSON matching them, a DB row pinning it with a genuine
    signature. Returns the configured module ready to call main()."""
    key_path, allowed_signers_path, identity = _make_test_key(tmp_path)
    tier1_files = _tier1_scenario(tmp_path)
    tier1_hashes = {name: _sha256(path) for name, path in tier1_files.items()}
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({"guardrails": {"tier1": tier1_hashes, "tier2": {}}}), encoding="utf-8")
    baseline_hash = _sha256(baseline_path)

    payload = (
        f"namespace: {mod.SIGNATURE_NAMESPACE}\n"
        f"guardrail_baseline_sha256: {baseline_hash}\n"
        f"baseline_file_relative_path: {baseline_path}\n"
    )
    signature = _sign(mod, payload, key_path)

    db_path = tmp_path / "live_slc.db"
    _seed_db(db_path, rows=[{
        "signed_payload": payload, "signature_blob": signature, "signer_identity": identity,
        "guardrail_baseline_sha256_at_transition": baseline_hash,
        "baseline_file_relative_path": str(baseline_path),
    }])

    mod.TIER1_FILES = tier1_files
    mod.LIVE_SLC_DB_PATH = db_path
    mod.ALLOWED_SIGNERS_PATH = allowed_signers_path
    return {"tier1_files": tier1_files, "baseline_path": baseline_path, "db_path": db_path}


def test_passes_against_a_fully_valid_signed_scenario(mod, tmp_path):
    _valid_scenario(mod, tmp_path)
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 0


def test_fails_when_db_missing(mod, tmp_path):
    mod.LIVE_SLC_DB_PATH = tmp_path / "does_not_exist.db"
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_fails_when_no_signed_event_exists(mod, tmp_path):
    scenario = _valid_scenario(mod, tmp_path)
    _seed_db(scenario["db_path"], rows=[])  # wipe the valid row
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_fails_on_tier1_drift(mod, tmp_path):
    scenario = _valid_scenario(mod, tmp_path)
    # Mutate a Tier-1 file after the baseline was signed.
    list(scenario["tier1_files"].values())[0].write_text("tampered content", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_fails_on_tampered_signature(mod, tmp_path):
    scenario = _valid_scenario(mod, tmp_path)
    conn = sqlite3.connect(scenario["db_path"])
    conn.execute("UPDATE slc_activation_events SET signature_blob = 'garbage not a real sshsig block'")
    conn.commit()
    conn.close()
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_fails_on_missing_tier1_file(mod, tmp_path):
    scenario = _valid_scenario(mod, tmp_path)
    missing_name = next(iter(scenario["tier1_files"]))
    scenario["tier1_files"][missing_name].unlink()
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_fails_on_baseline_file_tampering(mod, tmp_path):
    scenario = _valid_scenario(mod, tmp_path)
    scenario["baseline_path"].write_text('{"guardrails": {"tier1": {}, "tier2": {}}}', encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 1


def test_has_zero_live_slc_imports():
    """The whole point of an independent verifier - proven directly
    against the source text's actual import statements, not just by
    absence of an ImportError (which could pass even with an unused
    import) and not a blanket substring check (the docstring itself
    mentions "live_slc" in prose, explaining what it deliberately
    doesn't import)."""
    import re
    text = _SCRIPT_PATH.read_text(encoding="utf-8")
    import_lines = re.findall(r"^\s*(?:from|import)\s+\S+", text, re.MULTILINE)
    assert import_lines  # sanity: the file does import *something*
    assert not any("live_slc" in line for line in import_lines)


def test_runs_standalone_as_a_real_subprocess(tmp_path):
    """One genuine end-to-end proof this actually runs as the scheduler
    would invoke it - a real `python <script>` subprocess, not just an
    in-process function call."""
    result = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    # Against the real repo state (no signed event yet), this must fail closed.
    assert result.returncode == 1
    assert "FAIL" in result.stderr
