"""Hash-verify the frozen SLC rule documents live_slc depends on.

Duplicated from backtest/run_slc_backtest.py's verify_rule_freeze()
pattern, not imported from it - backtest/run_slc_backtest.py stays fully
untouched. Extends coverage to amendments 003-005.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PREREGISTRATION = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_preregistration.md"
AMENDMENT_001 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_001.md"
AMENDMENT_002 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_002.md"
AMENDMENT_003 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_003.md"
AMENDMENT_004 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_004.md"
AMENDMENT_005 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_005.md"

EXPECTED_HASHES = {
    "preregistration": "d453ae039c3d9145e986ff6cf27ce98af7418b7b5ea6da0d2ddb728e801c3e9d",
    "amendment_001": "3f46175fe2c801a2fed552e242dc97a963ec885ec9c544fd49bdddbd51d0bf03",
    "amendment_002": "f3774cf00c2fb6f7afa5d6399bb1a740303ee3a2151acbf7766bd13be652e2a2",
    "amendment_003": "36e666ec3c41c71aafb2d73df4e747c8b053aec4a399270a18eda013ae272cda",
    "amendment_004": "5981461c7bfbe89586098cfdae5eb35e96472221d6338e289f3d63f47230f76a",
    "amendment_005": "0ecf9ce77fecbe7f55a3478c192c5662fbd309fd86ac4863620b53e611885753",
}

_DOCUMENTS = {
    "preregistration": PREREGISTRATION,
    "amendment_001": AMENDMENT_001,
    "amendment_002": AMENDMENT_002,
    "amendment_003": AMENDMENT_003,
    "amendment_004": AMENDMENT_004,
    "amendment_005": AMENDMENT_005,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_rule_freeze() -> dict[str, str]:
    """Hard-stop if the preregistration or any of its 4 amendments drifted."""
    actual = {name: _sha256(path) for name, path in _DOCUMENTS.items()}
    if actual != EXPECTED_HASHES:
        raise RuntimeError(
            f"SLC frozen-rule hash mismatch: expected={EXPECTED_HASHES}, actual={actual}"
        )
    return actual
