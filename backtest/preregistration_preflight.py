"""Hash-based provenance checks for the stock_trend_momentum_v2 preregistration.

Stage 1 runs once, before any implementation edit, and persists its result -
this repo has no git history to reconstruct that starting state from later.
Stage 2 runs at harness runtime, in two parts: the preregistration's own
frozen-source scope (hard-requires the document, the cache, and the 3
untouched files to match; records but never rejects on the 3 files this
task intentionally modifies), and the deployment-scope guardrail files
(strategy_registry.py, run_pipeline.py, execution_node.py, the scheduler
launcher scripts) - none of those should ever change as a result of this
task, so any drift from the Stage-1 baseline is an unconditional hard-stop.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRADINGBOT_ROOT = REPO_ROOT.parent

MANIFEST_PATH = REPO_ROOT / "research" / "stock_trend_momentum_v2_preregistration.manifest.json"
DOCUMENT_PATH = REPO_ROOT / "research" / "stock_trend_momentum_v2_preregistration.md"
CACHE_DB_PATH = REPO_ROOT / "backtest" / "cache" / "bars_cache.db"
RECOVERY_AMENDMENT_PATH = (
    REPO_ROOT / "research" / "stock_trend_momentum_v2_cache_container_recovery_20260812.md"
)
ORIGINAL_FROZEN_CACHE_SHA256 = "2aa39029331c7dea6c620c8a23a43b70764fe40ad6761eda901da6775d83d841"
RECOVERED_CACHE_SHA256 = "abfeee4cae5c4edda4b8c3323d33174d484d2fe7e22d1e97f43ab87776f40ef1"
RECOVERY_AMENDMENT_SHA256 = "b331539f2ee46a8c9bfc9d9c2e0d8b265cbb7a2c83fd5bb99506ab69054d7b4b"

# Matches research/stock_trend_momentum_v2_preregistration.manifest.json's
# source_file_sha256 keys exactly.
FROZEN_SOURCE_FILES: dict[str, Path] = {
    "utils/strategy_signals.py": REPO_ROOT / "utils" / "strategy_signals.py",
    "utils/indicators.py": REPO_ROOT / "utils" / "indicators.py",
    "backtest/whole_bot_engine.py": REPO_ROOT / "backtest" / "whole_bot_engine.py",
    "backtest/whole_bot_metrics.py": REPO_ROOT / "backtest" / "whole_bot_metrics.py",
    "config/universe.py": REPO_ROOT / "config" / "universe.py",
    "utils/market_calendar.py": REPO_ROOT / "utils" / "market_calendar.py",
}
INTENTIONALLY_MODIFIED_SOURCE_FILES = frozenset({
    "utils/strategy_signals.py",
    "utils/indicators.py",
    "backtest/whole_bot_engine.py",
})
UNTOUCHED_SOURCE_FILES = frozenset(FROZEN_SOURCE_FILES) - INTENTIONALLY_MODIFIED_SOURCE_FILES

# Deployment-scope guardrail: this task must never change any of these,
# regardless of the ablation run's outcome. Not part of the preregistration
# manifest's own scope - captured fresh at Stage 1 since there is no prior
# frozen value, then re-checked at Stage 2 runtime and at final verification.
DEPLOYMENT_GUARDRAIL_FILES: dict[str, Path] = {
    "utils/strategy_registry.py": REPO_ROOT / "utils" / "strategy_registry.py",
    "run_pipeline.py": REPO_ROOT / "run_pipeline.py",
    "nodes/execution_node.py": REPO_ROOT / "nodes" / "execution_node.py",
}
# No Task Scheduler XML/config file exists in this repo (confirmed by
# direct search) - the OS-level registration is external and unhashable.
# These are the launcher scripts it invokes.
SCHEDULER_FILES: dict[str, Path] = {
    "run_bot.bat": TRADINGBOT_ROOT / "run_bot.bat",
    "scripts/run_daily.bat": REPO_ROOT / "scripts" / "run_daily.bat",
    "scripts/run_day_preflight.bat": REPO_ROOT / "scripts" / "run_day_preflight.bat",
    "scripts/run_day_open.bat": REPO_ROOT / "scripts" / "run_day_open.bat",
    "scripts/run_day_shadow.bat": REPO_ROOT / "scripts" / "run_day_shadow.bat",
    "scripts/run_monitor_only.bat": REPO_ROOT / "scripts" / "run_monitor_only.bat",
}


class FrozenHashMismatch(RuntimeError):
    """A preregistration-scope file (document, cache, or an untouched
    frozen source file) no longer matches its frozen manifest hash."""


class DeploymentGuardrailMismatch(RuntimeError):
    """A deployment/scheduler file has drifted from the Stage-1 baseline.
    None of these files are ever supposed to change as a result of this
    task - this is an unconditional hard-stop, not a recorded warning."""


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _verified_cache_recovery(frozen_hash: str, fresh_hash: str) -> dict | None:
    """Accept one documented SQLite-container recovery, never arbitrary drift.

    The logical v2 rows were restored after SLC smoke rows were isolated, but
    SQLite cannot recreate its previous byte layout. Both exact cache hashes
    and the append-only recovery document hash are therefore hard-coded.
    """
    if fresh_hash == frozen_hash:
        return None
    amendment_hash = (
        sha256_file(RECOVERY_AMENDMENT_PATH)
        if RECOVERY_AMENDMENT_PATH.exists() else None
    )
    if not (
        frozen_hash == ORIGINAL_FROZEN_CACHE_SHA256
        and fresh_hash == RECOVERED_CACHE_SHA256
        and amendment_hash == RECOVERY_AMENDMENT_SHA256
    ):
        raise FrozenHashMismatch(
            f"cache DB hash changed: frozen={frozen_hash} current={fresh_hash}"
        )
    return {
        "accepted": True,
        "reason": "documented SQLite container recovery; logical frozen rows restored",
        "amendment_path": str(RECOVERY_AMENDMENT_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "amendment_sha256": amendment_hash,
        "original_frozen_cache_sha256": frozen_hash,
        "recovered_cache_sha256": fresh_hash,
    }


def write_stage1_baseline_check(output_dir: Path | None = None) -> Path:
    """One-time, before-any-edit baseline record.

    Confirms the current repository still matches the frozen
    preregistration manifest (document, cache DB, all 6 source files),
    then independently captures fresh hashes of the deployment/scheduling
    guardrail files this task must never touch. Raises FrozenHashMismatch
    without writing anything if the preregistration scope itself has
    already drifted - there is nothing valid to persist in that case.

    Written to a timestamped, exclusively-created file so a second,
    accidental run cannot silently overwrite the first record.
    """
    output_dir = output_dir or (REPO_ROOT / "research")
    manifest = _load_manifest()

    fresh_document = sha256_file(DOCUMENT_PATH)
    if fresh_document != manifest["document_sha256"]:
        raise FrozenHashMismatch(
            f"preregistration document hash changed: "
            f"frozen={manifest['document_sha256']} current={fresh_document}"
        )

    fresh_cache = sha256_file(CACHE_DB_PATH)
    cache_recovery = _verified_cache_recovery(manifest["cache_db_sha256"], fresh_cache)

    fresh_sources = {name: sha256_file(path) for name, path in FROZEN_SOURCE_FILES.items()}
    for name, frozen_hash in manifest["source_file_sha256"].items():
        if fresh_sources.get(name) != frozen_hash:
            raise FrozenHashMismatch(
                f"{name} hash changed: frozen={frozen_hash} current={fresh_sources.get(name)}"
            )

    deployment_hashes = {name: sha256_file(path) for name, path in DEPLOYMENT_GUARDRAIL_FILES.items()}
    scheduler_hashes = {name: sha256_file(path) for name, path in SCHEDULER_FILES.items()}

    record = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "manifest_file_sha256": sha256_file(MANIFEST_PATH),
        "preregistration_scope": {
            "document_sha256_frozen": manifest["document_sha256"],
            "document_sha256_fresh": fresh_document,
            "cache_db_sha256_frozen": manifest["cache_db_sha256"],
            "cache_db_sha256_fresh": fresh_cache,
            "cache_container_recovery": cache_recovery,
            "source_file_sha256_frozen": manifest["source_file_sha256"],
            "source_file_sha256_fresh": fresh_sources,
        },
        "deployment_guardrail": {
            "strategy_registry_execution_pipeline_sha256": deployment_hashes,
            "scheduler_files_sha256": scheduler_hashes,
        },
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"
    out_path = output_dir / f"stock_trend_momentum_v2_implementation_baseline_check_{timestamp}.json"
    with open(out_path, "x", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return out_path


def verify_preregistration_scope() -> dict:
    """Stage 2, part 1 - the preregistration's own frozen scope.

    Hard-requires the document, the cache, and the 3 untouched frozen
    source files to match. Records (never raises on) the 3 intentionally
    modified files' frozen vs. current hashes.
    """
    manifest = _load_manifest()
    fresh_document = sha256_file(DOCUMENT_PATH)
    fresh_cache = sha256_file(CACHE_DB_PATH)
    fresh_sources = {name: sha256_file(path) for name, path in FROZEN_SOURCE_FILES.items()}

    if fresh_document != manifest["document_sha256"]:
        raise FrozenHashMismatch(
            f"preregistration document hash changed: "
            f"frozen={manifest['document_sha256']} current={fresh_document}"
        )
    cache_recovery = _verified_cache_recovery(manifest["cache_db_sha256"], fresh_cache)
    for name in UNTOUCHED_SOURCE_FILES:
        frozen_hash = manifest["source_file_sha256"][name]
        if fresh_sources[name] != frozen_hash:
            raise FrozenHashMismatch(
                f"{name} hash changed: frozen={frozen_hash} current={fresh_sources[name]}"
            )

    modified_report = {
        name: {
            "frozen": manifest["source_file_sha256"][name],
            "current": fresh_sources[name],
            "changed": fresh_sources[name] != manifest["source_file_sha256"][name],
        }
        for name in INTENTIONALLY_MODIFIED_SOURCE_FILES
    }
    return {
        "document_sha256": fresh_document,
        "cache_db_sha256_preflight": fresh_cache,
        "cache_container_recovery": cache_recovery,
        "untouched_source_files": {name: fresh_sources[name] for name in UNTOUCHED_SOURCE_FILES},
        "intentionally_modified_source_files": modified_report,
    }


def verify_deployment_guardrail(baseline_record_path: Path) -> dict:
    """Stage 2, part 2 - hard-stop if any deployment/scheduler file has
    drifted from the Stage-1 baseline record.
    """
    baseline = json.loads(Path(baseline_record_path).read_text(encoding="utf-8"))
    guardrail = baseline["deployment_guardrail"]

    current_deployment = {name: sha256_file(path) for name, path in DEPLOYMENT_GUARDRAIL_FILES.items()}
    for name, frozen_hash in guardrail["strategy_registry_execution_pipeline_sha256"].items():
        if current_deployment.get(name) != frozen_hash:
            raise DeploymentGuardrailMismatch(
                f"{name} changed since Stage-1 baseline: "
                f"baseline={frozen_hash} current={current_deployment.get(name)}"
            )

    current_scheduler = {name: sha256_file(path) for name, path in SCHEDULER_FILES.items()}
    for name, frozen_hash in guardrail["scheduler_files_sha256"].items():
        if current_scheduler.get(name) != frozen_hash:
            raise DeploymentGuardrailMismatch(
                f"{name} changed since Stage-1 baseline: "
                f"baseline={frozen_hash} current={current_scheduler.get(name)}"
            )

    return {
        "baseline_record_path": str(baseline_record_path),
        "deployment_files_sha256": current_deployment,
        "scheduler_files_sha256": current_scheduler,
    }


if __name__ == "__main__":
    path = write_stage1_baseline_check()
    print(f"Stage-1 baseline check written: {path}")
