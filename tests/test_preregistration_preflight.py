"""Tests for backtest/preregistration_preflight.py.

All frozen-scope paths (manifest, document, cache, source files,
deployment-guardrail files, scheduler files) are monkeypatched to point
at small, self-consistent temporary stand-ins - never the real repo's
frozen files - so these tests are fast and cannot themselves corrupt or
depend on the real frozen state.
"""
import json
from datetime import datetime

import pytest

import backtest.preregistration_preflight as preflight
from backtest.preregistration_preflight import (
    DeploymentGuardrailMismatch,
    FrozenHashMismatch,
    sha256_file,
    verify_deployment_guardrail,
    verify_preregistration_scope,
    write_stage1_baseline_check,
)


@pytest.fixture
def frozen_repo(tmp_path, monkeypatch):
    research_dir = tmp_path / "research"
    research_dir.mkdir()

    document_path = research_dir / "doc.md"
    document_path.write_text("frozen document content", encoding="utf-8")

    cache_path = tmp_path / "bars_cache.db"
    cache_path.write_bytes(b"fake cache content")

    source_paths = {}
    for name in preflight.FROZEN_SOURCE_FILES:
        p = tmp_path / (name.replace("/", "_") + ".src")
        p.write_text(f"source content for {name}", encoding="utf-8")
        source_paths[name] = p

    manifest = {
        "document_sha256": sha256_file(document_path),
        "cache_db_sha256": sha256_file(cache_path),
        "source_file_sha256": {name: sha256_file(p) for name, p in source_paths.items()},
    }
    manifest_path = research_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    deployment_paths = {}
    for name in preflight.DEPLOYMENT_GUARDRAIL_FILES:
        p = tmp_path / (name.replace("/", "_") + ".dep")
        p.write_text(f"deployment content for {name}", encoding="utf-8")
        deployment_paths[name] = p

    scheduler_paths = {}
    for name in preflight.SCHEDULER_FILES:
        p = tmp_path / (name.replace("/", "_") + ".bat")
        p.write_text(f"scheduler content for {name}", encoding="utf-8")
        scheduler_paths[name] = p

    monkeypatch.setattr(preflight, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(preflight, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(preflight, "DOCUMENT_PATH", document_path)
    monkeypatch.setattr(preflight, "CACHE_DB_PATH", cache_path)
    monkeypatch.setattr(preflight, "FROZEN_SOURCE_FILES", source_paths)
    monkeypatch.setattr(preflight, "DEPLOYMENT_GUARDRAIL_FILES", deployment_paths)
    monkeypatch.setattr(preflight, "SCHEDULER_FILES", scheduler_paths)
    return {
        "document_path": document_path, "cache_path": cache_path, "source_paths": source_paths,
        "deployment_paths": deployment_paths, "scheduler_paths": scheduler_paths,
    }


def test_write_stage1_baseline_check_succeeds_when_consistent(frozen_repo, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    path = write_stage1_baseline_check(output_dir=out_dir)
    record = json.loads(path.read_text(encoding="utf-8"))
    scope = record["preregistration_scope"]
    assert scope["document_sha256_fresh"] == scope["document_sha256_frozen"]
    assert scope["cache_db_sha256_fresh"] == scope["cache_db_sha256_frozen"]
    assert scope["source_file_sha256_fresh"] == scope["source_file_sha256_frozen"]
    assert set(record["deployment_guardrail"]["strategy_registry_execution_pipeline_sha256"]) == \
        set(preflight.DEPLOYMENT_GUARDRAIL_FILES)
    assert set(record["deployment_guardrail"]["scheduler_files_sha256"]) == set(preflight.SCHEDULER_FILES)


def test_write_stage1_baseline_check_raises_on_document_mismatch_and_writes_nothing(frozen_repo, tmp_path):
    frozen_repo["document_path"].write_text("tampered content", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(FrozenHashMismatch):
        write_stage1_baseline_check(output_dir=out_dir)
    assert list(out_dir.iterdir()) == []


def test_write_stage1_baseline_check_raises_on_source_file_mismatch(frozen_repo, tmp_path):
    any_source = next(iter(frozen_repo["source_paths"].values()))
    any_source.write_text("tampered", encoding="utf-8")
    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    with pytest.raises(FrozenHashMismatch):
        write_stage1_baseline_check(output_dir=out_dir)


def test_write_stage1_baseline_check_raises_on_cache_mismatch(frozen_repo, tmp_path):
    frozen_repo["cache_path"].write_bytes(b"mutated cache bytes")
    out_dir = tmp_path / "out3"
    out_dir.mkdir()
    with pytest.raises(FrozenHashMismatch):
        write_stage1_baseline_check(output_dir=out_dir)


class _FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 8, 12, 3, 0, 0, 123456, tzinfo=tz)


def test_write_stage1_baseline_check_is_exclusive_create(frozen_repo, tmp_path, monkeypatch):
    """A second attempt at the exact same (timestamped) path must raise,
    not silently overwrite the first record - there is no git history in
    this repo to reconstruct it otherwise."""
    monkeypatch.setattr(preflight, "datetime", _FixedDatetime)
    out_dir = tmp_path / "out4"
    out_dir.mkdir()
    first = write_stage1_baseline_check(output_dir=out_dir)
    assert first.exists()
    with pytest.raises(FileExistsError):
        write_stage1_baseline_check(output_dir=out_dir)


def test_verify_preregistration_scope_hard_requires_untouched_files(frozen_repo):
    untouched_name = next(iter(preflight.UNTOUCHED_SOURCE_FILES))
    frozen_repo["source_paths"][untouched_name].write_text("tampered", encoding="utf-8")
    with pytest.raises(FrozenHashMismatch):
        verify_preregistration_scope()


def test_verify_preregistration_scope_records_but_does_not_reject_modified_files(frozen_repo):
    modified_name = next(iter(preflight.INTENTIONALLY_MODIFIED_SOURCE_FILES))
    frozen_repo["source_paths"][modified_name].write_text("intentionally different now", encoding="utf-8")
    report = verify_preregistration_scope()
    entry = report["intentionally_modified_source_files"][modified_name]
    assert entry["changed"] is True
    assert entry["current"] != entry["frozen"]


def test_verify_preregistration_scope_passes_when_fully_consistent(frozen_repo):
    report = verify_preregistration_scope()
    for entry in report["intentionally_modified_source_files"].values():
        assert entry["changed"] is False


def test_verify_preregistration_scope_accepts_only_exact_documented_recovery(
    frozen_repo, tmp_path, monkeypatch
):
    original = sha256_file(frozen_repo["cache_path"])
    frozen_repo["cache_path"].write_bytes(b"logically restored sqlite container")
    recovered = sha256_file(frozen_repo["cache_path"])
    amendment = tmp_path / "recovery.md"
    amendment.write_text("audited recovery", encoding="utf-8")
    monkeypatch.setattr(preflight, "ORIGINAL_FROZEN_CACHE_SHA256", original)
    monkeypatch.setattr(preflight, "RECOVERED_CACHE_SHA256", recovered)
    monkeypatch.setattr(preflight, "RECOVERY_AMENDMENT_PATH", amendment)
    monkeypatch.setattr(preflight, "RECOVERY_AMENDMENT_SHA256", sha256_file(amendment))
    report = verify_preregistration_scope()
    assert report["cache_container_recovery"]["accepted"] is True


def test_recovery_path_still_rejects_any_unrecorded_cache_drift(
    frozen_repo, tmp_path, monkeypatch
):
    amendment = tmp_path / "recovery.md"
    amendment.write_text("audited recovery", encoding="utf-8")
    monkeypatch.setattr(preflight, "RECOVERY_AMENDMENT_PATH", amendment)
    monkeypatch.setattr(preflight, "RECOVERY_AMENDMENT_SHA256", sha256_file(amendment))
    frozen_repo["cache_path"].write_bytes(b"unknown mutation")
    with pytest.raises(FrozenHashMismatch):
        verify_preregistration_scope()


def _mkout(tmp_path, name):
    out_dir = tmp_path / name
    out_dir.mkdir()
    return out_dir


def test_verify_deployment_guardrail_passes_when_unchanged(frozen_repo, tmp_path):
    baseline_path = write_stage1_baseline_check(output_dir=_mkout(tmp_path, "out5"))
    result = verify_deployment_guardrail(baseline_path)
    assert result["deployment_files_sha256"]


def test_verify_deployment_guardrail_hard_stops_on_deployment_file_drift(frozen_repo, tmp_path):
    baseline_path = write_stage1_baseline_check(output_dir=_mkout(tmp_path, "out6"))
    any_deployment_file = next(iter(frozen_repo["deployment_paths"].values()))
    any_deployment_file.write_text("someone touched the registry", encoding="utf-8")
    with pytest.raises(DeploymentGuardrailMismatch):
        verify_deployment_guardrail(baseline_path)


def test_verify_deployment_guardrail_hard_stops_on_scheduler_file_drift(frozen_repo, tmp_path):
    baseline_path = write_stage1_baseline_check(output_dir=_mkout(tmp_path, "out7"))
    any_scheduler_file = next(iter(frozen_repo["scheduler_paths"].values()))
    any_scheduler_file.write_text("someone touched a scheduled task", encoding="utf-8")
    with pytest.raises(DeploymentGuardrailMismatch):
        verify_deployment_guardrail(baseline_path)
