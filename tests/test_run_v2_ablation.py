"""Tests for backtest/run_v2_ablation.py - qualification gating order,
output-artifact completeness, and import safety.

Does not run the full 48-slice pipeline (that needs real cached market
data and is exercised separately via `python -m backtest.run_v2_ablation
--smoke`) - these tests exercise decide_qualification() and
_write_results() directly with synthetic inputs, which is where the
gating/audit logic actually lives.
"""
import ast
import csv
import json
import subprocess
import sys
from unittest.mock import MagicMock

import pytest

import backtest.preregistration_preflight as preflight
import backtest.run_v2_ablation as run_v2_ablation
from backtest.run_v2_ablation import decide_qualification

REPO_ROOT = preflight.REPO_ROOT


def test_qualify_fn_never_called_when_cache_changed():
    qualify_fn = MagicMock()
    result = decide_qualification(
        cache_unchanged=False, determinism_passed=True,
        baseline_summary={}, stressed_summary={}, coverage_rate=0.99,
        pre_run_cache_sha256="a", post_run_cache_sha256="b", qualify_fn=qualify_fn,
    )
    qualify_fn.assert_not_called()
    assert result["qualification_status"] == "invalid"
    assert "verdict" not in result and "passed" not in result
    assert "cache" in result["reason"]


def test_qualify_fn_never_called_when_determinism_failed():
    qualify_fn = MagicMock()
    result = decide_qualification(
        cache_unchanged=True, determinism_passed=False,
        baseline_summary={}, stressed_summary={}, coverage_rate=0.99,
        pre_run_cache_sha256="a", post_run_cache_sha256="a", qualify_fn=qualify_fn,
    )
    qualify_fn.assert_not_called()
    assert result["qualification_status"] == "invalid"
    assert "determinism" in result["reason"]


def test_qualify_fn_never_called_when_both_fail():
    qualify_fn = MagicMock()
    result = decide_qualification(
        cache_unchanged=False, determinism_passed=False,
        baseline_summary={}, stressed_summary={}, coverage_rate=0.99,
        pre_run_cache_sha256="a", post_run_cache_sha256="b", qualify_fn=qualify_fn,
    )
    qualify_fn.assert_not_called()
    assert "cache" in result["reason"] and "determinism" in result["reason"]


def test_qualify_fn_called_with_the_full_v2_variant_summaries_when_clean():
    baseline_summary = {"tag": "full_v2_current_1pct_baseline"}
    stressed_summary = {"tag": "full_v2_current_1pct_stressed"}
    qualify_fn = MagicMock(return_value={"passed": True, "checks": {}, "failed_checks": []})

    result = decide_qualification(
        cache_unchanged=True, determinism_passed=True,
        baseline_summary=baseline_summary, stressed_summary=stressed_summary,
        coverage_rate=0.99, pre_run_cache_sha256="a", post_run_cache_sha256="a",
        qualify_fn=qualify_fn,
    )
    qualify_fn.assert_called_once_with(baseline_summary, stressed_summary, 0.99)
    assert result["qualification_status"] == "evaluated"
    assert result["passed"] is True


def _sample_write_results_kwargs(tmp_path):
    return dict(
        config={"label": "test"},
        preflight_report={
            "preregistration_scope": {"document_sha256": "doc-hash"},
            "cache_integrity": {"pre_run_sha256": "a", "post_run_sha256": "a", "unchanged": True},
        },
        data_coverage={"stock": {"coverage_rate": 0.99}},
        exclusions=[{"date": "2026-06-01", "symbol": "AAPL", "market": "stock", "reason": "warmup"}],
        all_trades=[{"variant_id": "d1_on_d2_on_d3_on", "symbol": "AAPL", "portfolio": "current_1pct"}],
        all_rejected=[{"variant_id": "d1_off_d2_off_d3_off", "symbol": "MSFT", "reason": "max_positions"}],
        all_missing=[{
            "variant_id": "d1_on_d2_on_d3_on", "delta1_expiration_sessions": 5,
            "delta2_enabled": True, "delta3_enabled": True,
            "symbol": "AAPL", "reason": "no relevant minute bars returned",
        }],
        all_equity=[{"variant_id": "d1_on_d2_on_d3_on", "date": "2026-06-01", "equity": 100_000.0}],
        cache_misses=[{"variant_id": "shared", "symbol": "AAPL", "interval": "research-stock-sip-1Hour"}],
        summaries={},
        temporal_diagnostics=[{"year": 2026, "closed_count": 3, "win_rate": 0.5}],
        qualification={"qualification_status": "invalid", "reason": "test"},
        determinism={"passed": True, "first_sha256": "x", "second_sha256": "x"},
        implementation_hashes={"stage1_baseline_check_path": "somewhere"},
    )


def test_write_results_manifest_written_last_and_excludes_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v2_ablation, "RESULTS_DIR", tmp_path)
    run_dir = run_v2_ablation._write_results(**_sample_write_results_kwargs(tmp_path))

    manifest = json.loads((run_dir / "results_manifest.json").read_text(encoding="utf-8"))
    assert "results_manifest.json" not in manifest["files_sha256"]
    for name, expected_hash in manifest["files_sha256"].items():
        assert preflight.sha256_file(run_dir / name) == expected_hash
    assert manifest["cache_db_sha256_pre_run"] == "a"
    assert manifest["cache_db_sha256_post_run"] == "a"
    assert manifest["preregistration_document_sha256"] == "doc-hash"


def test_write_results_missing_outcome_data_csv_carries_variant_and_delta_columns(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v2_ablation, "RESULTS_DIR", tmp_path)
    run_dir = run_v2_ablation._write_results(**_sample_write_results_kwargs(tmp_path))

    rows = list(csv.DictReader((run_dir / "missing_outcome_data.csv").open(encoding="utf-8")))
    assert rows[0]["variant_id"] == "d1_on_d2_on_d3_on"
    assert rows[0]["delta1_expiration_sessions"] == "5"
    assert rows[0]["delta2_enabled"] == "True"
    assert rows[0]["delta3_enabled"] == "True"


def test_write_results_cache_misses_csv_has_variant_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(run_v2_ablation, "RESULTS_DIR", tmp_path)
    run_dir = run_v2_ablation._write_results(**_sample_write_results_kwargs(tmp_path))

    rows = list(csv.DictReader((run_dir / "cache_misses.csv").open(encoding="utf-8")))
    assert rows[0]["variant_id"] == "shared"


def test_build_variants_produces_the_full_2x2x2_factorial():
    variants = run_v2_ablation.build_variants()
    assert len(variants) == 8
    assert len({v.variant_id for v in variants}) == 8
    full_v2 = [v for v in variants if v.variant_id == run_v2_ablation.FULL_V2_VARIANT_ID]
    assert len(full_v2) == 1
    assert full_v2[0].expiration_sessions == 5
    assert full_v2[0].delta2_enabled is True
    assert full_v2[0].delta3_enabled is True
    baseline = [v for v in variants if v.expiration_sessions is None
                and not v.delta2_enabled and not v.delta3_enabled]
    assert len(baseline) == 1


def test_subprocess_import_never_touches_forbidden_modules():
    script = (
        "import backtest.run_v2_ablation, sys\n"
        "assert 'alpaca.trading' not in sys.modules\n"
        "assert 'nodes.execution_node' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr


FORBIDDEN_IMPORT_PREFIXES = ("alpaca.trading", "nodes.execution_node")
AST_SCAN_FILES = [
    REPO_ROOT / "backtest" / "run_v2_ablation.py",
    REPO_ROOT / "backtest" / "preregistration_preflight.py",
    REPO_ROOT / "backtest" / "v2_readonly_adapters.py",
    REPO_ROOT / "utils" / "strategy_signals.py",
    REPO_ROOT / "utils" / "indicators.py",
    REPO_ROOT / "backtest" / "whole_bot_engine.py",
]


def _imported_module_names(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


def test_ast_scan_finds_no_forbidden_import_in_any_task_file():
    """Static, not import-time - catches a lazy import hidden inside a
    function body (e.g. main()) that a subprocess import test, which
    only imports the module, would never execute and so would miss."""
    for path in AST_SCAN_FILES:
        for name in _imported_module_names(path):
            for forbidden in FORBIDDEN_IMPORT_PREFIXES:
                assert not name.startswith(forbidden), f"{path}: forbidden import {name}"


def test_ast_scan_actually_detects_a_lazy_forbidden_import(tmp_path):
    """Proves the scan mechanism itself is sound - not a no-op that would
    trivially pass regardless of what it's pointed at."""
    poisoned = tmp_path / "poisoned.py"
    poisoned.write_text(
        "def main():\n"
        "    from alpaca.trading.client import TradingClient\n"
        "    return TradingClient\n",
        encoding="utf-8",
    )
    names = _imported_module_names(poisoned)
    assert any(name.startswith("alpaca.trading") for name in names)
