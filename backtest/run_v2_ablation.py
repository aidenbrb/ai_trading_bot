"""Ablation + temporal-stability harness for stock_trend_momentum_v2.

Implements exactly what research/stock_trend_momentum_v2_preregistration.md
specifies: the 8-variant (2^3) ablation matrix across Delta1 (finite
entry-order expiration), Delta2 (ADX trend-strength filter), and Delta3
(minimum expected-move-vs-cost filter), each across both portfolios and
all three cost tiers (48 simulation slices); a year-by-year
temporal-stability diagnostic derived from the full-period full-v2 run's
closed trades, grouped by exit year (Section 6 - not independent per-year
simulations, which would reset cash and force-exit positions at an
artificial year boundary); and qualification against
backtest/whole_bot_metrics.py::qualify_strategy for the full
Delta1+Delta2+Delta3 combination ONLY (Section 5) - the other 7
combinations are diagnostic-only and are never checked against the
qualification bar or substituted as "the real v2" after the fact.

Read-only, cache-only throughout (backtest/v2_readonly_adapters.py) - a
coverage gap in bars_cache.db becomes an exclusion or outcome_data_missing,
never a live fetch. A hash preflight (backtest/preregistration_preflight.py)
must pass before any simulation runs, and the cache's hash is rechecked
after every simulation and the determinism check complete; qualify_strategy
is never invoked if that recheck finds the cache changed.

This module imports market-data clients only. It cannot submit orders.
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.preregistration_preflight import (
    CACHE_DB_PATH,
    REPO_ROOT,
    sha256_file,
    verify_deployment_guardrail,
    verify_preregistration_scope,
)
from backtest.v2_readonly_adapters import make_hourly_fetcher, make_minute_fetcher
from backtest.whole_bot_engine import (
    COSTS,
    PORTFOLIOS,
    TECHNICAL_ONLY_DISCLOSURE,
    build_signal_calendar,
    load_research_data,
    simulate_order_outcome,
    simulate_portfolio,
)
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run
from config.universe import UNIVERSE
from utils.strategy_registry import registry_snapshot
from utils.strategy_signals import stock_trend_momentum_v2

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2026, 8, 9)
RESULTS_DIR = Path(__file__).parent / "results" / "whole_bot_v2_ablation"
FULL_V2_VARIANT_ID = "d1_on_d2_on_d3_on"

IMPLEMENTATION_FILES = {
    "backtest/run_v2_ablation.py": REPO_ROOT / "backtest" / "run_v2_ablation.py",
    "backtest/preregistration_preflight.py": REPO_ROOT / "backtest" / "preregistration_preflight.py",
    "backtest/v2_readonly_adapters.py": REPO_ROOT / "backtest" / "v2_readonly_adapters.py",
    "backtest/readonly_bar_cache.py": REPO_ROOT / "backtest" / "readonly_bar_cache.py",
    "backtest/data_cache.py": REPO_ROOT / "backtest" / "data_cache.py",
    "tests/test_indicators.py": REPO_ROOT / "tests" / "test_indicators.py",
    "tests/test_strategy_signals.py": REPO_ROOT / "tests" / "test_strategy_signals.py",
    "tests/test_whole_bot_engine.py": REPO_ROOT / "tests" / "test_whole_bot_engine.py",
    "tests/test_v2_readonly_adapters.py": REPO_ROOT / "tests" / "test_v2_readonly_adapters.py",
    "tests/test_preregistration_preflight.py": REPO_ROOT / "tests" / "test_preregistration_preflight.py",
    "tests/test_run_v2_ablation.py": REPO_ROOT / "tests" / "test_run_v2_ablation.py",
}


@dataclass(frozen=True)
class Variant:
    variant_id: str
    expiration_sessions: int | None
    delta2_enabled: bool
    delta3_enabled: bool


def build_variants() -> list[Variant]:
    """Full 2^3 factorial: baseline-v1-restated, each delta alone, each
    pair, and the full combination (preregistration Section 5)."""
    variants = []
    for expiration_sessions, delta2, delta3 in itertools.product((None, 5), (False, True), (False, True)):
        variant_id = (
            f"d1_{'on' if expiration_sessions else 'off'}_"
            f"d2_{'on' if delta2 else 'off'}_"
            f"d3_{'on' if delta3 else 'off'}"
        )
        variants.append(Variant(variant_id, expiration_sessions, delta2, delta3))
    return variants


def _make_miss_logger(cache_misses: list[dict], variant_id: str):
    def on_miss(miss: dict) -> None:
        cache_misses.append({**miss, "variant_id": variant_id})
    return on_miss


def _stable_hash(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(rendered.encode()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _temporal_diagnostics(trades: list[dict]) -> list[dict]:
    """Year-by-year diagnostic (preregistration Section 6): the full-period
    full-v2 run's closed trades, grouped by exit year post-hoc - not
    independent per-year simulations. Diagnostic only; no fold's pass/fail
    is itself a qualification criterion.
    """
    closed = [t for t in trades if t.get("status") == "closed" and t.get("net_pnl") is not None]
    by_year: dict[int, list[dict]] = {}
    for trade in closed:
        year = pd.Timestamp(trade["exit_time"]).year
        by_year.setdefault(year, []).append(trade)

    rows = []
    for year in sorted(by_year):
        year_trades = by_year[year]
        net = [float(t["net_pnl"]) for t in year_trades]
        wins = sum(1 for value in net if value > 0)
        gross_profit = sum(value for value in net if value > 0)
        gross_loss = -sum(value for value in net if value < 0)
        rows.append({
            "year": year,
            "closed_count": len(year_trades),
            "win_rate": wins / len(year_trades) if year_trades else None,
            "net_expectancy": sum(net) / len(net) if net else None,
            "total_net_pnl": sum(net),
            "profit_factor": gross_profit / gross_loss if gross_loss else (999_999.0 if gross_profit else None),
        })
    return rows


def decide_qualification(
    *,
    cache_unchanged: bool,
    determinism_passed: bool,
    baseline_summary: dict,
    stressed_summary: dict,
    coverage_rate: float | None,
    pre_run_cache_sha256: str,
    post_run_cache_sha256: str,
    qualify_fn=qualify_strategy,
) -> dict:
    """The required pipeline order (preregistration Section 12 / this
    plan's own audit requirement): qualify_fn is never invoked unless
    BOTH the post-run cache hash matches the preflight's recorded value
    AND the fresh-cache determinism check passed. On failure, there is no
    passing/failing verdict object at all - only an explicit "invalid"
    status and the reason(s) - nothing that could later be misread as a
    real qualification result.
    """
    if cache_unchanged and determinism_passed:
        qualification = qualify_fn(baseline_summary, stressed_summary, coverage_rate)
        qualification["qualification_status"] = "evaluated"
        return qualification

    reasons = []
    if not cache_unchanged:
        reasons.append("bars_cache.db hash changed between the Stage-2 preflight and the post-run recheck")
    if not determinism_passed:
        reasons.append("determinism check failed - a fresh-cache re-run produced a different result")
    return {
        "qualification_status": "invalid",
        "reason": "; ".join(reasons),
        "pre_run_cache_sha256": pre_run_cache_sha256,
        "post_run_cache_sha256": post_run_cache_sha256,
    }


def run_ablation(
    *,
    stock_symbols: list[str],
    start_date: date,
    end_date: date,
    baseline_record_path: Path,
    label: str | None = None,
) -> tuple[dict, Path]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    stock_symbols = sorted(set(s.upper() for s in stock_symbols))
    if "SPY" not in stock_symbols:
        stock_symbols.append("SPY")
        stock_symbols.sort()

    print("Stage-2 preflight ...", flush=True)
    preflight_preregistration = verify_preregistration_scope()
    preflight_deployment = verify_deployment_guardrail(baseline_record_path)
    preflight_report = {
        "preregistration_scope": preflight_preregistration,
        "deployment_guardrail": preflight_deployment,
    }
    pre_run_cache_sha256 = preflight_preregistration["cache_db_sha256_preflight"]

    variants = build_variants()
    cache_misses: list[dict] = []

    print("Loading hourly stock data (read-only, cache-only) ...", flush=True)
    hourly_stock_fetcher = make_hourly_fetcher("stock", on_miss=_make_miss_logger(cache_misses, "shared"))
    hourly_crypto_fetcher = make_hourly_fetcher("crypto", on_miss=_make_miss_logger(cache_misses, "shared"))
    stock_frames, crypto_frames = load_research_data(
        stock_symbols, [], start_date, end_date,
        fetch_stock=hourly_stock_fetcher, fetch_crypto=hourly_crypto_fetcher,
    )
    stock_benchmark = benchmark_summary(stock_frames.get("SPY"), start_date, end_date, "stock")

    outcome_cache: dict = {}
    results: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    signal_calendars: dict[str, dict] = {}
    indicator_frames_by_variant: dict[str, dict] = {}
    all_trades, all_rejected, all_missing, all_equity = [], [], [], []
    data_coverage, exclusions = None, None

    for variant in variants:
        print(f"  Variant {variant.variant_id} ...", flush=True)
        stock_signal_fn = functools.partial(
            stock_trend_momentum_v2,
            enable_adx_filter=variant.delta2_enabled,
            enable_cost_filter=variant.delta3_enabled,
        )
        signal_calendar, diagnostics = build_signal_calendar(
            stock_frames, crypto_frames, start_date, end_date, stock_signal_fn=stock_signal_fn,
        )
        indicator_frames = {
            "stock": diagnostics.pop("stock_indicator_frames"),
            "crypto": diagnostics.pop("crypto_indicator_frames"),
        }
        signal_calendars[variant.variant_id] = signal_calendar
        indicator_frames_by_variant[variant.variant_id] = indicator_frames
        if data_coverage is None:
            # Coverage/exclusions depend only on the shared hourly frames'
            # non-ADX indicator completeness (build_signal_calendar's own
            # `required` list never includes adx_14), so they are identical
            # across all 8 variants - computed once, from the first.
            data_coverage = diagnostics["coverage"]
            exclusions = diagnostics["exclusions"]

        outcome_simulator = functools.partial(
            simulate_order_outcome,
            fetch_stock=make_minute_fetcher("stock", on_miss=_make_miss_logger(cache_misses, variant.variant_id)),
            fetch_crypto=make_minute_fetcher("crypto", on_miss=_make_miss_logger(cache_misses, variant.variant_id)),
        )

        results[variant.variant_id] = {}
        summaries[variant.variant_id] = {}
        for portfolio_name, portfolio in PORTFOLIOS.items():
            results[variant.variant_id][portfolio_name] = {}
            summaries[variant.variant_id][portfolio_name] = {}
            for cost_name, cost in COSTS.items():
                result = simulate_portfolio(
                    signal_calendar, indicator_frames, start_date, end_date,
                    portfolio, cost, "stock_only", outcome_cache,
                    outcome_simulator=outcome_simulator,
                    expiration_sessions=variant.expiration_sessions,
                )
                summary = summarize_run(
                    result, starting_equity=portfolio.starting_equity,
                    start_date=start_date, end_date=end_date, benchmark=stock_benchmark,
                )
                results[variant.variant_id][portfolio_name][cost_name] = result
                summaries[variant.variant_id][portfolio_name][cost_name] = summary

                tag = {
                    "variant_id": variant.variant_id,
                    "delta1_expiration_sessions": variant.expiration_sessions,
                    "delta2_enabled": variant.delta2_enabled,
                    "delta3_enabled": variant.delta3_enabled,
                    "portfolio": portfolio_name,
                    "cost_model": cost_name,
                }
                all_trades.extend({**tag, **row} for row in result["trades"])
                all_rejected.extend({**tag, **row} for row in result["rejected"])
                all_missing.extend({**tag, **row} for row in result["missing_outcomes"])
                all_equity.extend({**tag, **row} for row in summary["equity_curve"])

    full_v2_variant = next(v for v in variants if v.variant_id == FULL_V2_VARIANT_ID)
    print("  Determinism check: full-v2/current_1pct/baseline, fresh cache ...", flush=True)
    fresh_cache: dict = {}
    fresh_outcome_simulator = functools.partial(
        simulate_order_outcome,
        fetch_stock=make_minute_fetcher("stock", on_miss=_make_miss_logger(cache_misses, "determinism_check")),
        fetch_crypto=make_minute_fetcher("crypto", on_miss=_make_miss_logger(cache_misses, "determinism_check")),
    )
    check_b = simulate_portfolio(
        signal_calendars[FULL_V2_VARIANT_ID], indicator_frames_by_variant[FULL_V2_VARIANT_ID],
        start_date, end_date, PORTFOLIOS["current_1pct"], COSTS["baseline"], "stock_only", fresh_cache,
        outcome_simulator=fresh_outcome_simulator, expiration_sessions=full_v2_variant.expiration_sessions,
    )
    check_a = results[FULL_V2_VARIANT_ID]["current_1pct"]["baseline"]
    hash_a, hash_b = _stable_hash(check_a), _stable_hash(check_b)
    determinism = {"passed": hash_a == hash_b, "first_sha256": hash_a, "second_sha256": hash_b}

    print("  Post-run cache-hash recheck ...", flush=True)
    post_run_cache_sha256 = sha256_file(CACHE_DB_PATH)
    cache_unchanged = post_run_cache_sha256 == pre_run_cache_sha256
    preflight_report["cache_integrity"] = {
        "pre_run_sha256": pre_run_cache_sha256,
        "post_run_sha256": post_run_cache_sha256,
        "unchanged": cache_unchanged,
    }

    qualification = decide_qualification(
        cache_unchanged=cache_unchanged,
        determinism_passed=determinism["passed"],
        baseline_summary=summaries[FULL_V2_VARIANT_ID]["current_1pct"]["baseline"],
        stressed_summary=summaries[FULL_V2_VARIANT_ID]["current_1pct"]["stressed"],
        coverage_rate=data_coverage["stock"]["coverage_rate"],
        pre_run_cache_sha256=pre_run_cache_sha256,
        post_run_cache_sha256=post_run_cache_sha256,
    )

    temporal_diagnostics = _temporal_diagnostics(results[FULL_V2_VARIANT_ID]["current_1pct"]["baseline"]["trades"])

    config = {
        "start": str(start_date),
        "end": str(end_date),
        "decision_time": "11:16 America/New_York",
        "stock_symbols": stock_symbols,
        "label": label,
        "portfolios": {name: asdict(value) for name, value in PORTFOLIOS.items()},
        "cost_models": {name: asdict(value) for name, value in COSTS.items()},
        "variants": [asdict(v) for v in variants],
        "strategy_registry": registry_snapshot(),
        "historical_scope": TECHNICAL_ONLY_DISCLOSURE,
        "data_policy": "read-only, cache-only (backtest/v2_readonly_adapters.py) - "
                       "a coverage gap never triggers a live fetch",
        "preregistration_document": "research/stock_trend_momentum_v2_preregistration.md",
    }

    implementation_hashes = {
        "preflight_report": preflight_report,
        "source_file_sha256": {name: sha256_file(path) for name, path in IMPLEMENTATION_FILES.items()},
        "stage1_baseline_check_path": str(baseline_record_path),
        "stage1_baseline_check_sha256": sha256_file(Path(baseline_record_path)),
    }

    run_dir = _write_results(
        config=config,
        preflight_report=preflight_report,
        data_coverage=data_coverage,
        exclusions=exclusions,
        all_trades=all_trades,
        all_rejected=all_rejected,
        all_missing=all_missing,
        all_equity=all_equity,
        cache_misses=cache_misses,
        summaries=summaries,
        temporal_diagnostics=temporal_diagnostics,
        qualification=qualification,
        determinism=determinism,
        implementation_hashes=implementation_hashes,
    )
    _print_summary(qualification, determinism, run_dir)
    comparison = {
        "config": config, "qualification": qualification, "determinism": determinism,
        "temporal_diagnostics": temporal_diagnostics, "data_coverage": data_coverage,
    }
    return comparison, run_dir


def _write_results(
    *,
    config: dict,
    preflight_report: dict,
    data_coverage: dict,
    exclusions: list[dict],
    all_trades: list[dict],
    all_rejected: list[dict],
    all_missing: list[dict],
    all_equity: list[dict],
    cache_misses: list[dict],
    summaries: dict,
    temporal_diagnostics: list[dict],
    qualification: dict,
    determinism: dict,
    implementation_hashes: dict,
) -> Path:
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "preflight_report.json", preflight_report)
    _write_json(run_dir / "data_coverage.json", data_coverage)
    _write_csv(run_dir / "exclusions.csv", exclusions)
    _write_csv(run_dir / "trades.csv", all_trades)
    _write_csv(run_dir / "rejected_orders.csv", all_rejected)
    _write_csv(run_dir / "equity_curves.csv", all_equity)
    _write_csv(run_dir / "cache_misses.csv", cache_misses)
    _write_csv(run_dir / "missing_outcome_data.csv", all_missing)
    _write_json(run_dir / "summaries.json", summaries)
    _write_csv(run_dir / "temporal_diagnostics.csv", temporal_diagnostics)
    _write_json(run_dir / "qualification_result.json", qualification)
    _write_json(run_dir / "determinism_check.json", determinism)
    _write_json(run_dir / "implementation_hashes.json", implementation_hashes)

    output_files = [
        "config.json", "preflight_report.json", "data_coverage.json", "exclusions.csv",
        "trades.csv", "rejected_orders.csv", "equity_curves.csv", "cache_misses.csv",
        "missing_outcome_data.csv", "summaries.json", "temporal_diagnostics.csv",
        "qualification_result.json", "determinism_check.json", "implementation_hashes.json",
    ]
    results_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_document_sha256": preflight_report["preregistration_scope"]["document_sha256"],
        "preregistration_manifest_path": "research/stock_trend_momentum_v2_preregistration.manifest.json",
        "cache_db_sha256_pre_run": preflight_report["cache_integrity"]["pre_run_sha256"],
        "cache_db_sha256_post_run": preflight_report["cache_integrity"]["post_run_sha256"],
        "files_sha256": {name: sha256_file(run_dir / name) for name in output_files},
    }
    # Written last, after every other artifact above - and never includes
    # its own hash, since hashing a file that contains its own hash is not
    # a fixed point this integrity check needs to solve.
    _write_json(run_dir / "results_manifest.json", results_manifest)
    return run_dir


def _print_summary(qualification: dict, determinism: dict, run_dir: Path) -> None:
    print("\n  Determinism check:", "PASS" if determinism["passed"] else "FAIL")
    print(f"  Qualification status: {qualification['qualification_status']}")
    if qualification["qualification_status"] == "evaluated":
        verdict = "PASS" if qualification["passed"] else "FAIL"
        print(f"  Full-v2 qualification: {verdict}")
        if qualification["failed_checks"]:
            print(f"    failed: {', '.join(qualification['failed_checks'])}")
    else:
        print(f"    reason: {qualification['reason']}")
    print(f"\n  Results written to {run_dir}")


def _find_latest_baseline_record() -> Path:
    candidates = sorted(
        (REPO_ROOT / "research").glob("stock_trend_momentum_v2_implementation_baseline_check_*.json")
    )
    if not candidates:
        raise FileNotFoundError(
            "No Stage-1 baseline check found. Run "
            "`python -m backtest.preregistration_preflight` once before implementing, "
            "or pass --baseline-record explicitly."
        )
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="stock_trend_momentum_v2 ablation + temporal-stability harness")
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--end", default=str(DEFAULT_END))
    parser.add_argument("--stock-symbols", nargs="+", default=None)
    parser.add_argument("--baseline-record", default=None,
                        help="Path to the Stage-1 baseline JSON; defaults to the most recent one in research/")
    parser.add_argument("--label", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="AAPL/MSFT/SPY over June-July 2026, for a fast dry run")
    args = parser.parse_args()

    if args.smoke:
        start_date, end_date = date(2026, 6, 1), date(2026, 7, 31)
        stock_symbols = ["AAPL", "MSFT", "SPY"]
        label = args.label or "smoke"
    else:
        start_date, end_date = date.fromisoformat(args.start), date.fromisoformat(args.end)
        stock_symbols = args.stock_symbols or list(UNIVERSE)
        label = args.label

    baseline_record_path = Path(args.baseline_record) if args.baseline_record else _find_latest_baseline_record()

    run_ablation(
        stock_symbols=stock_symbols,
        start_date=start_date,
        end_date=end_date,
        baseline_record_path=baseline_record_path,
        label=label,
    )


if __name__ == "__main__":
    main()
