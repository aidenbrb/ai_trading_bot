"""Qualification runner for etf_momentum_v1.

Implements research/etf_momentum_v1_preregistration.md end to end: loads
the frozen data snapshot (never fetches), sweeps the 18-cell grid on the
in-sample window selecting by baseline-cost Sharpe, evaluates the selected
cell on the full effective window (checks 1-13), on the out-of-sample
window (checks 14-15), and on its in-sample one-axis-step neighbors
(check 16), then runs the full 16-check qualification gate exactly once
(Section 9's one-shot rule) and writes the result to
backtest/results/etf_momentum_v1/<timestamp>/.

Run: python -m backtest.run_etf_momentum
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.etf_momentum_snapshot import load_snapshot, manifest_sha256
from backtest.whole_bot_engine import (
    COSTS,
    EtfMomentumConfig,
    simulate_etf_momentum_portfolio,
)
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run

STRATEGY_VERSION = "etf_momentum_v1"
PREREGISTRATION = Path(__file__).resolve().parents[1] / "research" / "etf_momentum_v1_preregistration.md"
PREREGISTRATION_MANIFEST = Path(__file__).resolve().parents[1] / "research" / "etf_momentum_v1_preregistration.manifest.json"
RESULTS_ROOT = Path(__file__).parent / "results" / "etf_momentum_v1"

FULL_START = date(2008, 7, 1)
FULL_END = date(2026, 8, 1)
IN_SAMPLE_START = date(2008, 7, 1)
IN_SAMPLE_END = date(2019, 12, 31)
OUT_OF_SAMPLE_START = date(2020, 1, 1)
OUT_OF_SAMPLE_END = date(2026, 8, 1)
STARTING_EQUITY = 100_000.0

LOOKBACK_MONTHS_GRID = (6, 9, 12)
SKIP_LAST_MONTH_GRID = (0, 1)
TOP_N_GRID = (2, 3, 4)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_document() -> dict:
    manifest = json.loads(PREREGISTRATION_MANIFEST.read_text(encoding="utf-8"))
    actual = _sha256(PREREGISTRATION)
    expected = manifest["document"]["sha256"]
    if actual != expected:
        raise RuntimeError(
            f"etf_momentum_v1 preregistration hash mismatch: expected={expected} actual={actual}"
        )
    return manifest


def _grid_cells() -> list[EtfMomentumConfig]:
    return [
        EtfMomentumConfig(lookback_months=lb, skip_last_month=sk, top_n=tn)
        for lb in LOOKBACK_MONTHS_GRID
        for sk in SKIP_LAST_MONTH_GRID
        for tn in TOP_N_GRID
    ]


def _neighbors(selected: EtfMomentumConfig) -> list[EtfMomentumConfig]:
    """One-axis-step neighbors, per preregistration Section 8: a neighbor
    changes exactly one of the three axes by one grid step, holding the
    other two fixed at the selected cell's values. Only in-grid results
    are returned (an edge value on an axis contributes fewer neighbors)."""
    axes = [
        ("lookback_months", LOOKBACK_MONTHS_GRID, selected.lookback_months),
        ("skip_last_month", SKIP_LAST_MONTH_GRID, selected.skip_last_month),
        ("top_n", TOP_N_GRID, selected.top_n),
    ]
    neighbors = []
    for field, grid, value in axes:
        idx = grid.index(value)
        for delta in (-1, 1):
            new_idx = idx + delta
            if 0 <= new_idx < len(grid):
                kwargs = asdict(selected)
                kwargs[field] = grid[new_idx]
                neighbors.append(EtfMomentumConfig(**kwargs))
    return neighbors


def _coverage_rate(result: dict) -> float:
    trades = len(result["trades"])
    rejected = len(result["rejected"])
    attempted = trades + rejected
    return trades / attempted if attempted else 1.0


def _annualized_turnover(result: dict, start: date, end: date) -> dict:
    years = max((end - start).days / 365.25, 1 / 365.25)
    trade_count = len(result["trades"])
    gross_notional = sum(
        abs(t["quantity"] * t["entry_price"]) + abs(t["quantity"] * t["exit_price"])
        for t in result["trades"]
    )
    avg_equity = (
        sum(row["equity"] for row in result["daily_equity"]) / len(result["daily_equity"])
        if result["daily_equity"] else STARTING_EQUITY
    )
    return {
        "closed_trades_per_year": trade_count / years,
        "gross_notional_per_year": gross_notional / years,
        "gross_notional_per_year_over_avg_equity": (gross_notional / years) / avg_equity if avg_equity else None,
    }


def main() -> dict:
    doc_manifest = verify_frozen_document()
    data_manifest_sha = manifest_sha256()

    snapshot = load_snapshot()
    adjusted_close = {ticker: frames["adjusted"]["Close"] for ticker, frames in snapshot.items()}

    spy_benchmark_frame = pd.DataFrame({"close": adjusted_close["SPY"]})

    # -- Step 1: 18-cell in-sample grid sweep, select by baseline Sharpe --------
    grid_results: list[dict] = []
    is_bench = benchmark_summary(spy_benchmark_frame, IN_SAMPLE_START, IN_SAMPLE_END, "stock")
    for config in _grid_cells():
        result = simulate_etf_momentum_portfolio(
            adjusted_close, IN_SAMPLE_START, IN_SAMPLE_END, STARTING_EQUITY, COSTS["baseline"], config,
        )
        summary = summarize_run(
            result, starting_equity=STARTING_EQUITY, start_date=IN_SAMPLE_START,
            end_date=IN_SAMPLE_END, benchmark=is_bench,
        )
        grid_results.append({
            "config": asdict(config),
            "sharpe": summary["sharpe"] if summary["sharpe"] is not None else float("-inf"),
            "closed_trades": summary["closed_count"],
        })
        print(f"IS grid: {asdict(config)} -> sharpe={summary['sharpe']} trades={summary['closed_count']}", flush=True)

    best = max(grid_results, key=lambda row: row["sharpe"])
    selected = EtfMomentumConfig(**best["config"])
    print(f"\nSelected cell (max IS Sharpe): {asdict(selected)} sharpe={best['sharpe']}", flush=True)

    # -- Step 2: full-window run for checks 1-13 (zero/baseline/stressed) -------
    full_bench = benchmark_summary(spy_benchmark_frame, FULL_START, FULL_END, "stock")
    full_results = {}
    full_summaries = {}
    for cost_name in ("zero", "baseline", "stressed"):
        result = simulate_etf_momentum_portfolio(
            adjusted_close, FULL_START, FULL_END, STARTING_EQUITY, COSTS[cost_name], selected,
        )
        full_results[cost_name] = result
        full_summaries[cost_name] = summarize_run(
            result, starting_equity=STARTING_EQUITY, start_date=FULL_START, end_date=FULL_END,
            benchmark=full_bench,
        )
    coverage_rate = _coverage_rate(full_results["baseline"])
    turnover = _annualized_turnover(full_results["baseline"], FULL_START, FULL_END)

    # -- Step 3: OOS-only run for checks 14-15 -----------------------------------
    oos_bench = benchmark_summary(spy_benchmark_frame, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, "stock")
    oos_result = simulate_etf_momentum_portfolio(
        adjusted_close, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, STARTING_EQUITY, COSTS["baseline"], selected,
    )
    oos_summary = summarize_run(
        oos_result, starting_equity=STARTING_EQUITY, start_date=OUT_OF_SAMPLE_START,
        end_date=OUT_OF_SAMPLE_END, benchmark=oos_bench,
    )
    walk_forward = {
        "oos_closed_trades": oos_summary["closed_count"],
        "oos_sharpe": oos_summary["sharpe"],
        "oos_benchmark_sharpe": oos_bench["sharpe"],
        "oos_profit_factor": oos_summary["profit_factor"],
    }

    # -- Step 4: IS-only neighbor cells for check 16 ------------------------------
    neighbor_rows = []
    selected_is_row = next(row for row in grid_results if row["config"] == asdict(selected))
    for neighbor_config in _neighbors(selected):
        row = next((r for r in grid_results if r["config"] == asdict(neighbor_config)), None)
        neighbor_rows.append({
            "config": asdict(neighbor_config),
            "in_grid": True,
            "sharpe": row["sharpe"] if row is not None else None,
            "closed_trades": row["closed_trades"] if row is not None else 0,
        })
    sensitivity = {
        "selected_sharpe": selected_is_row["sharpe"],
        "neighbors": neighbor_rows,
    }

    # -- Step 5: the one qualification call ---------------------------------------
    qualification = qualify_strategy(
        full_summaries["baseline"], full_summaries["stressed"], coverage_rate,
        walk_forward=walk_forward, sensitivity=sensitivity,
    )
    qualification["check_3_note"] = (
        "missing_outcomes_below_1pct is structurally vacuous for this engine "
        "(simulate_etf_momentum_portfolio always closes every position with a "
        "real price - missing_outcomes is hardcoded empty); trivial pass, not "
        "a meaningful result (preregistration Section 8)."
    )
    corner_cells = [
        row for row in grid_results
        if row["config"]["lookback_months"] in (6, 12) and row["config"]["top_n"] in (2, 4)
    ]
    qualification["check_16_corner_cell_count"] = len(corner_cells)
    qualification["check_16_selected_is_corner"] = asdict(selected) in [r["config"] for r in corner_cells]

    # -- Report / persist -----------------------------------------------------
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = RESULTS_ROOT / run_id
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "strategy_version": STRATEGY_VERSION,
        "preregistration_sha256": doc_manifest["document"]["sha256"],
        "data_snapshot_manifest_sha256": data_manifest_sha,
        "selected_cell": asdict(selected),
        "grid_results": grid_results,
        "checks": qualification["checks"],
        "failed_checks": qualification["failed_checks"],
        "passed": qualification["passed"],
        "check_3_note": qualification["check_3_note"],
        "check_16_corner_cell_count": qualification["check_16_corner_cell_count"],
        "check_16_selected_is_corner": qualification["check_16_selected_is_corner"],
        "full_window": {"start": str(FULL_START), "end": str(FULL_END)},
        "in_sample_window": {"start": str(IN_SAMPLE_START), "end": str(IN_SAMPLE_END)},
        "out_of_sample_window": {"start": str(OUT_OF_SAMPLE_START), "end": str(OUT_OF_SAMPLE_END)},
        "coverage_rate": coverage_rate,
        "turnover": turnover,
        "full_summaries": full_summaries,
        "oos_summary": oos_summary,
        "walk_forward": walk_forward,
        "sensitivity": sensitivity,
        "benchmark_full": full_bench,
        "benchmark_oos": oos_bench,
    }

    def _default(obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return str(obj)

    (output / "report.json").write_text(json.dumps(report, indent=2, default=_default), encoding="utf-8")

    def _csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = sorted({key for row in rows for key in row})
        import csv
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    _csv(output / "trades__baseline.csv", full_results["baseline"]["trades"])
    _csv(output / "trades__zero.csv", full_results["zero"]["trades"])
    _csv(output / "trades__stressed.csv", full_results["stressed"]["trades"])
    _csv(output / "oos_trades__baseline.csv", oos_result["trades"])
    _csv(output / "daily_equity__baseline.csv", full_results["baseline"]["daily_equity"])
    _csv(output / "daily_equity__oos_baseline.csv", oos_result["daily_equity"])
    _csv(output / "grid_results.csv", grid_results)

    manifest = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"sha256": manifest}, indent=2), encoding="utf-8")

    print(f"\nQUALIFICATION {'PASS' if qualification['passed'] else 'FAIL'}")
    print(f"selected cell: {asdict(selected)}")
    print(f"failed checks: {qualification['failed_checks']}")
    print(f"output: {output}")
    return report


if __name__ == "__main__":
    main()
