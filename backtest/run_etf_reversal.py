"""Qualification runner for etf_reversal_v1.

Implements research/etf_reversal_v1_preregistration.md end to end: loads
the etf_momentum_v1 data snapshot unchanged (never fetches), restricts to
the 16 equity tickers, sweeps the 18-cell grid on the in-sample window
selecting by baseline-cost Sharpe, evaluates the selected cell on the full
effective window (checks 1-13), on the out-of-sample window (checks
14-15), and on its in-sample one-axis-step neighbors (check 16), records
a supplementary zero-cost Sharpe ceiling (full-window and OOS, same run,
not part of the gate), then runs the 16-check qualification gate exactly
once (Section 10's one-shot rule) and writes the result to
backtest/results/etf_reversal_v1/<timestamp>/.

Run: python -m backtest.run_etf_reversal
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from backtest.etf_momentum_snapshot import load_snapshot, manifest_sha256
from backtest.whole_bot_engine import (
    COSTS,
    EtfReversalConfig,
    simulate_etf_reversal_portfolio,
)
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run

STRATEGY_VERSION = "etf_reversal_v1"
PREREGISTRATION = Path(__file__).resolve().parents[1] / "research" / "etf_reversal_v1_preregistration.md"
PREREGISTRATION_MANIFEST = Path(__file__).resolve().parents[1] / "research" / "etf_reversal_v1_preregistration.manifest.json"
RESULTS_ROOT = Path(__file__).parent / "results" / "etf_reversal_v1"

EQUITY_TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "XLB", "XLC", "XLE", "XLF",
    "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY",
]

FULL_START = date(2008, 7, 1)
FULL_END = date(2026, 8, 1)
IN_SAMPLE_START = date(2008, 7, 1)
IN_SAMPLE_END = date(2019, 12, 31)
OUT_OF_SAMPLE_START = date(2020, 1, 1)
OUT_OF_SAMPLE_END = date(2026, 8, 1)
STARTING_EQUITY = 100_000.0
MAX_POSITIONS = 5

ENTRY_GRID = (5, 10, 15)
EXIT_GRID = (55, 65, 75)
MAXHOLD_GRID = (5, 10)


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
        raise RuntimeError(f"etf_reversal_v1 preregistration hash mismatch: expected={expected} actual={actual}")
    return manifest


def _grid_cells() -> list[EtfReversalConfig]:
    return [
        EtfReversalConfig(entry_threshold=e, exit_threshold=x, max_hold=m)
        for e in ENTRY_GRID for x in EXIT_GRID for m in MAXHOLD_GRID
    ]


def _neighbors(selected: EtfReversalConfig) -> list[EtfReversalConfig]:
    axes = [
        ("entry_threshold", ENTRY_GRID, selected.entry_threshold),
        ("exit_threshold", EXIT_GRID, selected.exit_threshold),
        ("max_hold", MAXHOLD_GRID, selected.max_hold),
    ]
    neighbors = []
    for field, grid, value in axes:
        idx = grid.index(value)
        for delta in (-1, 1):
            new_idx = idx + delta
            if 0 <= new_idx < len(grid):
                kwargs = asdict(selected)
                kwargs[field] = grid[new_idx]
                neighbors.append(EtfReversalConfig(**kwargs))
    return neighbors


def _coverage_rate(result: dict) -> float:
    filled = len(result["trades"])
    rejected = len(result["rejected"])
    attempted = filled + rejected
    return filled / attempted if attempted else 1.0


def _annualized_turnover(result: dict, start: date, end: date) -> dict:
    years = max((end - start).days / 365.25, 1 / 365.25)
    closed = [t for t in result["trades"]]
    trade_count = len(closed)
    gross_notional = sum(
        abs(t["quantity"] * t["entry_price"]) + abs(t["quantity"] * t["exit_price"]) for t in closed
    )
    avg_equity = (
        sum(row["equity"] for row in result["daily_equity"]) / len(result["daily_equity"])
        if result["daily_equity"] else STARTING_EQUITY
    )
    holds = [(pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days for t in closed]
    avg_hold_calendar_days = sum(holds) / len(holds) if holds else None
    return {
        "closed_trades_per_year": trade_count / years,
        "gross_notional_per_year": gross_notional / years,
        "gross_notional_per_year_over_avg_equity": (gross_notional / years) / avg_equity if avg_equity else None,
        "average_hold_calendar_days": avg_hold_calendar_days,
    }


def main() -> dict:
    doc_manifest = verify_frozen_document()
    data_manifest_sha = manifest_sha256()

    snapshot = load_snapshot()
    adjusted = {ticker: snapshot[ticker]["adjusted"] for ticker in EQUITY_TICKERS}
    spy_benchmark_frame = pd.DataFrame({"close": adjusted["SPY"]["Close"]})

    # -- Step 1: 18-cell in-sample grid sweep, select by baseline Sharpe --------
    grid_results: list[dict] = []
    is_bench = benchmark_summary(spy_benchmark_frame, IN_SAMPLE_START, IN_SAMPLE_END, "stock")
    for config in _grid_cells():
        result = simulate_etf_reversal_portfolio(
            adjusted, IN_SAMPLE_START, IN_SAMPLE_END, STARTING_EQUITY, COSTS["baseline"], config, MAX_POSITIONS,
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
    selected = EtfReversalConfig(**best["config"])
    print(f"\nSelected cell (max IS Sharpe): {asdict(selected)} sharpe={best['sharpe']}", flush=True)

    # -- Step 2: full-window run for checks 1-13 (zero/baseline/stressed) -------
    full_bench = benchmark_summary(spy_benchmark_frame, FULL_START, FULL_END, "stock")
    full_results = {}
    full_summaries = {}
    for cost_name in ("zero", "baseline", "stressed"):
        result = simulate_etf_reversal_portfolio(
            adjusted, FULL_START, FULL_END, STARTING_EQUITY, COSTS[cost_name], selected, MAX_POSITIONS,
        )
        full_results[cost_name] = result
        full_summaries[cost_name] = summarize_run(
            result, starting_equity=STARTING_EQUITY, start_date=FULL_START, end_date=FULL_END,
            benchmark=full_bench,
        )
    coverage_rate = _coverage_rate(full_results["baseline"])
    turnover = _annualized_turnover(full_results["baseline"], FULL_START, FULL_END)

    # -- Step 3: OOS-only run for checks 14-15 (baseline cost, per the gate) ----
    oos_bench = benchmark_summary(spy_benchmark_frame, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, "stock")
    oos_result = simulate_etf_reversal_portfolio(
        adjusted, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, STARTING_EQUITY, COSTS["baseline"], selected, MAX_POSITIONS,
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

    # -- Step 3b: supplementary zero-cost ceiling, same run (not part of the gate) --
    oos_zero_result = simulate_etf_reversal_portfolio(
        adjusted, OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, STARTING_EQUITY, COSTS["zero"], selected, MAX_POSITIONS,
    )
    oos_zero_summary = summarize_run(
        oos_zero_result, starting_equity=STARTING_EQUITY, start_date=OUT_OF_SAMPLE_START,
        end_date=OUT_OF_SAMPLE_END, benchmark=oos_bench,
    )
    zero_cost_ceiling = {
        "note": (
            "Not part of the 16-check gate (checks 14-15 use baseline cost). "
            "Best-case Sharpe if trading were entirely free, same frozen "
            "selected cell, full-window and OOS."
        ),
        "full_window_sharpe": full_summaries["zero"]["sharpe"],
        "oos_sharpe": oos_zero_summary["sharpe"],
        "full_window_benchmark_sharpe": full_bench["sharpe"],
        "oos_benchmark_sharpe": oos_bench["sharpe"],
        "full_window_trails_benchmark": (
            full_summaries["zero"]["sharpe"] is not None and full_bench["sharpe"] is not None
            and full_summaries["zero"]["sharpe"] < full_bench["sharpe"]
        ),
        "oos_trails_benchmark": (
            oos_zero_summary["sharpe"] is not None and oos_bench["sharpe"] is not None
            and oos_zero_summary["sharpe"] < oos_bench["sharpe"]
        ),
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
    sensitivity = {"selected_sharpe": selected_is_row["sharpe"], "neighbors": neighbor_rows}

    # -- Step 5: the one qualification call ---------------------------------------
    qualification = qualify_strategy(
        full_summaries["baseline"], full_summaries["stressed"], coverage_rate,
        walk_forward=walk_forward, sensitivity=sensitivity,
    )
    qualification["check_3_note"] = (
        "missing_outcomes_below_1pct is structurally vacuous for this engine "
        "(every position closes with a real price by construction - trend/"
        "target/time exit or a forced end-of-test close); trivial pass, not "
        "a meaningful result (preregistration Section 9)."
    )
    corner_cells = [
        row for row in grid_results
        if row["config"]["entry_threshold"] in (5, 15) and row["config"]["exit_threshold"] in (55, 75)
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
        "equity_tickers": EQUITY_TICKERS,
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
        "zero_cost_ceiling": zero_cost_ceiling,
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
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})

    _csv(output / "trades__baseline.csv", full_results["baseline"]["trades"])
    _csv(output / "trades__zero.csv", full_results["zero"]["trades"])
    _csv(output / "trades__stressed.csv", full_results["stressed"]["trades"])
    _csv(output / "oos_trades__baseline.csv", oos_result["trades"])
    _csv(output / "oos_trades__zero.csv", oos_zero_result["trades"])
    _csv(output / "daily_equity__baseline.csv", full_results["baseline"]["daily_equity"])
    _csv(output / "daily_equity__zero.csv", full_results["zero"]["daily_equity"])
    _csv(output / "daily_equity__oos_baseline.csv", oos_result["daily_equity"])
    _csv(output / "daily_equity__oos_zero.csv", oos_zero_result["daily_equity"])
    _csv(output / "grid_results.csv", grid_results)

    manifest = {path.name: _sha256(path) for path in sorted(output.iterdir()) if path.is_file()}
    (output / "manifest.json").write_text(json.dumps({"sha256": manifest}, indent=2), encoding="utf-8")

    print(f"\nQUALIFICATION {'PASS' if qualification['passed'] else 'FAIL'}")
    print(f"selected cell: {asdict(selected)}")
    print(f"failed checks: {qualification['failed_checks']}")
    print(f"zero-cost ceiling: full={zero_cost_ceiling['full_window_sharpe']} "
          f"(benchmark {zero_cost_ceiling['full_window_benchmark_sharpe']}, "
          f"trails={zero_cost_ceiling['full_window_trails_benchmark']}) "
          f"OOS={zero_cost_ceiling['oos_sharpe']} "
          f"(benchmark {zero_cost_ceiling['oos_benchmark_sharpe']}, "
          f"trails={zero_cost_ceiling['oos_trails_benchmark']})")
    print(f"output: {output}")
    return report


if __name__ == "__main__":
    main()
