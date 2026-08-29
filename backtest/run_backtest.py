"""
CLI entry point for the 5-min ORB backtest.

Usage:
    # Primary out-of-sample evaluation (2024+) - this is the evidence that
    # matters; the strategy was developed/tested on 2016-2023 data in the
    # source research, so re-running on that window is a mechanics sanity
    # check only, not new evidence.
    python -m backtest.run_backtest --start 2024-01-01 --end 2025-12-31

    # Mechanics-reproduction check against the original research window
    # (explicitly NOT treated as out-of-sample evidence - see above):
    python -m backtest.run_backtest --start 2016-01-01 --end 2023-12-31 --label reproduction_check

    # Quick smoke test against a handful of liquid names:
    python -m backtest.run_backtest --symbols AAPL MSFT NVDA TSLA --start 2025-01-01 --end 2025-03-31

Every run writes backtest/results/{timestamp}/ with config.json, trades.csv,
equity_curve.csv, and summary.json - see backtest/metrics.py for what each
number means, including the caveats (142-symbol universe, long-only
actionability, IEX-vs-SIP overlap decision gate) that MUST be read alongside
any headline profitability number.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

from backtest.engine import run_backtest
from backtest.metrics import (
    UNIVERSE_CAVEAT,
    aggregate_overlap,
    equity_curve,
    missing_data_report,
    summarize,
    summarize_by_direction,
)
from backtest.portfolio import INTENDED_DEPLOYMENT, RESEARCH_FIDELITY
from config.universe import UNIVERSE
from utils.cost_model import BASELINE_COST, STRESSED_COST, ZERO_COST

_SCENARIOS = {"research_fidelity": RESEARCH_FIDELITY, "intended_deployment": INTENDED_DEPLOYMENT}
_COST_MODELS = {"zero": ZERO_COST, "baseline": BASELINE_COST, "stressed": STRESSED_COST}
_RESULTS_DIR = Path(__file__).parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description="5-min ORB backtest")
    parser.add_argument("--symbols", nargs="+", default=None,
                         help="limit to these symbols (default: full 142-symbol UNIVERSE)")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--label", default=None, help="e.g. 'reproduction_check' for the 2016-2023 window")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else list(UNIVERSE)
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    print(f"\n{'='*55}")
    print(f"  ORB BACKTEST   {start_date} -> {end_date}   symbols={len(symbols)}")
    if args.label:
        print(f"  label = {args.label}")
    print(f"{'='*55}")
    print(f"\n  {UNIVERSE_CAVEAT}\n")

    result = run_backtest(
        symbols=symbols, start_date=start_date, end_date=end_date,
        scenarios=_SCENARIOS, cost_models=_COST_MODELS, top_n=args.top_n,
    )
    trades = result["trades"]
    excluded = result["excluded"]
    rejected_orders = result["rejected_orders"]
    missing_outcome_data = result["missing_outcome_data"]
    overlap_records = result["overlap_records"]
    total_symbol_days = sum(c["attempted"] for c in result["daily_candidate_counts"])

    summaries = []
    directional = {}
    for scenario_name in _SCENARIOS:
        for cost_name in _COST_MODELS:
            summaries.append(summarize(trades, scenario_name, cost_name))
            directional[f"{scenario_name}/{cost_name}"] = summarize_by_direction(trades, scenario_name, cost_name)

    print("\n  Summary (scenario / cost_model / admitted / triggered / win_rate / expectancy / total_pnl):")
    for s in summaries:
        print(f"    {s['scenario']:<20} {s['cost_model']:<10} "
              f"admitted={s['admitted_count']:<5} triggered={s['triggered_count']:<5} "
              f"win_rate={s['win_rate']}  expectancy={s['expectancy']}  total_pnl={s['total_pnl']}")

    md_report = missing_data_report(excluded, total_symbol_days)
    print(f"\n  Missing-data: {md_report['excluded_count']} exclusions "
          f"({md_report['excluded_rate']} of symbol-days attempted)")
    print(f"  Rejected orders (position/exposure limits): {len(rejected_orders)}")
    print(f"  Missing outcome data (failed/empty 1-min fetch): {len(missing_outcome_data)} "
          f"selected candidates - see missing_outcome_data.csv, NOT counted as 'no trigger'")

    overlap_summary = aggregate_overlap(overlap_records)
    print(f"\n  IEX-vs-SIP overlap: avg_overlap_rate={overlap_summary['avg_overlap_rate']}  "
          f"avg_jaccard={overlap_summary['avg_jaccard']}  "
          f"(comparable={overlap_summary['comparable_sessions']}/{overlap_summary['valid_sessions']} valid sessions, "
          f"{overlap_summary['invalid_sessions']} invalid/excluded) - see overlap_records.csv for the full "
          f"per-session detail behind this average")
    if overlap_summary["avg_overlap_rate"] is not None and overlap_summary["avg_overlap_rate"] < 0.7:
        print(f"  *** LOW OVERLAP ({overlap_summary['avg_overlap_rate']:.1%}) - the free IEX feed selects "
              f"materially different candidates than full-market SIP would. Day-mode execution should "
              f"remain disabled regardless of headline profitability. ***")

    _write_results(args, symbols, start_date, end_date, trades, summaries, directional,
                    md_report, overlap_summary, rejected_orders, missing_outcome_data, overlap_records)


def _write_results(args, symbols, start_date, end_date, trades, summaries, directional,
                    md_report, overlap_summary, rejected_orders, missing_outcome_data, overlap_records) -> None:
    run_dir = _RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps({
        "symbols": symbols, "start": str(start_date), "end": str(end_date),
        "top_n": args.top_n, "label": args.label,
        "scenarios": list(_SCENARIOS.keys()), "cost_models": list(_COST_MODELS.keys()),
        "universe_caveat": UNIVERSE_CAVEAT,
    }, indent=2))

    if trades:
        with open(run_dir / "trades.csv", "w", newline="") as f:
            fieldnames = sorted({k for t in trades for k in t.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trades:
                writer.writerow({k: str(v) for k, v in t.items()})

    if rejected_orders:
        with open(run_dir / "rejected_orders.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["symbol", "session_date", "scenario", "cost_model", "reason"])
            writer.writeheader()
            for r in rejected_orders:
                writer.writerow(r)

    if missing_outcome_data:
        with open(run_dir / "missing_outcome_data.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["symbol", "session_date", "reason"])
            writer.writeheader()
            for m in missing_outcome_data:
                writer.writerow(m)

    if overlap_records:
        # The full per-session IEX-vs-SIP detail behind the averaged
        # overlap_rate/jaccard in summary.json - without this file the
        # average can't be audited or spot-checked against individual days.
        with open(run_dir / "overlap_records.csv", "w", newline="") as f:
            fieldnames = sorted({k for r in overlap_records for k in r.keys()})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in overlap_records:
                writer.writerow({k: str(v) for k, v in r.items()})

    for scenario_name, config in _SCENARIOS.items():
        for cost_name in _COST_MODELS:
            curve = equity_curve(trades, scenario_name, cost_name, config.starting_equity)
            if curve:
                with open(run_dir / f"equity_curve_{scenario_name}_{cost_name}.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["session_date", "equity", "drawdown_pct"])
                    writer.writeheader()
                    for row in curve:
                        writer.writerow(row)

    (run_dir / "summary.json").write_text(json.dumps({
        "summaries": summaries,
        "by_direction": directional,
        "missing_data": md_report,
        "missing_outcome_data_count": len(missing_outcome_data),
        "iex_sip_overlap": overlap_summary,
        "universe_caveat": UNIVERSE_CAVEAT,
        "long_only_note": (
            "Only long stock orders are executable live by this codebase today. "
            "Short-side results above are informational only."
        ),
        "portfolio_model_note": (
            "Admission is a rank-order reservation model at each session's shared "
            "9:35 decision point, not full trigger-time event simulation - see "
            "backtest/engine.py's module docstring."
        ),
    }, indent=2, default=str))

    print(f"\n  Results written to {run_dir}")


if __name__ == "__main__":
    main()
