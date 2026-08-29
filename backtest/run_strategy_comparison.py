"""CLI for evidence-first stock/crypto/combined strategy validation.

Examples
--------
Full locked evaluation::

    python -m backtest.run_strategy_comparison

Small real-data API smoke test::

    python -m backtest.run_strategy_comparison --smoke --determinism-check

This module imports market-data clients only.  It cannot submit orders.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from backtest.whole_bot_engine import (
    COSTS,
    PORTFOLIOS,
    TECHNICAL_ONLY_DISCLOSURE,
    build_signal_calendar,
    load_research_data,
    simulate_portfolio,
)
from backtest.whole_bot_metrics import (
    benchmark_summary,
    qualify_strategy,
    summarize_run,
)
from config.universe import CRYPTO, UNIVERSE
from utils.strategy_registry import registry_snapshot

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2026, 8, 9)
RESULTS_DIR = Path(__file__).parent / "results" / "whole_bot"


def run_comparison(
    *,
    stock_symbols: list[str],
    crypto_symbols: list[str],
    start_date: date,
    end_date: date,
    include_crypto_7day: bool = True,
    determinism_check: bool = False,
    label: str | None = None,
) -> tuple[dict, Path]:
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    stock_symbols = sorted(set(s.upper() for s in stock_symbols))
    crypto_symbols = sorted(set(s.upper() for s in crypto_symbols))
    if stock_symbols and "SPY" not in stock_symbols:
        stock_symbols.append("SPY")
        stock_symbols.sort()
    if crypto_symbols and "BTC-USD" not in crypto_symbols:
        crypto_symbols.append("BTC-USD")
        crypto_symbols.sort()

    config = {
        "start": str(start_date),
        "end": str(end_date),
        "decision_time": "11:16 America/New_York",
        "stock_symbols": stock_symbols,
        "crypto_symbols": crypto_symbols,
        "include_crypto_7day": include_crypto_7day,
        "label": label,
        "portfolios": {name: asdict(value) for name, value in PORTFOLIOS.items()},
        "cost_models": {name: asdict(value) for name, value in COSTS.items()},
        "strategy_registry": registry_snapshot(),
        "historical_scope": TECHNICAL_ONLY_DISCLOSURE,
        "data_policy": {
            "stocks": "Alpaca SIP, split-adjusted, no fallback",
            "crypto": "Alpaca US, no fallback",
        },
    }
    config_json = json.dumps(config, sort_keys=True, default=str)
    config["config_sha256"] = hashlib.sha256(config_json.encode()).hexdigest()

    print("\n" + "=" * 68)
    print("  EVIDENCE-FIRST WHOLE-BOT STRATEGY COMPARISON")
    print(f"  {start_date} -> {end_date}")
    print(f"  stocks={len(stock_symbols)} crypto={len(crypto_symbols)}")
    print("  RESEARCH ONLY - no trading client is imported")
    print("=" * 68)
    print(f"\n  {TECHNICAL_ONLY_DISCLOSURE}\n")

    stock_frames, crypto_frames = load_research_data(
        stock_symbols, crypto_symbols, start_date, end_date
    )
    signal_calendar, diagnostics = build_signal_calendar(
        stock_frames, crypto_frames, start_date, end_date
    )
    indicator_frames = {
        "stock": diagnostics.pop("stock_indicator_frames"),
        "crypto": diagnostics.pop("crypto_indicator_frames"),
    }

    stock_benchmark = benchmark_summary(
        stock_frames.get("SPY"), start_date, end_date, "stock"
    )
    crypto_benchmark = benchmark_summary(
        crypto_frames.get("BTC-USD"), start_date, end_date, "crypto"
    )

    modes = ["stock_only", "crypto_weekends"]
    if include_crypto_7day:
        modes.append("crypto_7day")
    results, summaries = {}, {}
    outcome_cache: dict = {}
    for mode in modes:
        print(f"  Simulating {mode} ...", flush=True)
        results[mode], summaries[mode] = {}, {}
        for portfolio_name, portfolio in PORTFOLIOS.items():
            results[mode][portfolio_name], summaries[mode][portfolio_name] = {}, {}
            for cost_name, cost in COSTS.items():
                result = simulate_portfolio(
                    signal_calendar,
                    indicator_frames,
                    start_date,
                    end_date,
                    portfolio,
                    cost,
                    mode,
                    outcome_cache,
                )
                benchmark = stock_benchmark if mode == "stock_only" else crypto_benchmark
                summary = summarize_run(
                    result,
                    starting_equity=portfolio.starting_equity,
                    start_date=start_date,
                    end_date=end_date,
                    benchmark=benchmark,
                )
                results[mode][portfolio_name][cost_name] = result
                summaries[mode][portfolio_name][cost_name] = summary

    qualifications = {
        "stock_only": qualify_strategy(
            summaries["stock_only"]["current_1pct"]["baseline"],
            summaries["stock_only"]["current_1pct"]["stressed"],
            diagnostics["coverage"]["stock"]["coverage_rate"],
        ),
        "crypto_weekends": qualify_strategy(
            summaries["crypto_weekends"]["current_1pct"]["baseline"],
            summaries["crypto_weekends"]["current_1pct"]["stressed"],
            diagnostics["coverage"]["crypto"]["coverage_rate"],
        ),
    }

    passing_markets = []
    if qualifications["stock_only"]["passed"]:
        passing_markets.append("stock")
    if qualifications["crypto_weekends"]["passed"]:
        passing_markets.append("crypto")

    if len(passing_markets) == 2:
        mode = "combined_primary"
        print("  Both strategies passed individually; simulating combined portfolio ...")
        results[mode], summaries[mode] = {}, {}
        for portfolio_name, portfolio in PORTFOLIOS.items():
            results[mode][portfolio_name], summaries[mode][portfolio_name] = {}, {}
            for cost_name, cost in COSTS.items():
                result = simulate_portfolio(
                    signal_calendar, indicator_frames, start_date, end_date,
                    portfolio, cost, mode, outcome_cache,
                )
                # Combined has no honest single-asset benchmark; both
                # individual benchmark gates already had to pass.
                combined_benchmark = {"symbol": "SPY/BTC individual gates",
                                      "total_return": None, "sharpe": None,
                                      "max_drawdown": None}
                results[mode][portfolio_name][cost_name] = result
                summaries[mode][portfolio_name][cost_name] = summarize_run(
                    result, starting_equity=portfolio.starting_equity,
                    start_date=start_date, end_date=end_date,
                    benchmark=combined_benchmark,
                )

    comparison = {
        "config": config,
        "data_coverage": diagnostics["coverage"],
        "benchmarks": {"stock": stock_benchmark, "crypto": crypto_benchmark},
        "summaries": summaries,
        "qualifications": qualifications,
        "passing_markets": passing_markets,
        "historical_qualification_passed": bool(passing_markets),
        "paper_rollout_candidate": bool(passing_markets),
        "automatic_entry_eligible": False,
        "automatic_entry_note": (
            "Historical qualification alone does not satisfy the required "
            "90-day/30-closed-trade forward paper gate."
        ),
        "combined_simulated": "combined_primary" in results,
        "historical_scope": TECHNICAL_ONLY_DISCLOSURE,
    }

    if determinism_check:
        print("  Re-running an in-memory baseline slice for determinism ...")
        check_a = results["stock_only"]["current_1pct"]["baseline"]
        check_b = simulate_portfolio(
            signal_calendar, indicator_frames, start_date, end_date,
            PORTFOLIOS["current_1pct"], COSTS["baseline"], "stock_only", outcome_cache,
        )
        hash_a = _stable_hash(check_a)
        hash_b = _stable_hash(check_b)
        comparison["determinism"] = {
            "passed": hash_a == hash_b, "first_sha256": hash_a, "second_sha256": hash_b,
        }
        if hash_a != hash_b:
            raise RuntimeError("determinism check failed")

    run_dir = _write_results(config, comparison, diagnostics, results)
    _print_qualification(comparison, run_dir)
    return comparison, run_dir


def _stable_hash(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(rendered.encode()).hexdigest()


def _write_results(config: dict, comparison: dict, diagnostics: dict, results: dict) -> Path:
    run_dir = RESULTS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "summary.json", comparison)
    _write_json(run_dir / "data_coverage.json", diagnostics["coverage"])
    _write_csv(run_dir / "exclusions.csv", diagnostics["exclusions"])

    all_trades, all_rejected, all_missing = [], [], []
    quarterly, yearly, equity = [], [], []
    for mode, by_portfolio in results.items():
        for portfolio, by_cost in by_portfolio.items():
            for cost, result in by_cost.items():
                all_trades.extend(result["trades"])
                all_rejected.extend({"mode": mode, "portfolio": portfolio,
                                     "cost_model": cost, **row}
                                    for row in result["rejected"])
                all_missing.extend({"mode": mode, "portfolio": portfolio,
                                    "cost_model": cost, **row}
                                   for row in result["missing_outcomes"])
                summary = comparison["summaries"][mode][portfolio][cost]
                quarterly.extend({"mode": mode, "portfolio": portfolio,
                                  "cost_model": cost, **row}
                                 for row in summary["quarterly"])
                yearly.extend({"mode": mode, "portfolio": portfolio,
                               "cost_model": cost, **row}
                              for row in summary["yearly"])
                equity.extend({"mode": mode, "portfolio": portfolio,
                               "cost_model": cost, **row}
                              for row in summary["equity_curve"])

    _write_csv(run_dir / "trades.csv", all_trades)
    _write_csv(run_dir / "rejected_orders.csv", all_rejected)
    _write_csv(run_dir / "missing_outcome_data.csv", all_missing)
    _write_csv(run_dir / "quarterly.csv", quarterly)
    _write_csv(run_dir / "yearly.csv", yearly)
    _write_csv(run_dir / "equity_curves.csv", equity)
    return run_dir


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


def _print_qualification(comparison: dict, run_dir: Path) -> None:
    print("\n  Qualification:")
    for name, result in comparison["qualifications"].items():
        verdict = "PASS" if result["passed"] else "FAIL"
        print(f"    {name:<18} {verdict}")
        if result["failed_checks"]:
            print(f"      failed: {', '.join(result['failed_checks'])}")
    if not comparison["passing_markets"]:
        print("\n  No strategy qualified. Automatic entry remains off.")
    print(f"\n  Results written to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence-first whole-bot strategy comparison")
    parser.add_argument("--start", default=str(DEFAULT_START))
    parser.add_argument("--end", default=str(DEFAULT_END))
    parser.add_argument("--stock-symbols", nargs="+", default=None)
    parser.add_argument("--crypto-symbols", nargs="+", default=None)
    parser.add_argument("--skip-crypto-7day", action="store_true")
    parser.add_argument("--determinism-check", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="AAPL/MSFT/SPY + BTC/ETH over June 2026")
    args = parser.parse_args()

    if args.smoke:
        start_date, end_date = date(2026, 6, 1), date(2026, 6, 30)
        stock_symbols = ["AAPL", "MSFT", "SPY"]
        crypto_symbols = ["BTC-USD", "ETH-USD"]
        label = args.label or "api_smoke"
    else:
        start_date, end_date = date.fromisoformat(args.start), date.fromisoformat(args.end)
        stock_symbols = args.stock_symbols or list(UNIVERSE)
        crypto_symbols = args.crypto_symbols or list(CRYPTO)
        label = args.label

    run_comparison(
        stock_symbols=stock_symbols,
        crypto_symbols=crypto_symbols,
        start_date=start_date,
        end_date=end_date,
        include_crypto_7day=not args.skip_crypto_7day,
        determinism_check=args.determinism_check,
        label=label,
    )


if __name__ == "__main__":
    main()
