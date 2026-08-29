"""Read-only, post-hoc failure analysis for stock_only v1 (rejected).

POST-HOC EXPLORATORY ANALYSIS of the same dataset that already produced the
qualification verdict (run 20260811_142144). Any pattern surfaced here is a
HYPOTHESIS for stock_trend_momentum_v2, not a confirmed finding - v2's
actual rules must be predefined before looking at new data and evaluated
through walk-forward or forward-paper evidence, never justified by
re-pointing back at this same 2022-2026 dataset.

Read-only: never imports strategy_registry, run_pipeline, any node, or a
trading client. Never writes inside the source run directory - all output
goes to backtest/results/failure_analysis/<run_id>_<analysis_timestamp>/.
Bar data for MFE/MAE comes exclusively from backtest/readonly_bar_cache.py,
which cannot fetch from Alpaca.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.readonly_bar_cache import read_cached_bars_or_none
from backtest.whole_bot_engine import _snapshot, _stock_regular_bars, indicator_frame
from config.universe import SECTOR_MAP

DEFAULT_RUN_DIR = Path(__file__).parent / "results" / "whole_bot" / "20260811_142144"
OUTPUT_ROOT = Path(__file__).parent / "results" / "failure_analysis"
RECONCILE_TOLERANCE = 0.01


# -- Loading -------------------------------------------------------------

DATE_COLUMNS = ["decision_time", "signal_bar_end", "filled_at", "exit_time"]


def load_run(run_dir: Path) -> dict:
    # read_csv's parse_dates is not relied on here - its cross-pandas-version
    # behavior is inconsistent (e.g. silently a no-op on pandas 3.0.3), so
    # date columns are converted explicitly and verified below instead.
    trades = pd.read_csv(run_dir / "trades.csv")
    for column in DATE_COLUMNS:
        # format="mixed": end_of_test exits use datetime.combine(end_date,
        # time.max), which serializes WITH microseconds
        # ("...23:59:59.999999"), unlike every other exit_time value - a
        # single inferred format chokes on the mix.
        trades[column] = pd.to_datetime(trades[column], format="mixed")
        if column not in ("exit_time", "filled_at"):
            assert trades[column].notna().all(), f"{column} must never be missing"
    with open(run_dir / "summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    with open(run_dir / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    return {"trades": trades, "summary": summary, "config": config}


def filter_slice(trades: pd.DataFrame, *, mode: str, portfolio: str, cost_model: str) -> pd.DataFrame:
    return trades[
        (trades["mode"] == mode)
        & (trades["portfolio"] == portfolio)
        & (trades["cost_model"] == cost_model)
    ].copy()


def assert_reconciles(computed: float, authoritative: float, label: str) -> None:
    if abs(computed - authoritative) > RECONCILE_TOLERANCE:
        raise AssertionError(f"{label}: computed {computed:.2f} != authoritative {authoritative:.2f}")


# -- Section 1: performance by period / regime / symbol / sector --------

def _period_stats(closed: pd.DataFrame, period: pd.Series) -> pd.DataFrame:
    is_win = closed["net_pnl"] > 0
    out = closed.groupby(period).agg(
        count=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
        expectancy=("net_pnl", "mean"),
    )
    out["win_rate"] = is_win.groupby(period).mean()
    return out.reset_index(names="period")


def realized_performance_by_period(closed: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """Buckets by exit_time - matches summarize_run's yearly/quarterly attribution."""
    return _period_stats(closed, closed["exit_time"].dt.to_period(freq))


def entry_cohort_performance_by_period(closed: pd.DataFrame, freq: str = "Q") -> pd.DataFrame:
    """Buckets by decision_time - diagnostic view, NOT the qualification-authoritative one."""
    return _period_stats(closed, closed["decision_time"].dt.to_period(freq))


def load_spy_hourly(config: dict) -> pd.DataFrame:
    start_date = date.fromisoformat(config["start"])
    end_date = date.fromisoformat(config["end"])
    start = datetime.combine(start_date - timedelta(days=90), datetime.min.time())
    end = datetime.combine(end_date + timedelta(days=2), datetime.min.time())
    bars = read_cached_bars_or_none("SPY", "research-stock-sip-1Hour", start, end)
    if bars is None:
        raise RuntimeError(
            "SPY hourly bars are not fully cached for this run's date range - "
            "cannot compute market regime without a live fetch, which this "
            "script will never perform."
        )
    return bars


def compute_spy_regime(spy_hourly: pd.DataFrame, decision_times: pd.Series) -> pd.Series:
    """UPTREND/DOWNTREND/SIDEWAYS per decision date, via the exact same
    indicator_frame()/_snapshot() classifier this engine uses everywhere,
    including this strategy's own monitor_reversal exit. Keyed by
    decision_time deliberately - regime is a one-time, per-trade
    entry-conditions attribute, not a period-bucketing choice."""
    ind = indicator_frame(spy_hourly)
    labels = []
    for dt in decision_times:
        row = _snapshot(ind, dt.date(), "stock")
        labels.append(str(row["trend"]) if row is not None else "UNKNOWN")
    return pd.Series(labels, index=decision_times.index, name="regime")


def build_sector_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for sector, symbols in SECTOR_MAP.items():
        for symbol in symbols:
            lookup[symbol] = sector
    return lookup


def performance_by_symbol_and_sector(closed: pd.DataFrame, sector_lookup: dict[str, str]) -> pd.DataFrame:
    is_win = closed["net_pnl"] > 0
    out = closed.groupby("symbol").agg(
        count=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
        expectancy=("net_pnl", "mean"),
    )
    out["win_rate"] = is_win.groupby(closed["symbol"]).mean()
    out["sector"] = [sector_lookup.get(sym, "UNKNOWN") for sym in out.index]
    return out.reset_index(names="symbol")


# -- Section 2: entry wait time & unfilled-order capital usage -----------

def entry_wait_time_stats(trades: pd.DataFrame, as_of: datetime) -> dict:
    filled = trades[trades["filled_at"].notna()]
    wait = filled["filled_at"] - filled["decision_time"]
    unfilled = trades[trades["filled_at"].isna()]
    open_days = (as_of - unfilled["decision_time"]).dt.days
    return {
        "filled_wait_min": wait.min(),
        "filled_wait_median": wait.median(),
        "filled_wait_p90": wait.quantile(0.9) if len(wait) else None,
        "filled_wait_max": wait.max(),
        "unfilled_count": int(len(unfilled)),
        "unfilled_roster": [
            {"symbol": sym, "decision_time": str(dec), "days_open_as_of_run_end": int(days)}
            for sym, dec, days in zip(unfilled["symbol"], unfilled["decision_time"], open_days)
        ],
    }


def reserved_capital_ledger(trades: pd.DataFrame) -> pd.DataFrame:
    """Chronological reserved-notional ledger. Release-before-reservation
    tie-breaking at equal timestamps, matching simulate_portfolio's own
    lifecycle: each day's realization loop frees exited positions' cash
    BEFORE that same day's candidate-approval loop reserves new notional.
    """
    events: list[tuple[pd.Timestamp, int, float, str]] = []
    for _, t in trades.iterrows():
        events.append((t["decision_time"], 1, float(t["reserved_notional"]), t["symbol"]))
        if pd.notna(t["exit_time"]):
            events.append((t["exit_time"], 0, -float(t["reserved_notional"]), t["symbol"]))
    events.sort(key=lambda e: (e[0], e[1]))  # marker 0 (release) before 1 (reserve) at equal ts

    running = 0.0
    rows = []
    for ts, marker, delta, symbol in events:
        running += delta
        rows.append({
            "timestamp": ts, "event": "release" if marker == 0 else "reserve",
            "symbol": symbol, "delta": delta, "reserved_total": running,
        })
    return pd.DataFrame(rows)


# -- Section 3: exit reason & holding duration ---------------------------

def exit_reason_stats(closed: pd.DataFrame) -> pd.DataFrame:
    is_win = closed["net_pnl"] > 0
    out = closed.groupby("exit_reason").agg(
        count=("net_pnl", "size"),
        net_pnl=("net_pnl", "sum"),
        avg_pnl_r=("pnl_r", "mean"),
    )
    out["win_rate"] = is_win.groupby(closed["exit_reason"]).mean()
    return out.reset_index()


def holding_duration_buckets(closed: pd.DataFrame) -> pd.DataFrame:
    duration_days = (closed["exit_time"] - closed["filled_at"]).dt.total_seconds() / 86400.0
    bucket = pd.cut(
        duration_days, bins=[-0.001, 1, 3, 7, 30, np.inf],
        labels=["<1d", "1-3d", "3-7d", "7-30d", "30d+"],
    )
    is_win = closed["net_pnl"] > 0
    out = closed.groupby(bucket, observed=False).agg(count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"))
    out["win_rate"] = is_win.groupby(bucket, observed=False).mean()
    return out.reset_index(names="duration_bucket")


# -- Section 4: MFE / MAE - bar-based, cache-only, exit-bar-inclusive -----

@dataclass
class ExcursionResult:
    symbol: str
    mfe_r: float | None
    mae_r: float | None
    missing: bool


def exit_bar_inclusive_end(exit_time: datetime) -> datetime:
    """Floor exit_time to its own bar-minute, then add one minute, so the
    exclusive upper bound always lands one full minute past the START of
    the exit bar - including that bar. Querying [filled_at, exit_time)
    directly would silently exclude the exit bar, which is exactly the bar
    most likely to hold the true extreme (it's where the stop/target was
    reached). Degrades sensibly for end_of_test's artificial
    datetime.combine(end_date, time.max) marker too, rolling to midnight of
    the next calendar day and including every available bar through the
    end of the run.
    """
    floored = exit_time.replace(second=0, microsecond=0)
    return floored + timedelta(minutes=1)


def compute_trade_excursion(trade: pd.Series) -> ExcursionResult:
    risk_per_unit = float(trade["entry"]) - float(trade["stop"])
    if risk_per_unit <= 0:
        return ExcursionResult(trade["symbol"], None, None, True)

    interval = "research-stock-sip-1Minute" if trade["market"] == "stock" else "research-crypto-us-1Minute"
    start = trade["filled_at"]
    end = exit_bar_inclusive_end(trade["exit_time"])
    bars = read_cached_bars_or_none(trade["symbol"], interval, start, end)
    if bars is None or bars.empty:
        return ExcursionResult(trade["symbol"], None, None, True)
    if trade["market"] == "stock":
        bars = _stock_regular_bars(bars)
        if bars.empty:
            return ExcursionResult(trade["symbol"], None, None, True)

    fill_price = float(trade["fill_price"])
    mfe_r = (float(bars["high"].max()) - fill_price) / risk_per_unit
    mae_r = (fill_price - float(bars["low"].min())) / risk_per_unit
    return ExcursionResult(trade["symbol"], mfe_r, mae_r, False)


def mfe_mae_report(closed: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Returns (per-trade excursion table, cache-coverage fraction). Every
    trade is a long (this strategy has no short logic), so MFE/MAE are
    signed accordingly. These are BAR-BASED excursions: minute OHLC does
    not reveal the exact intrabar high/low sequence, so this is a
    bar-resolution approximation, not exact tick-level MFE/MAE."""
    results = [compute_trade_excursion(t) for _, t in closed.iterrows()]
    df = pd.DataFrame([asdict(r) for r in results])
    df["net_pnl"] = closed["net_pnl"].to_numpy()
    coverage = 1.0 - float(df["missing"].mean()) if len(df) else 0.0
    usable = df[~df["missing"]].copy()
    usable["group"] = np.where(usable["net_pnl"] > 0, "winner", "loser")
    return usable, coverage


def summarize_mfe_mae(usable: pd.DataFrame) -> pd.DataFrame:
    if usable.empty:
        return pd.DataFrame(columns=["group", "mfe_r_mean", "mfe_r_median", "mae_r_mean", "mae_r_median", "count"])
    out = usable.groupby("group").agg(
        mfe_r_mean=("mfe_r", "mean"), mfe_r_median=("mfe_r", "median"),
        mae_r_mean=("mae_r", "mean"), mae_r_median=("mae_r", "median"),
        count=("mfe_r", "size"),
    )
    return out.reset_index()


# -- Section 5: cost impact - path-dependent view + fixed-cohort isolation

def path_dependent_cost_comparison(summary: dict) -> pd.DataFrame:
    rows = []
    for cost in ("zero", "baseline", "stressed"):
        s = summary["summaries"]["stock_only"]["current_1pct"][cost]
        rows.append({
            "cost_model": cost, "closed_count": s["closed_count"],
            "net_expectancy": s["net_expectancy"], "total_net_pnl": s["total_net_pnl"],
        })
    return pd.DataFrame(rows)


def fixed_cohort_cost_isolation(baseline_closed: pd.DataFrame) -> pd.DataFrame:
    """Reprices the IDENTICAL baseline closed-trade cohort (same quantity,
    fill_price, exit_price, gross_pnl - no re-simulation) at 0/5/13 bps per
    leg, using the same formula _transaction_cost already uses:
    (entry_notional + exit_notional) * bps / 10_000. This isolates pure
    cost drag from the admission-set differences across cost tiers, which
    the path-dependent comparison above cannot do on its own."""
    rows = []
    for label, bps in (("zero_0bps", 0.0), ("baseline_5bps", 5.0), ("stressed_13bps", 13.0)):
        entry_notional = baseline_closed["quantity"] * baseline_closed["fill_price"]
        exit_notional = baseline_closed["quantity"] * baseline_closed["exit_price"]
        cost = (entry_notional + exit_notional) * bps / 10_000.0
        net_pnl = baseline_closed["gross_pnl"] - cost
        rows.append({
            "label": label, "cost_bps_per_leg": bps,
            "total_net_pnl": float(net_pnl.sum()), "expectancy": float(net_pnl.mean()),
            "count": int(len(baseline_closed)),
        })
    return pd.DataFrame(rows)


# -- Section 6: loss clustering by regime & volatility --------------------

def loss_clustering_by_regime(closed_with_regime: pd.DataFrame) -> pd.DataFrame:
    is_win = closed_with_regime["net_pnl"] > 0
    out = closed_with_regime.groupby("regime").agg(count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"))
    out["win_rate"] = is_win.groupby(closed_with_regime["regime"]).mean()
    return out.reset_index()


def loss_clustering_by_volatility(closed: pd.DataFrame) -> pd.DataFrame:
    # No explicit labels: with few trades or ties, qcut may drop duplicate
    # bin edges and produce fewer than 4 bins, which would mismatch a fixed
    # 4-label list and raise. Auto-generated interval labels stay correct
    # regardless of how many distinct bins survive.
    vol_ratio = closed["atr"] / closed["entry"]
    bucket = pd.qcut(vol_ratio, 4, duplicates="drop")
    is_win = closed["net_pnl"] > 0
    out = closed.groupby(bucket, observed=False).agg(count=("net_pnl", "size"), net_pnl=("net_pnl", "sum"))
    out["win_rate"] = is_win.groupby(bucket, observed=False).mean()
    return out.reset_index(names="volatility_quartile")


# -- Manifest --------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(run_dir: Path, config: dict, cache_db_path: Path | None = None) -> dict:
    hashed_files = ["trades.csv", "rejected_orders.csv", "summary.json", "data_coverage.json", "config.json"]
    if cache_db_path is None:
        cache_db_path = Path(__file__).parent / "cache" / "bars_cache.db"
    return {
        "source_run_dir": str(run_dir),
        "analyzed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file_sha256": {name: sha256_file(run_dir / name) for name in hashed_files},
        "config_internal_sha256_field": config.get("config_sha256"),
        "cache_db_path": str(cache_db_path),
        "cache_db_sha256": sha256_file(cache_db_path),
    }


# -- Orchestration ----------------------------------------------------------

def _df_to_md(df: pd.DataFrame) -> str:
    """Dependency-free markdown table renderer (avoids adding a `tabulate`
    dependency just for report formatting)."""
    if df.empty:
        return "_(no rows)_"
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt_cell(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:,.4f}"
    return str(value)


REPORT_HEADER = """# stock_only v1 failure analysis

**POST-HOC EXPLORATORY ANALYSIS.** This report examines the same dataset
that already produced the qualification verdict for `stock_trend_momentum_v1`
(run `{run_id}`, rejected). Any pattern below - regime clustering, MFE/MAE
shape, cost drag, anything else - is a **hypothesis** for a separately
versioned `stock_trend_momentum_v2`, not a confirmed finding. v2's actual
rules must be predefined before looking at new data and evaluated through
walk-forward or forward-paper evidence. Nothing here justifies a rule by
re-pointing back at this same 2022-2026 dataset.

Automatic stock entries remain disabled regardless of anything in this
report. See `manifest.json` for exact source-file and cache hashes.
"""


def run_analysis(run_dir: Path, output_root: Path = OUTPUT_ROOT) -> Path:
    data = load_run(run_dir)
    trades, summary, config = data["trades"], data["summary"], data["config"]

    current_baseline = filter_slice(trades, mode="stock_only", portfolio="current_1pct", cost_model="baseline")
    closed = current_baseline[current_baseline["status"] == "closed"].copy()

    baseline_summary = summary["summaries"]["stock_only"]["current_1pct"]["baseline"]
    assert_reconciles(float(closed["net_pnl"].sum()), baseline_summary["total_net_pnl"], "baseline total_net_pnl")
    assert len(closed) == baseline_summary["closed_count"], (
        f"closed trade count mismatch: computed {len(closed)} != "
        f"summary.json {baseline_summary['closed_count']}"
    )

    # Section 1
    realized_period = realized_performance_by_period(closed)
    entry_cohort_period = entry_cohort_performance_by_period(closed)
    realized_year_total = float(closed.groupby(closed["exit_time"].dt.year)["net_pnl"].sum().sum())
    assert_reconciles(realized_year_total, baseline_summary["total_net_pnl"], "realized-by-period rollup")

    spy_hourly = load_spy_hourly(config)
    regime = compute_spy_regime(spy_hourly, closed["decision_time"])
    closed_with_regime = closed.assign(regime=regime)
    sector_lookup = build_sector_lookup()
    by_symbol_sector = performance_by_symbol_and_sector(closed, sector_lookup)

    # Section 2
    as_of = date.fromisoformat(config["end"])
    wait_stats = entry_wait_time_stats(current_baseline, datetime.combine(as_of, datetime.min.time()))
    ledger = reserved_capital_ledger(current_baseline)

    # Section 3
    exit_stats = exit_reason_stats(closed)
    duration_buckets = holding_duration_buckets(closed)

    # Section 4
    excursions, coverage = mfe_mae_report(closed)
    mfe_mae_summary = summarize_mfe_mae(excursions)

    # Section 5
    path_dependent = path_dependent_cost_comparison(summary)
    cost_isolation = fixed_cohort_cost_isolation(closed)

    # Section 6
    loss_by_regime = loss_clustering_by_regime(closed_with_regime)
    loss_by_vol = loss_clustering_by_volatility(closed)

    # safe_0_25pct headline-only comparison
    safe_summary = summary["summaries"]["stock_only"]["safe_0_25pct"]["baseline"]

    out_dir = output_root / f"{run_dir.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=False)

    manifest = build_manifest(run_dir, config)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    realized_period.to_csv(out_dir / "by_period_realized.csv", index=False)
    entry_cohort_period.to_csv(out_dir / "by_period_entry_cohort.csv", index=False)
    loss_by_regime.to_csv(out_dir / "by_regime.csv", index=False)
    by_symbol_sector.to_csv(out_dir / "by_symbol_sector.csv", index=False)
    ledger.to_csv(out_dir / "entry_wait_and_capital.csv", index=False)
    pd.concat([exit_stats.assign(_section="exit_reason"),
               duration_buckets.assign(_section="holding_duration")],
              ignore_index=True).to_csv(out_dir / "by_exit_reason_duration.csv", index=False)
    excursions.to_csv(out_dir / "mfe_mae.csv", index=False)
    cost_isolation.to_csv(out_dir / "cost_isolation_fixed_cohort.csv", index=False)
    loss_by_vol.to_csv(out_dir / "loss_clustering_by_volatility.csv", index=False)

    report = [REPORT_HEADER.format(run_id=run_dir.name)]
    report.append(f"## 1. Performance by period, regime, symbol, sector\n")
    report.append(f"Realized (exit-time) net P&L by quarter reconciles to summary.json "
                  f"(total ${realized_year_total:,.2f}).\n")
    report.append(_df_to_md(realized_period))
    report.append("\n**Entry-cohort (decision-time) view - diagnostic, not authoritative:**\n")
    report.append(_df_to_md(entry_cohort_period))
    report.append("\n**Market regime (SPY, decision-time):**\n")
    report.append(_df_to_md(loss_by_regime))
    report.append("\n**By symbol/sector:**\n")
    report.append(_df_to_md(by_symbol_sector.sort_values("net_pnl")))

    report.append("\n## 2. Entry wait time & unfilled-order capital usage\n")
    report.append(f"Filled trades: median wait {wait_stats['filled_wait_median']}, "
                  f"p90 {wait_stats['filled_wait_p90']}, max {wait_stats['filled_wait_max']}.\n")
    report.append(f"Unfilled (censored) orders: {wait_stats['unfilled_count']}\n")
    for row in wait_stats["unfilled_roster"]:
        report.append(f"- {row['symbol']}: pending since {row['decision_time']}, "
                      f"{row['days_open_as_of_run_end']} days open as of run end\n")
    report.append(f"\nPeak reserved capital: ${ledger['reserved_total'].max():,.2f} of "
                  f"$100,000 starting equity. Ending reserved: ${ledger['reserved_total'].iloc[-1]:,.2f}.\n")

    report.append("\n## 3. Exit reason & holding duration\n")
    report.append(_df_to_md(exit_stats))
    report.append("\n")
    report.append(_df_to_md(duration_buckets))

    report.append("\n## 4. Winners'/losers' MFE and MAE (bar-based excursions)\n")
    report.append(
        "Bar-based excursions from minute OHLC - intrabar high/low sequence is not "
        "observable, so this is a resolution approximation, not exact tick-level "
        f"MFE/MAE. Normalized to planned risk (entry - stop). Cache coverage: "
        f"{coverage:.1%} of closed trades had fully cached minute data; the rest "
        "are excluded from this table but counted in the coverage figure.\n"
    )
    report.append(_df_to_md(mfe_mae_summary))

    report.append("\n## 5. Cost impact on expectancy\n")
    report.append("**(a) Path-dependent (what actually happened - trade sets differ "
                  "per cost tier because cost feeds equity -> sizing -> admission):**\n")
    report.append(_df_to_md(path_dependent))
    report.append("\n**(b) Fixed-cohort cost isolation (pure cost drag on one identical "
                  "set of trades):**\n")
    report.append(_df_to_md(cost_isolation))
    report.append(f"\nFor reference, `safe_0_25pct`/baseline: {safe_summary['closed_count']} "
                  f"closed, net P&L ${safe_summary['total_net_pnl']:,.2f}, expectancy "
                  f"${safe_summary['net_expectancy']:.2f}/trade (headline only).\n")

    report.append("\n## 6. Loss clustering by regime & volatility\n")
    report.append(_df_to_md(loss_by_regime))
    report.append("\n")
    report.append(_df_to_md(loss_by_vol))

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only failure analysis for stock_only v1")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    args = parser.parse_args()
    out_dir = run_analysis(Path(args.run_dir))
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
