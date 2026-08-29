"""
Phase 2 Step 5: backtest comparison for crypto_trend_daily_v1 and
crypto_xsec_momentum_v1 against the locked crypto_trend_momentum_v1
results (20260811_102403). Research-only informational report - never
promotes or unlocks any strategy. utils/strategy_registry.py (Tier-1 for
the live SLC system) and utils/research_strategy_registry.py are both
untouched by this script; it only reads utils/strategy_signals.py and
backtest/whole_bot_engine.py.

Run: python -m backtest.run_crypto_variant_comparison
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, time as dtime
from functools import partial
from pathlib import Path

import pandas as pd

import backtest.whole_bot_engine as engine
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run
from config.universe import CRYPTO
from utils.strategy_signals import crypto_trend_daily_v1

DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2026, 8, 9)
RESULTS_DIR = Path(__file__).parent / "results" / "crypto_variant_comparison"

DEAD_SYMBOLS = {
    "FET-USD", "INJ-USD", "JUP-USD", "OP-USD", "SEI-USD", "SUI-USD", "TAO-USD", "TON-USD",
}
BTC_ETH_SOL = ["BTC-USD", "ETH-USD", "SOL-USD"]


def _universe_coverage(daily_ind: dict, start: date, end: date) -> float:
    """Universe-wide usable-daily-bar coverage over [start, end], same
    per-symbol methodology as the Step 0 report. Reported uniformly for
    every variant drawing on the same daily_ind frames (crypto_trend_daily_v1
    AND crypto_xsec_momentum_v1) - neither strategy's own admission/
    rebalance accounting maps cleanly onto the gate's "attempted vs usable"
    concept the way the per-symbol daily gate does, so this is the
    consistent, defensible number to report for both."""
    n_days = (end - start).days + 1
    attempted = usable = 0
    day = start
    while day <= end:
        cutoff = datetime.combine(day - timedelta(days=1), dtime.min)
        for symbol, frame in daily_ind.items():
            # A symbol with ZERO rows ever (e.g. the 8 dead-on-Alpaca
            # symbols from Step 0) has no "pre-inception" to skip past -
            # matching build_signal_calendar()'s own first_usable=None
            # handling, it's ATTEMPTED every day and simply never usable.
            # Only genuine pre-inception (frame exists, starts later) is
            # skipped from the denominator entirely. Silently excluding
            # empty frames here previously inflated coverage by dropping
            # the worst-performing symbols out of the denominator.
            if not frame.empty and cutoff < frame.index[0]:
                continue
            attempted += 1
            if not frame.empty and cutoff in frame.index:
                row = frame.loc[cutoff]
                if pd.notna(row.get("close")) and pd.notna(row.get("atr_14")):
                    usable += 1
        day += timedelta(days=1)
    return usable / attempted if attempted else 0.0


def _bench(hourly_frames, start, end):
    return benchmark_summary(hourly_frames.get("BTC-USD"), start, end, "crypto")


def run_daily_v1(daily_ind, hourly, start, end, entry_mode, mode, portfolio, cost):
    fn = partial(crypto_trend_daily_v1, entry_mode=entry_mode)
    calendar, meta = engine.build_daily_crypto_calendar(daily_ind, hourly, start, end, fn)
    signal_calendar = {d: {"stock": [], "crypto": cands} for d, cands in calendar.items()}
    indicator_frames = {"stock": {}, "crypto": daily_ind}
    result = engine.simulate_portfolio(
        signal_calendar, indicator_frames, start, end, portfolio, cost, mode,
        outcome_simulator=engine.simulate_daily_crypto_order_outcome,
    )
    return result


def run_xsec(daily_ind, hourly, start, end, lookback_days, rebalance_weekday, portfolio, cost):
    config = engine.XsecMomentumConfig(lookback_days=lookback_days, rebalance_weekday=rebalance_weekday)
    return engine.simulate_xsec_momentum_portfolio(daily_ind, hourly, start, end, portfolio, cost, config)


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def build_hourly_btc(start, end):
    hourly = engine.get_crypto_research_bars_multi(
        ["BTC-USD"],
        datetime.combine(start - timedelta(days=90), dtime.min),
        datetime.combine(end, dtime.min),
        amount=1, unit="Hour",
    )
    return {s: engine._normalise_frame(df) for s, df in hourly.items()}


def main(symbols=None, start=DEFAULT_START, end=DEFAULT_END, label="full_universe"):
    symbols = symbols or CRYPTO
    print(f"[{label}] fetching daily bars for {len(symbols)} symbols {start}..{end}")
    daily_bars = engine.fetch_daily_crypto_frames(symbols, start, end)
    daily_ind = engine.build_daily_crypto_indicator_frames(daily_bars)
    hourly = build_hourly_btc(start, end)
    coverage = _universe_coverage(daily_ind, start, end)
    bench = _bench(hourly, start, end)
    print(f"[{label}] universe coverage: {coverage:.1%}")

    rows = []

    # -- crypto_trend_daily_v1: 3 entry_modes x 2 modes x 2 portfolios x 3 costs
    for entry_mode in ("strict_stack", "sma50_rising", "donchian"):
        for mode, mode_label in (("crypto_weekends", "weekend_only"), ("crypto_7day", "all_days")):
            for portfolio in engine.PORTFOLIOS.values():
                for cost in engine.COSTS.values():
                    result = run_daily_v1(daily_ind, hourly, start, end, entry_mode, mode, portfolio, cost)
                    summary = summarize_run(
                        result, starting_equity=portfolio.starting_equity,
                        start_date=start, end_date=end, benchmark=bench,
                    )
                    rows.append({
                        "strategy": "crypto_trend_daily_v1", "variant": entry_mode,
                        "day_mode": mode_label, "portfolio": portfolio.name, "cost_model": cost.name,
                        "coverage_rate": coverage, "summary": summary,
                    })

    # -- crypto_xsec_momentum_v1: 2 lookbacks x 2 portfolios x 3 costs
    # "weekend-only" has no natural meaning for a weekly-rebalance strategy
    # (see the written report) - substituted with rebalance_weekday
    # Monday (0, "weekday_rebalance") vs Saturday (5, "weekend_rebalance")
    # as the closest analogous ablation, clearly labeled as a substitution.
    for lookback_days in (30, 90):
        for weekday, weekday_label in ((0, "weekday_rebalance"), (5, "weekend_rebalance")):
            for portfolio in engine.PORTFOLIOS.values():
                for cost in engine.COSTS.values():
                    result = run_xsec(daily_ind, hourly, start, end, lookback_days, weekday, portfolio, cost)
                    summary = summarize_run(
                        result, starting_equity=portfolio.starting_equity,
                        start_date=start, end_date=end, benchmark=bench,
                    )
                    rows.append({
                        "strategy": "crypto_xsec_momentum_v1", "variant": f"lookback_{lookback_days}",
                        "day_mode": weekday_label, "portfolio": portfolio.name, "cost_model": cost.name,
                        "coverage_rate": coverage, "summary": summary,
                    })

    # Qualification: pair each row's baseline-cost-model summary with its
    # own stressed-cost-model summary (same strategy/variant/day_mode/portfolio).
    by_key = {}
    for row in rows:
        key = (row["strategy"], row["variant"], row["day_mode"], row["portfolio"])
        by_key.setdefault(key, {})[row["cost_model"]] = row
    for key, by_cost in by_key.items():
        if "baseline" in by_cost and "stressed" in by_cost:
            qual = qualify_strategy(by_cost["baseline"]["summary"], by_cost["stressed"]["summary"], coverage)
            by_cost["baseline"]["qualification"] = qual
            by_cost["stressed"]["qualification"] = qual
            if "zero" in by_cost:
                by_cost["zero"]["qualification"] = qual

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{timestamp}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "label": label, "symbols": symbols, "start": str(start), "end": str(end),
            "universe_coverage": coverage, "benchmark": bench, "rows": rows,
        }, f, indent=2, default=_json_default)
    print(f"[{label}] wrote {out_dir / 'results.json'} ({len(rows)} rows)")
    return out_dir


if __name__ == "__main__":
    main()
