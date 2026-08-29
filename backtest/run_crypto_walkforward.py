"""
Phase 3 Step 5: walk-forward out-of-sample validation for the two leading
crypto strategy variants. Parameters (crypto_trend_daily_v1's entry_mode,
crypto_xsec_momentum_v1's lookback_days) are selected using ONLY the
in-sample window (2022-01-01 - 2024-06-30), by baseline-cost Sharpe, then
evaluated on the out-of-sample window (2024-07-01 - 2026-08-09) WITHOUT
retuning - the selected parameters are frozen before the OOS window is
ever touched.

Universe is the 14-symbol "live" universe (22 minus the 8 symbols with
zero Alpaca coverage - FET/INJ/JUP/OP/SEI/SUI/TAO/TON) for
crypto_trend_daily_v1, and the 10 highest-coverage symbols for
crypto_xsec_momentum_v1 (matching the genuine, non-tautological N=3
selection rerun from the Step 3 addendum).

Run: python -m backtest.run_crypto_walkforward
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, time as dtime
from functools import partial
from pathlib import Path

import backtest.whole_bot_engine as engine
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run
from utils.strategy_signals import crypto_trend_daily_v1

IN_SAMPLE_START = date(2022, 1, 1)
IN_SAMPLE_END = date(2024, 6, 30)
OUT_OF_SAMPLE_START = date(2024, 7, 1)
OUT_OF_SAMPLE_END = date(2026, 8, 9)
FULL_START = IN_SAMPLE_START
FULL_END = OUT_OF_SAMPLE_END

DEAD_SYMBOLS = {
    "FET-USD", "INJ-USD", "JUP-USD", "OP-USD", "SEI-USD", "SUI-USD", "TAO-USD", "TON-USD",
}
RESULTS_DIR = Path(__file__).parent / "results" / "crypto_walkforward"


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return str(obj)


def _half_year_pnl(trades: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in trades:
        if t.get("net_pnl") is None:
            continue
        d = t.get("exit_time") or t.get("exit_date")
        if isinstance(d, datetime):
            d = d.date()
        half = f"{d.year}H{1 if d.month <= 6 else 2}"
        bucket = out.setdefault(half, {"net_pnl": 0.0, "n": 0})
        bucket["net_pnl"] += float(t["net_pnl"])
        bucket["n"] += 1
    return dict(sorted(out.items()))


def _run_daily_v1(daily_ind, hourly, universe_live, entry_mode, start, end, portfolio, cost):
    fn = partial(crypto_trend_daily_v1, entry_mode=entry_mode)
    calendar, meta = engine.build_daily_crypto_calendar(daily_ind, hourly, start, end, fn)
    signal_calendar = {d: {"stock": [], "crypto": cands} for d, cands in calendar.items()}
    indicator_frames = {"stock": {}, "crypto": daily_ind}
    result = engine.simulate_portfolio(
        signal_calendar, indicator_frames, start, end, portfolio, cost, "crypto_weekends",
        outcome_simulator=engine.simulate_daily_crypto_order_outcome,
    )
    return result, meta["coverage"]


def _run_xsec(daily_ind, hourly, lookback_days, start, end, portfolio, cost):
    config = engine.XsecMomentumConfig(lookback_days=lookback_days, rebalance_weekday=5, top_n=3)
    return engine.simulate_xsec_momentum_portfolio(daily_ind, hourly, start, end, portfolio, cost, config)


def main():
    from config.universe import CRYPTO

    live_universe = [s for s in CRYPTO if s not in DEAD_SYMBOLS]
    top10_universe = [
        "BTC-USD", "ETH-USD", "AAVE-USD", "AVAX-USD", "DOGE-USD",
        "LINK-USD", "UNI-USD", "XRP-USD", "PEPE-USD", "ARB-USD",
    ]
    portfolio = engine.PORTFOLIOS["current_1pct"]
    cost = engine.COSTS["baseline"]

    daily_bars_live = engine.fetch_daily_crypto_frames(live_universe, FULL_START, FULL_END)
    daily_ind_live = engine.build_daily_crypto_indicator_frames(daily_bars_live)
    daily_bars_top10 = engine.fetch_daily_crypto_frames(top10_universe, FULL_START, FULL_END)
    daily_ind_top10 = engine.build_daily_crypto_indicator_frames(daily_bars_top10)
    hourly = engine.get_crypto_research_bars_multi(
        ["BTC-USD"],
        datetime.combine(FULL_START - timedelta(days=90), dtime.min),
        datetime.combine(FULL_END, dtime.min),
        amount=1, unit="Hour",
    )
    hourly = {s: engine._normalise_frame(df) for s, df in hourly.items()}

    bench_is = benchmark_summary(hourly.get("BTC-USD"), IN_SAMPLE_START, IN_SAMPLE_END, "crypto")
    bench_oos = benchmark_summary(hourly.get("BTC-USD"), OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, "crypto")

    # -- Step 1: select entry_mode / lookback_days on the in-sample window only --
    is_selection = {}
    for entry_mode in ("strict_stack", "sma50_rising", "donchian"):
        result, coverage = _run_daily_v1(
            daily_ind_live, hourly, live_universe, entry_mode,
            IN_SAMPLE_START, IN_SAMPLE_END, portfolio, cost,
        )
        s = summarize_run(result, starting_equity=100_000.0, start_date=IN_SAMPLE_START, end_date=IN_SAMPLE_END, benchmark=bench_is)
        is_selection[("daily_v1", entry_mode)] = s["sharpe"] if s["sharpe"] is not None else float("-inf")

    for lookback_days in (30, 90):
        result = _run_xsec(daily_ind_top10, hourly, lookback_days, IN_SAMPLE_START, IN_SAMPLE_END, portfolio, cost)
        s = summarize_run(result, starting_equity=100_000.0, start_date=IN_SAMPLE_START, end_date=IN_SAMPLE_END, benchmark=bench_is)
        is_selection[("xsec", lookback_days)] = s["sharpe"] if s["sharpe"] is not None else float("-inf")

    best_daily_v1_mode = max(
        ("strict_stack", "sma50_rising", "donchian"),
        key=lambda m: is_selection[("daily_v1", m)],
    )
    best_xsec_lookback = max((30, 90), key=lambda lb: is_selection[("xsec", lb)])

    print("In-sample parameter selection (frozen before touching OOS):")
    print(f"  daily_v1 entry_mode -> {best_daily_v1_mode} (Sharpe {is_selection[('daily_v1', best_daily_v1_mode)]:.3f})")
    print(f"  xsec lookback_days  -> {best_xsec_lookback} (Sharpe {is_selection[('xsec', best_xsec_lookback)]:.3f})")

    # -- Step 2: evaluate the FROZEN selections on both windows -----------------
    cells = {}

    for label, start, end, bench in (
        ("in_sample", IN_SAMPLE_START, IN_SAMPLE_END, bench_is),
        ("out_of_sample", OUT_OF_SAMPLE_START, OUT_OF_SAMPLE_END, bench_oos),
    ):
        result, coverage = _run_daily_v1(
            daily_ind_live, hourly, live_universe, best_daily_v1_mode, start, end, portfolio, cost,
        )
        s = summarize_run(result, starting_equity=100_000.0, start_date=start, end_date=end, benchmark=bench)
        q = qualify_strategy(s, s, coverage["coverage_rate"] or 0.0)
        cells[("daily_v1", label)] = {
            "summary": s, "qualification": q, "coverage": coverage,
            "half_year_pnl": _half_year_pnl(result["trades"]),
        }

        result_x = _run_xsec(daily_ind_top10, hourly, best_xsec_lookback, start, end, portfolio, cost)
        s_x = summarize_run(result_x, starting_equity=100_000.0, start_date=start, end_date=end, benchmark=bench)
        cells[("xsec", label)] = {
            "summary": s_x, "half_year_pnl": _half_year_pnl(result_x["trades"]),
        }

    # -- Report ------------------------------------------------------------------
    for strat in ("daily_v1", "xsec"):
        print(f"\n=== {strat} ({'sma50_rising' if strat=='daily_v1' else f'lookback_{best_xsec_lookback}'}) ===")
        for label in ("in_sample", "out_of_sample"):
            s = cells[(strat, label)]["summary"]
            print(
                f"  {label:15s} trades={s['closed_count']:4d} net_pnl={s['total_net_pnl']:10.2f} "
                f"sharpe={s['sharpe']!r:>8} maxdd={s['max_drawdown']!r} "
                f"recent_12m={s['recent_12m_net_pnl']!r} pos_qtr={s['positive_quarter_fraction']!r} "
                f"bootstrap_lower={s['bootstrap_95pct_lower_mean_r']!r}"
            )
        print("  OOS half-year P&L:", cells[(strat, "out_of_sample")]["half_year_pnl"])

    print(f"\nBTC buy-and-hold OOS: sharpe={bench_oos['sharpe']} total_return={bench_oos['total_return']} maxdd={bench_oos['max_drawdown']}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {
        f"{strat}__{label}": data for (strat, label), data in cells.items()
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump({
            "in_sample_window": [str(IN_SAMPLE_START), str(IN_SAMPLE_END)],
            "out_of_sample_window": [str(OUT_OF_SAMPLE_START), str(OUT_OF_SAMPLE_END)],
            "selected_daily_v1_entry_mode": best_daily_v1_mode,
            "selected_xsec_lookback_days": best_xsec_lookback,
            "in_sample_sharpe_by_candidate": {f"{k[0]}:{k[1]}": v for k, v in is_selection.items()},
            "btc_buy_and_hold_oos": bench_oos,
            "cells": serializable,
        }, f, indent=2, default=_json_default)
    print(f"\nWrote {out_dir / 'results.json'}")
    return out_dir


if __name__ == "__main__":
    main()
