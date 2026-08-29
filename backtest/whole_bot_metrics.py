"""Qualification metrics for the evidence-first whole-bot simulator."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def benchmark_summary(frame: pd.DataFrame, start_date: date, end_date: date, market: str) -> dict:
    if frame is None or frame.empty:
        return {"symbol": "SPY" if market == "stock" else "BTC-USD",
                "total_return": None, "sharpe": None, "max_drawdown": None}
    close = frame.loc[
        (frame.index.date >= start_date) & (frame.index.date <= end_date), "close"
    ].resample("1D").last().dropna()
    if len(close) < 2:
        return {"symbol": "SPY" if market == "stock" else "BTC-USD",
                "total_return": None, "sharpe": None, "max_drawdown": None}
    returns = close.pct_change().dropna()
    periods = 252 if market == "stock" else 365
    sharpe = _sharpe(returns.to_numpy(), periods)
    drawdown = close / close.cummax() - 1.0
    return {
        "symbol": "SPY" if market == "stock" else "BTC-USD",
        "total_return": float(close.iloc[-1] / close.iloc[0] - 1.0),
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def summarize_run(
    result: dict,
    *,
    starting_equity: float,
    start_date: date,
    end_date: date,
    benchmark: dict,
) -> dict:
    trades = result["trades"]
    closed = [t for t in trades if t.get("status") == "closed" and t.get("net_pnl") is not None]
    net = np.array([float(t["net_pnl"]) for t in closed], dtype=float)
    pnl_r = np.array([float(t["pnl_r"]) for t in closed if t.get("pnl_r") is not None], dtype=float)
    gross_profit = float(net[net > 0].sum()) if net.size else 0.0
    gross_loss = float(-net[net < 0].sum()) if net.size else 0.0
    total_net = float(net.sum()) if net.size else 0.0

    daily_index = pd.date_range(start_date, end_date, freq="D")
    daily_pnl = pd.Series(0.0, index=daily_index)
    for trade in closed:
        exit_day = pd.Timestamp(trade["exit_time"]).normalize()
        if exit_day in daily_pnl.index:
            daily_pnl.loc[exit_day] += float(trade["net_pnl"])
    marked_rows = result.get("daily_equity") or []
    if marked_rows:
        equity = pd.Series(
            [float(row["equity"]) for row in marked_rows],
            index=pd.to_datetime([row["date"] for row in marked_rows]),
        ).sort_index()
        equity = equity[~equity.index.duplicated(keep="last")]
    else:
        equity = starting_equity + daily_pnl.cumsum()
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mode = result.get("mode") or (trades[0]["mode"] if trades else "unknown")
    periods = 252 if mode == "stock_only" else 365
    sharpe = _sharpe(returns.to_numpy(), periods)
    sortino = _sortino(returns.to_numpy(), periods)
    drawdown = equity / equity.cummax().clip(lower=starting_equity) - 1.0

    quarterly_series = daily_pnl.resample("QE").sum()
    yearly_series = daily_pnl.resample("YE").sum()
    positive_quarter_fraction = (
        float((quarterly_series > 0).sum() / len(quarterly_series))
        if len(quarterly_series) else None
    )
    recent_start = pd.Timestamp(end_date - timedelta(days=365))
    recent_12m = float(daily_pnl[daily_pnl.index >= recent_start].sum())

    per_symbol: dict[str, float] = {}
    for trade in closed:
        per_symbol[trade["symbol"]] = per_symbol.get(trade["symbol"], 0.0) + float(trade["net_pnl"])
    positive_total = sum(value for value in per_symbol.values() if value > 0)
    max_symbol_contribution = (
        max((value for value in per_symbol.values() if value > 0), default=0.0) / positive_total
        if positive_total > 0 else None
    )

    total_traded_notional = sum(
        abs(float(t["quantity"]) * float(t["fill_price"]))
        + abs(float(t["quantity"]) * float(t["exit_price"]))
        for t in closed
    )
    total_gross = sum(float(t["gross_pnl"]) for t in closed)
    break_even_cost_bps = (
        total_gross / total_traded_notional * 10_000.0
        if total_traded_notional else None
    )

    elapsed_years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    ending_equity = starting_equity + total_net
    annualized_return = (
        (ending_equity / starting_equity) ** (1 / elapsed_years) - 1.0
        if ending_equity > 0 else -1.0
    )
    bootstrap_lower = _bootstrap_lower_mean(pnl_r)
    missing_count = len(result.get("missing_outcomes", []))
    approved_count = len(trades)

    return {
        "mode": mode,
        "portfolio": result.get("portfolio") or (trades[0]["portfolio"] if trades else None),
        "cost_model": result.get("cost_model") or (trades[0]["cost_model"] if trades else None),
        "approved_count": approved_count,
        "closed_count": len(closed),
        "win_rate": float((net > 0).mean()) if net.size else None,
        "net_expectancy": float(net.mean()) if net.size else None,
        "average_pnl_r": float(pnl_r.mean()) if pnl_r.size else None,
        "bootstrap_95pct_lower_mean_r": bootstrap_lower,
        "total_net_pnl": total_net,
        "total_return": total_net / starting_equity,
        "annualized_return": annualized_return,
        "profit_factor": gross_profit / gross_loss if gross_loss else (999_999.0 if gross_profit else None),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else None,
        "recent_12m_net_pnl": recent_12m,
        "positive_quarter_fraction": positive_quarter_fraction,
        "max_symbol_profit_contribution": max_symbol_contribution,
        "break_even_cost_bps_per_leg_pool": break_even_cost_bps,
        "missing_outcome_count": missing_count,
        "missing_outcome_rate": missing_count / approved_count if approved_count else 0.0,
        "ambiguous_count": sum(1 for t in closed if t.get("ambiguous")),
        "benchmark": benchmark,
        "quarterly": [
            {"period": str(idx.to_period("Q")), "net_pnl": float(value)}
            for idx, value in quarterly_series.items()
        ],
        "yearly": [
            {"period": str(idx.year), "net_pnl": float(value)}
            for idx, value in yearly_series.items()
        ],
        "per_symbol_net_pnl": per_symbol,
        "equity_curve": [
            {"date": idx.date(), "equity": float(value),
             "drawdown": float(drawdown.loc[idx])}
            for idx, value in equity.items()
        ],
    }


MIN_OOS_CLOSED_TRADES = 30
MIN_PLATEAU_NEIGHBORS = 4
MIN_NEIGHBOR_CLOSED_TRADES = 30
PLATEAU_NEIGHBOR_MEDIAN_FRACTION = 0.75


def _walk_forward_checks(walk_forward: dict) -> dict[str, bool]:
    """Phase 5 Step 4 (approved amendment b): both checks additionally
    require >= MIN_OOS_CLOSED_TRADES OOS closed trades - a nominal OOS
    Sharpe/profit-factor pass on a handful of trades is not evidence,
    matching this project's Phase 3 finding that an OOS pass built on too
    few trades can be a single-period artifact rather than a real result."""
    oos_trades = walk_forward.get("oos_closed_trades") or 0
    enough_trades = oos_trades >= MIN_OOS_CLOSED_TRADES
    oos_sharpe = walk_forward.get("oos_sharpe")
    oos_benchmark_sharpe = walk_forward.get("oos_benchmark_sharpe")
    oos_profit_factor = walk_forward.get("oos_profit_factor")
    return {
        "walk_forward_oos_sharpe_beats_benchmark": (
            enough_trades
            and oos_sharpe is not None
            and oos_benchmark_sharpe is not None
            and oos_sharpe >= oos_benchmark_sharpe
        ),
        "walk_forward_oos_profit_factor_at_least_1_0": (
            enough_trades
            and oos_profit_factor is not None
            and oos_profit_factor >= 1.0
        ),
    }


def _sensitivity_plateau_check(sensitivity: dict) -> bool:
    """Phase 5 Step 4 (approved amendment a): plateau check anchored to the
    selected cell, not a ratio that breaks down near zero/negative Sharpe -
    passes iff the evaluable neighbors' median Sharpe is at least
    PLATEAU_NEIGHBOR_MEDIAN_FRACTION of the selected cell's own Sharpe.

    Only in-grid neighbors are evaluable (an edge/corner selected cell has
    fewer than 4 in-grid neighbors and fails outright - a plateau claim
    needs a real surrounding grid, not extrapolation past its edge). A
    neighbor with fewer than MIN_NEIGHBOR_CLOSED_TRADES trades is not
    dropped from the evaluable count ("missing") - its thin sample fails
    the whole check outright ("failing"), since folding an unreliable
    Sharpe into the median would silently launder a low-confidence result
    into a passing plateau claim."""
    selected_sharpe = sensitivity["selected_sharpe"]
    evaluable = [n for n in sensitivity["neighbors"] if n.get("in_grid")]
    if len(evaluable) < MIN_PLATEAU_NEIGHBORS:
        return False
    if any((n.get("closed_trades") or 0) < MIN_NEIGHBOR_CLOSED_TRADES for n in evaluable):
        return False
    neighbor_sharpes = [n["sharpe"] for n in evaluable if n.get("sharpe") is not None]
    if len(neighbor_sharpes) < MIN_PLATEAU_NEIGHBORS:
        return False
    median_neighbor_sharpe = float(np.median(neighbor_sharpes))
    return median_neighbor_sharpe >= PLATEAU_NEIGHBOR_MEDIAN_FRACTION * selected_sharpe


def qualify_strategy(
    baseline: dict,
    stressed: dict,
    coverage_rate: float | None,
    *,
    walk_forward: dict | None = None,
    sensitivity: dict | None = None,
) -> dict:
    """walk_forward and sensitivity are optional (Phase 5 Step 4) - omitting
    either (the default) reproduces every pre-existing call site and every
    historical qualification run's shape exactly, with none of the 3 new
    checks present. Passing them adds:
      walk_forward: {"oos_closed_trades": int, "oos_sharpe": float|None,
                      "oos_benchmark_sharpe": float|None,
                      "oos_profit_factor": float|None}
      sensitivity: {"selected_sharpe": float, "neighbors": [
                      {"in_grid": bool, "sharpe": float|None,
                       "closed_trades": int}, ...]}
    """
    checks = {
        "closed_trades_at_least_100": baseline["closed_count"] >= 100,
        "data_coverage_at_least_95pct": coverage_rate is not None and coverage_rate >= 0.95,
        "missing_outcomes_below_1pct": baseline["missing_outcome_rate"] < 0.01,
        "baseline_expectancy_positive": (baseline["net_expectancy"] or 0.0) > 0,
        "stressed_expectancy_positive": (stressed["net_expectancy"] or 0.0) > 0,
        "baseline_profit_factor_at_least_1_15": (
            baseline["profit_factor"] is not None and baseline["profit_factor"] >= 1.15
        ),
        "sharpe_at_least_1": baseline["sharpe"] is not None and baseline["sharpe"] >= 1.0,
        "sharpe_beats_benchmark": (
            baseline["sharpe"] is not None
            and baseline["benchmark"].get("sharpe") is not None
            and baseline["sharpe"] > baseline["benchmark"]["sharpe"]
        ),
        "max_drawdown_no_more_than_15pct": (
            baseline["max_drawdown"] is not None and baseline["max_drawdown"] >= -0.15
        ),
        "recent_12m_positive": baseline["recent_12m_net_pnl"] > 0,
        "positive_quarters_at_least_60pct": (
            baseline["positive_quarter_fraction"] is not None
            and baseline["positive_quarter_fraction"] >= 0.60
        ),
        "bootstrap_lower_mean_r_positive": (
            baseline["bootstrap_95pct_lower_mean_r"] is not None
            and baseline["bootstrap_95pct_lower_mean_r"] > 0
        ),
        "single_symbol_profit_no_more_than_25pct": (
            baseline["max_symbol_profit_contribution"] is not None
            and baseline["max_symbol_profit_contribution"] <= 0.25
        ),
    }
    if walk_forward is not None:
        checks.update(_walk_forward_checks(walk_forward))
    if sensitivity is not None:
        checks["sensitivity_plateau_within_25pct_of_neighbor_median"] = _sensitivity_plateau_check(sensitivity)
    failed = [name for name, passed in checks.items() if not passed]
    return {"passed": not failed, "checks": checks, "failed_checks": failed}


def _sharpe(values: np.ndarray, annual_periods: int) -> float | None:
    clean = values[np.isfinite(values)]
    if clean.size < 2 or np.std(clean, ddof=1) == 0:
        return None
    return float(np.mean(clean) / np.std(clean, ddof=1) * np.sqrt(annual_periods))


def _sortino(values: np.ndarray, annual_periods: int) -> float | None:
    clean = values[np.isfinite(values)]
    downside = clean[clean < 0]
    if clean.size < 2 or downside.size < 2 or np.std(downside, ddof=1) == 0:
        return None
    return float(np.mean(clean) / np.std(downside, ddof=1) * np.sqrt(annual_periods))


def _bootstrap_lower_mean(values: np.ndarray, samples: int = 10_000) -> float | None:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return None
    rng = np.random.default_rng(42)
    means = np.empty(samples, dtype=float)
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        draw = rng.choice(clean, size=(count, clean.size), replace=True)
        means[start:start + count] = draw.mean(axis=1)
    return float(np.quantile(means, 0.025))
