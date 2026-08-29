"""
Reporting for backtest/engine.py results - aggregate stats, long/short
split, missing-data reporting, and IEX-vs-SIP overlap aggregation.

Terminology note: this is "rolling out-of-sample evaluation," not
walk-forward parameter optimization - see backtest/engine.py's module
docstring for why. Trade-level terminology (eligible_signals / admitted_orders
/ triggered_trades, reserved vs. realized exposure) is also defined there -
`trades` here means admitted orders (rejections are tracked separately in
engine.py's `rejected_orders`), and `admitted_count`/`triggered_count` below
follow that same distinction.
"""
from __future__ import annotations

from typing import Optional

UNIVERSE_CAVEAT = (
    "This backtest evaluates the bot's own static 142-symbol universe "
    "(config/universe.py::UNIVERSE), NOT the ~7,000-symbol universe the "
    "original ORB research used. Results characterize whether this works "
    "for THIS bot's existing symbol list - a materially narrower claim than "
    "'this is a generally profitable market-wide strategy.'"
)


def summarize(trades: list[dict], scenario: str, cost_model: str) -> dict:
    """Aggregate stats for one (scenario, cost_model) slice of ADMITTED orders."""
    subset = [t for t in trades if t["scenario"] == scenario and t["cost_model"] == cost_model]
    triggered = [t for t in subset if t.get("breakout_triggered")]
    closed = [t for t in triggered if t.get("cost_adjusted_pnl") is not None]

    if not closed:
        return {
            "scenario": scenario, "cost_model": cost_model,
            "admitted_count": len(subset), "triggered_count": len(triggered), "closed_count": 0,
            "win_rate": None, "avg_pnl_r": None, "expectancy": None, "total_pnl": None,
            "reservation_overrun_count": sum(1 for t in subset if t.get("reservation_overrun")),
        }

    wins = [t for t in closed if t["cost_adjusted_pnl"] > 0]
    total_pnl = sum(t["cost_adjusted_pnl"] for t in closed)
    avg_pnl_r = sum(t["pnl_r"] for t in closed if t["pnl_r"] is not None) / len(closed)

    return {
        "scenario": scenario,
        "cost_model": cost_model,
        "admitted_count": len(subset),
        "triggered_count": len(triggered),
        "closed_count": len(closed),
        "win_rate": len(wins) / len(closed),
        "avg_pnl_r": avg_pnl_r,
        "expectancy": total_pnl / len(closed),
        "total_pnl": total_pnl,
        "ambiguous_count": sum(1 for t in closed if t.get("outcome_ambiguous")),
        "reservation_overrun_count": sum(1 for t in subset if t.get("reservation_overrun")),
        "exposure_cap_exceeded_sessions": len({
            t["session_date"] for t in subset if t.get("session_exposure_cap_exceeded")
        }),
    }


def summarize_by_direction(trades: list[dict], scenario: str, cost_model: str) -> dict:
    """
    Long vs. short performance, reported SEPARATELY - required because this
    bot can currently only submit long stock orders live, so short-side
    edge (if any) is not actionable without separate engineering.
    """
    subset = [t for t in trades if t["scenario"] == scenario and t["cost_model"] == cost_model]
    result = {}
    for direction in ("long", "short"):
        directional = [t for t in subset if t.get("direction") == direction]
        closed = [t for t in directional if t.get("cost_adjusted_pnl") is not None]
        if not closed:
            result[direction] = {"admitted_count": len(directional), "closed_count": 0, "win_rate": None, "expectancy": None}
            continue
        wins = [t for t in closed if t["cost_adjusted_pnl"] > 0]
        result[direction] = {
            "admitted_count": len(directional),
            "closed_count": len(closed),
            "win_rate": len(wins) / len(closed),
            "expectancy": sum(t["cost_adjusted_pnl"] for t in closed) / len(closed),
        }
    return result


def missing_data_report(excluded: list[dict], total_symbol_days: int) -> dict:
    """
    Symbol-day exclusion rate - the denominator MUST be the full attempted
    population (see backtest/engine.py's `daily_candidate_counts[i]["attempted"]`,
    summed by the caller), not just successfully-built candidates, or this
    silently understates (or wildly overstates) the true exclusion rate.
    """
    if total_symbol_days == 0:
        return {"excluded_count": len(excluded), "excluded_rate": None, "reasons": {}}
    reasons: dict[str, int] = {}
    for e in excluded:
        reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    return {
        "excluded_count": len(excluded),
        "excluded_rate": len(excluded) / total_symbol_days,
        "reasons": reasons,
    }


def equity_curve(trades: list[dict], scenario: str, cost_model: str, starting_equity: float) -> list[dict]:
    """Chronological equity curve for one (scenario, cost_model) slice."""
    subset = sorted(
        (t for t in trades if t["scenario"] == scenario and t["cost_model"] == cost_model
         and t.get("cost_adjusted_pnl") is not None),
        key=lambda t: t["session_date"],
    )
    equity = starting_equity
    curve = []
    peak = starting_equity
    for t in subset:
        equity += t["cost_adjusted_pnl"]
        peak = max(peak, equity)
        drawdown_pct = (equity - peak) / peak if peak else 0.0
        curve.append({"session_date": t["session_date"], "equity": equity, "drawdown_pct": drawdown_pct})
    return curve


def aggregate_overlap(overlap_records: list[dict]) -> dict:
    """
    Aggregates backtest/engine.py's per-session IEX-vs-SIP overlap records
    (see engine.py::_session_overlap_record) into a run-level summary.

    Averaging rule (precise, see the day-trading-mode plan §3): only
    `valid=True` sessions are eligible at all. Of those, sessions where
    BOTH feeds selected zero candidates (`overlap_rate is None`) are
    excluded from the average - genuinely nothing to compare. Sessions
    where only one feed selected zero (`overlap_rate == 0.0`) ARE included
    and pull the average down - that's real disagreement, not a data gap.
    Sessions with a failed/incomplete fetch (`valid=False`) are excluded
    entirely, distinct from both of the above.
    """
    total = len(overlap_records)
    valid = [r for r in overlap_records if r.get("valid")]
    invalid_count = total - len(valid)
    comparable = [r for r in valid if r.get("overlap_rate") is not None]
    both_empty_count = len(valid) - len(comparable)

    avg_overlap_rate: Optional[float] = None
    avg_jaccard: Optional[float] = None
    if comparable:
        avg_overlap_rate = sum(r["overlap_rate"] for r in comparable) / len(comparable)
        avg_jaccard = sum(r["jaccard"] for r in comparable) / len(comparable)

    return {
        "total_sessions": total,
        "valid_sessions": len(valid),
        "invalid_sessions": invalid_count,
        "both_empty_sessions": both_empty_count,
        "comparable_sessions": len(comparable),
        "avg_overlap_rate": avg_overlap_rate,
        "avg_jaccard": avg_jaccard,
    }
