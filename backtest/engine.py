"""
Chronological ORB backtest simulation engine.

Reuses, byte-for-byte, the same signal logic (utils/orb_signals.py), cost
model (utils/cost_model.py), and Alpaca feed/adjustment rules
(utils/alpaca_bars.py via backtest/data_cache.py) as the live day-trading
nodes, so backtested and live results can never diverge on the signal itself
- only on how far back in history this is run.

Fetch design (session-date-major, multi-symbol batched - see the
day-trading-mode plan's fetch-batching fix): daily reference bars are
fetched ONCE per symbol for the whole span (including a 14-trading-day
warm-up before start_date). Opening 5-min windows (both IEX and SIP - SIP
for the #3 overlap check) are fetched ONE multi-symbol request per session
date, walking chronologically, so an early session's own opening bar is
already cached by the time a later session needs it as one of its own
"prior 14" baseline inputs. 1-minute outcome bars are fetched, batched
across symbols, only for that session's top-N SELECTED candidates, after
ranking - never per-candidate, never a wide per-symbol range. Total request
count for a full 2-year, 142-symbol run is on the order of 1,500 requests
(a handful of daily-history requests + ~500 IEX opening + ~500 SIP opening
+ up to ~500 one-minute requests), not tens of millions of bars downloaded.

Portfolio accounting (see the day-trading-mode plan's portfolio-terminology
fix): this is a RANK-ORDER RESERVATION MODEL, not full trigger-time event
simulation. At each session's shared 9:35 decision point, `admitted_orders`
are reserved in rank order against `max_concurrent_positions` and
`max_gross_exposure_x` (evolving equity, not a fixed starting balance).
`triggered_trades` are the subset of admitted orders whose entry actually
triggered intraday - this reflects that orders are ARMED at 9:35, not
guaranteed filled. This is legitimate for this specific strategy only
because there is a single decision point per session (no new entries are
ever generated after 9:35, so nothing is ever "queued behind" an early
exit) - true concurrent-position enforcement in general would require
event-time simulation, which this deliberately is not.

Scope caveats, stated here so every caller/report inherits them:
  - This backtest tests the bot's own static 142-symbol universe
    (config/universe.py::UNIVERSE), not the ~7,000-symbol universe the
    original ORB research used.
  - The primary out-of-sample evaluation should cover 2024 onward.
  - This is "rolling out-of-sample evaluation," not walk-forward parameter
    optimization - no parameters are fit/trained; the exact spec rules are
    fixed inputs, evaluated across many periods to see if performance holds.
"""
from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Optional

import pandas as pd

from backtest.data_cache import get_daily_bars_multi, get_intraday_bars_multi
from backtest.portfolio import PortfolioConfig, size_trade
from utils.alpaca_bars import first_regular_session_bar
from utils.cost_model import CostModel, apply_entry_cost, apply_exit_cost
from utils.market_calendar import prior_trading_days, session_for, trading_days_between
from utils.orb_signals import (
    build_candidate_fields,
    compute_daily_reference_stats,
    rank_and_select,
    simulate_intraday_outcome,
)

_LOOKBACK_SESSIONS = 14
_TOP_N = 20


# -- Candidate construction (pure, given already-fetched local data) -----------

def _build_candidate(
    symbol: str,
    session_date: date,
    prior_sessions: list[date],
    full_daily_df: pd.DataFrame,
    opening_history_for_symbol: dict[date, pd.Series],
) -> Optional[dict]:
    """
    Builds one candidate from already-fetched, already-local data - no I/O.
    `full_daily_df` is sliced to strictly-prior sessions here (the actual
    look-ahead guard); `opening_history_for_symbol` is a {date: bar} map
    already populated by the session-date-major fetch loop below.
    """
    if full_daily_df is None or full_daily_df.empty:
        return None

    prior_daily = full_daily_df[full_daily_df.index < pd.Timestamp(session_date)]
    opening_volumes = [
        float(opening_history_for_symbol[d]["volume"])
        for d in prior_sessions if d in opening_history_for_symbol
    ]
    stats = compute_daily_reference_stats(prior_daily, opening_volumes, lookback=_LOOKBACK_SESSIONS)
    if stats is None:
        return None

    today_bar = opening_history_for_symbol.get(session_date)
    if today_bar is None:
        return None

    opening_open = float(today_bar["open"])
    opening_high = float(today_bar["high"])
    opening_low = float(today_bar["low"])
    opening_close = float(today_bar["close"])
    opening_volume = float(today_bar["volume"])

    fields = build_candidate_fields(
        opening_open=opening_open, opening_high=opening_high,
        opening_low=opening_low, opening_close=opening_close,
        opening_volume=opening_volume,
        avg_daily_volume_14d=stats["avg_daily_volume_14d"],
        daily_atr_14=stats["daily_atr_14"],
        avg_opening_volume_14d=stats["avg_opening_volume_14d"],
    )

    bar_time = today_bar.name
    bar_time = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time

    return {
        "symbol": symbol,
        "session_date": session_date,
        "opening_bar_time": bar_time,
        "opening_price": opening_open,
        "opening_open": opening_open,
        "opening_high": opening_high,
        "opening_low": opening_low,
        "opening_close": opening_close,
        "opening_volume": opening_volume,
        **stats,
        **fields,
    }


def _price_pnl(outcome: dict, direction: str, cost_model: CostModel) -> dict:
    """Gross and cost-adjusted per-unit P&L, before quantity is applied."""
    if not outcome.get("breakout_triggered"):
        return {"gross_pnl_per_unit": None, "cost_adjusted_pnl_per_unit": None}

    entry = outcome["simulated_entry_price"]
    exit_ = outcome["exit_price"]
    entry_adj = apply_entry_cost(entry, direction, cost_model)
    exit_adj = apply_exit_cost(exit_, direction, cost_model)

    if direction == "long":
        gross = exit_ - entry
        cost_adjusted = exit_adj - entry_adj
    else:
        gross = entry - exit_
        cost_adjusted = entry_adj - exit_adj
    return {"gross_pnl_per_unit": gross, "cost_adjusted_pnl_per_unit": cost_adjusted}


# -- Fetch orchestration (session-date-major, multi-symbol batched) ------------

def _fetch_and_record_opening(
    symbols: list[str], session_date: date, feed: str, history: dict[str, dict[date, pd.Series]]
) -> bool:
    """
    One multi-symbol batch request for `session_date`'s opening 5-min window.
    Returns False (and records nothing) on a request-level failure - the
    caller treats that session as having failed/incomplete data for this
    feed, distinct from "the request succeeded but found nothing," which is
    a normal, informative result and not a failure at all.
    """
    session = session_for(session_date)
    if session is None:
        return False
    window_start = session["open"]
    window_end = window_start + timedelta(minutes=5)
    try:
        bars_by_symbol = get_intraday_bars_multi(symbols, window_start, window_end, minutes=5, feed=feed)
    except Exception:
        return False
    for symbol in symbols:
        bar = first_regular_session_bar(bars_by_symbol.get(symbol, pd.DataFrame()), window_start)
        if bar is not None:
            history[symbol][session_date] = bar
    return True


def _session_overlap_record(
    symbols: list[str],
    session_date: date,
    prior_sessions: list[date],
    daily_bars_by_symbol: dict[str, pd.DataFrame],
    iex_history: dict[str, dict[date, pd.Series]],
    sip_history: dict[str, dict[date, pd.Series]],
    top_n: int,
    iex_fetch_ok: bool,
    sip_fetch_ok: bool,
) -> dict:
    """
    IEX-vs-SIP ranking overlap for one session, using the SAME
    build_candidate_fields() gate the real strategy uses (so dojis and
    filter-failing candidates are excluded identically on both sides), with
    the corrected overlap ratio and precise empty-set handling - see
    backtest/metrics.py's module docstring and the day-trading-mode plan §3.
    """
    if not iex_fetch_ok or not sip_fetch_ok:
        return {
            "session_date": session_date, "valid": False,
            "reason": "missing/incomplete feed data",
            "overlap_rate": None, "jaccard": None,
        }

    def eligible_for(history):
        out = []
        for symbol in symbols:
            candidate = _build_candidate(
                symbol, session_date, prior_sessions,
                daily_bars_by_symbol.get(symbol, pd.DataFrame()), history[symbol],
            )
            if candidate is not None and candidate["passed_filters"]:
                out.append(candidate)
        return out

    iex_eligible = eligible_for(iex_history)
    sip_eligible = eligible_for(sip_history)
    iex_selected = {c["symbol"] for c in rank_and_select(iex_eligible, top_n=top_n) if c["selected"]}
    sip_selected = {c["symbol"] for c in rank_and_select(sip_eligible, top_n=top_n) if c["selected"]}

    intersection = len(iex_selected & sip_selected)
    union = len(iex_selected | sip_selected)

    if not iex_selected and not sip_selected:
        overlap_rate, jaccard = None, None
    elif not iex_selected or not sip_selected:
        overlap_rate, jaccard = 0.0, 0.0
    else:
        overlap_rate = intersection / max(len(iex_selected), len(sip_selected))
        jaccard = intersection / union

    return {
        "session_date": session_date,
        "valid": True,
        "iex_eligible_count": len(iex_eligible),
        "sip_eligible_count": len(sip_eligible),
        "iex_selected_count": len(iex_selected),
        "sip_selected_count": len(sip_selected),
        "intersection_count": intersection,
        "overlap_rate": overlap_rate,
        "jaccard": jaccard,
    }


# -- Main entry point ------------------------------------------------------------

def run_backtest(
    symbols: list[str],
    start_date: date,
    end_date: date,
    scenarios: dict[str, PortfolioConfig],
    cost_models: dict[str, CostModel],
    top_n: int = _TOP_N,
) -> dict:
    """
    Simulate every trading session in [start_date, end_date] across
    `symbols`, reporting results for every (scenario, cost_model)
    combination.

    Returns
    -------
    dict with keys:
      trades              - one row per (admitted order, scenario, cost_model);
                             see the module docstring for eligible/admitted/
                             triggered terminology
      rejected_orders     - admission-time rejections (position-count or
                             exposure-cap cutoffs), not missing-data exclusions
      excluded            - {symbol, session_date, reason} for missing/failed
                             candidate construction (feeds #5's denominator)
      missing_outcome_data - {symbol, session_date, reason} for a selected
                             candidate whose 1-minute outcome data failed to
                             fetch or came back empty - these trades are
                             recorded with exit_reason="outcome_data_missing"
                             and excluded from win-rate/expectancy exactly
                             like a real "no_trigger", but are separately
                             auditable here rather than silently conflated
                             with a genuine "never triggered" result
      daily_candidate_counts - per-session attempted/considered/passing/selected
      overlap_records     - per-session IEX-vs-SIP overlap analysis (#3)
      final_equity        - ending equity per (scenario, cost_model)
    """
    warmup_dates = prior_trading_days(start_date, _LOOKBACK_SESSIONS)
    warmup_start = warmup_dates[0] if warmup_dates else start_date

    daily_bars_by_symbol = get_daily_bars_multi(symbols, warmup_start, end_date)

    iex_history: dict[str, dict[date, pd.Series]] = {s: {} for s in symbols}
    sip_history: dict[str, dict[date, pd.Series]] = {s: {} for s in symbols}

    for wd in warmup_dates:
        _fetch_and_record_opening(symbols, wd, "iex", iex_history)
        _fetch_and_record_opening(symbols, wd, "sip", sip_history)

    trades: list[dict] = []
    rejected_orders: list[dict] = []
    excluded: list[dict] = []
    missing_outcome_data: list[dict] = []
    daily_candidate_counts: list[dict] = []
    overlap_records: list[dict] = []

    equity_state: dict[tuple[str, str], float] = {
        (sname, cname): config.starting_equity
        for sname, config in scenarios.items()
        for cname in cost_models
    }

    total_trading_days = len(trading_days_between(start_date, end_date))
    processed_sessions = 0
    started_at = time.time()

    session_date = start_date
    while session_date <= end_date:
        session = session_for(session_date)
        if session is None:
            session_date += timedelta(days=1)
            continue

        processed_sessions += 1
        elapsed = time.time() - started_at
        avg_per_session = elapsed / processed_sessions
        remaining = max(total_trading_days - processed_sessions, 0)
        eta_seconds = avg_per_session * remaining
        print(f"  [{session_date}] session {processed_sessions}/{total_trading_days}  "
              f"elapsed={elapsed:.0f}s  eta={eta_seconds:.0f}s", flush=True)

        prior_sessions = prior_trading_days(session_date, _LOOKBACK_SESSIONS)
        if len(prior_sessions) < _LOOKBACK_SESSIONS:
            session_date += timedelta(days=1)
            continue

        iex_fetch_ok = _fetch_and_record_opening(symbols, session_date, "iex", iex_history)
        sip_fetch_ok = _fetch_and_record_opening(symbols, session_date, "sip", sip_history)

        candidates = []
        for symbol in symbols:
            try:
                candidate = _build_candidate(
                    symbol, session_date, prior_sessions,
                    daily_bars_by_symbol.get(symbol, pd.DataFrame()), iex_history[symbol],
                )
            except Exception as exc:
                excluded.append({"symbol": symbol, "session_date": session_date, "reason": str(exc)})
                continue
            if candidate is None:
                excluded.append({
                    "symbol": symbol, "session_date": session_date,
                    "reason": "missing opening bar or reference stats",
                })
                continue
            candidates.append(candidate)

        passing = [c for c in candidates if c["passed_filters"]]
        ranked = rank_and_select(passing, top_n=top_n)
        selected = [c for c in ranked if c["selected"]]

        daily_candidate_counts.append({
            "session_date": session_date,
            "attempted": len(symbols),
            "considered": len(candidates),
            "passing": len(passing),
            "selected": len(selected),
        })

        overlap_records.append(_session_overlap_record(
            symbols, session_date, prior_sessions, daily_bars_by_symbol,
            iex_history, sip_history, top_n, iex_fetch_ok, sip_fetch_ok,
        ))

        # Outcome simulation once per selected candidate, batched, shared
        # across every scenario/cost-model (entry/exit price and time don't
        # depend on either - only the dollar P&L scaling does).
        #
        # A failed or empty 1-minute fetch is NOT the same thing as "the
        # breakout genuinely never triggered" - the latter is a real,
        # informative result; the former means we simply don't know what
        # happened. Conflating them would silently bias results toward
        # "no trigger" (zero P&L impact) whenever data happens to be missing,
        # rather than making that gap visible and excludable from analysis.
        outcomes: dict[str, dict] = {}
        if selected:
            window_start = selected[0]["opening_bar_time"] + timedelta(minutes=5)
            try:
                minute_bars_by_symbol = get_intraday_bars_multi(
                    [c["symbol"] for c in selected], window_start, session["close"], minutes=1,
                )
                fetch_failed = False
            except Exception:
                minute_bars_by_symbol = {}
                fetch_failed = True
            for c in selected:
                bars = minute_bars_by_symbol.get(c["symbol"], pd.DataFrame())
                if fetch_failed or bars.empty:
                    outcomes[c["symbol"]] = {
                        "breakout_triggered": False,
                        "exit_reason": "outcome_data_missing",
                        "outcome_data_missing": True,
                    }
                    missing_outcome_data.append({
                        "symbol": c["symbol"], "session_date": session_date,
                        "reason": "1-minute outcome fetch failed" if fetch_failed else "no 1-minute bars returned",
                    })
                else:
                    outcome = simulate_intraday_outcome(
                        bars, entry_trigger=c["entry_trigger_price"],
                        stop_price=c["stop_price"], direction=c["direction"],
                    )
                    outcome["outcome_data_missing"] = False
                    outcomes[c["symbol"]] = outcome

        # Per-(scenario, cost_model) rank-order admission and evolving equity.
        for scenario_name, config in scenarios.items():
            for cost_name, cost_model in cost_models.items():
                key = (scenario_name, cost_name)
                equity = equity_state[key]
                admitted = 0
                reserved_exposure = 0.0
                realized_exposure = 0.0
                day_pnl = 0.0
                session_trade_start = len(trades)

                for c in selected:
                    sizing = size_trade(config, c["entry_trigger_price"], c["stop_price"], equity=equity)
                    proposed_exposure = reserved_exposure + sizing["position_value"]
                    over_position_limit = admitted >= config.max_concurrent_positions
                    over_exposure_limit = proposed_exposure > equity * config.max_gross_exposure_x
                    if over_position_limit or over_exposure_limit:
                        rejected_orders.append({
                            "symbol": c["symbol"], "session_date": session_date,
                            "scenario": scenario_name, "cost_model": cost_name,
                            "reason": "max_concurrent_positions" if over_position_limit else "max_gross_exposure",
                        })
                        continue

                    admitted += 1
                    reserved_exposure = proposed_exposure

                    outcome = outcomes.get(c["symbol"], {
                        "breakout_triggered": False, "exit_reason": "outcome_data_missing", "outcome_data_missing": True,
                    })
                    triggered = bool(outcome.get("breakout_triggered"))
                    pnl_per_unit = _price_pnl(outcome, c["direction"], cost_model)
                    gross_pnl = (
                        pnl_per_unit["gross_pnl_per_unit"] * sizing["quantity"]
                        if pnl_per_unit["gross_pnl_per_unit"] is not None else None
                    )
                    cost_adjusted_pnl = (
                        pnl_per_unit["cost_adjusted_pnl_per_unit"] * sizing["quantity"]
                        if pnl_per_unit["cost_adjusted_pnl_per_unit"] is not None else None
                    )
                    pnl_r = (
                        cost_adjusted_pnl / sizing["risk_amount"]
                        if (cost_adjusted_pnl is not None and sizing["risk_amount"]) else None
                    )

                    realized_position_value = None
                    reservation_overrun = False
                    if triggered:
                        realized_position_value = sizing["quantity"] * outcome["simulated_entry_price"]
                        reservation_overrun = realized_position_value > sizing["position_value"]
                        realized_exposure += realized_position_value
                        day_pnl += cost_adjusted_pnl or 0.0

                    trades.append({
                        **c, **outcome,
                        "scenario": scenario_name,
                        "cost_model": cost_name,
                        "quantity": sizing["quantity"],
                        "reserved_risk_amount": sizing["risk_amount"],
                        "reserved_position_value": sizing["position_value"],
                        "realized_position_value": realized_position_value,
                        "reservation_overrun": reservation_overrun,
                        "triggered": triggered,
                        "gross_pnl": gross_pnl,
                        "cost_adjusted_pnl": cost_adjusted_pnl,
                        "pnl_r": pnl_r,
                        "equity_at_arm_time": equity,
                    })

                # Session-level, conservative worst-case exposure check -
                # NOT a real-time/event-driven measurement, see module docstring.
                exposure_cap_exceeded = realized_exposure > equity * config.max_gross_exposure_x
                if admitted > 0:
                    for t in trades[session_trade_start:]:
                        t["session_worst_case_realized_gross_exposure"] = realized_exposure
                        t["session_exposure_cap_exceeded"] = exposure_cap_exceeded

                equity_state[key] = equity + day_pnl

        session_date += timedelta(days=1)

    return {
        "trades": trades,
        "rejected_orders": rejected_orders,
        "excluded": excluded,
        "missing_outcome_data": missing_outcome_data,
        "daily_candidate_counts": daily_candidate_counts,
        "overlap_records": overlap_records,
        "final_equity": dict(equity_state),
    }
