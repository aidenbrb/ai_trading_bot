"""Research-only runner for the frozen SLC 4-hour/5-minute stock strategy.

This module can read or acquire historical market data. It cannot import or
call an Alpaca trading/order client and never changes the live strategy
registry, pipeline, execution node, or schedulers.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest.slc_data import SLC_CACHE_PATH, load_session_minutes, load_stock_bars
from backtest.whole_bot_engine import COSTS, PORTFOLIOS, TECHNICAL_ONLY_DISCLOSURE
from backtest.whole_bot_metrics import benchmark_summary, qualify_strategy, summarize_run
from config.universe import UNIVERSE
from utils.market_calendar import session_for, trading_days_between
from utils.slc_signals import SlcSignal, generate_signals, simulate_intraday_outcome

REPO_ROOT = Path(__file__).resolve().parents[1]
TRADINGBOT_ROOT = REPO_ROOT.parent
RESULTS_ROOT = Path(__file__).parent / "results" / "slc_4h_5m_stock_v1"
PREREGISTRATION = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_preregistration.md"
AMENDMENT = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_001.md"
EXPECTED_PREREGISTRATION_SHA256 = "d453ae039c3d9145e986ff6cf27ce98af7418b7b5ea6da0d2ddb728e801c3e9d"
EXPECTED_AMENDMENT_SHA256 = "3f46175fe2c801a2fed552e242dc97a963ec885ec9c544fd49bdddbd51d0bf03"
AMENDMENT_002 = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_amendment_002.md"
EXPECTED_AMENDMENT_002_SHA256 = "f3774cf00c2fb6f7afa5d6399bb1a740303ee3a2151acbf7766bd13be652e2a2"
DEFAULT_START = date(2022, 1, 1)
DEFAULT_END = date(2026, 8, 9)
DEPLOYMENT_BASELINE = (
    REPO_ROOT / "research" / "slc_4h_5m_stock_v1_deployment_baseline_20260812.json"
)
EXPECTED_DEPLOYMENT_BASELINE_SHA256 = (
    "67be6b6e64892e1f87e221fb7ff7d24ac114e592245e16415505e39afee649ea"
)

GUARDRAILS = {
    "../run_bot.bat": TRADINGBOT_ROOT / "run_bot.bat",
    "run_pipeline.py": REPO_ROOT / "run_pipeline.py",
    "utils/strategy_registry.py": REPO_ROOT / "utils" / "strategy_registry.py",
    "nodes/execution_node.py": REPO_ROOT / "nodes" / "execution_node.py",
    **{
        f"scripts/{path.name}": path
        for path in sorted((REPO_ROOT / "scripts").glob("*.bat"))
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_digest(value) -> str:
    rendered = json.dumps(value, sort_keys=True, default=str, allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def verify_rule_freeze() -> dict:
    actual = {
        "preregistration": _sha256(PREREGISTRATION),
        "amendment_001": _sha256(AMENDMENT),
        "amendment_002": _sha256(AMENDMENT_002),
    }
    expected = {
        "preregistration": EXPECTED_PREREGISTRATION_SHA256,
        "amendment_001": EXPECTED_AMENDMENT_SHA256,
        "amendment_002": EXPECTED_AMENDMENT_002_SHA256,
    }
    if actual != expected:
        raise RuntimeError(f"SLC frozen-rule hash mismatch: expected={expected}, actual={actual}")
    return actual


def _guardrail_hashes() -> dict[str, str]:
    missing = [name for name, path in GUARDRAILS.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"deployment guardrail file missing: {missing}")
    return {name: _sha256(path) for name, path in GUARDRAILS.items()}


def verify_deployment_baseline() -> dict:
    """Hard-stop on drift that occurred before this research run began."""
    baseline_hash = _sha256(DEPLOYMENT_BASELINE)
    if baseline_hash != EXPECTED_DEPLOYMENT_BASELINE_SHA256:
        raise RuntimeError(
            "SLC deployment baseline file hash mismatch: "
            f"expected={EXPECTED_DEPLOYMENT_BASELINE_SHA256} actual={baseline_hash}"
        )
    baseline = json.loads(DEPLOYMENT_BASELINE.read_text(encoding="utf-8"))
    expected = baseline.get("guardrails")
    current = _guardrail_hashes()
    if expected != current:
        changed = sorted(
            name for name in set(expected or {}) | set(current)
            if (expected or {}).get(name) != current.get(name)
        )
        raise RuntimeError(f"deployment guardrail drift since frozen baseline: {changed}")
    return {
        "baseline_path": str(DEPLOYMENT_BASELINE.relative_to(REPO_ROOT)).replace("\\", "/"),
        "baseline_sha256": baseline_hash,
        "baseline_type": baseline["baseline_type"],
        "guardrails": current,
    }


def _json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding="utf-8")


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in fields})


def _cell(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _rth(frame: pd.DataFrame, days: list[date]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="bar_time"),
        )
    parts = []
    for day in days:
        session = session_for(day)
        if session:
            part = frame.loc[
                (frame.index >= pd.Timestamp(session["open"]))
                & (frame.index < pd.Timestamp(session["close"]))
            ]
            if not part.empty:
                parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.concat(parts).sort_index()


def coverage_report(
    frames: dict[str, pd.DataFrame], symbols: list[str], days: list[date]
) -> tuple[float, list[dict]]:
    rows: list[dict] = []
    usable = 0
    total = len(symbols) * len(days)
    for symbol in symbols:
        frame = frames.get(symbol, pd.DataFrame())
        for day in days:
            session = session_for(day)
            expected = int((session["close"] - session["open"]).total_seconds() // 300)
            observed = len(frame.loc[
                (frame.index >= pd.Timestamp(session["open"]))
                & (frame.index < pd.Timestamp(session["close"]))
            ]) if not frame.empty else 0
            valid = observed == expected
            usable += int(valid)
            if not valid:
                rows.append({
                    "symbol": symbol, "session": day, "expected_bars": expected,
                    "observed_bars": observed, "usable": False,
                })
    return (usable / total if total else 0.0), rows


def _load_outcomes(
    signals: list[SlcSignal], *, data_mode: str
) -> tuple[dict[tuple, dict], list[dict]]:
    outcomes: dict[tuple, dict] = {}
    misses: list[dict] = []
    by_day: dict[date, list[SlcSignal]] = {}
    for signal in signals:
        by_day.setdefault(pd.Timestamp(signal.entry_time).date(), []).append(signal)
    for number, day in enumerate(sorted(by_day), start=1):
        session = session_for(day)
        day_signals = by_day[day]
        symbols = sorted(set(signal.symbol for signal in day_signals))
        print(f"outcomes {number}/{len(by_day)}: {day} ({len(symbols)} symbols, {data_mode})", flush=True)
        frames, day_misses = load_session_minutes(
            symbols, session["open"], session["close"], mode=data_mode
        )
        misses.extend(day_misses)
        for signal in day_signals:
            outcomes[_signal_key(signal)] = simulate_intraday_outcome(
                signal, frames.get(signal.symbol), session_close=pd.Timestamp(session["close"])
            )
    return outcomes, misses


def _signal_key(signal: SlcSignal) -> tuple:
    return signal.symbol, signal.level_id, signal.entry_time.isoformat()


def simulate_portfolio(
    signals: list[SlcSignal], outcomes: dict[tuple, dict], *, portfolio, cost,
    start_date: date | None = None, end_date: date | None = None,
) -> dict:
    """Chronological cash-only simulation for intraday SLC positions."""
    cash = float(portfolio.starting_equity)
    realized = 0.0
    active: list[dict] = []
    trades: list[dict] = []
    rejected: list[dict] = []
    missing: list[dict] = []
    daily_equity: list[dict] = []
    signals_by_day: dict[date, list[SlcSignal]] = {}
    for signal in signals:
        signals_by_day.setdefault(pd.Timestamp(signal.entry_time).date(), []).append(signal)

    first_day = start_date or (min(signals_by_day) if signals_by_day else DEFAULT_START)
    last_day = end_date or (max(signals_by_day) if signals_by_day else first_day)
    all_days = trading_days_between(first_day, last_day)
    for day in all_days:
        session = session_for(day)
        day_start_equity = portfolio.starting_equity + realized
        day_realized = 0.0
        approved = 0
        day_signals = sorted(
            signals_by_day.get(day, []),
            key=lambda s: (s.entry_time, -s.level_active_time.value, -s.impulse_atr, s.symbol),
        )
        for signal in day_signals:
            now = pd.Timestamp(signal.entry_time)
            still_active: list[dict] = []
            for item in active:
                if item["exit_time"] <= now:
                    cash += item["reserved_notional"] + item["net_pnl"]
                    realized += item["net_pnl"]
                    day_realized += item["net_pnl"]
                else:
                    still_active.append(item)
            active = still_active

            reason = None
            if day_realized <= -(day_start_equity * portfolio.daily_loss_limit):
                reason = "daily_loss_limit"
            elif any(item["symbol"] == signal.symbol for item in active):
                reason = "already_held"
            elif len(active) >= portfolio.max_positions:
                reason = "max_positions"
            elif approved >= portfolio.max_new_trades_per_day:
                reason = "max_new_trades_per_day"
            equity = portfolio.starting_equity + realized
            risk_per_share = signal.initial_risk
            max_notional = min(equity * portfolio.max_position_pct, cash)
            quantity = math.floor(min(
                equity * portfolio.risk_per_trade / risk_per_share,
                max_notional / signal.entry,
            )) if risk_per_share > 0 and signal.entry > 0 else 0
            if reason is None and quantity <= 0:
                reason = "insufficient_cash_or_quantity"
            if reason:
                rejected.append({"date": day, "symbol": signal.symbol, "reason": reason,
                                 "entry_time": signal.entry_time})
                continue

            approved += 1
            reserved = quantity * signal.entry
            cash -= reserved
            outcome = outcomes.get(_signal_key(signal), {
                "status": "outcome_data_missing", "reason": "outcome_not_loaded"
            })
            trade = {
                **signal.to_dict(),
                "mode": "stock_only",
                "portfolio": portfolio.name,
                "cost_model": cost.name,
                "quantity": quantity,
                "risk_amount": quantity * risk_per_share,
                "reserved_notional": reserved,
                "fill_price": signal.entry,
                **outcome,
                "gross_pnl": None,
                "transaction_cost": None,
                "net_pnl": None,
                "pnl_r": None,
            }
            if outcome.get("status") == "closed":
                gross = quantity * float(outcome["gross_per_share"])
                transaction_cost = quantity * (
                    signal.entry + float(outcome["exit_price"])
                ) * cost.stock_bps_per_leg / 10_000.0
                net = gross - transaction_cost
                trade.update(
                    gross_pnl=gross,
                    transaction_cost=transaction_cost,
                    net_pnl=net,
                    pnl_r=net / (quantity * risk_per_share),
                )
                exit_time = pd.Timestamp(outcome["exit_time"])
            else:
                missing.append({"date": day, "symbol": signal.symbol,
                                "entry_time": signal.entry_time,
                                "reason": outcome.get("reason", "unknown")})
                exit_time = pd.Timestamp(session["close"])
                net = 0.0
            active.append({
                "symbol": signal.symbol, "exit_time": exit_time,
                "reserved_notional": reserved, "net_pnl": net,
            })
            trades.append(trade)

        for item in active:
            cash += item["reserved_notional"] + item["net_pnl"]
            realized += item["net_pnl"]
        active = []
        daily_equity.append({"date": day, "equity": portfolio.starting_equity + realized})
    return {
        "mode": "stock_only", "portfolio": portfolio.name, "cost_model": cost.name,
        "trades": trades, "rejected": rejected, "missing_outcomes": missing,
        "daily_equity": daily_equity, "final_equity": portfolio.starting_equity + realized,
    }


def run(
    *, start_date: date, end_date: date, symbols: list[str], data_mode: str,
    label: str | None = None,
) -> tuple[dict, Path]:
    if start_date > end_date:
        raise ValueError("start must be on or before end")
    freeze = verify_rule_freeze()
    deployment_preflight = verify_deployment_baseline()
    guardrails_before = deployment_preflight["guardrails"]
    cache_before = _sha256(SLC_CACHE_PATH) if SLC_CACHE_PATH.exists() else None
    symbols = sorted(set(symbol.upper() for symbol in symbols))
    if "SPY" not in symbols:
        symbols.append("SPY")
        symbols.sort()
    sessions = trading_days_between(start_date, end_date)
    if not sessions:
        raise ValueError("date range contains no NYSE sessions")
    lookback_start = datetime.combine(start_date - timedelta(days=120), datetime.min.time())
    data_end = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
    frames, data_misses = load_stock_bars(
        symbols, lookback_start, data_end, minutes=5, mode=data_mode,
        progress=lambda message: print(message, flush=True),
    )
    rth_frames = {symbol: _rth(frame, trading_days_between(start_date - timedelta(days=120), end_date))
                  for symbol, frame in frames.items()}
    coverage, incomplete = coverage_report(rth_frames, symbols, sessions)

    signals: list[SlcSignal] = []
    research_symbols = [symbol for symbol in symbols if symbol != "SPY"]
    for number, symbol in enumerate(research_symbols, start=1):
        print(f"signals {number}/{len(research_symbols)}: {symbol}", flush=True)
        symbol_signals = generate_signals(symbol, rth_frames.get(symbol, pd.DataFrame()))
        signals.extend(
            signal for signal in symbol_signals
            if start_date <= pd.Timestamp(signal.entry_time).date() <= end_date
        )
    signals.sort(key=lambda signal: (signal.entry_time, signal.symbol, signal.level_id))
    outcomes, outcome_misses = _load_outcomes(signals, data_mode=data_mode)
    benchmark = benchmark_summary(rth_frames.get("SPY"), start_date, end_date, "stock")

    scenario_results: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for portfolio_name in ("safe_0_25pct", "current_1pct"):
        for cost_name in ("zero", "baseline", "stressed"):
            key = f"{portfolio_name}__{cost_name}"
            result = simulate_portfolio(
                signals, outcomes, portfolio=PORTFOLIOS[portfolio_name], cost=COSTS[cost_name],
                start_date=start_date, end_date=end_date,
            )
            scenario_results[key] = result
            summaries[key] = summarize_run(
                result, starting_equity=PORTFOLIOS[portfolio_name].starting_equity,
                start_date=start_date, end_date=end_date, benchmark=benchmark,
            )
    repeated_primary = simulate_portfolio(
        signals, outcomes, portfolio=PORTFOLIOS["safe_0_25pct"],
        cost=COSTS["baseline"], start_date=start_date, end_date=end_date,
    )
    determinism_passed = (
        _stable_digest(repeated_primary)
        == _stable_digest(scenario_results["safe_0_25pct__baseline"])
    )
    if determinism_passed:
        qualification = qualify_strategy(
            summaries["safe_0_25pct__baseline"],
            summaries["safe_0_25pct__stressed"],
            coverage,
        )
        qualification["qualification_status"] = "evaluated"
    else:
        qualification = {
            "passed": False, "qualification_status": "invalid",
            "failed_checks": ["deterministic_repeatability"],
            "checks": {"deterministic_repeatability": False},
        }
    qualification["historical_pass_does_not_enable_execution"] = True
    qualification["shortability_assumption_unverified"] = True
    guardrails_after = _guardrail_hashes()
    if guardrails_after != guardrails_before:
        raise RuntimeError("deployment guardrail file changed during research run")
    cache_after = _sha256(SLC_CACHE_PATH) if SLC_CACHE_PATH.exists() else None

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    if label:
        safe_label = "".join(ch for ch in label if ch.isalnum() or ch in "-_" )[:40]
        run_id = f"{run_id}_{safe_label}"
    output = RESULTS_ROOT / run_id
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "strategy_version": "slc_4h_5m_stock_v1",
        "start": start_date, "end": end_date, "symbols": symbols,
        "data_mode": data_mode, "primary_risk": "safe_0_25pct",
        "costs_bps_per_leg": {name: COSTS[name].stock_bps_per_leg for name in COSTS},
        "technical_only_disclosure": TECHNICAL_ONLY_DISCLOSURE,
        "fixed_universe_survivorship_bias": True,
        "shorts_assume_borrow_available": True,
        "execution_capable": False,
        "deterministic_repeatability": determinism_passed,
        "rule_hashes": freeze,
        "deployment_guardrail_preflight": deployment_preflight,
        "guardrail_hashes": guardrails_after,
        "cache_sha256_before": cache_before,
        "cache_sha256_after": cache_after,
    }
    _json(output / "config.json", config)
    _json(output / "summary.json", {
        "qualification": qualification, "coverage_rate": coverage,
        "signal_count": len(signals), "deterministic_repeatability": determinism_passed,
        "summaries": summaries,
    })
    _json(output / "data_coverage.json", {
        "coverage_rate": coverage, "incomplete_count": len(incomplete),
        "data_cache_misses": data_misses, "outcome_cache_misses": outcome_misses,
    })
    _csv(output / "signals.csv", [signal.to_dict() for signal in signals])
    _csv(output / "incomplete_sessions.csv", incomplete)
    _csv(output / "outcome_cache_misses.csv", outcome_misses)
    for key, result in scenario_results.items():
        _csv(output / f"trades__{key}.csv", result["trades"])
        _csv(output / f"equity__{key}.csv", result["daily_equity"])
        _csv(output / f"rejections__{key}.csv", result["rejected"])
        _csv(output / f"missing_outcomes__{key}.csv", result["missing_outcomes"])
        _csv(output / f"quarterly__{key}.csv", summaries[key]["quarterly"])
        _csv(output / f"yearly__{key}.csv", summaries[key]["yearly"])
    manifest = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir()) if path.is_file()
    }
    _json(output / "manifest.json", {"sha256": manifest})
    report = {
        "qualification": qualification,
        "coverage_rate": coverage,
        "signal_count": len(signals),
        "primary_baseline": summaries["safe_0_25pct__baseline"],
        "primary_stressed": summaries["safe_0_25pct__stressed"],
        "output": str(output),
    }
    return report, output


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only SLC 4h/5m stock backtest")
    parser.add_argument("--start", type=_parse_date, default=DEFAULT_START)
    parser.add_argument("--end", type=_parse_date, default=DEFAULT_END)
    parser.add_argument("--symbols", help="comma-separated symbols; default is the full bot universe")
    parser.add_argument("--data-mode", choices=("cache-only", "fetch"), default="cache-only")
    parser.add_argument("--label")
    args = parser.parse_args()
    symbols = [part.strip() for part in args.symbols.split(",") if part.strip()] if args.symbols else UNIVERSE
    report, output = run(
        start_date=args.start, end_date=args.end, symbols=symbols,
        data_mode=args.data_mode, label=args.label,
    )
    print("\nSLC RESEARCH COMPLETE")
    print(f"  signals:       {report['signal_count']}")
    print(f"  coverage:      {report['coverage_rate']:.1%}")
    print(f"  qualification: {'PASS' if report['qualification']['passed'] else 'FAIL'}")
    print(f"  output:        {output}")
    print("  execution:     disabled by architecture")


if __name__ == "__main__":
    main()
