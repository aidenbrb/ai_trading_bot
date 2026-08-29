"""
Pipeline runner - executes all nodes in sequence.

Usage:
    python run_pipeline.py                         # run all nodes
    python run_pipeline.py --nodes data indicator  # run specific nodes only
    python run_pipeline.py --tickers AAPL MSFT     # limit to specific tickers

Rule-based pipeline only - no LLM/Claude nodes exist in this bot.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from utils.timeutil import utcnow

from config.settings import settings
from db.connection import get_session, init_db
from db.models import RunLog
from utils.market_session import resolve_market
from utils.strategy_registry import registration
from utils.strategy_signals import CRYPTO_STRATEGY_VERSION, STOCK_STRATEGY_VERSION


def _check_market_regime() -> dict:
    """
    Check SPY's trend to classify the broad market regime.
    Returns label, a quality_boost added to MIN_SETUP_QUALITY, and a description.

    BULL    - SPY above both SMA-20 and SMA-50  -> run normally
    CAUTION - SPY below SMA-20 but above SMA-50 -> raise bar by 10 pts
    BEAR    - SPY below SMA-50                  -> raise bar by 20 pts
    """
    try:
        import yfinance as yf

        spy = yf.Ticker("SPY").history(period="65d", interval="1d")
        if spy.empty or len(spy) < 21:
            return {"label": "UNKNOWN", "quality_boost": 0,
                    "description": "SPY data unavailable - running at default thresholds"}

        close   = spy["Close"]
        current = float(close.iloc[-1])
        sma20   = float(close.rolling(20).mean().iloc[-1])
        sma50   = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None

        above_sma20 = current > sma20
        above_sma50 = sma50 is None or current > sma50

        if above_sma20 and above_sma50:
            return {"label": "BULL", "quality_boost": 0,
                    "description": f"SPY ${current:.2f} above SMA20 ${sma20:.2f} - full signal strength"}
        elif above_sma50:
            desc = f"SPY ${current:.2f} below SMA20 ${sma20:.2f} but above SMA50"
            return {"label": "CAUTION", "quality_boost": 10, "description": desc}
        else:
            s50_str = f"${sma50:.2f}" if sma50 else "n/a"
            desc = f"SPY ${current:.2f} below SMA50 {s50_str} - bear market filter active"
            return {"label": "BEAR", "quality_boost": 20, "description": desc}

    except Exception as exc:
        return {"label": "UNKNOWN", "quality_boost": 0,
                "description": f"Regime check error: {exc}"}

PHASES = ["monitor", "data", "indicator", "news", "strategy", "options", "risk", "execution"]
_ENTRY_PHASES = {"risk", "execution"}


def _failed_deployment_strategies(
    market_mode: str,
    stock_strategy: str,
):
    """Return selected strategies that evidence has made research-only."""
    versions = []
    if stock_strategy == "swing" and market_mode in {"stocks", "all"}:
        versions.append(STOCK_STRATEGY_VERSION)
    if market_mode in {"crypto", "all"}:
        versions.append(CRYPTO_STRATEGY_VERSION)
    return [
        registration(version)
        for version in versions
        if not registration(version).execution_eligible
    ]


def _run_phase(name, func, **kwargs):
    """Run one phase and persist an auditable crash record before re-raising."""
    started = utcnow()
    try:
        return func(**kwargs)
    except Exception as exc:
        finished = utcnow()
        print(f"\n{'!'*70}")
        print(f"  PHASE '{name}' CRASHED - remaining phases will not run")
        print(f"  Error: {exc}")
        print(f"{'!'*70}\n")
        try:
            init_db()
            with get_session() as session:
                session.add(RunLog(
                    run_id=str(uuid.uuid4()),
                    node_name=name,
                    status="error",
                    error_message=str(exc),
                    duration_seconds=(finished - started).total_seconds(),
                    started_at=started,
                    finished_at=finished,
                ))
        except Exception as log_exc:
            print(f"  WARNING: could not persist crash RunLog: {log_exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description="ai_trading_bot pipeline runner")
    parser.add_argument(
        "--nodes",
        nargs="+",
        choices=PHASES,
        default=None,
        help="which nodes to run (default: all)",
    )
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    parser.add_argument(
        "--market",
        choices=["auto", "stocks", "crypto", "all"],
        default="auto",
        help="market universe (default: auto = stocks weekdays, crypto weekends)",
    )
    parser.add_argument(
        "--stock-strategy",
        choices=["swing", "day"],
        default="swing",
        help=(
            "stock strategy: swing (hourly, existing, default) or day "
            "(5-min ORB, signal-only - cannot reach risk/execution/options)"
        ),
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.llm:
        parser.error("--llm has been removed; Claude is no longer used anywhere in this bot.")

    settings.validate()

    market_mode = resolve_market(
        args.market,
        auto_enabled=settings.AUTO_MARKET_SCHEDULE,
    )
    active_phases = list(args.nodes or PHASES)

    if args.stock_strategy == "day":
        if market_mode != "stocks":
            parser.error(
                f"--stock-strategy day requires --market stocks (resolved "
                f"market is '{market_mode}'). Day mode only ever affects the "
                f"stock side of the pipeline. Pass --market stocks "
                f"explicitly, or drop --stock-strategy day to run "
                f"crypto/all/auto normally."
            )
        _DAY_MODE_FORBIDDEN_PHASES = {"risk", "execution", "options"}
        if args.nodes:
            blocked = _DAY_MODE_FORBIDDEN_PHASES & set(active_phases)
            if blocked:
                parser.error(
                    f"--stock-strategy day is signal-only: {sorted(blocked)} "
                    f"cannot be combined with it (no live/paper order path "
                    f"exists for day mode yet). Remove --stock-strategy day "
                    f"or drop these from --nodes."
                )
        else:
            active_phases = [
                p for p in active_phases if p not in _DAY_MODE_FORBIDDEN_PHASES
            ]

    # Evidence gate: a strategy that failed its locked comparison may continue
    # producing auditable research signals, but it cannot reach sizing or order
    # submission. This applies even when execution is enabled in .env and makes
    # a manually requested entry phase fail loudly instead of silently running.
    deployment_locks = _failed_deployment_strategies(
        market_mode, args.stock_strategy
    )
    locked_entry_phases = _ENTRY_PHASES & set(active_phases)
    if deployment_locks and args.nodes and locked_entry_phases:
        locked_versions = ", ".join(item.version for item in deployment_locks)
        parser.error(
            f"{sorted(locked_entry_phases)} cannot run: {locked_versions} "
            "failed the locked evidence-first comparison and is research-only. "
            "Monitoring existing positions remains available with --nodes monitor."
        )
    if deployment_locks:
        active_phases = [
            phase for phase in active_phases if phase not in _ENTRY_PHASES
        ]

    crypto_research_only = (
        market_mode == "crypto" and not settings.CRYPTO_EXECUTION_ENABLED
    )
    if market_mode == "crypto":
        if settings.WEEKEND_CRYPTO_RESEARCH:
            # Options are equity contracts and do not belong in weekend crypto
            # mode. Risk/execution/monitor remain active only when Alpaca paper
            # crypto execution is explicitly enabled.
            active_phases = [p for p in active_phases if p != "options"]
            if crypto_research_only:
                active_phases = [
                    p for p in active_phases
                    if p not in {"monitor", "risk", "execution"}
                ]
        else:
            active_phases = []

    pipeline_id = str(uuid.uuid4())
    print(f"\n{'='*60}")
    print(f"  ai_trading_bot  PIPELINE  {utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  pipeline_id = {pipeline_id[:8]}")
    print(f"  mode        = Rule-based (no LLM)")
    mode_note = (
        "weekend research only; no crypto orders"
        if crypto_research_only else
        "Alpaca paper crypto"
        if market_mode == "crypto" else
        "Alpaca paper stocks"
        if market_mode == "stocks" else
        "manual mixed universe"
    )
    print(f"  market      = {market_mode} ({mode_note})")
    print(f"  nodes       = {active_phases}")
    if args.tickers:
        print(f"  tickers     = {args.tickers}")
    print(f"{'='*60}")

    if deployment_locks:
        print("\n  *** NEW ENTRY EXECUTION LOCKED BY EVIDENCE GATE ***")
        for item in deployment_locks:
            print(f"  {item.version}: {item.reason}")
        print("  Research signals may run; existing positions remain monitored.\n")

    settings.print_summary()

    # -- Market regime check ---------------------------------------------------
    # Informational only - printed before strategy runs so the operator can see
    # broad market context alongside the per-ticker BUY/NONE decisions.
    if "strategy" in active_phases and market_mode in {"stocks", "all"}:
        regime = _check_market_regime()
        print(f"\n  MARKET REGIME : {regime['label']}")
        print(f"  {regime['description']}")
        print()

    from config.universe import CRYPTO, CRYPTO_SET, UNIVERSE
    if args.tickers:
        selected_tickers = [t.upper() for t in args.tickers]
    elif market_mode == "crypto":
        selected_tickers = list(CRYPTO)
    elif market_mode == "stocks":
        selected_tickers = list(UNIVERSE)
    else:
        selected_tickers = list(UNIVERSE) + list(CRYPTO)

    results = {}

    if "monitor" in active_phases:
        from nodes.monitor_node import run as monitor_run
        # Existing holdings must remain managed regardless of which universe
        # is eligible for new entries today (crypto positions stay open 24/7).
        results["monitor"] = _run_phase(
            "monitor", monitor_run, tickers=args.tickers
        )

    if "data" in active_phases:
        from nodes.data_node import run as data_run
        results["data"] = _run_phase(
            "data", data_run, tickers=selected_tickers
        )

    if "indicator" in active_phases:
        from nodes.indicator_node import run as ind_run
        results["indicator"] = _run_phase(
            "indicator", ind_run, tickers=selected_tickers
        )

    if "news" in active_phases:
        from nodes.news_node import run as news_run
        results["news"] = _run_phase(
            "news", news_run, tickers=selected_tickers
        )

    if "strategy" in active_phases:
        shared_strat_id = str(uuid.uuid4())
        crypto_tickers = [t for t in selected_tickers if t in CRYPTO_SET]
        stock_tickers = [t for t in selected_tickers if t not in CRYPTO_SET]

        if crypto_tickers:
            from nodes.crypto_strategy_node import run as crypto_strat_run
            crypto_result = _run_phase(
                "crypto_strategy",
                crypto_strat_run,
                tickers=crypto_tickers,
                strategies_run_id=shared_strat_id,
            )
            results["crypto_strategy"] = crypto_result

        # Run stock strategy only when stock tickers were explicitly requested
        # or when no specific tickers were given (full universe run)
        if stock_tickers:
            if args.stock_strategy == "day":
                # Signal-only 5-min ORB day mode. Deliberately does NOT use
                # shared_strat_id and writes to IntradaySignal, never
                # Strategy - results["day_strategy"] has no strategies_run_id
                # key at all, so risk/execution (already hard-blocked from
                # this invocation entirely - see the day-mode safety gate
                # above) would find nothing to act on even if reached.
                from nodes.day_strategy_node import run as day_strat_run
                results["day_strategy"] = _run_phase(
                    "day_strategy", day_strat_run, tickers=stock_tickers,
                )
            else:
                from nodes.stock_strategy_node import run as stock_strat_run
                results["strategy"] = _run_phase(
                    "strategy",
                    stock_strat_run,
                    tickers=stock_tickers,
                    strategies_run_id=shared_strat_id,
                )

    if "risk" in active_phases:
        strategy_run_id = (
            (results.get("strategy") or {}).get("strategies_run_id")
            or (results.get("crypto_strategy") or {}).get("strategies_run_id")
        )
        from nodes.risk_node import run as risk_run
        results["risk"] = _run_phase(
            "risk",
            risk_run,
            strategies_run_id=strategy_run_id,
            tickers=args.tickers,
        )

    if "options" in active_phases:
        strategy_run_id = (results.get("strategy") or {}).get("strategies_run_id")
        from nodes.options_research_node import run as options_run
        results["options"] = _run_phase(
            "options",
            options_run,
            strategies_run_id=strategy_run_id,
            tickers=args.tickers,
        )

    if "execution" in active_phases:
        risk_run_id = (results.get("risk") or {}).get("run_id")
        from nodes.execution_node import run as exec_run
        results["execution"] = _run_phase(
            "execution",
            exec_run,
            risk_run_id=risk_run_id,
            tickers=args.tickers,
        )

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    for node, r in results.items():
        if not r:
            continue
        if node == "news":
            print(f"  {node:12}  status={r.get('status','')}  "
                  f"overall_risk={r.get('overall_market_risk','')}")
        elif node == "options":
            print(f"  {node:12}  selected={len(r.get('selected',[]))}  "
                  f"skipped={len(r.get('skipped',[]))}  "
                  f"failed={len(r.get('failed',[]))}")
        elif node == "monitor":
            print(f"  {node:12}  held={len(r.get('held',[]))}  "
                  f"tightened={len(r.get('tightened',[]))}  "
                  f"closed={len(r.get('closed',[]))}  "
                  f"reconciled={len(r.get('reconciled',[]))}")
        elif node in ("risk",):
            print(f"  {node:12}  approved={len(r.get('approved',[]))}  "
                  f"rejected={len(r.get('rejected',[]))}")
        elif node in ("execution",):
            print(f"  {node:12}  submitted={len(r.get('submitted',[]))}  "
                  f"dry_run={len(r.get('dry_run',[]))}  "
                  f"failed={len(r.get('failed',[]))}")
        elif node == "day_strategy":
            print(f"  {node:12}  generated={len(r.get('generated',[]))}  "
                  f"excluded={len(r.get('excluded',[]))}  "
                  f"blocked={r.get('blocked')}")
        else:
            print(f"  {node:12}  passed={len(r.get('passed',[]))}  "
                  f"skipped={len(r.get('skipped',[]))}  "
                  f"failed={len(r.get('failed',[]))}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
