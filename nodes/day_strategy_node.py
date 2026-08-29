"""
Day Strategy Node - 5-min Opening Range Breakout (ORB), signal-only.

Part of the day-trading mode's structural isolation: this node writes ONLY
IntradaySignal rows. It NEVER writes to Strategy or RiskApproval, and never
returns a strategies_run_id - so even if risk_node/execution_node were
somehow invoked in the same pipeline run, they would have nothing from this
node to act on. run_pipeline.py's --stock-strategy day flag additionally
hard-blocks those phases outright (see run_pipeline.py's day-mode safety
gate) - this is defense in depth, not the primary guarantee.

At 9:35am ET (see scripts/run_day_open.bat), fetches each candidate's first
5-min opening bar (Alpaca IEX, split-adjusted), combines it with
IntradayDailyStats (computed the prior session by intraday_reference_node),
applies the exact-spec ORB filters/direction/stop (utils/orb_signals.py),
ranks by opening relative volume, and upserts one IntradaySignal row per
considered symbol for today's session (every considered candidate gets a
row, passing or not, for full auditability).

Fails CLOSED for the WHOLE session if the morning news report is missing or
stale (utils/market_intelligence.py::news_gate_status) - same fail-closed
convention as stock_strategy_node.py. A single symbol's missing opening bar
or reference stats excludes ONLY that symbol (recorded with a reason) - it
does not block the rest of the day's candidates.

Run standalone:
    python -m nodes.day_strategy_node
    python -m nodes.day_strategy_node --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date, timedelta
from typing import Optional

from sqlmodel import select

from config.universe import UNIVERSE
from db.connection import get_session, init_db
from db.models import IntradayDailyStats, IntradaySignal, RunLog, Ticker
from utils.alpaca_bars import fetch_intraday_bars, first_regular_session_bar
from utils.market_calendar import is_trading_day, session_for
from utils.market_intelligence import news_gate_status
from utils.orb_signals import build_candidate_fields, rank_and_select
from utils.timeutil import utcnow

_NODE = "day_strategy_node"
_INTERVAL = "5Min"
_STRATEGY_VERSION = "orb_v1"
_TOP_N = 20

# INTENDED_DEPLOYMENT sizing (locked in - see the day-trading-mode plan §6):
# 0.25% risk per trade, $100k assumed equity, no leverage. Live shadow rows
# always size using this scenario - RESEARCH_FIDELITY (1% risk, $25k equity)
# is a backtest-report-only comparison, never used for live rows.
_SIZING_EQUITY = 100_000.0
_SIZING_RISK_PCT = 0.0025
_COST_MODEL_VERSION = "baseline_v1"


def run(tickers: Optional[list[str]] = None, as_of: Optional[date] = None) -> dict:
    """
    Generate today's ORB day-mode signals. Signal-only: never touches
    Strategy/RiskApproval, never returns a strategies_run_id.

    Returns
    -------
    dict with keys: run_id, generated, excluded, blocked
    """
    run_id = str(uuid.uuid4())
    target = as_of or date.today()
    symbols = [t.upper() for t in (tickers or UNIVERSE)]

    print(f"\n{'='*55}")
    print(f"  DAY STRATEGY NODE (5-min ORB, signal-only)   run_id={run_id[:8]}")
    print(f"  Tickers: {len(symbols)}   Date: {target}")
    print(f"{'='*55}")

    init_db()

    if not is_trading_day(target):
        print(f"  {target} is not an NYSE trading day - no signals generated.")
        return {"run_id": run_id, "generated": [], "excluded": symbols, "blocked": "not_a_trading_day"}

    # Session-wide fail-closed gate (same convention as stock_strategy_node.py).
    _, news_blocked, news_reason = news_gate_status(target.isoformat())
    if news_blocked:
        print(f"\n  *** NEWS GATE: {news_reason} - blocking ALL day-mode signals ***")
        return {"run_id": run_id, "generated": [], "excluded": symbols,
                "blocked": f"news_gate: {news_reason}"}

    session = session_for(target)
    if session is None:
        return {"run_id": run_id, "generated": [], "excluded": symbols, "blocked": "no_session"}

    candidates, excluded = [], []
    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {symbol:<8} ", end="", flush=True)
        try:
            candidate = _build_candidate(symbol, target, session)
        except Exception as exc:
            print(f"ERR  {exc}")
            excluded.append({"symbol": symbol, "reason": str(exc)})
            continue
        if candidate is None:
            print("SKIP  missing opening bar or reference stats")
            excluded.append({"symbol": symbol, "reason": "missing opening bar or reference stats"})
            continue
        candidates.append(candidate)
        if candidate["passed_filters"]:
            print(f"PASS  candle={candidate['candle_type']}  "
                  f"rel_vol={candidate['opening_rel_volume']:.2f}x")
        else:
            print(f"none  {candidate['rejection_reason']}")

    passing = [c for c in candidates if c["passed_filters"]]
    not_passing = [c for c in candidates if not c["passed_filters"]]
    ranked = rank_and_select(passing, top_n=_TOP_N)
    all_rows = ranked + [dict(c, rank=None, selected=False) for c in not_passing]

    generated = []
    for c in all_rows:
        _upsert_signal(run_id, target, c)
        generated.append(c["symbol"])

    _write_log(run_id, generated, excluded)
    selected_count = sum(1 for c in ranked if c.get("selected"))
    print(f"\n  Done - considered={len(candidates)}  selected={selected_count}  excluded={len(excluded)}")
    return {"run_id": run_id, "generated": generated, "excluded": excluded, "blocked": None}


# -- Per-symbol candidate construction ------------------------------------------

def _get_ticker_id(symbol: str) -> Optional[str]:
    with get_session() as session:
        row = session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()
        return row.id if row else None


def _get_daily_stats(ticker_id: str, as_of_session: date) -> Optional[IntradayDailyStats]:
    with get_session() as session:
        return session.exec(
            select(IntradayDailyStats)
            .where(IntradayDailyStats.ticker_id == ticker_id)
            .where(IntradayDailyStats.as_of_session == as_of_session)
            .where(IntradayDailyStats.interval == _INTERVAL)
        ).first()


def _compute_sizing(entry_trigger: float, stop_price: float) -> dict:
    risk_per_unit = abs(entry_trigger - stop_price)
    if risk_per_unit <= 0:
        return {}
    risk_budget = _SIZING_EQUITY * _SIZING_RISK_PCT
    quantity = risk_budget / risk_per_unit
    return {
        "simulated_quantity": quantity,
        "simulated_risk_amount": risk_budget,
        "simulated_position_value": quantity * entry_trigger,
        "account_equity_used": _SIZING_EQUITY,
        "cost_model_version": _COST_MODEL_VERSION,
    }


def _build_candidate(symbol: str, target: date, session: dict) -> Optional[dict]:
    ticker_id = _get_ticker_id(symbol)
    if not ticker_id:
        return None

    stats = _get_daily_stats(ticker_id, target)
    if stats is None:
        return None

    window_start = session["open"]
    window_end = window_start + timedelta(minutes=5)
    bars = fetch_intraday_bars(symbol, window_start, window_end, minutes=5)
    bar = first_regular_session_bar(bars, window_start)
    if bar is None:
        return None

    opening_open = float(bar["open"])
    opening_high = float(bar["high"])
    opening_low = float(bar["low"])
    opening_close = float(bar["close"])
    opening_volume = float(bar["volume"])

    fields = build_candidate_fields(
        opening_open=opening_open, opening_high=opening_high,
        opening_low=opening_low, opening_close=opening_close,
        opening_volume=opening_volume,
        avg_daily_volume_14d=stats.avg_daily_volume_14d,
        daily_atr_14=stats.daily_atr_14,
        avg_opening_volume_14d=stats.avg_opening_volume_14d,
    )

    sizing: dict = {}
    if fields["passed_filters"]:
        sizing = _compute_sizing(fields["entry_trigger_price"], fields["stop_price"])

    bar_time = bar.name
    bar_time = bar_time.to_pydatetime() if hasattr(bar_time, "to_pydatetime") else bar_time

    return {
        "symbol": symbol,
        "ticker_id": ticker_id,
        "opening_bar_time": bar_time,
        "opening_open": opening_open,
        "opening_high": opening_high,
        "opening_low": opening_low,
        "opening_close": opening_close,
        "opening_volume": opening_volume,
        "opening_price": opening_open,
        "avg_daily_volume_14d": stats.avg_daily_volume_14d,
        "daily_atr_14": stats.daily_atr_14,
        "avg_opening_volume_14d": stats.avg_opening_volume_14d,
        **fields,
        **sizing,
    }


# -- Persistence (upsert) --------------------------------------------------------

def _upsert_signal(run_id: str, session_date: date, c: dict) -> None:
    with get_session() as session:
        existing = session.exec(
            select(IntradaySignal)
            .where(IntradaySignal.ticker_id == c["ticker_id"])
            .where(IntradaySignal.session_date == session_date)
            .where(IntradaySignal.interval == _INTERVAL)
            .where(IntradaySignal.strategy_version == _STRATEGY_VERSION)
        ).first()

        fields = dict(
            run_id=run_id,
            opening_bar_time=c["opening_bar_time"],
            opening_open=c["opening_open"], opening_high=c["opening_high"],
            opening_low=c["opening_low"], opening_close=c["opening_close"],
            opening_volume=c["opening_volume"],
            candle_type=c["candle_type"], direction=c["direction"],
            opening_price=c["opening_price"],
            avg_daily_volume_14d=c["avg_daily_volume_14d"],
            daily_atr_14=c["daily_atr_14"],
            avg_opening_volume_14d=c["avg_opening_volume_14d"],
            opening_rel_volume=c["opening_rel_volume"],
            passed_filters=c["passed_filters"],
            rank=c.get("rank"),
            selected=bool(c.get("selected", False)),
            entry_trigger_price=c["entry_trigger_price"],
            stop_price=c["stop_price"],
            rejection_reason=c["rejection_reason"],
            simulated_quantity=c.get("simulated_quantity"),
            simulated_risk_amount=c.get("simulated_risk_amount"),
            simulated_position_value=c.get("simulated_position_value"),
            account_equity_used=c.get("account_equity_used"),
            cost_model_version=c.get("cost_model_version"),
        )
        if existing:
            for k, v in fields.items():
                setattr(existing, k, v)
            session.add(existing)
        else:
            session.add(IntradaySignal(
                ticker_id=c["ticker_id"],
                session_date=session_date,
                interval=_INTERVAL,
                strategy_version=_STRATEGY_VERSION,
                **fields,
            ))


def _write_log(run_id: str, generated: list[str], excluded: list[dict]) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success",
            tickers_processed=len(generated) + len(excluded),
            records_written=len(generated),
            error_message=(
                "; ".join(f"{e['symbol']}: {e['reason']}" for e in excluded)
                if excluded else None
            ),
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day Strategy Node - 5-min ORB (signal-only)")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
