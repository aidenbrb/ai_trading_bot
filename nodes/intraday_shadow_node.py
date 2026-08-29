"""
Intraday Shadow Node - day-trading (ORB) mode, post-close outcome reconstruction.

Runs AFTER the actual session close (~4:10pm ET, see scripts/run_day_shadow.bat)
- not before it, so it never reads an incomplete candle and can correctly
reproduce the research strategy's market-close exit. Uses
utils/market_calendar.py to get the session's true close time (handles early
closes correctly, since this only reads completed historical data, not
acting on anything in real time).

For every SELECTED (top-20, passed-filters) IntradaySignal row from today's
day_strategy_node run, reconstructs entry/stop/exit from completed 1-minute
bars (utils/orb_signals.py::simulate_intraday_outcome - 1-minute resolution
disambiguates same-bar entry/stop touches that a 5-min bar cannot; if still
ambiguous even at 1-minute resolution, the ADVERSE outcome is assumed, never
the favorable one), computes gross/cost-adjusted P&L and P&L-in-R using the
sizing already stored on the signal row (from day_strategy_node, always
INTENDED_DEPLOYMENT sizing), and upserts the outcome fields onto that same
IntradaySignal row.

Writes ONLY to IntradaySignal - same structural isolation as day_strategy_node.

Run standalone:
    python -m nodes.intraday_shadow_node
    python -m nodes.intraday_shadow_node --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Optional

from sqlmodel import select

from db.connection import get_session, init_db
from db.models import IntradaySignal, RunLog, Ticker
from utils.alpaca_bars import fetch_intraday_bars
from utils.cost_model import apply_entry_cost, apply_exit_cost, by_version
from utils.market_calendar import is_trading_day, session_for
from utils.orb_signals import simulate_intraday_outcome
from utils.timeutil import utcnow

_NODE = "intraday_shadow_node"
_INTERVAL = "5Min"
_STRATEGY_VERSION = "orb_v1"


def run(tickers: Optional[list[str]] = None, as_of: Optional[date] = None) -> dict:
    """
    Reconstruct today's shadow outcomes for every selected IntradaySignal row.

    Returns
    -------
    dict with keys: reconstructed, skipped
    """
    target = as_of or date.today()

    print(f"\n{'='*55}")
    print(f"  INTRADAY SHADOW NODE   date={target}")
    print(f"{'='*55}")

    init_db()

    if not is_trading_day(target):
        print(f"  {target} is not an NYSE trading day - nothing to reconstruct.")
        return {"reconstructed": [], "skipped": []}

    session = session_for(target)
    if session is None:
        return {"reconstructed": [], "skipped": []}

    rows = _load_selected_signals(target, tickers)
    print(f"  Selected signals to reconstruct: {len(rows)}")

    reconstructed, skipped = [], []
    for row in rows:
        symbol = _symbol_for(row.ticker_id)
        print(f"  {symbol:<8} ", end="", flush=True)
        try:
            outcome = _reconstruct_outcome(row, session)
        except Exception as exc:
            print(f"ERR  {exc}")
            skipped.append(symbol)
            continue
        if outcome is None:
            print("SKIP  no completed bars available")
            skipped.append(symbol)
            continue
        _apply_outcome(row.id, outcome)
        reconstructed.append(symbol)
        print(f"OK    {outcome['exit_reason']:<12} pnl_r={outcome.get('pnl_r')}")

    _write_log(reconstructed, skipped)
    print(f"\n  Done - reconstructed={len(reconstructed)}  skipped={len(skipped)}")
    return {"reconstructed": reconstructed, "skipped": skipped}


def _load_selected_signals(target: date, tickers: Optional[list[str]]) -> list[IntradaySignal]:
    with get_session() as session:
        query = (
            select(IntradaySignal)
            .where(IntradaySignal.session_date == target)
            .where(IntradaySignal.interval == _INTERVAL)
            .where(IntradaySignal.strategy_version == _STRATEGY_VERSION)
            .where(IntradaySignal.selected == True)  # noqa: E712
        )
        rows = session.exec(query).all()
    if tickers:
        upper = {t.upper() for t in tickers}
        rows = [r for r in rows if _symbol_for(r.ticker_id) in upper]
    return list(rows)


_ticker_cache: dict[str, str] = {}

def _symbol_for(ticker_id: str) -> str:
    if ticker_id not in _ticker_cache:
        with get_session() as session:
            t = session.exec(select(Ticker).where(Ticker.id == ticker_id)).first()
            _ticker_cache[ticker_id] = t.symbol if t else "UNKNOWN"
    return _ticker_cache[ticker_id]


def _reconstruct_outcome(row: IntradaySignal, session: dict) -> Optional[dict]:
    symbol = _symbol_for(row.ticker_id)
    window_start = row.opening_bar_time + timedelta(minutes=5)  # right after the opening range
    window_end = session["close"]

    bars = fetch_intraday_bars(symbol, window_start, window_end, minutes=1)
    if bars.empty:
        return None

    sim = simulate_intraday_outcome(
        bars, entry_trigger=row.entry_trigger_price, stop_price=row.stop_price, direction=row.direction,
    )

    if not sim["breakout_triggered"]:
        return sim

    cost_model = by_version(row.cost_model_version or "baseline_v1")
    entry_cost_adj = apply_entry_cost(sim["simulated_entry_price"], row.direction, cost_model)
    exit_cost_adj = apply_exit_cost(sim["exit_price"], row.direction, cost_model)

    qty = row.simulated_quantity or 0.0
    if row.direction == "long":
        gross_pnl = (sim["exit_price"] - sim["simulated_entry_price"]) * qty
        cost_adjusted_pnl = (exit_cost_adj - entry_cost_adj) * qty
    else:
        gross_pnl = (sim["simulated_entry_price"] - sim["exit_price"]) * qty
        cost_adjusted_pnl = (entry_cost_adj - exit_cost_adj) * qty

    pnl_r = None
    if row.simulated_risk_amount:
        pnl_r = cost_adjusted_pnl / row.simulated_risk_amount

    sim["gross_pnl"] = gross_pnl
    sim["cost_adjusted_pnl"] = cost_adjusted_pnl
    sim["pnl_r"] = pnl_r
    return sim


def _apply_outcome(signal_id: str, outcome: dict) -> None:
    with get_session() as session:
        row = session.exec(select(IntradaySignal).where(IntradaySignal.id == signal_id)).first()
        if not row:
            return
        row.breakout_triggered = outcome["breakout_triggered"]
        row.trigger_time = outcome["trigger_time"]
        row.simulated_entry_price = outcome["simulated_entry_price"]
        row.stop_hit = outcome["stop_hit"]
        row.exit_time = outcome["exit_time"]
        row.exit_price = outcome["exit_price"]
        row.exit_reason = outcome["exit_reason"]
        row.outcome_ambiguous = outcome["outcome_ambiguous"]
        row.gross_pnl = outcome.get("gross_pnl")
        row.cost_adjusted_pnl = outcome.get("cost_adjusted_pnl")
        row.pnl_r = outcome.get("pnl_r")
        session.add(row)


def _write_log(reconstructed: list[str], skipped: list[str]) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=_NODE,
            node_name=_NODE,
            status="success",
            tickers_processed=len(reconstructed) + len(skipped),
            records_written=len(reconstructed),
            error_message=("; ".join(skipped) if skipped else None),
            finished_at=utcnow(),
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday Shadow Node - post-close outcome reconstruction")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
