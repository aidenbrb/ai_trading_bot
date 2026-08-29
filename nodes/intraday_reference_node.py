"""
Intraday Reference Node - day-trading (ORB) mode, prior-session-only stats.

Computes IntradayDailyStats (14-day avg daily volume, daily ATR-14, 14-session
avg opening-bar volume) from data through the PREVIOUS trading session only -
never today's still-forming session - so the time-critical opening job
(day_strategy_node, ~9:36am) only has to fetch today's single opening bar.

Meant to run once, any time after the previous session's close and before
today's 9:30 open (see scripts/run_day_preflight.bat).

Part of the signal-only day-trading mode: writes ONLY IntradayDailyStats.
Never touches Strategy/RiskApproval/Order/Position.

Run standalone:
    python -m nodes.intraday_reference_node
    python -m nodes.intraday_reference_node --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Optional

from sqlmodel import select

from config.universe import UNIVERSE
from db.connection import get_session, init_db
from db.models import IntradayDailyStats, RunLog, Ticker
from utils.alpaca_bars import fetch_daily_bars, fetch_intraday_bars, first_regular_session_bar
from utils.market_calendar import is_trading_day, prior_trading_days, session_for
from utils.orb_signals import compute_daily_reference_stats
from utils.timeutil import utcnow

_NODE = "intraday_reference_node"
_INTERVAL = "5Min"
_LOOKBACK_SESSIONS = 14
_DAILY_FETCH_CALENDAR_DAYS = 45  # comfortably covers 14+ trading days incl. holidays


def run(tickers: Optional[list[str]] = None, as_of: Optional[date] = None) -> dict:
    """
    Compute/refresh IntradayDailyStats for `as_of` (defaults to today) from
    prior-session-only data.

    Returns
    -------
    dict with keys: computed, skipped, failed
    """
    target = as_of or date.today()
    symbols = [t.upper() for t in (tickers or UNIVERSE)]

    print(f"\n{'='*55}")
    print(f"  INTRADAY REFERENCE NODE   as_of={target}")
    print(f"  Tickers: {len(symbols)}")
    print(f"{'='*55}")

    init_db()

    if not is_trading_day(target):
        print(f"  {target} is not an NYSE trading day - nothing to compute.")
        return {"computed": [], "skipped": symbols, "failed": []}

    prior_sessions = prior_trading_days(target, _LOOKBACK_SESSIONS)
    if len(prior_sessions) < _LOOKBACK_SESSIONS:
        print(f"  WARNING: only {len(prior_sessions)}/{_LOOKBACK_SESSIONS} prior "
              f"sessions available (early in the calendar) - stats will use "
              f"whatever history exists.")

    daily_start = prior_sessions[0] if prior_sessions else target - timedelta(days=_DAILY_FETCH_CALENDAR_DAYS)
    daily_end = target  # exclusive per Alpaca convention - never includes today

    computed, skipped, failed = [], [], []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {symbol:<8} ", end="", flush=True)
        try:
            stats = _compute_stats(symbol, prior_sessions, daily_start, daily_end)
            if stats is None:
                print("SKIP  insufficient prior-session data")
                skipped.append(symbol)
                continue
            _upsert(symbol, target, stats)
            computed.append(symbol)
            print(f"OK    avg_vol_14d={stats['avg_daily_volume_14d']:,.0f}  "
                  f"atr_14=${stats['daily_atr_14']:.2f}  "
                  f"avg_open_vol_14d={stats['avg_opening_volume_14d']:,.0f}")
        except Exception as exc:
            print(f"ERR   {exc}")
            failed.append(symbol)

    _write_log(computed, skipped, failed)
    print(f"\n  Done - computed={len(computed)}  skipped={len(skipped)}  failed={len(failed)}")
    return {"computed": computed, "skipped": skipped, "failed": failed}


def _fetch_prior_opening_volumes(symbol: str, prior_sessions: list[date]) -> list[float]:
    opening_volumes = []
    for session_date in prior_sessions:
        session = session_for(session_date)
        if session is None:
            continue
        window_start = session["open"]
        window_end = session["open"] + timedelta(minutes=5)
        bars = fetch_intraday_bars(symbol, window_start, window_end, minutes=5)
        bar = first_regular_session_bar(bars, session["open"])
        if bar is not None:
            opening_volumes.append(float(bar["volume"]))
    return opening_volumes


def _compute_stats(symbol: str, prior_sessions: list[date], daily_start: date, daily_end: date) -> Optional[dict]:
    if not prior_sessions:
        return None
    daily = fetch_daily_bars(symbol, daily_start, daily_end)
    # Daily data is required for every reference record.  Fail immediately so
    # an unavailable daily source cannot trigger fourteen unnecessary live
    # intraday requests (and cannot make an otherwise isolated unit test touch
    # the network).
    if daily is None or daily.empty:
        return None
    opening_volumes = _fetch_prior_opening_volumes(symbol, prior_sessions)
    return compute_daily_reference_stats(daily, opening_volumes, lookback=_LOOKBACK_SESSIONS)


def _get_ticker_id(symbol: str) -> Optional[str]:
    with get_session() as session:
        row = session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()
        return row.id if row else None


def _upsert(symbol: str, as_of_session: date, stats: dict) -> None:
    ticker_id = _get_ticker_id(symbol)
    if not ticker_id:
        return
    with get_session() as session:
        existing = session.exec(
            select(IntradayDailyStats)
            .where(IntradayDailyStats.ticker_id == ticker_id)
            .where(IntradayDailyStats.as_of_session == as_of_session)
            .where(IntradayDailyStats.interval == _INTERVAL)
        ).first()
        if existing:
            existing.avg_daily_volume_14d = stats["avg_daily_volume_14d"]
            existing.daily_atr_14 = stats["daily_atr_14"]
            existing.avg_opening_volume_14d = stats["avg_opening_volume_14d"]
            session.add(existing)
        else:
            session.add(IntradayDailyStats(
                ticker_id=ticker_id,
                as_of_session=as_of_session,
                interval=_INTERVAL,
                **stats,
            ))


def _write_log(computed, skipped, failed) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=_NODE,
            node_name=_NODE,
            status="success" if not failed else "partial",
            tickers_processed=len(computed) + len(skipped) + len(failed),
            records_written=len(computed),
            error_message=("; ".join(failed) if failed else None),
            finished_at=utcnow(),
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday Reference Node - prior-session ORB stats")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
