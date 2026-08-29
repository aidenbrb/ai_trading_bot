"""
Indicator Node - Phase 2 of the pipeline.

Responsibilities:
  1. Read OHLCV bars from the ohlcv table (written by data_node)
  2. Compute all technical indicators for each ticker
  3. Write one Indicator row per ticker per date to the indicators table
     (skips dates already computed)
  4. Log results to run_log

Run standalone:
    python -m nodes.indicator_node
    python -m nodes.indicator_node --tickers AAPL MSFT SPY
    python -m nodes.indicator_node --tickers AAPL --date 2024-06-01

This node makes NO calls to Claude and NO calls to the broker.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date, datetime
from utils.timeutil import utcnow
from typing import Optional

import pandas as pd
from sqlmodel import select

from config.settings import settings
from config.universe import UNIVERSE
from db.connection import get_session, init_db
from db.models import OHLCV, Indicator, RunLog, Ticker
from utils.indicators import compute_all

_NODE = "indicator_node"


# -- Public entry point --------------------------------------------------------

def run(
    tickers:  list[str] | None = None,
    as_of:    Optional[date]   = None,
) -> dict:
    """
    Compute and store indicators.

    Parameters
    ----------
    tickers : list[str] | None
        Symbols to process. Defaults to the full universe.
    as_of : date | None
        Compute indicators for this date only. Defaults to today.

    Returns
    -------
    dict  with keys: passed, failed, skipped, run_id
    """
    run_id   = str(uuid.uuid4())
    started  = utcnow()
    symbols  = [t.upper() for t in (tickers or UNIVERSE)]
    target   = as_of or date.today()

    print(f"\n{'='*55}")
    print(f"  INDICATOR NODE   run_id={run_id[:8]}")
    print(f"  Tickers: {len(symbols)}   Date: {target}")
    print(f"{'='*55}")

    init_db()

    passed, failed, skipped = [], [], []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{total}] {symbol:<6} ", end="", flush=True)
        result = _process(symbol, target)
        if result["status"] == "ok":
            passed.append(symbol)
            print(f"OK  indicators written")
        else:
            failed.append({"symbol": symbol, "reason": result["reason"]})
            print(f"ERR  {result['reason']}")

    duration = (utcnow() - started).total_seconds()
    _write_log(run_id, passed, failed, skipped, duration)

    print(f"\n  Done in {duration:.1f}s - "
          f"computed={len(passed)}  skipped={len(skipped)}  failed={len(failed)}")

    return {
        "run_id":  run_id,
        "passed":  passed,
        "failed":  failed,
        "skipped": skipped,
    }


# -- Private helpers -----------------------------------------------------------

def _get_ticker_id(symbol: str) -> Optional[str]:
    with get_session() as session:
        row = session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()
        return row.id if row else None


def _load_ohlcv(ticker_id: str, min_rows: int = 200) -> pd.DataFrame | None:
    """Load OHLCV bars (hourly) sorted oldest-first. Returns None if not enough history."""
    with get_session() as session:
        rows = session.exec(
            select(OHLCV)
            .where(OHLCV.ticker_id == ticker_id)
            .order_by(OHLCV.bar_time)   # type: ignore[attr-defined]
        ).all()

        if len(rows) < min_rows:
            return None

        data = [{
            "date":   r.bar_time,
            "open":   r.open,
            "high":   r.high,
            "low":    r.low,
            "close":  r.close,
            "volume": r.volume,
        } for r in rows]

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def _process(symbol: str, target: date) -> dict:
    ticker_id = _get_ticker_id(symbol)
    if ticker_id is None:
        return {"status": "error", "reason": "ticker not in DB - run data_node first"}

    df = _load_ohlcv(ticker_id, min_rows=settings.MIN_HISTORY_BARS)
    if df is None:
        return {"status": "error", "reason": f"< {settings.MIN_HISTORY_BARS} bars in DB"}

    # Only use data up to and including target date
    df = df[df.index.date <= target]  # type: ignore[attr-defined]
    if df.empty:
        return {"status": "error", "reason": "no bars on or before target date"}

    try:
        ind_values = compute_all(df)
    except Exception as exc:
        return {"status": "error", "reason": f"compute error: {exc}"}

    # Upsert - update existing row for today or insert new one
    with get_session() as session:
        existing = session.exec(
            select(Indicator)
            .where(Indicator.ticker_id == ticker_id)
            .where(Indicator.bar_date == target)
        ).first()
        if existing:
            for k, v in ind_values.items():
                setattr(existing, k, v)
        else:
            session.add(Indicator(ticker_id=ticker_id, bar_date=target, **ind_values))

    return {"status": "ok"}


def _write_log(run_id, passed, failed, skipped, duration) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success" if not failed else "partial",
            tickers_processed=len(passed) + len(skipped),
            records_written=len(passed),
            error_message=(
                "; ".join(f"{f['symbol']}: {f['reason']}" for f in failed)
                if failed else None
            ),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indicator Node - compute technicals")
    parser.add_argument("--tickers", nargs="+", help="symbols to process")
    parser.add_argument("--date",    type=str,  help="target date (YYYY-MM-DD), default=today")
    args = parser.parse_args()

    target = date.fromisoformat(args.date) if args.date else None
    run(tickers=args.tickers, as_of=target)
