"""
Stock Strategy Node - pure rule-based trade setups, no LLM calls.

Strategy: Trend-Following + Momentum (stocks)
---------------------------------------------
Gate 1  Uptrend structure   price > SMA-20 > SMA-50 (trend="UPTREND")
Gate 2  Momentum zone       RSI-14 between 50 and 68
Gate 3  MACD confirmation   MACD histogram > 0
Gate 4  Volume confirmation  relative volume >= 1.2x 20-day average
Gate 5  Minimum price       close >= settings.MIN_PRICE

Trade parameters (tighter than crypto - stocks are less volatile):
  Entry  = current close
  Stop   = entry - 1.5 * ATR(14)
  Target = entry + 3.0 * ATR(14)   => R:R of 2.0 (meets MIN_RISK_REWARD)

Writes Strategy rows with the shared strategies_run_id so risk_node
processes stock and crypto strategies together.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from utils.timeutil import utcnow
from typing import Optional

from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET, UNIVERSE
from db.connection import get_session, init_db
from db.models import Indicator, OHLCV, RunLog, Strategy, Ticker
from utils.market_intelligence import check_ticker, get_warnings, news_gate_status, overall_risk
from utils.strategy_signals import (
    STOCK_STRATEGY_VERSION,
    stock_trend_momentum_v1,
)

_NODE = "stock_strategy_node"

# -- Public entry point --------------------------------------------------------

def run(
    tickers:           list[str] | None = None,
    strategies_run_id: Optional[str]   = None,
    as_of:             Optional[date]  = None,
) -> dict:
    """
    Generate rule-based trade strategies for stock tickers.

    Parameters
    ----------
    tickers : list[str] | None
        Symbols to process. Crypto tickers are auto-skipped.
        Defaults to full UNIVERSE.
    strategies_run_id : str | None
        Shared run_id written into strategies table. Generated if None.
    as_of : date | None
        Indicator/price date to use. Defaults to today.

    Returns
    -------
    dict  with keys: passed, failed, skipped, strategies_run_id
    """
    run_id  = strategies_run_id or str(uuid.uuid4())
    started = utcnow()
    target  = as_of or date.today()

    raw_symbols = [t.upper() for t in (tickers or UNIVERSE)]
    symbols     = [s for s in raw_symbols if s not in CRYPTO_SET]

    print(f"\n{'='*55}")
    print(f"  STOCK STRATEGY NODE   run_id={run_id[:8]}")
    print(f"  Tickers: {len(symbols)}   Date: {target}")
    print(f"  Strategy: Trend-Following + Momentum (rule-based)")
    print(f"  Min setup quality skipped - using indicator gates directly")
    print(f"{'='*55}")

    init_db()

    # Fail-closed morning news gate: refuse ALL new BUY signals if today's
    # report is missing, unreadable, or stale (see news_gate_status()).
    mi_data, news_blocked, news_reason = news_gate_status(target.isoformat())
    if news_blocked:
        print(f"\n  *** NEWS GATE: {news_reason} - blocking ALL new BUY signals ***")
        skipped = list(symbols)
        for symbol in symbols:
            _write_strategy(run_id, symbol, target, signal="NONE",
                            thesis=f"Blocked: {news_reason}")
        duration = (utcnow() - started).total_seconds()
        _write_log(run_id, [], [], skipped, duration)
        print(f"\n  Done in {duration:.1f}s - BUY=0  NONE={len(skipped)}  failed=0")
        return {"strategies_run_id": run_id, "passed": [], "failed": [], "skipped": skipped}

    if mi_data:
        risk = overall_risk(mi_data)
        items_count = len(mi_data.get("items", []))
        print(f"\n  Market Intelligence: {items_count} items  overall_risk={risk.upper()}")
        print(f"  {mi_data.get('stock_market_summary', '')}")
        print(f"  {mi_data.get('macro_summary', '')}")

    passed, failed, skipped = [], [], []

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{len(symbols)}] {symbol:<8} ", end="", flush=True)

        ticker_id = _get_ticker_id(symbol)
        if not ticker_id:
            print(f"ERR  not in DB (run data_node first)")
            failed.append(symbol)
            continue

        ind = _get_indicator(ticker_id, target)
        if not ind:
            print(f"ERR  no indicator row (run indicator_node first)")
            failed.append(symbol)
            continue

        close = _get_latest_close(ticker_id, target)
        if not close:
            print(f"ERR  no OHLCV data")
            failed.append(symbol)
            continue

        # News gate: block if Market Analysis Bot flagged this ticker
        if mi_data:
            mi_ok, mi_reason = check_ticker(symbol, mi_data, crypto=False)
            if not mi_ok:
                skipped.append(symbol)
                print(f"SKIP  {mi_reason}")
                _write_strategy(run_id, symbol, target, signal="NONE",
                                ticker_id=ticker_id, thesis=f"Blocked by {mi_reason}")
                continue

        ok, reason, params = _evaluate(symbol, ind, close)

        if ok:
            # Annotate thesis with any non-blocking warnings
            warnings = get_warnings(symbol, mi_data) if mi_data else []
            if warnings:
                params["thesis"] += f"  [MI warnings: {', '.join(warnings)}]"
            passed.append(symbol)
            print(f"BUY   entry={params['entry']:.2f}  "
                  f"stop={params['stop']:.2f}  "
                  f"target={params['target']:.2f}  "
                  f"R:R={params['rr']:.2f}  "
                  f"conviction={params['conviction_score']}")
            _write_strategy(run_id, symbol, target, signal="BUY",
                            ticker_id=ticker_id, **params)
        else:
            skipped.append(symbol)
            print(f"NONE  {reason}")
            _write_strategy(run_id, symbol, target, signal="NONE",
                            ticker_id=ticker_id, thesis=f"Gates not met: {reason}")

    duration = (utcnow() - started).total_seconds()
    _write_log(run_id, passed, failed, skipped, duration)

    print(f"\n  Done in {duration:.1f}s - "
          f"BUY={len(passed)}  NONE={len(skipped)}  failed={len(failed)}")

    return {
        "strategies_run_id": run_id,
        "passed":  passed,
        "failed":  failed,
        "skipped": skipped,
    }


# -- Rule evaluation -----------------------------------------------------------

def _evaluate(
    symbol: str,
    ind:    Indicator,
    close:  float,
) -> tuple[bool, str, dict]:
    """Compatibility wrapper around the shared frozen v1 rule."""
    return stock_trend_momentum_v1(
        symbol,
        ind,
        close,
        min_price=settings.MIN_PRICE,
        min_rr=settings.MIN_RISK_REWARD,
    ).legacy_tuple()


# -- DB helpers ----------------------------------------------------------------

def _get_ticker_id(symbol: str) -> Optional[str]:
    with get_session() as session:
        row = session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()
        return row.id if row else None


def _get_indicator(ticker_id: str, bar_date: date) -> Optional[Indicator]:
    with get_session() as session:
        return session.exec(
            select(Indicator)
            .where(Indicator.ticker_id == ticker_id)
            .where(Indicator.bar_date  == bar_date)
        ).first()


def _get_latest_close(ticker_id: str, bar_date: date) -> Optional[float]:
    with get_session() as session:
        row = session.exec(
            select(OHLCV)
            .where(OHLCV.ticker_id == ticker_id)
            .where(OHLCV.bar_time  <= datetime.combine(bar_date, datetime.max.time()))
            .order_by(OHLCV.bar_time.desc())  # type: ignore[attr-defined]
        ).first()
        return float(row.close) if row else None


def _write_strategy(
    run_id:          str,
    symbol:          str,
    bar_date:        date,
    signal:          str,
    ticker_id:       Optional[str]   = None,
    entry:           Optional[float] = None,
    stop:            Optional[float] = None,
    target:          Optional[float] = None,
    rr:              Optional[float] = None,
    conviction_score: Optional[int]  = None,
    thesis:          Optional[str]   = None,
) -> None:
    if not ticker_id:
        ticker_id = _get_ticker_id(symbol)
    if not ticker_id:
        return
    with get_session() as session:
        session.add(Strategy(
            run_id=run_id,
            ticker_id=ticker_id,
            bar_date=bar_date,
            strategy_name="StockTrend-Momentum",
            signal=signal,
            entry=entry,
            stop=stop,
            target=target,
            rr=rr,
            conviction_score=conviction_score,
            thesis=thesis,
            model_used=STOCK_STRATEGY_VERSION,
        ))


def _write_log(run_id, passed, failed, skipped, duration) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success" if not failed else "partial",
            tickers_processed=len(passed) + len(skipped) + len(failed),
            records_written=len(passed) + len(skipped),
            error_message=("; ".join(failed) if failed else None),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))
