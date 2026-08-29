"""
Crypto Strategy Node - Phase 5c of the pipeline (crypto tickers only).

Pure rule-based trend-following momentum strategy tuned for 24/7 crypto markets.
Does NOT call Claude - deterministic rules generate trade setups consistently.

Strategy: Trend-Following + Momentum Confirmation
---------------------------------------------------
Gate 1  BTC macro filter  BTC-USD must be above its 20-day SMA.
        (skipped for BTC itself - BTC IS the macro)
Gate 2  Uptrend structure  price > SMA-20 > SMA-50 (stored as trend="UPTREND")
Gate 3  Momentum zone      RSI-14 between 50 and 75
Gate 4  MACD confirmation  MACD histogram > 0 (positive momentum)
Gate 5  Volume surge       relative volume >= 1.3x 20-day average

Trade parameters (ATR-based, wider than stocks to survive crypto noise):
  Entry  = current close
  Stop   = entry - 2.5 * ATR(14)
  Target = entry + 5.0 * ATR(14)   => R:R of 2.0 (meets MIN_RISK_REWARD)

Writes one Strategy row per ticker (signal=BUY or signal=NONE) using the
shared strategies_run_id so risk_node processes crypto and stocks together.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from types import SimpleNamespace
from utils.timeutil import utcnow
from typing import Optional

import pandas as pd
from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET
from db.connection import get_session, init_db
from db.models import Indicator, OHLCV, RunLog, Strategy, Ticker
from utils.market_intelligence import check_ticker, get_warnings, news_gate_status, overall_risk
from utils.strategy_signals import (
    CRYPTO_STRATEGY_VERSION,
    crypto_trend_momentum_v1,
)

_NODE = "crypto_strategy_node"

# -- Strategy constants --------------------------------------------------------
# -- Public entry point --------------------------------------------------------

def run(
    tickers:          list[str],
    strategies_run_id: Optional[str] = None,
    as_of:            Optional[date] = None,
) -> dict:
    """
    Generate rule-based trade strategies for crypto tickers.

    Parameters
    ----------
    tickers : list[str]
        Crypto symbols to process (e.g. ["BTC-USD", "ETH-USD"]).
    strategies_run_id : str | None
        Shared run_id to write into strategies table. Generated here if None.
        Pass the same value used by strategy_node so risk_node sees both.
    as_of : date | None
        Indicator/price date to use. Defaults to today.

    Returns
    -------
    dict  with keys: passed, failed, skipped, strategies_run_id
    """
    run_id  = strategies_run_id or str(uuid.uuid4())
    started = utcnow()
    target  = as_of or date.today()

    crypto_tickers = [t.upper() for t in tickers if t.upper() in CRYPTO_SET]
    if not crypto_tickers:
        return {"strategies_run_id": run_id, "passed": [], "failed": [], "skipped": []}

    print(f"\n{'='*55}")
    print(f"  CRYPTO STRATEGY NODE   run_id={run_id[:8]}")
    print(f"  Tickers: {len(crypto_tickers)}   Date: {target}")
    print(f"  Strategy: Trend-Following + Momentum (rule-based)")
    print(f"{'='*55}")

    init_db()

    # Fail-closed morning news gate: refuse ALL new BUY signals if today's
    # report is missing, unreadable, or stale (see news_gate_status()).
    mi_data, news_blocked, news_reason = news_gate_status(target.isoformat())
    if news_blocked:
        print(f"\n  *** NEWS GATE: {news_reason} - blocking ALL new BUY signals ***")
        skipped = list(crypto_tickers)
        for symbol in crypto_tickers:
            _write_strategy(run_id, symbol, target, signal="NONE",
                            thesis=f"Blocked: {news_reason}")
        duration = (utcnow() - started).total_seconds()
        _write_log(run_id, [], [], skipped, duration)
        print(f"\n  Done in {duration:.1f}s - BUY=0  NONE={len(skipped)}  failed=0")
        return {"strategies_run_id": run_id, "passed": [], "failed": [], "skipped": skipped}

    if mi_data:
        risk = overall_risk(mi_data)
        print(f"\n  Market Intelligence: {len(mi_data.get('items', []))} items  overall_risk={risk.upper()}")
        print(f"  {mi_data.get('crypto_market_summary', '')}")

    # -- BTC macro check (shared across all altcoins) --------------------------
    btc_ok, btc_note = _btc_macro_check()
    print(f"\n  BTC Macro: {btc_note}")

    passed, failed, skipped = [], [], []

    for i, symbol in enumerate(crypto_tickers, 1):
        print(f"\n  [{i}/{len(crypto_tickers)}] {symbol}")

        # BTC itself skips the macro gate
        if symbol != "BTC-USD" and not btc_ok:
            print(f"    SKIP - BTC below SMA-20, altcoin trades paused")
            skipped.append(symbol)
            _write_strategy(run_id, symbol, target, signal="NONE",
                            thesis=f"Skipped: BTC macro filter - {btc_note}")
            continue

        ticker_id = _get_ticker_id(symbol)
        if not ticker_id:
            print(f"    ERR - ticker not in DB (run data_node first)")
            failed.append(symbol)
            continue

        ind = _get_indicator(ticker_id, target)
        if not ind:
            print(f"    ERR - no indicator row (run indicator_node first)")
            failed.append(symbol)
            continue

        close = _get_latest_close(ticker_id, target)
        if not close:
            print(f"    ERR - no OHLCV data")
            failed.append(symbol)
            continue

        # News gate: block if Market Analysis Bot flagged this crypto
        if mi_data:
            mi_ok, mi_reason = check_ticker(symbol, mi_data, crypto=True)
            if not mi_ok:
                skipped.append(symbol)
                print(f"    SKIP  {mi_reason}")
                _write_strategy(run_id, symbol, target, signal="NONE",
                                ticker_id=ticker_id, thesis=f"Blocked by {mi_reason}")
                continue

        ok, reason, params = _evaluate(symbol, ind, close)

        if ok:
            warnings = get_warnings(symbol, mi_data) if mi_data else []
            if warnings:
                params["thesis"] += f"  [MI warnings: {', '.join(warnings)}]"
            passed.append(symbol)
            print(f"    BUY   entry={params['entry']:.4f}  "
                  f"stop={params['stop']:.4f}  "
                  f"target={params['target']:.4f}  "
                  f"R:R={params['rr']:.2f}  "
                  f"conviction={params['conviction_score']}")
            _write_strategy(run_id, symbol, target, signal="BUY",
                            ticker_id=ticker_id, **params)
        else:
            skipped.append(symbol)
            print(f"    NONE  {reason}")
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


# -- Daily-bar indicator fetch (research use only - not wired into run()) ------
#
# Added after a Phase 1 review found the existing hourly path's SMA20/50/200
# trend calculation actually runs on 20/50/200 HOURLY bars (~0.8/2.1/8.3
# calendar days), not the daily periods the names imply. This gives a live
# equivalent of backtest/whole_bot_engine.py's fetch_daily_crypto_frames() +
# build_daily_crypto_indicator_frames() for the upcoming daily-timeframe
# strategies (crypto_trend_daily_v1, crypto_xsec_momentum_v1). Deliberately
# NOT called from run() above: run() writes Strategy rows that feed the live
# risk/execution gating for the already-registered v1 strategy, and this
# function has no business touching that path while the new strategies are
# still research-only (REGISTRY execution_eligible=False). Data source is
# yfinance (matching _btc_macro_check's existing daily-fetch precedent
# above), not Alpaca (the backtest's exclusive source) - so this is the
# same indicator MATH as the backtest (both ultimately call
# utils.indicators.compute_all()/trend_label()), on a different, real-time
# data source. Daily boundary is UTC midnight, same convention as the
# backtest path.

def _fetch_daily_ohlcv_frame(symbol: str, *, period: str = "2y"):
    """Real daily OHLCV via yfinance, normalized to the UTC-midnight daily
    boundary (same convention as the backtest's native Alpaca daily bars).
    Returns None if unavailable or under 200 rows (not enough for SMA200)."""
    import yfinance as yf

    try:
        raw = yf.Ticker(symbol).history(period=period, interval="1d")
    except Exception:
        return None
    if raw.empty or len(raw) < 200:
        return None

    raw = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    raw.index = raw.index.normalize()  # UTC-midnight daily boundary
    df = raw[["open", "high", "low", "close", "volume"]].dropna()
    return df if len(df) >= 200 else None


def fetch_daily_indicator(symbol: str, *, period: str = "2y") -> Optional[dict]:
    """Compute the same indicator set as utils.indicators.compute_all()
    (SMA20/50/200, RSI14, MACD, ATR14, relative volume, trend label) from
    real daily OHLCV - the live-side counterpart to the backtest's daily
    indicator frame. Returns None if data is unavailable or too short for a
    valid SMA200."""
    from utils.indicators import compute_all

    df = _fetch_daily_ohlcv_frame(symbol, period=period)
    if df is None:
        return None
    values = compute_all(df)
    if values["sma_200"] is None:
        return None
    return values


def evaluate_crypto_trend_daily_v1(
    symbol: str, *, entry_mode: str = "strict_stack", min_rr: float = 2.0, period: str = "2y",
):
    """Live-side evaluation of utils.strategy_signals.crypto_trend_daily_v1
    against real daily OHLCV - research use only (see the module note above
    fetch_daily_indicator: never wired into run(), the new strategy is
    still execution_eligible=False in the registry). Returns None only when
    daily data itself is unavailable/too short; otherwise returns the
    SignalDecision (passed or rejected - callers distinguish via
    .passed, matching every other strategy_signals function's contract)."""
    from utils.indicators import compute_all, sma
    from utils.strategy_signals import (
        CRYPTO_DAILY_DONCHIAN_LOOKBACK,
        CRYPTO_DAILY_SMA50_RISING_LOOKBACK,
        crypto_trend_daily_v1,
    )

    df = _fetch_daily_ohlcv_frame(symbol, period=period)
    if df is None:
        return None
    values = compute_all(df)
    if values["sma_200"] is None:
        return None

    indicator = SimpleNamespace(
        trend=values["trend"], rsi_14=values["rsi_14"], macd_hist=values["macd_hist"],
        rel_volume=values["rel_volume"], atr_14=values["atr_14"],
        sma_20=values["sma_20"], sma_50=values["sma_50"],
    )

    sma_50_series = sma(df["close"], 50)
    prior_idx = len(sma_50_series) - 1 - CRYPTO_DAILY_SMA50_RISING_LOOKBACK
    indicator.sma_50_prior = (
        float(sma_50_series.iloc[prior_idx])
        if prior_idx >= 0 and pd.notna(sma_50_series.iloc[prior_idx]) else None
    )

    window = df["high"].iloc[-1 - CRYPTO_DAILY_DONCHIAN_LOOKBACK:-1]
    indicator.high_20 = float(window.max()) if len(window) == CRYPTO_DAILY_DONCHIAN_LOOKBACK else None

    close = float(df["close"].iloc[-1])
    return crypto_trend_daily_v1(symbol, indicator, close, min_rr=min_rr, entry_mode=entry_mode)


# -- BTC macro filter ----------------------------------------------------------

def _btc_macro_check() -> tuple[bool, str]:
    """Returns (passes, description). True = BTC is above SMA-20 (bullish macro)."""
    try:
        import yfinance as yf
        btc = yf.Ticker("BTC-USD").history(period="30d", interval="1d")
        if btc.empty or len(btc) < 21:
            return False, "BTC data unavailable - crypto entries blocked"
        close   = btc["Close"]
        current = float(close.iloc[-1])
        sma20   = float(close.rolling(20).mean().iloc[-1])
        pct     = (current / sma20 - 1) * 100
        if current > sma20:
            return True, f"BTC ${current:,.0f} above SMA-20 ${sma20:,.0f} (+{pct:.1f}%) - altcoins enabled"
        else:
            return False, f"BTC ${current:,.0f} below SMA-20 ${sma20:,.0f} ({pct:.1f}%) - altcoins paused"
    except Exception as exc:
        return False, f"BTC check error ({exc}) - crypto entries blocked"


# -- Rule evaluation -----------------------------------------------------------

def _evaluate(
    symbol: str,
    ind:    Indicator,
    close:  float,
) -> tuple[bool, str, dict]:
    """Compatibility wrapper around the shared frozen v1 rule."""
    return crypto_trend_momentum_v1(
        symbol,
        ind,
        close,
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
            .order_by(OHLCV.bar_time.desc())   # type: ignore[attr-defined]
        ).first()
        return float(row.close) if row else None


def _write_strategy(
    run_id:          str,
    symbol:          str,
    bar_date:        date,
    signal:          str,
    ticker_id:       Optional[str] = None,
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
            strategy_name="CryptoTrend-Momentum",
            signal=signal,
            entry=entry,
            stop=stop,
            target=target,
            rr=rr,
            conviction_score=conviction_score,
            thesis=thesis,
            model_used=CRYPTO_STRATEGY_VERSION,
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
