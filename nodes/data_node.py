"""
Data Node - Phase 1 of the pipeline.

Responsibilities:
  1. Seed the tickers table from config/universe.py (first run only)
  2. Download OHLCV bars for every active ticker:
       Crypto  -> yfinance (fallback: Coinbase public market-data API)
       Stocks  -> Alpaca Market Data API       (fallback: yfinance)
  3. Write new bars to the ohlcv table (skips bars already stored)
  4. Log results to run_log

Run standalone:
    python -m nodes.data_node
    python -m nodes.data_node --tickers AAPL MSFT SPY
    python -m nodes.data_node --tickers AAPL --days 400

This node makes NO calls to Claude.
"""
from __future__ import annotations

import argparse
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from utils.timeutil import utcnow

import pandas as pd
import yfinance as yf
from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET, SECTOR_MAP, UNIVERSE
from db.connection import get_session, init_db
from db.models import OHLCV, RunLog, Ticker

_NODE = "data_node"


# -- Public entry point --------------------------------------------------------

def run(tickers: list[str] | None = None, days: int | None = None) -> dict:
    """
    Download and store OHLCV data.

    Parameters
    ----------
    tickers : list[str] | None
        Symbols to process. Defaults to the full UNIVERSE.
    days : int | None
        How many calendar days of history to request.
        Defaults to settings.DOWNLOAD_DAYS (800).

    Returns
    -------
    dict  with keys: passed, failed, skipped, run_id
    """
    run_id   = str(uuid.uuid4())
    started  = utcnow()
    symbols  = [t.upper() for t in (tickers or UNIVERSE)]
    dl_days  = days or settings.DOWNLOAD_DAYS

    print(f"\n{'='*55}")
    print(f"  DATA NODE   run_id={run_id[:8]}")
    print(f"  Tickers: {len(symbols)}   History: {dl_days} days")
    print(f"{'='*55}")

    init_db()
    _seed_tickers()

    passed, failed, skipped = [], [], []
    total = len(symbols)

    for i, symbol in enumerate(symbols, 1):
        print(f"  [{i:>3}/{total}] {symbol:<6} ", end="", flush=True)
        result = _process(symbol, dl_days)
        if result["status"] == "ok":
            passed.append(symbol)
            print(f"OK  {result['new_bars']} new bars")
        elif result["status"] == "skipped":
            skipped.append(symbol)
            print(f"--  up to date")
        else:
            failed.append({"symbol": symbol, "reason": result["reason"]})
            print(f"ERR  {result['reason']}")

    duration = (utcnow() - started).total_seconds()

    _write_log(run_id, passed, failed, skipped, duration)

    print(f"\n  Done in {duration:.1f}s - "
          f"downloaded={len(passed)}  skipped={len(skipped)}  failed={len(failed)}")

    return {
        "run_id":  run_id,
        "passed":  passed,
        "failed":  failed,
        "skipped": skipped,
    }


# -- Private helpers -----------------------------------------------------------

def _seed_tickers() -> None:
    """Insert any tickers from the universe that aren't in the DB yet."""
    from config.universe import CRYPTO
    with get_session() as session:
        existing = {t.symbol for t in session.exec(select(Ticker)).all()}
        seen_this_batch: set[str] = set()
        new_tickers = []
        for sector, symbols in SECTOR_MAP.items():
            for sym in symbols:
                if sym not in existing and sym not in seen_this_batch:
                    new_tickers.append(Ticker(symbol=sym, sector=sector))
                    seen_this_batch.add(sym)
        for sym in CRYPTO:
            if sym not in existing and sym not in seen_this_batch:
                new_tickers.append(Ticker(symbol=sym, sector="Crypto"))
                seen_this_batch.add(sym)
        if new_tickers:
            for t in new_tickers:
                session.add(t)
            print(f"  Seeded {len(new_tickers)} new ticker(s) into DB")


def _get_ticker_id(symbol: str) -> str | None:
    with get_session() as session:
        row = session.exec(select(Ticker).where(Ticker.symbol == symbol)).first()
        return row.id if row else None


def _latest_time(ticker_id: str) -> datetime | None:
    """Return the most recent bar timestamp already stored for this ticker."""
    with get_session() as session:
        row = session.exec(
            select(OHLCV)
            .where(OHLCV.ticker_id == ticker_id)
            .order_by(OHLCV.bar_time.desc())  # type: ignore[attr-defined]
        ).first()
        return row.bar_time if row else None


def _process(symbol: str, dl_days: int) -> dict:
    ticker_id = _get_ticker_id(symbol)
    if ticker_id is None:
        with get_session() as session:
            t = Ticker(symbol=symbol, sector="Unknown")
            session.add(t)
        ticker_id = _get_ticker_id(symbol)

    latest = _latest_time(ticker_id)
    end    = date.today() + timedelta(days=1)

    if latest:
        # Skip if last bar is less than 30 minutes old (already current)
        if (utcnow() - latest).total_seconds() < 1800:
            return {"status": "skipped", "new_bars": 0}
        # Re-fetch from the date of the last bar (catches new bars added today)
        start = latest.date()
    else:
        start = date.today() - timedelta(days=dl_days)

    try:
        df = _download(symbol, start, end)
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}

    if df.empty:
        return {"status": "skipped", "new_bars": 0}

    try:
        new_bars = _upsert_bars(ticker_id, df)
    except Exception as exc:
        return {"status": "error", "reason": str(exc)}
    if new_bars == 0:
        return {"status": "skipped", "new_bars": 0}
    return {"status": "ok", "new_bars": new_bars}


def _download(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Route to the best available data source with yfinance fallback."""
    if symbol.upper() in CRYPTO_SET:
        # Yahoo is much faster for long history. Some valid Coinbase products
        # have no Yahoo history, so fall back to Coinbase's unauthenticated,
        # read-only public candles endpoint. No Coinbase key is needed.
        try:
            df = _download_yfinance(symbol, start, end)
            if not df.empty:
                return df
        except Exception as exc:
            print(f"(yfinance failed: {exc} - trying public Coinbase) ", end="")
        return _download_coinbase(symbol, start, end)
    else:
        if settings.ALPACA_API_KEY and settings.ALPACA_SECRET_KEY:
            try:
                df = _download_alpaca(symbol, start, end)
                if not df.empty:
                    return df
            except Exception as exc:
                print(f"(Alpaca failed: {exc} - falling back to yfinance) ", end="")
        return _download_yfinance(symbol, start, end)


def _download_coinbase(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly OHLCV from Coinbase's public, unauthenticated API."""
    import httpx

    # Coinbase max 300 candles per request at ONE_HOUR granularity
    step_secs  = 3600
    chunk_size = 290
    start_ts   = int(datetime.combine(start, datetime.min.time())
                     .replace(tzinfo=timezone.utc).timestamp())
    end_ts     = int(datetime.combine(end, datetime.min.time())
                     .replace(tzinfo=timezone.utc).timestamp())

    rows = []
    failures = []
    current = start_ts
    url = (
        "https://api.coinbase.com/api/v3/brokerage/market/products/"
        f"{symbol}/candles"
    )
    with httpx.Client(
        timeout=20,
        headers={"User-Agent": "ai-trading-bot/1.0"},
    ) as client:
        while current < end_ts:
            chunk_end = min(current + chunk_size * step_secs, end_ts)
            try:
                resp = client.get(
                    url,
                    params={
                        "start": str(current),
                        "end": str(chunk_end),
                        "granularity": "ONE_HOUR",
                        "limit": chunk_size,
                    },
                )
                resp.raise_for_status()
                candles = resp.json().get("candles", [])
                for candle in candles:
                    rows.append({
                        "bar_time": datetime.fromtimestamp(
                            int(candle["start"]), tz=timezone.utc
                        ).replace(tzinfo=None),
                        "open":   float(candle["open"]),
                        "high":   float(candle["high"]),
                        "low":    float(candle["low"]),
                        "close":  float(candle["close"]),
                        "volume": float(candle["volume"]),
                    })
            except Exception as exc:
                failures.append(str(exc))
            time.sleep(0.08)   # ~12 req/s - within Coinbase limits
            current = chunk_end

    if not rows:
        detail = failures[0] if failures else "no candles returned"
        raise RuntimeError(f"public Coinbase data unavailable for {symbol}: {detail}")

    if failures:
        print(f"(public Coinbase missed {len(failures)} chunk(s)) ", end="")

    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="bar_time").sort_values("bar_time")
    df = df.set_index("bar_time")
    df.index.name = "bar_time"
    return df[["open", "high", "low", "close", "volume"]]


def _download_alpaca(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly OHLCV from Alpaca Market Data API."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
    )

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=datetime.combine(start, datetime.min.time()),
        end=datetime.combine(end, datetime.min.time()),
        feed="iex",   # free tier; upgrade to "sip" with paid plan
    )

    bars = client.get_stock_bars(request)
    df   = bars.df

    if df is None or df.empty:
        return pd.DataFrame()

    # bars.df has MultiIndex (symbol, timestamp) - drop the symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0)

    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    df.index.name = "bar_time"

    df = df.rename(columns={
        "open": "open", "high": "high",
        "low":  "low",  "close": "close", "volume": "volume",
    })
    return df[["open", "high", "low", "close", "volume"]].copy()


def _download_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly OHLCV from yfinance (fallback)."""
    raw = yf.Ticker(symbol).history(
        start=str(start),
        end=str(end),
        interval=settings.BAR_INTERVAL,
        auto_adjust=True,
        back_adjust=False,
    )
    if raw.empty:
        return raw

    if raw.index.tz is not None:
        raw.index = raw.index.tz_convert("UTC").tz_localize(None)
    raw.index.name = "bar_time"
    raw = raw.rename(columns={
        "Open": "open", "High": "high",
        "Low": "low", "Close": "close", "Volume": "volume",
    })
    return raw[["open", "high", "low", "close", "volume"]].copy()


def _upsert_bars(ticker_id: str, df: pd.DataFrame) -> int:
    """Insert bars that don't already exist. Returns count of new rows."""
    if df.empty:
        return 0

    # Drop bars with any missing/NaN OHLCV value. A NaN float bound to a
    # NOT NULL SQLite column raises "NOT NULL constraint failed" (it is NOT
    # a silent no-op), which previously crashed the whole data_node run - and
    # every ticker after the offending one - the moment yfinance returned one
    # bad bar (illiquid pre-market bar, data-provider gap, etc).
    clean = df.dropna(subset=["open", "high", "low", "close", "volume"])
    dropped = len(df) - len(clean)
    if dropped:
        print(f"(dropped {dropped} bar(s) with missing OHLCV data) ", end="")
    df = clean
    if df.empty:
        return 0

    min_t = df.index.min().to_pydatetime().replace(tzinfo=None)
    max_t = df.index.max().to_pydatetime().replace(tzinfo=None)

    with get_session() as session:
        existing_times = {
            row.bar_time for row in session.exec(
                select(OHLCV)
                .where(OHLCV.ticker_id == ticker_id)
                .where(OHLCV.bar_time >= min_t)
                .where(OHLCV.bar_time <= max_t)
            ).all()
        }
        new = 0
        for ts, row in df.iterrows():
            t = ts.to_pydatetime().replace(tzinfo=None, minute=0, second=0, microsecond=0)
            if t not in existing_times:
                session.add(OHLCV(
                    ticker_id=ticker_id,
                    bar_time=t,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
                new += 1
        return new


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
    parser = argparse.ArgumentParser(description="Data Node - download OHLCV")
    parser.add_argument("--tickers", nargs="+", help="symbols to process")
    parser.add_argument("--days", type=int, help="days of history to fetch")
    args = parser.parse_args()
    run(tickers=args.tickers, days=args.days)
