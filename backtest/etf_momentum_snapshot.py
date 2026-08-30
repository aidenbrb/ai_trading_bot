"""Data snapshot for etf_momentum_v1 (preregistration Section 4).

Fetches each of the 21 universe tickers from yfinance EXACTLY ONCE, both
raw (auto_adjust=False, with dividends) and adjusted (auto_adjust=True,
genuine split+dividend total-return series), and persists both to
parquet under backtest/data/etf_momentum_v1/. Writes a manifest recording
each file's SHA-256, the pinned yfinance version, and the fetch
timestamp.

Every other etf_momentum_v1 module reads exclusively from this snapshot
via load_snapshot() below - none of them call yfinance directly. Running
this script again overwrites the snapshot with a fresh fetch (a new
manifest, new timestamp); it is not re-run automatically by anything else
in this package.

Run: python -m backtest.etf_momentum_snapshot
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKERS: list[str] = [
    "SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU",
    "XLV", "XLY", "QQQ", "IWM", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "BIL",
]

# Effective window start (2008-07-01) minus 13 months' lead time for the
# deepest grid lookback (lookback_months=12, skip_last_month=1).
FETCH_START = date(2007, 6, 1)
FETCH_END = date(2026, 8, 1)

DATA_DIR = Path(__file__).parent / "data" / "etf_momentum_v1"
MANIFEST_PATH = DATA_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    frame = frame.copy()
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index = frame.index.normalize()
    frame.index.name = "date"
    return frame


def fetch_snapshot() -> dict:
    """Fetches and persists the snapshot. Returns the manifest dict."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = datetime.now(timezone.utc).isoformat()
    files: dict[str, str] = {}
    per_ticker: dict[str, dict] = {}

    for ticker in TICKERS:
        raw = _normalize(yf.Ticker(ticker).history(
            start=FETCH_START.isoformat(), end=FETCH_END.isoformat(), auto_adjust=False,
        ))
        adjusted = _normalize(yf.Ticker(ticker).history(
            start=FETCH_START.isoformat(), end=FETCH_END.isoformat(), auto_adjust=True,
        ))
        raw_path = DATA_DIR / f"{ticker}_raw.parquet"
        adjusted_path = DATA_DIR / f"{ticker}_adjusted.parquet"
        raw.to_parquet(raw_path)
        adjusted.to_parquet(adjusted_path)
        files[raw_path.relative_to(DATA_DIR).as_posix()] = _sha256(raw_path)
        files[adjusted_path.relative_to(DATA_DIR).as_posix()] = _sha256(adjusted_path)
        per_ticker[ticker] = {
            "requested_start": FETCH_START.isoformat(),
            "requested_end": FETCH_END.isoformat(),
            "raw_rows": int(len(raw)),
            "adjusted_rows": int(len(adjusted)),
            "first_bar": str(raw.index.min().date()) if not raw.empty else None,
            "last_bar": str(raw.index.max().date()) if not raw.empty else None,
        }
        print(f"{ticker}: raw={len(raw)} adjusted={len(adjusted)} "
              f"first={per_ticker[ticker]['first_bar']}", flush=True)

    manifest = {
        "strategy_version": "etf_momentum_v1",
        "yfinance_version": yf.__version__,
        "fetched_at_utc": fetched_at,
        "tickers": TICKERS,
        "requested_start": FETCH_START.isoformat(),
        "requested_end": FETCH_END.isoformat(),
        "per_ticker": per_ticker,
        "files": files,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_sha256"] = _sha256(MANIFEST_PATH)
    return manifest


def load_snapshot() -> dict[str, dict[str, pd.DataFrame]]:
    """Loads the persisted snapshot - {ticker: {"raw": df, "adjusted": df}}.
    Fails closed (raises) if the manifest or any file is missing; never
    fetches from yfinance itself."""
    if not MANIFEST_PATH.exists():
        raise RuntimeError(
            f"etf_momentum_v1 data snapshot not found at {MANIFEST_PATH} - "
            "run `python -m backtest.etf_momentum_snapshot` first. This "
            "loader never fetches from yfinance itself (preregistration "
            "Section 4)."
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for ticker in manifest["tickers"]:
        raw_path = DATA_DIR / f"{ticker}_raw.parquet"
        adjusted_path = DATA_DIR / f"{ticker}_adjusted.parquet"
        if not raw_path.exists() or not adjusted_path.exists():
            raise RuntimeError(f"etf_momentum_v1 snapshot missing files for {ticker}")
        result[ticker] = {
            "raw": pd.read_parquet(raw_path),
            "adjusted": pd.read_parquet(adjusted_path),
        }
    return result


def manifest_sha256() -> str:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"manifest not found at {MANIFEST_PATH}")
    return _sha256(MANIFEST_PATH)


if __name__ == "__main__":
    result = fetch_snapshot()
    print(f"\nSnapshot written to {DATA_DIR}")
    print(f"manifest sha256: {manifest_sha256()}")
