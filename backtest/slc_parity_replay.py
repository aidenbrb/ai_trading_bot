"""Diagnostic: replay slc_4h_5m_stock_v1's offline generate_signals()
pipeline on IEX-sourced bars (matching the live feed) instead of the SIP
bars the historical backtest uses, and compare candidate-by-candidate
against the live paper-trading record.

Origin: an initial parity check (offline/SIP engine vs. the live paper
log) found zero admitted-trade matches over 2026-08-17..08-28. The
data-feed difference (offline reads Alpaca SIP; live reads Alpaca IEX,
per amendment_004) was the leading suspect. This script tested that
directly - replaying on the literal bars live consumed (live_slc.db's own
persisted bar cache) plus a fresh IEX fetch as a stated proxy for the
pre-window warmup (live's bootstrap() never persists its own raw
bootstrap bars, only the derived ReducerState).

Result (see the Crypto Evidence Ledger, SLC section, for the full
writeup): the feed switch alone raised the admitted-trade match rate
(0/16 -> 4/16) but did not change the clustering pattern, ruling out
"wrong feed" as the explanation. Tracing individual candidates found
byte-exact agreement between the offline engine and live's own reducer
on symbol/level_id/confirmation-minute in every case checked - the
signal-generation rules are consistent. The remaining admission gap
traced to two real mechanisms with no offline analog: deployment
downtime (dry_run/suspended stretches in the signed activation-event
log) and broker-side execution rejections (rounded_bracket_invalid_pre_
submission, stale_or_missing_data, canceled_unfilled) that consume no
capacity slot, letting a later-ranked candidate fill in a top pick's
place.

STANDING RULE this established for any future strategy's live/offline
parity work: compare pre-cap CANDIDATE streams, never admitted fills.
Any per-day (or per-symbol) admission cap makes "admitted trades"
order-sensitive - small timing differences, ranking ties, deployment
downtime, or execution-layer rejections can each independently swap
which candidates get the scarce slots, producing large apparent
divergence even when the underlying signal logic is identical. This
script's own admitted-trade comparison is kept only as a secondary
check; the candidate-stream counts and the byte-exact per-candidate
traces are what actually settled the question. And: backtest on the
feed the strategy will trade - a backtest on a different feed than the
live system uses is evidence about a different, related engine, not the
one running live.

Read-only against live_slc/live_slc.db (its own dedicated read-only
sqlite3 connection, never live_slc.models' read-write ORM session).
Never imports or calls a trading/order client.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from config.universe import UNIVERSE
from live_slc.models import LIVE_SLC_DB_PATH
from utils.alpaca_bars import fetch_bars
from utils.market_calendar import session_for
from utils.slc_signals import generate_signals
from backtest.run_slc_backtest import simulate_portfolio
from backtest.whole_bot_engine import COSTS, PORTFOLIOS

START = date(2026, 8, 17)
END = date(2026, 8, 28)
LOOKBACK_DAYS = 120
IEX_CACHE_PATH = Path(__file__).parent / "cache" / "slc_iex_proxy_cache.db"
REPORT_PATH = Path(__file__).parent / "results" / "slc_4h_5m_stock_v1" / "parity_replay_iex_report.json"

SYMBOLS = sorted(set(UNIVERSE) | {"SPY"})


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"],
                         index=pd.DatetimeIndex([], name="bar_time"))


def _rth_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    keep = []
    for ts in frame.index:
        session = session_for(pd.Timestamp(ts).date())
        keep.append(session is not None and session["open"] <= ts < session["close"])
    return frame.loc[keep]


def _iex_cache_connect() -> sqlite3.Connection:
    IEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(IEX_CACHE_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS bars (
        symbol TEXT, bar_time TEXT, open REAL, high REAL, low REAL,
        close REAL, volume REAL, PRIMARY KEY (symbol, bar_time))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS fetched (
        symbol TEXT PRIMARY KEY, start TEXT, end TEXT)""")
    return conn


def fetch_iex_lookback(symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """Fresh Alpaca IEX fetch, cached - the stated proxy for live's
    pre-window bootstrap bars (see module docstring: live never persists
    those raw)."""
    conn = _iex_cache_connect()
    try:
        done = {row[0] for row in conn.execute(
            "SELECT symbol FROM fetched WHERE start=? AND end=?", (str(start), str(end))
        )}
        needed = [s for s in symbols if s not in done]
        if needed:
            batch_size = 25
            for offset in range(0, len(needed), batch_size):
                batch = needed[offset: offset + batch_size]
                print(f"  iex lookback fetch {offset + len(batch)}/{len(needed)}", flush=True)
                fresh = fetch_bars(batch, start, end, amount=5, unit="Minute", feed="iex")
                with conn:
                    for symbol in batch:
                        frame = fresh.get(symbol, _empty())
                        frame = _rth_filter(frame)
                        if frame is not None and not frame.empty:
                            conn.executemany(
                                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)",
                                [(symbol, str(ts), r.open, r.high, r.low, r.close, r.volume)
                                 for ts, r in frame.iterrows()],
                            )
                        conn.execute("INSERT OR REPLACE INTO fetched VALUES (?,?,?)",
                                     (symbol, str(start), str(end)))
        result = {}
        for symbol in symbols:
            rows = conn.execute(
                "SELECT bar_time,open,high,low,close,volume FROM bars WHERE symbol=? "
                "AND bar_time>=? AND bar_time<? ORDER BY bar_time",
                (symbol, str(start), str(end)),
            ).fetchall()
            if not rows:
                result[symbol] = _empty()
                continue
            frame = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
            frame["bar_time"] = pd.to_datetime(frame["bar_time"], format="mixed")
            result[symbol] = frame.set_index("bar_time")
        return result
    finally:
        conn.close()


def load_live_persisted_bars(symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
    """The literal bars live's reducer consumed - a dedicated read-only
    connection to live_slc.db, never the read-write ORM session."""
    uri = f"file:{Path(LIVE_SLC_DB_PATH).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        result = {}
        for symbol in symbols:
            rows = conn.execute(
                "SELECT bar_time,open,high,low,close,volume FROM slc_five_min_bars "
                "WHERE symbol=? AND bar_time>=? AND bar_time<? ORDER BY bar_time",
                (symbol, str(start), str(end)),
            ).fetchall()
            if not rows:
                result[symbol] = _empty()
                continue
            frame = pd.DataFrame(rows, columns=["bar_time", "open", "high", "low", "close", "volume"])
            frame["bar_time"] = pd.to_datetime(frame["bar_time"], format="mixed")
            result[symbol] = frame.set_index("bar_time")
        return result
    finally:
        conn.close()


def load_real_live_trades() -> list[dict]:
    uri = f"file:{Path(LIVE_SLC_DB_PATH).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute("""
            SELECT p.symbol, p.direction, p.confirmation_time, p.level_id,
                   t.exit_time, t.net_pnl
            FROM slc_positions p JOIN slc_trades t ON t.position_id = p.id
            WHERE p.confirmation_time >= ? AND p.confirmation_time < ?
            ORDER BY p.confirmation_time
        """, (str(START), str(END + timedelta(days=1)))).fetchall()
        return [
            {"symbol": r[0], "direction": r[1], "confirmation_time": r[2],
             "level_id": r[3], "exit_time": r[4], "net_pnl": r[5]}
            for r in rows
        ]
    finally:
        conn.close()


def main() -> None:
    lookback_start = datetime.combine(START - timedelta(days=LOOKBACK_DAYS), datetime.min.time())
    window_start = datetime.combine(START, datetime.min.time())
    window_end = datetime.combine(END + timedelta(days=1), datetime.min.time())

    print(f"Fetching IEX proxy lookback bars: {lookback_start} .. {window_start} for {len(SYMBOLS)} symbols")
    lookback_frames = fetch_iex_lookback(SYMBOLS, lookback_start, window_start)

    print(f"Loading literal live-persisted bars: {window_start} .. {window_end}")
    live_frames = load_live_persisted_bars(SYMBOLS, window_start, window_end)

    merged: dict[str, pd.DataFrame] = {}
    for symbol in SYMBOLS:
        parts = [p for p in (lookback_frames.get(symbol), live_frames.get(symbol)) if p is not None and not p.empty]
        if not parts:
            merged[symbol] = _empty()
            continue
        frame = pd.concat(parts).sort_index()
        merged[symbol] = frame[~frame.index.duplicated(keep="last")]

    coverage_rows = []
    for symbol in SYMBOLS:
        coverage_rows.append({
            "symbol": symbol,
            "lookback_bars": len(lookback_frames.get(symbol, _empty())),
            "live_window_bars": len(live_frames.get(symbol, _empty())),
        })

    print("Running generate_signals() (unmodified) per symbol...")
    all_signals = []
    research_symbols = [s for s in SYMBOLS if s != "SPY"]
    for i, symbol in enumerate(research_symbols, 1):
        if i % 20 == 0 or i == len(research_symbols):
            print(f"  signals {i}/{len(research_symbols)}", flush=True)
        symbol_signals = generate_signals(symbol, merged.get(symbol, _empty()))
        for signal in symbol_signals:
            entry_date = pd.Timestamp(signal.entry_time).date()
            if START <= entry_date <= END:
                all_signals.append(signal)

    all_signals.sort(key=lambda s: (s.entry_time, s.symbol, s.level_id))

    candidates_by_day: dict[str, int] = {}
    for signal in all_signals:
        day = str(pd.Timestamp(signal.entry_time).date())
        candidates_by_day[day] = candidates_by_day.get(day, 0) + 1

    # Admission pipeline: no outcome data needed for admission itself, only
    # entry-time/rank/capacity - outcomes are irrelevant to the parity
    # question (which trades get ADMITTED), so pass an empty outcomes map;
    # simulate_portfolio() marks every admitted trade "outcome_data_missing"
    # which is fine, we only read entry-side fields off `trades`. This
    # admitted-trade comparison is secondary - see module docstring's
    # standing rule: candidate streams are what actually settle parity.
    result = simulate_portfolio(
        all_signals, {}, portfolio=PORTFOLIOS["safe_0_25pct"], cost=COSTS["baseline"],
        start_date=START, end_date=END,
    )
    admitted = result["trades"]

    real_trades = load_real_live_trades()

    def _minute(ts) -> str:
        return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")

    admitted_index = {(t["symbol"], _minute(t["entry_time"])) for t in admitted}
    matches = []
    misses = []
    for rt in real_trades:
        key = (rt["symbol"], _minute(rt["confirmation_time"]))
        if key in admitted_index:
            matches.append(rt)
        else:
            misses.append(rt)

    entry_hours = [pd.Timestamp(t["entry_time"]).strftime("%H:%M") for t in admitted]

    report = {
        "window": {"start": str(START), "end": str(END)},
        "universe_size": len(SYMBOLS),
        "caveat": (
            "Pre-2026-08-17 bars are a fresh IEX historical fetch (proxy for "
            "live's real bootstrap, which is not persisted raw) - not proven "
            "byte-identical to what live's bootstrap actually consumed. "
            "2026-08-17..08-28 bars are the literal live-persisted bars "
            "(live_slc.db::slc_five_min_bars), gaps included. No intraday "
            "outcome simulation was run (admission-only test): every admitted "
            "trade uses the existing outcome_data_missing fallback (held to "
            "session close, net_pnl=0), which can block a legitimate second "
            "same-symbol same-day entry via the already_held check - a small, "
            "known approximation, irrelevant to P&L, immaterial to the 2/day "
            "portfolio-wide cap that drives most rejections."
        ),
        "pre_cap_candidates_total": len(all_signals),
        "pre_cap_candidates_by_day": candidates_by_day,
        "admitted_trades_total": len(admitted),
        "admitted_entry_times": sorted(entry_hours)[:30],
        "real_live_trades_total": len(real_trades),
        "matches": len(matches),
        "match_rate": len(matches) / len(real_trades) if real_trades else None,
        "matched_trades": matches,
        "missed_real_trades": misses,
        "admitted_trades_sample": [
            {"symbol": t["symbol"], "direction": t["direction"], "entry_time": str(t["entry_time"]),
             "level_id": t["level_id"]}
            for t in admitted
        ],
        "coverage": coverage_rows,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nDONE. Report: {REPORT_PATH}")
    print(f"pre-cap candidates total: {len(all_signals)}")
    print(f"admitted trades: {len(admitted)}")
    print(f"real live trades: {len(real_trades)}")
    print(f"matches: {len(matches)} ({report['match_rate']})")


if __name__ == "__main__":
    main()
