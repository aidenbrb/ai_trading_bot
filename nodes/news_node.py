"""
News Node - morning news/RSS gate, phase "news" of the pipeline.

Builds the first-party morning market-intelligence report that
utils/market_intelligence.py's news_gate_status() reads before any BUY
strategy is generated. Runs once per pipeline invocation, before "strategy".

Sources (all free, no API key required - see utils/news.py):
  - Per-ticker headlines : yfinance.Ticker(symbol).news, keyword-classified
  - Macro / market-wide  : CNBC + Federal Reserve RSS, keyword-classified
  - Volatility proxy     : ^VIX level + SPY ATR-14 vs its own 20-day average

Writes data/market_intelligence/YYYY-MM-DD.json + data/run_status.json in the
same schema the old external "Market Analysis Bot" reader already understood,
so stock_strategy_node.py / crypto_strategy_node.py / risk_node.py need no
changes to how they consume the report - only to how strictly they enforce it
being present and fresh (see utils/market_intelligence.py::news_gate_status).

On any uncaught failure this still writes run_status.json with status="error"
rather than leaving a stale/missing status file that could look "ready" -
fail loud AND fail closed.

Run standalone:
    python -m nodes.news_node
    python -m nodes.news_node --tickers AAPL MSFT
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import date
from utils.timeutil import utcnow

from db.connection import get_session, init_db
from db.models import RunLog
from config.settings import settings
from utils.market_intelligence import data_dir, status_file
from utils.news import (
    MACRO_FEED_COUNT,
    classify_ticker_headline,
    classify_macro_headline,
    fetch_macro_feeds,
    fetch_ticker_news,
    get_volatility_regime,
    is_recent_headline,
)

_NODE = "news_node"


def run(tickers: list[str] | None = None) -> dict:
    """
    Build today's morning news/market-intelligence report.

    Returns
    -------
    dict with keys: run_id, status, overall_market_risk, items, source_errors
    """
    run_id  = str(uuid.uuid4())
    started = utcnow()
    today   = date.today()

    print(f"\n{'='*55}")
    print(f"  NEWS NODE   run_id={run_id[:8]}")
    print(f"{'='*55}")

    if tickers:
        symbols = [t.upper() for t in tickers]
    else:
        from config.universe import CRYPTO, UNIVERSE
        symbols = list(UNIVERSE) + (
            CRYPTO if settings.CRYPTO_EXECUTION_ENABLED else []
        )

    try:
        items: list[dict] = []

        print(f"  Fetching per-ticker news for {len(symbols)} symbols ...")
        from config.universe import ETFS
        etf_symbols = set(ETFS)
        for i, symbol in enumerate(symbols, 1):
            for headline in fetch_ticker_news(symbol):
                if not is_recent_headline(headline["pub_date"]):
                    continue
                label = classify_ticker_headline(
                    symbol,
                    headline["headline"],
                    headline["summary"],
                    is_etf=symbol in etf_symbols,
                )
                if label:
                    items.append({
                        "symbol":            symbol,
                        "related_symbols":   headline["related_symbols"],
                        "importance_score":  70 if label == "avoid_new_trade" else 50,
                        "catalyst_type":     label,
                        "bot_context_label": label,
                        "sentiment":         "negative",
                        "headline":          headline["headline"],
                        "pub_date":          headline["pub_date"],
                        "relevance_source":  headline["relevance_source"],
                    })
            if i % 20 == 0:
                print(f"    ... {i}/{len(symbols)}")

        print("  Fetching macro/market-wide feeds ...")
        macro_items, source_errors = fetch_macro_feeds()
        macro_severity = "low"
        macro_outage_reason = ""
        if MACRO_FEED_COUNT and len(source_errors) >= MACRO_FEED_COUNT:
            macro_severity = "high"
            macro_outage_reason = (
                "All configured macro news feeds failed; macro visibility "
                "is unavailable, so new trades are blocked."
            )
            print(f"  *** MACRO NEWS OUTAGE: {macro_outage_reason} ***")
            items.append({
                "symbol": None,
                "related_symbols": [],
                "importance_score": 100,
                "catalyst_type": "macro_data_outage",
                "bot_context_label": "avoid_new_trade",
                "sentiment": "negative",
                "headline": macro_outage_reason,
            })
        for m in macro_items:
            label = classify_macro_headline(f"{m['headline']} {m['summary']}")
            if label == "high":
                macro_severity = "high"
                items.append({
                    "symbol": None, "related_symbols": [], "importance_score": 90,
                    "catalyst_type": "macro", "bot_context_label": "avoid_new_trade",
                    "sentiment": "negative", "headline": m["headline"],
                })
            elif label == "warn":
                if macro_severity == "low":
                    macro_severity = "medium"
                items.append({
                    "symbol": None, "related_symbols": [], "importance_score": 40,
                    "catalyst_type": "macro", "bot_context_label": "macro_risk",
                    "sentiment": "neutral", "headline": m["headline"],
                })

        print("  Checking volatility regime ...")
        vol = get_volatility_regime()
        print(f"  Volatility: {vol['regime'].upper()}  "
              f"VIX={vol['vix']}  SPY ATR ratio={vol['spy_atr_ratio']}")

        if vol["regime"] == "high" or macro_severity == "high":
            overall_risk = "high"
        elif vol["regime"] == "elevated" or macro_severity == "medium":
            overall_risk = "medium"
        else:
            overall_risk = "low"

        report = {
            "report_date":           today.isoformat(),
            "generated_at":          utcnow().isoformat(),
            "overall_market_risk":   overall_risk,
            "macro_summary":         (
                macro_outage_reason
                or f"Volatility={vol['regime']}  VIX={vol['vix']}"
            ),
            "stock_market_summary":  "",
            "crypto_market_summary": "",
            "commodity_summary":     "",
            "items":                 items,
            "source_errors":         source_errors,
        }

        data_dir().mkdir(parents=True, exist_ok=True)
        (data_dir() / f"{today.isoformat()}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        status_file().write_text(
            json.dumps({"status": "success", "date": today.isoformat(),
                        "generated_at": report["generated_at"]}, indent=2),
            encoding="utf-8",
        )

        duration = (utcnow() - started).total_seconds()
        _write_log(run_id, "success", len(items), duration)

        print(f"\n  Done in {duration:.1f}s - items={len(items)}  "
              f"overall_risk={overall_risk.upper()}  "
              f"source_errors={len(source_errors)}")

        return {
            "run_id":              run_id,
            "status":              "success",
            "overall_market_risk": overall_risk,
            "items":               len(items),
            "source_errors":       source_errors,
        }

    except Exception as exc:
        # Fail loud AND fail closed: never leave a stale/missing status file
        # that could accidentally look "ready" to news_gate_status().
        print(f"\n  *** NEWS NODE FAILED: {exc} ***")
        try:
            status_file().parent.mkdir(parents=True, exist_ok=True)
            status_file().write_text(
                json.dumps({"status": "error", "date": today.isoformat(),
                            "error": str(exc)}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

        duration = (utcnow() - started).total_seconds()
        _write_log(run_id, "error", 0, duration, error_message=str(exc))

        return {
            "run_id":              run_id,
            "status":              "error",
            "overall_market_risk": "unknown",
            "items":               0,
            "source_errors":       [],
        }


def _write_log(run_id: str, status: str, items: int, duration: float,
               error_message: str | None = None) -> None:
    init_db()
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status=status,
            tickers_processed=None,
            records_written=items,
            error_message=error_message,
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="News Node - morning news/RSS gate")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
