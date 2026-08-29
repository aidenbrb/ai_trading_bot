"""
Intraday Preflight Node - day-trading (ORB) mode, morning setup.

Runs ~8:30-8:45am ET (see scripts/run_day_preflight.bat), before the 9:30
open, so the time-critical 9:36 job (day_strategy_node) only has to fetch
today's single opening bar. Three steps, each able to fail loudly on its own:

    1. Confirm today is an NYSE trading day (utils/market_calendar).
    2. Refresh the morning news report (nodes/news_node) - this machine has
       no other scheduled full-daily/news run beyond the 30-min monitor-only
       job, so day-mode cannot assume the news gate is already fresh.
    3. Compute/refresh IntradayDailyStats (nodes/intraday_reference_node)
       from prior-session-only data.

Any incomplete step fails the whole preflight loudly (non-zero-looking
result, clear log message) - day_strategy_node's own fail-closed news-gate
check is the actual enforcement point, but a failed preflight should be
visible well before 9:36, not discovered then.

Run standalone:
    python -m nodes.intraday_preflight_node
"""
from __future__ import annotations

import argparse
from datetime import date
from typing import Optional

from utils.market_calendar import is_trading_day


def run(tickers: Optional[list[str]] = None, as_of: Optional[date] = None) -> dict:
    target = as_of or date.today()

    print(f"\n{'='*55}")
    print(f"  INTRADAY PREFLIGHT NODE   date={target}")
    print(f"{'='*55}")

    if not is_trading_day(target):
        print(f"  {target} is not an NYSE trading day - nothing to prepare.")
        return {"status": "skipped", "reason": "not_a_trading_day"}

    print("\n  Step 1/2: refreshing morning news report ...")
    from nodes.news_node import run as news_run
    news_result = news_run(tickers=tickers)
    if news_result.get("status") != "success":
        print(f"  *** PREFLIGHT FAILED: news_node status={news_result.get('status')} ***")
        return {"status": "error", "reason": "news_node_failed", "news_result": news_result}

    print("\n  Step 2/2: computing intraday daily reference stats ...")
    from nodes.intraday_reference_node import run as reference_run
    reference_result = reference_run(tickers=tickers, as_of=target)
    if reference_result.get("failed"):
        print(f"  WARNING: {len(reference_result['failed'])} symbol(s) failed reference "
              f"computation - they will be excluded from today's day-mode signals.")

    print(f"\n  Preflight complete - news OK, "
          f"reference computed={len(reference_result.get('computed', []))}")
    return {
        "status": "success",
        "news_result": news_result,
        "reference_result": reference_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Intraday Preflight Node - morning setup for day mode")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
