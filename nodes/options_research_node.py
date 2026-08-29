"""Options research node: selects long-call contracts and never submits orders."""
from __future__ import annotations

import argparse
import uuid
from datetime import date, timedelta
from typing import Optional

from sqlmodel import select

from config.settings import settings
from db.connection import get_session, init_db
from db.models import OptionsResearch, RunLog, Strategy, Ticker
from utils.options import OptionCandidate, select_candidate, size_long_option
from utils.timeutil import utcnow

_NODE = "options_research_node"


def run(
    strategies_run_id: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    as_of: Optional[date] = None,
) -> dict:
    run_id = str(uuid.uuid4())
    started = utcnow()
    target = as_of or date.today()
    print(f"\n{'='*55}\n  OPTIONS RESEARCH NODE   run_id={run_id[:8]}\n{'='*55}")
    print("  Research only: this node has no order-submission code.")

    init_db()
    if settings.OPTIONS_EXECUTION_ENABLED:
        raise RuntimeError("Options execution is not implemented; set OPTIONS_EXECUTION_ENABLED=false.")
    if not settings.OPTIONS_RESEARCH_ENABLED:
        print("  Disabled (OPTIONS_RESEARCH_ENABLED=false)")
        _write_log(run_id, [], [], 0, started)
        return {"run_id": run_id, "selected": [], "skipped": [], "failed": [], "disabled": True}
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        reason = "Alpaca paper credentials are required for options chain research"
        print(f"  BLOCKED: {reason}")
        _write_log(run_id, [], [{"symbol": "*", "reason": reason}], 0, started)
        return {"run_id": run_id, "selected": [], "skipped": [], "failed": [reason]}

    source_run = strategies_run_id or _latest_strategy_run_id()
    strategies = _load_buy_strategies(source_run, tickers, target) if source_run else []
    if not strategies:
        print("  No bullish stock strategies to research.")
        _write_log(run_id, [], [], 0, started)
        return {"run_id": run_id, "selected": [], "skipped": [], "failed": []}

    equity = _research_equity()
    selected, skipped, failed = [], [], []
    for strat, underlying in strategies:
        try:
            candidates = _fetch_candidates(underlying, target)
            pick = select_candidate(
                candidates,
                underlying_price=float(strat.entry or 0),
                as_of=target,
                min_dte=settings.OPTIONS_MIN_DTE,
                max_dte=settings.OPTIONS_MAX_DTE,
                min_delta=settings.OPTIONS_MIN_DELTA,
                max_delta=settings.OPTIONS_MAX_DELTA,
                min_open_interest=settings.OPTIONS_MIN_OPEN_INTEREST,
                max_spread_pct=settings.OPTIONS_MAX_SPREAD_PCT,
            )
            if not pick:
                skipped.append(underlying)
                print(f"  {underlying:<6} SKIP  no contract passed liquidity/risk filters")
                continue
            sizing = size_long_option(
                pick.midpoint, pick.multiplier, equity,
                settings.OPTIONS_MAX_RISK_PER_TRADE,
                settings.OPTIONS_MAX_PREMIUM_PCT,
            )
            if sizing["contracts"] < 1:
                skipped.append(underlying)
                print(f"  {underlying:<6} SKIP  premium exceeds per-trade research budget")
                continue
            _save(run_id, strat.id, pick, target, sizing)
            selected.append(underlying)
            print(
                f"  {underlying:<6} CALL  {pick.symbol}  DTE={pick.dte(target)}  "
                f"delta={pick.delta:.2f}  mid=${pick.midpoint:.2f}  "
                f"spread={pick.spread_pct:.1%}  OI={pick.open_interest}  "
                f"contracts={sizing['contracts']}  max_loss=${sizing['max_loss']:.2f}"
            )
        except Exception as exc:
            failed.append({"symbol": underlying, "reason": str(exc)})
            print(f"  {underlying:<6} ERR   {exc}")

    _write_log(run_id, selected, failed, len(strategies), started)
    return {"run_id": run_id, "selected": selected, "skipped": skipped, "failed": failed}


def _fetch_candidates(underlying: str, as_of: date) -> list[OptionCandidate]:
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import OptionChainRequest
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import ContractType
    from alpaca.trading.requests import GetOptionContractsRequest

    trading = TradingClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY, paper=True)
    start = as_of + timedelta(days=settings.OPTIONS_MIN_DTE)
    end = as_of + timedelta(days=settings.OPTIONS_MAX_DTE)
    response = trading.get_option_contracts(GetOptionContractsRequest(
        underlying_symbols=[underlying],
        type=ContractType.CALL,
        expiration_date_gte=start,
        expiration_date_lte=end,
        status="active",
        limit=10000,
    ))
    contracts = getattr(response, "option_contracts", None) or []
    if not contracts:
        return []

    data = OptionHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)
    snapshots = data.get_option_chain(OptionChainRequest(
        underlying_symbol=underlying,
        type=ContractType.CALL,
        expiration_date_gte=start,
        expiration_date_lte=end,
    ))
    result = []
    for contract in contracts:
        snap = snapshots.get(contract.symbol)
        quote = getattr(snap, "latest_quote", None) if snap else None
        greeks = getattr(snap, "greeks", None) if snap else None
        if not quote:
            continue
        result.append(OptionCandidate(
            symbol=contract.symbol,
            underlying=underlying,
            expiration_date=contract.expiration_date,
            strike_price=float(contract.strike_price),
            bid=float(quote.bid_price or 0),
            ask=float(quote.ask_price or 0),
            open_interest=int(contract.open_interest or 0),
            delta=float(greeks.delta) if greeks and greeks.delta is not None else None,
            implied_volatility=(
                float(snap.implied_volatility)
                if snap.implied_volatility is not None else None
            ),
            multiplier=int(contract.size or 100),
        ))
    return result


def _research_equity() -> float:
    """Use paper equity when reachable; otherwise the explicit hypothetical value."""
    try:
        from utils.account import get_account_state
        acct = get_account_state()
        if acct.get("connected") and float(acct.get("equity") or 0) > 0:
            return float(acct["equity"])
    except Exception:
        pass
    return settings.OPTIONS_RESEARCH_EQUITY


def _latest_strategy_run_id() -> Optional[str]:
    with get_session() as session:
        row = session.exec(
            select(Strategy)
            .where(Strategy.signal == "BUY")
            .order_by(Strategy.created_at.desc())  # type: ignore[attr-defined]
        ).first()
        return row.run_id if row else None


def _load_buy_strategies(
    run_id: str, tickers: Optional[list[str]], as_of: date
) -> list[tuple[Strategy, str]]:
    upper = {t.upper() for t in tickers} if tickers else None
    with get_session() as session:
        rows = session.exec(
            select(Strategy)
            .where(Strategy.run_id == run_id)
            .where(Strategy.signal == "BUY")
            .where(Strategy.bar_date == as_of)
        ).all()
        result = []
        for strategy in rows:
            ticker = session.exec(select(Ticker).where(Ticker.id == strategy.ticker_id)).first()
            if not ticker or "-" in ticker.symbol:
                continue
            if upper and ticker.symbol.upper() not in upper:
                continue
            result.append((strategy, ticker.symbol.upper()))
        return result


def _save(run_id: str, strategy_id: str, pick: OptionCandidate, as_of: date, sizing: dict) -> None:
    rationale = (
        f"Research-only long call: delta {pick.delta:.2f}, spread {pick.spread_pct:.1%}, "
        f"open interest {pick.open_interest}, DTE {pick.dte(as_of)}. "
        f"Maximum theoretical loss is premium paid."
    )
    with get_session() as session:
        session.add(OptionsResearch(
            run_id=run_id,
            strategy_id=strategy_id,
            underlying=pick.underlying,
            contract_symbol=pick.symbol,
            contract_type=pick.contract_type,
            expiration_date=pick.expiration_date,
            strike_price=pick.strike_price,
            multiplier=pick.multiplier,
            bid=pick.bid,
            ask=pick.ask,
            midpoint=round(pick.midpoint, 4),
            spread_pct=pick.spread_pct,
            delta=pick.delta,
            implied_volatility=pick.implied_volatility,
            open_interest=pick.open_interest,
            days_to_expiration=pick.dte(as_of),
            contracts=sizing["contracts"],
            premium_per_contract=sizing["premium_per_contract"],
            total_premium=sizing["total_premium"],
            max_loss=sizing["max_loss"],
            rationale=rationale,
        ))


def _write_log(run_id: str, selected: list, failed: list, total: int, started) -> None:
    duration = (utcnow() - started).total_seconds()
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success" if not failed else "partial",
            tickers_processed=total,
            records_written=len(selected),
            error_message=(
                "; ".join(
                    f"{f.get('symbol', '*')}: {f.get('reason', f)}" if isinstance(f, dict) else str(f)
                    for f in failed
                ) or None
            ),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Research long-call options; never submits orders")
    parser.add_argument("--run-id", help="stock strategy run id (default: latest BUY run)")
    parser.add_argument("--tickers", nargs="+")
    args = parser.parse_args()
    run(strategies_run_id=args.run_id, tickers=args.tickers)
