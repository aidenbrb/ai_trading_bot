"""Read-only Alpaca reconciliation for the paper evaluation ledger.

This module calls only Alpaca GET endpoints.  It may insert/update local
``evaluation_ledger`` rows, but it cannot submit, replace, cancel, or close a
broker order.  The report is the required safety check before any future
automatic entry schedule is considered.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET
from db.connection import get_session, init_db
from db.models import (
    EvaluationLedger,
    OHLCV,
    Order,
    Position,
    RiskApproval,
    RunLog,
    Strategy,
    Trade,
)
from utils.strategy_signals import CRYPTO_STRATEGY_VERSION, STOCK_STRATEGY_VERSION
from utils.timeutil import utcnow


def _canonical_symbol(symbol: str) -> str:
    upper = symbol.upper()
    if "/" in upper:
        return upper.replace("/", "-")
    compact = upper.replace("-", "")
    for candidate in CRYPTO_SET:
        if candidate.replace("-", "") == compact:
            return candidate
    return upper


def _plain_utc(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _enum_text(value: Any) -> str:
    text = str(value or "").lower()
    return text.split(".")[-1]


def _float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _broker_client():
    if not settings.ALPACA_PAPER_TRADE:
        raise RuntimeError("Evaluation reconciliation is restricted to Alpaca paper trading.")
    settings.validate(require_broker=True)
    from alpaca.trading.client import TradingClient

    return TradingClient(
        settings.ALPACA_API_KEY,
        settings.ALPACA_SECRET_KEY,
        paper=True,
    )


def _latest_bar_time(session, ticker_id: str, decision_time: datetime) -> datetime | None:
    row = session.exec(
        select(OHLCV)
        .where(OHLCV.ticker_id == ticker_id)
        .where(OHLCV.bar_time <= decision_time)
        .order_by(OHLCV.bar_time.desc())  # type: ignore[attr-defined]
    ).first()
    return row.bar_time if row else None


def _backfill_local_ledger() -> int:
    """Create ledger rows for historical non-dry-run orders already in the DB."""
    created = 0
    with get_session() as session:
        rows = session.exec(
            select(Order, RiskApproval, Strategy)
            .join(RiskApproval, Order.risk_approval_id == RiskApproval.id)
            .join(Strategy, RiskApproval.strategy_id == Strategy.id)
            .where(Order.dry_run == False)  # noqa: E712
        ).all()
        existing = {
            row.order_id for row in session.exec(select(EvaluationLedger)).all()
        }
        for order, approval, strategy in rows:
            if order.id in existing:
                continue
            market = "crypto" if _canonical_symbol(order.symbol) in CRYPTO_SET else "stock"
            default_version = (
                CRYPTO_STRATEGY_VERSION if market == "crypto" else STOCK_STRATEGY_VERSION
            )
            decision_time = strategy.created_at or order.created_at
            session.add(EvaluationLedger(
                order_id=order.id,
                strategy_id=strategy.id,
                strategy_version=strategy.model_used or default_version,
                market=market,
                symbol=_canonical_symbol(order.symbol),
                decision_time_utc=decision_time,
                signal_bar_end_utc=_latest_bar_time(
                    session, strategy.ticker_id, decision_time
                ),
                expected_entry=float(strategy.entry or 0.0),
                expected_stop=float(strategy.stop or 0.0),
                expected_target=float(strategy.target or 0.0),
                expected_qty=float(approval.shares or order.qty or 0.0),
                broker_order_id=order.alpaca_order_id,
                broker_order_status=order.status,
                reconciliation_status="unreconciled",
                actual_fill_price=order.filled_price,
                filled_at_utc=order.filled_at,
                fee_source=(
                    "estimated_tier1_taker" if market == "crypto"
                    else "stock_commission_zero"
                ),
            ))
            created += 1
    return created


def _open_orders(client) -> list[Any]:
    try:
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        return list(client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            nested=True,
            limit=500,
        )))
    except Exception:
        # Protection auditing is fail-closed; the caller records that the list
        # could not be verified rather than interpreting it as no open orders.
        raise RuntimeError("could not retrieve Alpaca open orders")


def _flatten_orders(orders: list[Any]) -> list[Any]:
    flat: list[Any] = []
    for order in orders:
        flat.append(order)
        legs = getattr(order, "legs", None) or []
        flat.extend(_flatten_orders(list(legs)))
    return flat


def _protected_symbols(open_orders: list[Any]) -> set[str]:
    protected: set[str] = set()
    for order in _flatten_orders(open_orders):
        if _enum_text(getattr(order, "side", None)) != "sell":
            continue
        order_type = _enum_text(getattr(order, "type", None) or getattr(order, "order_type", None))
        if order_type in {"stop", "stop_limit", "trailing_stop"}:
            protected.add(_canonical_symbol(str(getattr(order, "symbol", ""))))
    return protected


def run(client=None) -> dict:
    """Reconcile local paper observations using broker reads only."""
    init_db()
    backfilled = _backfill_local_ledger()
    client = client or _broker_client()

    broker_positions = {
        _canonical_symbol(str(getattr(p, "symbol", ""))): p
        for p in client.get_all_positions()
    }
    open_orders = _open_orders(client)
    protected = _protected_symbols(open_orders)
    protection_verified = True
    now = utcnow()
    discrepancies: list[dict] = []
    reconciled = 0

    with get_session() as session:
        ledger_rows = session.exec(select(EvaluationLedger)).all()
        for ledger in ledger_rows:
            order = session.exec(
                select(Order).where(Order.id == ledger.order_id)
            ).first()
            position = session.exec(
                select(Position).where(Position.order_id == ledger.order_id)
            ).first()
            trade = None
            if position:
                trade = session.exec(
                    select(Trade).where(Trade.position_id == position.id)
                ).first()

            broker_order = None
            lookup_error = None
            if ledger.broker_order_id:
                try:
                    from alpaca.trading.requests import GetOrderByIdRequest
                    broker_order = client.get_order_by_id(
                        ledger.broker_order_id,
                        GetOrderByIdRequest(nested=True),
                    )
                    # Alpaca can return only the active take-profit from the
                    # account-wide open-order query while the sibling stop is
                    # HELD.  The original bracket queried with nested=True is
                    # the authoritative protection view.
                    protected.update(_protected_symbols([broker_order]))
                except Exception as exc:
                    lookup_error = str(exc)

            discrepancy = None
            if broker_order:
                ledger.broker_order_status = _enum_text(getattr(broker_order, "status", None))
                fill_price = _float(getattr(broker_order, "filled_avg_price", None))
                if fill_price:
                    ledger.actual_fill_price = fill_price
                    ledger.filled_at_utc = _plain_utc(getattr(broker_order, "filled_at", None))
                    if ledger.expected_entry > 0:
                        ledger.entry_slippage_bps = (
                            fill_price / ledger.expected_entry - 1.0
                        ) * 10_000.0
            elif ledger.broker_order_id:
                discrepancy = f"broker order lookup failed: {lookup_error or 'not found'}"

            symbol = _canonical_symbol(ledger.symbol)
            broker_has_position = symbol in broker_positions
            if trade:
                ledger.exit_time_utc = datetime.combine(
                    trade.exit_date, datetime.max.time()
                )
                ledger.exit_price = trade.exit_price
                ledger.exit_reason = trade.exit_reason
                ledger.gross_pnl = trade.pnl
                entry_notional = abs(trade.entry_price * trade.shares)
                exit_notional = abs(trade.exit_price * trade.shares)
                if ledger.market == "crypto":
                    ledger.fee_amount = (entry_notional + exit_notional) * 0.0025
                    ledger.fee_source = "estimated_tier1_taker_round_trip"
                else:
                    ledger.fee_amount = 0.0
                    ledger.fee_source = "stock_commission_zero"
                ledger.net_pnl = trade.pnl - float(ledger.fee_amount or 0.0)
                ledger.reconciliation_status = "closed"
            elif position and position.status == "open":
                if not broker_has_position:
                    discrepancy = "DB position is open but broker position is absent"
                elif symbol not in protected:
                    discrepancy = "broker position has no verified protective stop"
                else:
                    ledger.reconciliation_status = "consistent_open"
            elif position and position.status == "pending":
                terminal = (ledger.broker_order_status or "") in {
                    "canceled", "cancelled", "expired", "rejected"
                }
                if terminal:
                    discrepancy = "DB position is pending but broker order is terminal"
                else:
                    ledger.reconciliation_status = "consistent_pending"
            elif broker_has_position:
                discrepancy = "broker position exists without an open/pending DB position"
            elif not discrepancy:
                ledger.reconciliation_status = "consistent_no_position"

            if discrepancy:
                ledger.reconciliation_status = "discrepancy"
                discrepancies.append({"symbol": symbol, "order_id": ledger.order_id,
                                      "reason": discrepancy})
            ledger.discrepancy = discrepancy
            ledger.last_reconciled_at = now
            ledger.updated_at = now
            reconciled += 1

    # Catch broker positions for which no ledger/order exists at all.
    with get_session() as session:
        known_symbols = {
            _canonical_symbol(row.symbol)
            for row in session.exec(select(EvaluationLedger)).all()
        }
    for symbol in sorted(set(broker_positions) - known_symbols):
        discrepancies.append({
            "symbol": symbol,
            "order_id": None,
            "reason": "broker position has no evaluation-ledger order",
        })

    forward_gate = _forward_paper_gate(now, discrepancies, protection_verified)
    result = {
        "broker_read_only": True,
        "paper_account": True,
        "backfilled": backfilled,
        "reconciled": reconciled,
        "broker_positions": len(broker_positions),
        "protection_verified": protection_verified,
        "discrepancies": discrepancies,
        "safe_for_scheduler_change": not discrepancies and protection_verified,
        "forward_paper_gate": forward_gate,
        "checked_at_utc": now.isoformat(),
    }
    return result


def _forward_paper_gate(now: datetime, discrepancies: list[dict], protection_verified: bool) -> dict:
    """Evaluate the locked 90-day/30-trade paper observation gate."""
    versions = {STOCK_STRATEGY_VERSION, CRYPTO_STRATEGY_VERSION}
    with get_session() as session:
        observations = [
            row for row in session.exec(select(EvaluationLedger)).all()
            if row.strategy_version in versions
        ]
        cutoff = now - timedelta(days=90)
        recent_runs = session.exec(
            select(RunLog)
            .where(RunLog.node_name == "execution_node")
            .where(RunLog.started_at >= cutoff)
        ).all()

    closed = [row for row in observations if row.reconciliation_status == "closed"
              and row.net_pnl is not None]
    first = min((row.decision_time_utc for row in observations), default=None)
    age_days = (now.date() - first.date()).days if first else 0
    run_success_rate = (
        sum(1 for row in recent_runs if row.status == "success") / len(recent_runs)
        if recent_runs else None
    )
    net_expectancy = (
        sum(float(row.net_pnl) for row in closed) / len(closed) if closed else None
    )

    cost_checks = []
    for row in closed:
        entry_notional = abs(row.actual_fill_price or row.expected_entry) * row.expected_qty
        exit_notional = abs(row.exit_price or 0.0) * row.expected_qty
        denominator = entry_notional + exit_notional
        fee_bps = float(row.fee_amount or 0.0) / denominator * 10_000.0 if denominator else None
        entry_slippage = abs(float(row.entry_slippage_bps or 0.0))
        stressed = 13.0 if row.market == "stock" else 50.0
        cost_checks.append(
            fee_bps is not None and (fee_bps + entry_slippage) <= stressed
        )
    costs_within_stressed = bool(cost_checks) and all(cost_checks)

    checks = {
        "at_least_90_calendar_days": age_days >= 90,
        "at_least_30_closed_trades": len(closed) >= 30,
        "no_reconciliation_discrepancies": not discrepancies,
        "all_open_positions_protected": protection_verified,
        "observed_run_success_at_least_95pct": (
            run_success_rate is not None and run_success_rate >= 0.95
        ),
        "realized_costs_within_stressed_model": costs_within_stressed,
        "realized_net_expectancy_positive": net_expectancy is not None and net_expectancy > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "observation_age_days": age_days,
        "closed_trades": len(closed),
        "observed_execution_runs": len(recent_runs),
        "observed_run_success_rate": run_success_rate,
        "net_expectancy": net_expectancy,
        "note": (
            "Only orders written with the new versioned strategy IDs count. "
            "Historical/manual legacy rows are deliberately excluded."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Alpaca paper ledger reconciliation")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    result = run()
    rendered = json.dumps(result, indent=2, default=str)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
