"""
Entrypoint for the live_slc paper-forward automation:

    python -m live_slc.run_slc_live --stage {preflight,cycle,closeout}

This is the direct, non-guardrail-violating replacement for
`run_pipeline.py --stock-strategy slc` (rejected in review because it
would have required editing run_pipeline.py's hardcoded
`choices=["swing","day"]`). Entirely outside run_pipeline.py's argparse
surface - run_pipeline.py never needs a third value.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from sqlmodel import delete, select

from config.universe import UNIVERSE
from live_slc import bar_cache, closeout as closeout_mod, execution, guardrails, ranking, reducer, risk
from live_slc.authorization import MAX_CYCLE_SECONDS, get_current_deployment_record
from live_slc.models import (
    SlcCycleRun,
    SlcFiveMinBar,
    SlcOrder,
    SlcPosition,
    SlcReducerState,
    SlcSignalRecord,
    get_live_slc_session,
    init_live_slc_db,
)
from live_slc.process_lock import LockAlreadyHeld, acquire_process_lock
from live_slc.settings import live_slc_settings
from utils.market_calendar import is_trading_day, session_for

BOOTSTRAP_DAYS = 120
EASTERN = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now_naive() -> pd.Timestamp:
    """Current UTC time without tzinfo, matching the DB's existing format."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def compute_entry_cutoff(day: date) -> Optional[pd.Timestamp]:
    """ET-aware. None if `day` isn't a trading day.

    min(3:30 PM ET, official_close - 30min), per amendment_004/005."""
    session = session_for(day)
    if session is None:
        return None
    close_et = pd.Timestamp(session["close"]).tz_localize("UTC").tz_convert(EASTERN)
    regular_cutoff_et = pd.Timestamp.combine(day, dt_time(15, 30)).tz_localize(EASTERN)
    return min(regular_cutoff_et, close_et - timedelta(minutes=30))


def confirmation_within_entry_cutoff(confirmation, day: date) -> bool:
    """Per-confirmation check, never a whole-cycle wall-clock gate (rev. 11
    point 1 fix): the 3:31 PM cycle legitimately processes the
    confirmation whose entry_time is exactly 3:30 PM - comparing the
    cycle's own wall-clock invocation time against the cutoff would
    incorrectly suppress that last permitted signal of the session, and
    the same bug would occur on early-close days. Bar ingestion/state
    maintenance is never gated by this at all - only whether THIS
    confirmation may still be acted on."""
    cutoff = compute_entry_cutoff(day)
    if cutoff is None:
        return False
    entry_time_et = pd.Timestamp(confirmation.entry_time).tz_localize("UTC").tz_convert(EASTERN)
    return entry_time_et <= cutoff


class AccountSnapshotUnusable(RuntimeError):
    """Raised, never silently defaulted, when the freshly-read broker
    account state cannot be trusted for a new-entry sizing/risk decision
    (rev. 11 Step 6). Deliberately does not fall back to any placeholder
    account state - a broker outage must fail the cycle's new-entry path,
    never fabricate a plausible-looking account."""


def _finite(value, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise AccountSnapshotUnusable(f"{name} missing or non-numeric: {value!r}")
    if f != f or f in (float("inf"), float("-inf")):  # NaN/inf check without importing math here
        raise AccountSnapshotUnusable(f"{name} is NaN/inf: {value!r}")
    return f


def build_account_snapshot(client, *, quote_fn=None) -> risk.AccountSnapshot:
    """Builds the account snapshot directly and only from live_slc's own
    dedicated client (rev. 11 Step 6 - never utils.account.get_account_state(),
    which constructs its own client from the SHARED bot's config.settings
    and silently falls back to a fabricated $100k account on any read
    failure). Fails closed via AccountSnapshotUnusable on any missing/
    non-numeric/NaN/infinite value, any broker-call exception, or a
    missing quote for an in-flight order (never a $0-notional fallback,
    which would understate exposure).

    Precise value rules (rev. 11 point 7, corrected from an earlier
    blanket "non-positive raises" draft): only equity and
    start_of_day_equity must be positive - cash/non_marginable_buying_power
    being zero or negative is a legitimate account state (margin debit)
    and is only required to be finite; risk.available_buying_power()
    floors it at zero itself.
    """
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest

    quote_fn = quote_fn or (lambda symbol: _get_fresh_quote(symbol, max_age_seconds=60.0))

    try:
        account = client.get_account()
    except Exception as exc:  # noqa: BLE001
        raise AccountSnapshotUnusable(f"broker account read failed: {exc}") from exc

    account_id = str(getattr(account, "id", "") or "")
    if not account_id:
        raise AccountSnapshotUnusable("broker did not return an account id")
    equity = _finite(getattr(account, "equity", None), "equity")
    if equity <= 0:
        raise AccountSnapshotUnusable(f"equity is non-positive: {equity}")
    cash = _finite(getattr(account, "cash", None), "cash")
    non_marginable = _finite(getattr(account, "non_marginable_buying_power", None), "non_marginable_buying_power")
    last_equity = _finite(getattr(account, "last_equity", None), "last_equity")
    if last_equity <= 0:
        raise AccountSnapshotUnusable(f"last_equity (start-of-day equity) is non-positive: {last_equity}")

    try:
        raw_positions = client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        raise AccountSnapshotUnusable(f"broker positions read failed: {exc}") from exc
    positions = [
        risk.PositionInfo(
            symbol=str(p.symbol), qty=_finite(p.qty, f"{p.symbol}.qty"),
            market_value=abs(_finite(p.market_value, f"{p.symbol}.market_value")),
        )
        for p in raw_positions
    ]

    today_start = _utc_now_naive().normalize()
    try:
        raw_orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.ALL, after=today_start.to_pydatetime()))
    except Exception as exc:  # noqa: BLE001
        raise AccountSnapshotUnusable(f"broker order history read failed: {exc}") from exc

    today_orders = []
    for o in raw_orders or []:
        status = execution.order_status_value(o)
        record = {
            "client_order_id": getattr(o, "client_order_id", None),
            "parent_id": getattr(o, "parent_id", None) or getattr(o, "parent_order_id", None),
            # Alpaca enums stringify as e.g. ``OrderClass.BRACKET`` and
            # ``PositionIntent.SELL_TO_CLOSE``.  Classification operates
            # on their wire values, so unwrap them exactly as order-status
            # decisions do (the 2026-08-18 incident proved bare str(enum)
            # is not a safe SDK boundary).
            "order_class": execution.enum_value(getattr(o, "order_class", "")),
            "type": execution.enum_value(getattr(o, "type", "")),
            "position_intent": execution.enum_value(getattr(o, "position_intent", "") or ""),
            "legs": getattr(o, "legs", None),
        }
        classification = risk.classify_order(record)
        symbol = str(getattr(o, "symbol", ""))
        limit_price = getattr(o, "limit_price", None)
        filled_avg_price = getattr(o, "filled_avg_price", None)
        qty = _finite(getattr(o, "qty", 0) or 0, f"{symbol}.order_qty")
        filled_qty = _finite(getattr(o, "filled_qty", 0) or 0, f"{symbol}.filled_qty")

        # ``status=ALL`` also returns canceled/rejected/expired/replaced
        # orders.  A terminal zero-fill opening order is neither one of
        # today's trades nor pending exposure, so omit it.  A filled or
        # partially-filled terminal entry still consumes the daily entry
        # slot, but its live exposure is already represented by the broker
        # position; keeping its pending notional at zero avoids counting it
        # twice.  Exit legs never consume entry capacity and likewise never
        # need a quote merely to build an account snapshot.
        terminal = status in ("filled", "canceled", "expired", "rejected", "replaced")
        has_fill = filled_qty > 0 or status == "filled"
        if classification in ("entry", "unclassifiable") and terminal and not has_fill:
            continue

        pending_qty = max(0.0, abs(qty) - abs(filled_qty)) if not terminal else 0.0
        reference_price = 0.0
        if classification in ("entry", "unclassifiable") and pending_qty > 0:
            if limit_price:
                reference_price = _finite(limit_price, f"{symbol}.limit_price")
            elif filled_avg_price:
                reference_price = _finite(filled_avg_price, f"{symbol}.filled_avg_price")
            else:
                # An in-flight MarketOrderRequest (SLC's own entry shape)
                # has neither - a missing quote here makes the WHOLE
                # snapshot unusable, never a $0-notional fallback that
                # would understate actual pending exposure.
                quote = quote_fn(symbol)
                if quote is None:
                    raise AccountSnapshotUnusable(f"no quote available to value in-flight order for {symbol}")
                reference_price = _finite(quote, f"{symbol}.quote")
        today_orders.append(risk.OrderInfo(
            symbol=symbol, classification=classification, notional=abs(pending_qty * reference_price),
            is_slc=bool(record["client_order_id"] and str(record["client_order_id"]).startswith("slc-")),
        ))

    # Single computation, no double-counting (rev. 11 point 7): the
    # (0.0, delta) split is the canonical convention documented here so
    # nothing elsewhere independently computes and ADDS a realized-P&L
    # figure on top of this already-comprehensive equity delta.
    daily_equity_change = equity - last_equity
    return risk.AccountSnapshot(
        account_id=account_id, equity=equity, cash=cash,
        non_marginable_buying_power=non_marginable, start_of_day_equity=last_equity,
        daily_realized_pnl=0.0, daily_unrealized_pnl=daily_equity_change,
        positions=positions, today_orders=today_orders,
    )


def _load_reducer_state(symbol: str) -> reducer.ReducerState:
    with get_live_slc_session() as session:
        row = session.get(SlcReducerState, symbol)
        if row is None:
            return reducer.ReducerState(symbol=symbol)
        data = {
            "bootstrap_completed": row.bootstrap_completed,
            "last_processed_bar_time": row.last_processed_bar_time.isoformat() if row.last_processed_bar_time else None,
            "raw_bar_tail": json.loads(row.raw_bar_tail_json),
            "in_progress_4h": json.loads(row.in_progress_4h_bar_json),
            "completed_4h_bars": json.loads(row.completed_4h_bars_json),
            "active_levels": json.loads(row.active_levels_json),
            # rev. 11 Step 2 fix: this was hardcoded to [] on every load,
            # which meant the "at most one signal per symbol per session"
            # guarantee didn't actually hold across the real multi-process
            # operating model (every cycle is a fresh process) - only in a
            # single continuous in-memory run.
            "signaled_sessions": json.loads(row.signaled_sessions_json),
        }
        return reducer.ReducerState.from_json(symbol, data)


def _save_reducer_state(state: reducer.ReducerState) -> None:
    data = state.to_json()
    with get_live_slc_session() as session:
        row = session.get(SlcReducerState, state.symbol)
        if row is None:
            row = SlcReducerState(symbol=state.symbol)
        row.bootstrap_completed = data["bootstrap_completed"]
        row.last_processed_bar_time = (
            pd.Timestamp(data["last_processed_bar_time"]).to_pydatetime()
            if data["last_processed_bar_time"] else None
        )
        row.raw_bar_tail_json = json.dumps(data["raw_bar_tail"])
        row.in_progress_4h_bar_json = json.dumps(data["in_progress_4h"])
        row.completed_4h_bars_json = json.dumps(data["completed_4h_bars"])
        row.active_levels_json = json.dumps(data["active_levels"])
        row.signaled_sessions_json = json.dumps(data["signaled_sessions"])
        session.add(row)


def _symbols_with_split_pending() -> set[str]:
    with get_live_slc_session() as session:
        return {
            row.symbol for row in session.exec(
                select(SlcReducerState).where(SlcReducerState.split_pending == True)  # noqa: E712
            )
        }


def _get_fresh_quote(symbol: str, *, max_age_seconds: float) -> Optional[float]:
    """Latest IEX quote, rejected if older than max_age_seconds - the
    fresh-quote check immediately before submission (amendment_004)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockLatestQuoteRequest

    client = StockHistoricalDataClient(
        api_key=live_slc_settings.ALPACA_API_KEY, secret_key=live_slc_settings.ALPACA_SECRET_KEY,
    )
    try:
        quotes = client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol, feed="iex"))
    except Exception:  # noqa: BLE001 - a quote failure means "don't submit," not "crash the cycle"
        return None
    quote = quotes.get(symbol) if quotes else None
    if quote is None:
        return None
    age = (_utc_now_naive() - pd.Timestamp(quote.timestamp).tz_localize(None)).total_seconds()
    if age > max_age_seconds:
        return None
    mid = (float(quote.bid_price) + float(quote.ask_price)) / 2.0
    return mid if mid > 0 else None


def _record_order(
    confirmation, *, dry_run: bool, client_order_id: Optional[str] = None,
    alpaca_order_id: Optional[str] = None, status: str = "submitted",
    qty: float = 0.0, expected_quote: Optional[float] = None,
    stop_submitted: Optional[float] = None, target_submitted: Optional[float] = None,
    notional: Optional[float] = None, monetary_risk: Optional[float] = None,
    effective_reward_risk: Optional[float] = None,
) -> None:
    """Find-or-update on client_order_id (unique), never a blind insert -
    the same signal identity may legitimately be revisited across the
    lifecycle (submission_intent_pending -> filled, etc.)."""
    oc_id = client_order_id or f"dry-run-{confirmation.symbol}-{confirmation.confirmation_time.isoformat()}"
    with get_live_slc_session() as session:
        row = session.exec(select(SlcOrder).where(SlcOrder.client_order_id == oc_id)).first()
        if row is None:
            # signal_id links back to the SlcSignalRecord already written
            # for this confirmation earlier in run_cycle() - the only way
            # delayed reconciliation (Step 5c/9) can later reconstruct
            # enough identity (level_id, confirmation_time, direction) to
            # open/flag an SlcPosition for an order discovered after a
            # crash, since SlcOrder itself doesn't carry those fields.
            signal = session.exec(
                select(SlcSignalRecord).where(
                    SlcSignalRecord.symbol == confirmation.symbol,
                    SlcSignalRecord.level_id == confirmation.level_id,
                    SlcSignalRecord.confirmation_time == confirmation.confirmation_time.to_pydatetime(),
                )
            ).first()
            row = SlcOrder(
                client_order_id=oc_id, symbol=confirmation.symbol, leg="entry",
                side="buy" if confirmation.direction == "long" else "sell",
                position_intent="buy_to_open" if confirmation.direction == "long" else "sell_to_open",
                order_class="bracket", dry_run=dry_run, qty=0.0,
                signal_id=signal.id if signal is not None else None,
            )
        row.alpaca_order_id = alpaca_order_id or row.alpaca_order_id
        row.status = status
        # NOT "qty or row.qty" - an explicit qty=0.0 is a legitimate value
        # (e.g. an existing_broker_order_detected record never got sized),
        # and `0.0 or x` incorrectly falls through to x since 0.0 is falsy.
        if qty:
            row.qty = qty
        row.expected_quote = expected_quote if expected_quote is not None else row.expected_quote
        row.stop_submitted = stop_submitted if stop_submitted is not None else row.stop_submitted
        row.target_submitted = target_submitted if target_submitted is not None else row.target_submitted
        row.notional = notional if notional is not None else row.notional
        row.monetary_risk = monetary_risk if monetary_risk is not None else row.monetary_risk
        row.effective_reward_risk = effective_reward_risk if effective_reward_risk is not None else row.effective_reward_risk
        session.add(row)


def _update_order_status(client_order_id: str, status: str, **fields) -> None:
    with get_live_slc_session() as session:
        row = session.exec(select(SlcOrder).where(SlcOrder.client_order_id == client_order_id)).first()
        if row is None:
            return
        row.status = status
        for key, value in fields.items():
            setattr(row, key, value)
        session.add(row)


def _open_position(confirmation, *, qty, entry_price, stop_price, target_price,
                    entry_order_id=None, expected_quote=None, status="open",
                    protective_order_id=None, target_order_id=None) -> str:
    """Find-or-update on the (symbol, level_id, confirmation_time) natural
    key (rev. 11 point 4) - idempotent across a crash-and-retry.
    protective_order_id/target_order_id must be saved here (found via
    review: they never were, leaving closeout's cancel-then-close with no
    id to cancel and no way to discover the bracket that reserves the
    shares - see execution.protective_leg()/take_profit_leg())."""
    created = False
    session_date = confirmation.confirmation_time.date()
    with get_live_slc_session() as session:
        signal = session.exec(
            select(SlcSignalRecord).where(
                SlcSignalRecord.symbol == confirmation.symbol,
                SlcSignalRecord.level_id == confirmation.level_id,
                SlcSignalRecord.confirmation_time == confirmation.confirmation_time.to_pydatetime(),
            )
        ).first()
        row = session.exec(
            select(SlcPosition).where(
                SlcPosition.symbol == confirmation.symbol,
                SlcPosition.level_id == confirmation.level_id,
                SlcPosition.confirmation_time == confirmation.confirmation_time.to_pydatetime(),
            )
        ).first()
        if row is None:
            created = True
            row = SlcPosition(
                symbol=confirmation.symbol, level_id=confirmation.level_id,
                confirmation_time=confirmation.confirmation_time.to_pydatetime(),
                direction=confirmation.direction, session_date=session_date,
            )
        row.qty = qty
        row.entry_price = entry_price
        row.stop_price = stop_price
        row.target_price = target_price
        row.status = status
        if signal is not None:
            row.signal_id = signal.id
        row.entry_order_id = entry_order_id
        row.expected_quote = expected_quote
        row.actual_fill = entry_price
        row.notional = float(qty) * float(entry_price)
        row.monetary_risk = float(qty) * abs(float(entry_price) - float(stop_price))
        row.slippage = (
            float(entry_price) - float(expected_quote)
            if expected_quote is not None else None
        )
        if protective_order_id is not None:
            row.protective_order_id = protective_order_id
        if target_order_id is not None:
            row.target_order_id = target_order_id
        session.add(row)
        session.flush()
        session.refresh(row)
        position_id = row.id
    if created:
        _bump_session_stat_counter(session_date, "trades_opened")
    return position_id


def _bump_session_stat_counter(session_date, field_name: str, *, delta: int = 1) -> None:
    with get_live_slc_session() as session:
        stat = _get_or_create_session_stat(session, session_date)
        setattr(stat, field_name, getattr(stat, field_name) + delta)
        session.add(stat)


def _mark_position_protected_degraded(position_id: str, reason: str) -> None:
    session_date = None
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is None:
            return
        row.status = "protected_degraded"
        session.add(row)
        session_date = row.session_date
        from live_slc.models import SlcAuditEvent
        session.add(SlcAuditEvent(
            event_type="protected_degraded", symbol=row.symbol,
            payload_json=json.dumps({"position_id": position_id, "reason": reason}),
        ))
    if session_date is not None:
        _bump_session_stat_counter(session_date, "unprotected_position_incident_count")


def _mark_position_open(position_id: str) -> None:
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is not None:
            row.status = "open"
            session.add(row)


def _update_position_order_ids(position_id: str, *, protective_order_id=None, target_order_id=None) -> None:
    """target_order_id is overwritten on every successful replace_order_by_id
    (a new id each time, per the SDK's replaces/replaced_by semantics)."""
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is None:
            return
        if protective_order_id is not None:
            row.protective_order_id = protective_order_id
        if target_order_id is not None:
            row.target_order_id = target_order_id
        session.add(row)


def _mark_position_closed(
    position_id: str, *, exit_price, exit_reason: str, exit_order_id: Optional[str] = None,
    exit_time=None,
) -> Optional[object]:
    """Marks the position closed AND writes its one-time SlcTrade record
    using the ACTUAL exit fill price (found via review: this never
    happened at all outside the end-of-day closeout stage, and even
    there it fabricated P&L from the position's theoretical target price
    instead of the real close fill). Returns the position row (for
    inspection/logging) or None if it doesn't exist."""
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is None:
            return None
        row.status = "closed"
        resolved_exit_time = (
            pd.Timestamp(exit_time).tz_convert("UTC").tz_localize(None).to_pydatetime()
            if exit_time is not None and pd.Timestamp(exit_time).tzinfo is not None
            else pd.Timestamp(exit_time).to_pydatetime() if exit_time is not None
            else _utc_now_naive().to_pydatetime()
        )
        row.closed_at = resolved_exit_time
        session.add(row)
        session.flush()
        session.refresh(row)
        symbol, direction, entry_price, qty, session_date = (
            row.symbol, row.direction, row.entry_price, row.qty, row.session_date,
        )
        position_snapshot = row
    _write_trade_from_closed_position(
        position_id=position_id, symbol=symbol, direction=direction, entry_price=entry_price,
        qty=qty, session_date=session_date, exit_price=exit_price, exit_reason=exit_reason,
        exit_order_id=exit_order_id, exit_time=resolved_exit_time,
    )
    return position_snapshot


def _mark_position_ambiguous(position_id: str, reason: str) -> None:
    """The quantity-mismatch counterpart to _mark_position_protected_degraded
    - a DIFFERENT problem (the broker-reported quantity itself can't be
    reconciled with what SLC expected, vs. protected_degraded's "the
    stop/target geometry couldn't be confirmed"). Both feed
    risk.system_wide_entry_block_reasons() as separate reasons; closeout.py
    already skips (never force-flattens) an "ambiguous" position."""
    session_date = None
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is None:
            return
        row.status = "ambiguous"
        session.add(row)
        session_date = row.session_date
        from live_slc.models import SlcAuditEvent
        session.add(SlcAuditEvent(
            event_type="ambiguous_quantity", symbol=row.symbol,
            payload_json=json.dumps({"position_id": position_id, "reason": reason}),
        ))
    if session_date is not None:
        _bump_session_stat_counter(session_date, "reconciliation_discrepancy_count")


def _mark_position_reconciliation_ambiguous(position_id: str, reason: str) -> None:
    """Block entries when broker position state and local ownership cannot
    be reconciled, without mislabeling the problem as a quantity-only
    ambiguity.  This status is intentionally the same system-wide block
    consumed by risk.system_wide_entry_block_reasons()."""
    session_date = None
    with get_live_slc_session() as session:
        row = session.get(SlcPosition, position_id)
        if row is None or row.status == "ambiguous":
            return
        row.status = "ambiguous"
        session.add(row)
        session_date = row.session_date
        from live_slc.models import SlcAuditEvent
        session.add(SlcAuditEvent(
            event_type="broker_position_reconciliation_ambiguous", symbol=row.symbol,
            payload_json=json.dumps({"position_id": position_id, "reason": reason}),
        ))
    if session_date is not None:
        _bump_session_stat_counter(session_date, "reconciliation_discrepancy_count")


def _mark_signal_skipped(confirmation, reason: str) -> None:
    _mark_signal_result(confirmation, reason, acted_on=False)


def _mark_signal_result(confirmation, result: str, *, acted_on: bool) -> None:
    with get_live_slc_session() as session:
        row = session.exec(
            select(SlcSignalRecord)
            .where(
                SlcSignalRecord.symbol == confirmation.symbol,
                SlcSignalRecord.level_id == confirmation.level_id,
                SlcSignalRecord.confirmation_time == confirmation.confirmation_time.to_pydatetime(),
            )
        ).first()
        if row is not None:
            row.acted_on = acted_on
            row.action_result = result
            session.add(row)


def _recalculate_session_acted_count(session_date: date) -> None:
    """Set, rather than increment, the audit count from persisted signal
    decisions. This repairs a cycle that crashes after broker action but
    before its normal end-of-cycle telemetry accumulator runs."""
    with get_live_slc_session() as session:
        acted = sum(
            1 for row in session.exec(select(SlcSignalRecord)).all()
            if row.direction and row.confirmation_time.date() == session_date and row.acted_on
        )
        stat = _get_or_create_session_stat(session, session_date)
        stat.signals_acted_on = acted
        session.add(stat)


def _dry_run_proposals_for_session(session_date: date) -> int:
    """Count prior simulated entries for the whole session.

    Broker account snapshots cannot contain dry-run proposals because those
    orders are deliberately never sent to Alpaca. Without this DB-side count,
    every five-minute process restarted with two fresh daily slots and today's
    first live dry run produced 38 proposals despite the fixed two/day limit.
    Client-order IDs are unique, so each row consumes exactly one slot.
    """
    with get_live_slc_session() as session:
        rows = list(session.exec(
            select(SlcOrder).where(
                SlcOrder.dry_run == True,  # noqa: E712
                SlcOrder.status == "dry_run_proposed",
            )
        ))
    return sum(1 for row in rows if row.submitted_at.date() == session_date)


def _duplicate_signal_count_for_session(session_date: date) -> int:
    """Return only genuine duplicate confirmation identities.

    Missing-bar audit rows have an empty direction/level and are intentionally
    excluded. Data coverage already accounts for those observations; treating
    every missing bar as a duplicate signal made the engineering gate
    contradict its own 95% coverage allowance.
    """
    with get_live_slc_session() as session:
        rows = list(session.exec(select(SlcSignalRecord)))
    counts: dict[tuple, int] = {}
    for row in rows:
        if not row.direction or row.confirmation_time.date() != session_date:
            continue
        key = (row.symbol, row.level_id, row.confirmation_time)
        counts[key] = counts.get(key, 0) + 1
    return sum(count - 1 for count in counts.values() if count > 1)


def _start_cycle_run(stage: str) -> str:
    with get_live_slc_session() as session:
        run = SlcCycleRun(stage=stage, status="running")
        session.add(run)
        session.flush()
        session.refresh(run)
        return run.id


def _finish_cycle_run(run_id: str, **fields) -> None:
    """status defaults to "completed" only if the caller didn't pass one
    explicitly (found via review: this previously overwrote status
    UNCONDITIONALLY, silently discarding e.g. status="failed" the one
    place it was ever passed - run_closeout_stage()'s account-unreachable
    path - meaning SlcCycleRun.status could never actually record a
    failure)."""
    with get_live_slc_session() as session:
        run = session.get(SlcCycleRun, run_id)
        if run is None:
            return
        for key, value in fields.items():
            setattr(run, key, value)
        if "status" not in fields:
            run.status = "completed"
        session.add(run)


def _cached_bars(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    with get_live_slc_session() as session:
        rows = list(session.exec(
            select(SlcFiveMinBar).where(
                SlcFiveMinBar.symbol == symbol,
                SlcFiveMinBar.bar_time >= start.to_pydatetime(),
                SlcFiveMinBar.bar_time <= end.to_pydatetime(),
            ).order_by(SlcFiveMinBar.bar_time)
        ))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    idx = [pd.Timestamp(r.bar_time) for r in rows]
    return pd.DataFrame({
        "open": [r.open for r in rows], "high": [r.high for r in rows],
        "low": [r.low for r in rows], "close": [r.close for r in rows],
        "volume": [r.volume for r in rows],
    }, index=idx)


# -- Delayed reconciliation of ambiguous order states (rev. 11 Step 5c/9) --
# Strictly read-only relative to the broker: queries, records, and flags
# only - NEVER calls submit_order/replace_order_by_id/cancel_order_by_id/
# cancel_orders/close_position. All risk-reducing mutation happens
# exclusively through --stage closeout or --stage cycle's own guarded
# post-fill path, never preflight.

_AMBIGUOUS_ORDER_STATUSES = (
    "submission_intent_pending", "submitted_unresolved",
    "ambiguous_submission", "existing_broker_order_detected",
)
DELAYED_RECONCILIATION_MIN_AGE = timedelta(minutes=5)


class _HistoryScanUnavailable(Exception):
    """Raised when the broker-history scan itself couldn't run - never
    treated as evidence of absence (a failed scan proves nothing)."""


@dataclass(frozen=True)
class _RecoveredConfirmationRef:
    """The minimal confirmation-shaped surface _open_position() needs,
    reconstructed from a SlcSignalRecord via SlcOrder.signal_id - lets
    delayed reconciliation open/flag a position for an order discovered
    after a crash without changing _open_position()'s own signature."""
    symbol: str
    level_id: str
    confirmation_time: pd.Timestamp
    direction: str


def _scan_order_history(client, client_order_id: str, *, since) -> Optional[object]:
    """The SECOND, independent check beyond the direct
    get_order_by_client_id lookup - a single fresh 404 alone is not
    sufficient evidence given Alpaca's own eventual-consistency lag."""
    from alpaca.trading.enums import QueryOrderStatus
    from alpaca.trading.requests import GetOrdersRequest
    try:
        orders = client.get_orders(GetOrdersRequest(
            status=QueryOrderStatus.ALL, after=pd.Timestamp(since).to_pydatetime(), nested=True,
        ))
    except Exception as exc:  # noqa: BLE001
        raise _HistoryScanUnavailable(str(exc)) from exc
    for order in orders or []:
        if str(getattr(order, "client_order_id", "")) == client_order_id:
            return order
    return None


def _apply_discovered_order_state(client, row: SlcOrder, order) -> None:
    """Read-only relative to the broker - `order` was already fetched by
    the caller. filled-and-protected -> a normal tracked SlcPosition,
    eligible for ordinary closeout management later. filled-and-
    unprotected -> protected_degraded, flagged for the guardian. A
    quantity mismatch (includes a still partially-filled order) ->
    ambiguous, flagged for manual/closeout attention - never guessed at.
    Never filled -> confirmed_no_order_resulted."""
    status = execution.order_status_value(order)
    filled_qty = float(getattr(order, "filled_qty", 0) or 0)
    alpaca_order_id = str(getattr(order, "id", "")) or None

    if status in ("canceled", "expired", "rejected") or filled_qty == 0:
        _update_order_status(row.client_order_id, "confirmed_no_order_resulted", alpaca_order_id=alpaca_order_id)
        return

    # `order` may have come from reconcile_by_client_id()'s
    # get_order_by_client_id() call, which has NO way to request nested
    # bracket legs at all (found via review: this endpoint's SDK method
    # takes no filter parameter whatsoever) - re-fetch via get_order_by_id
    # with nested=True so the protected/unprotected classification below
    # can actually see the legs, regardless of which discovery path
    # produced `order`.
    if alpaca_order_id:
        try:
            order = execution._fetch_order_with_legs(client, alpaca_order_id)
        except Exception:  # noqa: BLE001
            pass  # fall back to whatever `order` already had - read-only, never worse than before

    fill_price = float(getattr(order, "filled_avg_price", 0) or 0) or None
    _update_order_status(
        row.client_order_id, "filled", alpaca_order_id=alpaca_order_id,
        fill_price=fill_price, filled_at=_utc_now_naive().to_pydatetime(),
    )

    with get_live_slc_session() as session:
        signal = session.get(SlcSignalRecord, row.signal_id) if row.signal_id else None
        signal_ref = None
        if signal is not None:
            signal_ref = _RecoveredConfirmationRef(
                symbol=signal.symbol, level_id=signal.level_id,
                confirmation_time=pd.Timestamp(signal.confirmation_time), direction=signal.direction,
            )
    if signal_ref is None:
        # Can't reconstruct enough identity (level_id/confirmation_time)
        # to open a proper SlcPosition - flag for manual review rather
        # than guess. Only a pre-existing/orphaned order missing
        # signal_id can reach this (every new order going forward has it
        # populated by _record_order()).
        with get_live_slc_session() as session:
            from live_slc.models import SlcAuditEvent
            session.add(SlcAuditEvent(
                event_type="discovered_fill_unresolvable_identity", symbol=row.symbol,
                payload_json=json.dumps({"client_order_id": row.client_order_id, "alpaca_order_id": alpaca_order_id}),
            ))
        return

    requested_qty = float(getattr(order, "qty", 0) or row.qty or 0)
    quantity_ambiguous = bool(requested_qty) and filled_qty != requested_qty
    legs = getattr(order, "legs", None) or []
    # Identified by TYPE (stop/stop_limit), never by "any leg that isn't
    # itself canceled" (found via review of the analogous bug in
    # _process_admitted_confirmation: a still-live take-profit leg could
    # otherwise be mistaken for the protective stop) - AND must actually
    # be live (not itself canceled/expired/rejected), or it provides no
    # real protection despite still appearing in the leg list.
    protective_leg_obj = execution.protective_leg(legs)
    protective_live = protective_leg_obj is not None and execution.order_status_value(protective_leg_obj) not in ("canceled", "expired", "rejected")
    target_leg_obj = execution.take_profit_leg(legs)
    protective_id = str(getattr(protective_leg_obj, "id", "")) if protective_live else None
    target_id = str(getattr(target_leg_obj, "id", "")) if target_leg_obj is not None else None

    position_id = _open_position(
        signal_ref, qty=filled_qty, entry_price=fill_price or 0.0,
        stop_price=row.stop_submitted or 0.0, target_price=row.target_submitted or 0.0,
        entry_order_id=alpaca_order_id, expected_quote=row.expected_quote,
        protective_order_id=protective_id, target_order_id=target_id,
    )
    if quantity_ambiguous:
        _mark_position_ambiguous(position_id, "discovered_fill_quantity_mismatch")
    elif not protective_live:
        _mark_position_protected_degraded(position_id, "discovered_fill_unprotected")


def _reconcile_ambiguous_orders(client) -> dict:
    now = _utc_now_naive()
    resolved: list[str] = []
    still_ambiguous: list[str] = []
    with get_live_slc_session() as session:
        rows = list(session.exec(select(SlcOrder).where(SlcOrder.status.in_(_AMBIGUOUS_ORDER_STATUSES))))
    for row in rows:
        age = now - pd.Timestamp(row.submitted_at)
        if age < DELAYED_RECONCILIATION_MIN_AGE:
            still_ambiguous.append(row.client_order_id)
            continue
        direct = execution.reconcile_by_client_id(client, row.client_order_id)
        if direct.outcome == execution.ReconcileOutcome.CONFIRMED_PRESENT:
            # Already definitive - no need for the history scan too.
            _apply_discovered_order_state(client, row, direct.order)
            resolved.append(row.client_order_id)
            continue
        if direct.outcome == execution.ReconcileOutcome.AMBIGUOUS_UNREACHABLE:
            still_ambiguous.append(row.client_order_id)
            continue
        # direct.outcome == CONFIRMED_ABSENT - a single fresh 404 alone is
        # not sufficient evidence given Alpaca's own eventual-consistency
        # lag; the independent history scan must ALSO agree before
        # concluding no order ever resulted.
        try:
            history_match = _scan_order_history(client, row.client_order_id, since=row.submitted_at)
        except _HistoryScanUnavailable:
            still_ambiguous.append(row.client_order_id)
            continue
        if history_match is not None:
            _apply_discovered_order_state(client, row, history_match)
        else:
            _update_order_status(row.client_order_id, "confirmed_no_order_resulted")
        resolved.append(row.client_order_id)
    return {"resolved": resolved, "still_ambiguous": still_ambiguous}


# -- Split detection and atomic staged rebuild (rev. 11 Step 9) -------------

SPLIT_OVERLAP_LOOKBACK_DAYS = 5
MIN_REBUILD_BARS = 1000  # a low floor (~13 trading days of 5-min bars) -
                          # only meant to catch a badly truncated/failed
                          # fetch, not a tight data-quality gate


def _get_corporate_actions_client():
    from alpaca.data.historical.corporate_actions import CorporateActionsClient
    return CorporateActionsClient(
        api_key=live_slc_settings.ALPACA_API_KEY, secret_key=live_slc_settings.ALPACA_SECRET_KEY,
    )


def _record_split_check(symbol: str, *, latest_close: Optional[float], split_pending: Optional[bool] = None) -> None:
    """Bookkeeping for every candidate symbol split-detection examines,
    not just ones where a split was found - when it was last checked and
    the close price observed at that time. `split_pending=True` is set
    the instant validated (non-conflicting) evidence is found, BEFORE the
    rebuild is even attempted (SlcReducerState.split_pending's own
    docstring: blocks this symbol's reducer processing and entries until
    a full rebuild succeeds) - so a rebuild that fails on this attempt
    correctly leaves the symbol blocked for a later preflight to retry,
    rather than silently continuing to run on stale-scale state."""
    with get_live_slc_session() as session:
        row = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == symbol)).first()
        if row is None:
            return
        row.last_split_check_at = _utc_now_naive().to_pydatetime()
        if latest_close is not None:
            row.last_split_check_close = latest_close
        if split_pending is not None:
            row.split_pending = split_pending
        session.add(row)


def _record_split_audit_event(symbol: str, event_type: str, corporate_evidence, price_evidence) -> None:
    with get_live_slc_session() as session:
        from live_slc.models import SlcAuditEvent
        session.add(SlcAuditEvent(
            event_type=event_type, symbol=symbol,
            payload_json=json.dumps({
                "corporate_actions": corporate_evidence.detail if corporate_evidence else None,
                "price_ratio": price_evidence.detail if price_evidence else None,
            }),
        ))


def _atomic_split_rebuild(symbol: str, evidence) -> bool:
    """Staged: fetch a fresh 120-day (current-basis) history and bootstrap
    a brand-new ReducerState IN MEMORY first - the DB row is untouched
    until that fully succeeds. Only then are the old cached SlcFiveMinBar
    rows for this symbol replaced and the reducer state overwritten, all
    inside ONE transaction, so a mid-swap failure leaves the last usable
    (if stale-scale) state completely intact rather than half-updated.
    Returns False (never raises) on any failure - the symbol simply keeps
    running on its last-known-good state until the next preflight
    retries. Rebuilds onto the SAME 120-day window bootstrap() always
    uses from scratch (BOOTSTRAP_DAYS), not an attempt to preserve
    whatever longer history happened to accumulate - consistent with
    bootstrap()'s one already-established contract."""
    end = _utc_now_naive()
    start = end - timedelta(days=BOOTSTRAP_DAYS)
    try:
        frames = bar_cache._default_fetch([symbol], start, end)
        frame = frames.get(symbol)
        if frame is None or frame.empty or len(frame) < MIN_REBUILD_BARS:
            return False
        new_state = reducer.bootstrap(symbol, frame)
    except Exception:  # noqa: BLE001
        return False

    try:
        data = new_state.to_json()
        with get_live_slc_session() as session:
            session.exec(delete(SlcFiveMinBar).where(SlcFiveMinBar.symbol == symbol))
            for bar_time, row in frame.iterrows():
                session.add(SlcFiveMinBar(
                    symbol=symbol, bar_time=pd.Timestamp(bar_time).to_pydatetime(),
                    open=float(row["open"]), high=float(row["high"]),
                    low=float(row["low"]), close=float(row["close"]),
                    volume=float(row["volume"]),
                ))
            reducer_row = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == symbol)).first()
            if reducer_row is None:
                reducer_row = SlcReducerState(symbol=symbol)
            reducer_row.bootstrap_completed = data["bootstrap_completed"]
            reducer_row.last_processed_bar_time = (
                pd.Timestamp(data["last_processed_bar_time"]).to_pydatetime()
                if data["last_processed_bar_time"] else None
            )
            reducer_row.raw_bar_tail_json = json.dumps(data["raw_bar_tail"])
            reducer_row.in_progress_4h_bar_json = json.dumps(data["in_progress_4h"])
            reducer_row.completed_4h_bars_json = json.dumps(data["completed_4h_bars"])
            reducer_row.active_levels_json = json.dumps(data["active_levels"])
            reducer_row.signaled_sessions_json = json.dumps(data["signaled_sessions"])
            reducer_row.split_pending = False
            session.add(reducer_row)
            from live_slc.models import SlcAuditEvent
            session.add(SlcAuditEvent(
                event_type="split_rebuild_applied", symbol=symbol,
                payload_json=json.dumps({
                    "scale_factor": str(evidence.scale_factor), "source": evidence.source, "detail": evidence.detail,
                }),
            ))
        return True
    except Exception:  # noqa: BLE001
        return False


def _run_split_detection_and_rebuild(client, symbols: list[str]) -> dict:
    """Both detection sources are pure reads; only a symbol with fully
    validated, non-conflicting evidence ever reaches the atomic rebuild.
    Symbols where the two sources disagree are flagged for manual review
    and left completely untouched - never guessed at. Only ever
    considers symbols that already have cached bars (a not-yet-
    bootstrapped symbol has nothing to rebuild - the ordinary bootstrap
    loop in run_preflight() handles those)."""
    with get_live_slc_session() as session:
        bootstrapped = {
            row.symbol for row in session.exec(
                select(SlcReducerState).where(SlcReducerState.bootstrap_completed == True)  # noqa: E712
            )
        }
    candidates = [s for s in symbols if s in bootstrapped]
    if not candidates:
        return {"rebuilt": [], "conflicting": [], "failed": []}

    from live_slc import split_detection

    try:
        corp_client = _get_corporate_actions_client()
        corporate_evidence = split_detection.corporate_action_split_evidence(
            corp_client, candidates, lookback_days=SPLIT_OVERLAP_LOOKBACK_DAYS,
        )
    except Exception:  # noqa: BLE001
        corporate_evidence = {}

    end = _utc_now_naive()
    start = end - timedelta(days=SPLIT_OVERLAP_LOOKBACK_DAYS)
    fresh_frames = bar_cache._default_fetch(candidates, start, end)

    rebuilt, conflicting, failed = [], [], []
    for symbol in candidates:
        fresh = fresh_frames.get(symbol)
        cached = _cached_bars(symbol, start, end)
        price_evidence = (
            split_detection.price_ratio_split_evidence(symbol, cached, fresh)
            if fresh is not None else None
        )
        latest_close = None
        if fresh is not None and not fresh.empty:
            latest_close = float(fresh.iloc[-1]["close"])
        elif not cached.empty:
            latest_close = float(cached.iloc[-1]["close"])

        evidence, conflict = split_detection.reconcile_evidence(corporate_evidence.get(symbol), price_evidence)
        if conflict:
            conflicting.append(symbol)
            _record_split_audit_event(symbol, "split_evidence_conflict", corporate_evidence.get(symbol), price_evidence)
            _record_split_check(symbol, latest_close=latest_close)
            continue
        if evidence is None:
            _record_split_check(symbol, latest_close=latest_close)
            continue
        _record_split_check(symbol, latest_close=latest_close, split_pending=True)
        if _atomic_split_rebuild(symbol, evidence):
            rebuilt.append(symbol)
        else:
            failed.append(symbol)
            _record_split_audit_event(symbol, "split_rebuild_failed", evidence, None)
            # split_pending stays True - correctly blocks this symbol's
            # cycle processing until a LATER preflight's rebuild attempt
            # succeeds; the prior (stale-scale) state is left completely
            # untouched by _atomic_split_rebuild's own failure path.
    return {"rebuilt": rebuilt, "conflicting": conflicting, "failed": failed}


# -- Engine-parity self-check (rev. 12: frozen validation-corpus source) ----
#
# The original design sampled a rotation of the live universe, reading
# each symbol's bars via _cached_bars() (SlcFiveMinBar). That is broken by
# construction immediately after a real bootstrap: bootstrap() persists
# only the serialized ReducerState, never the ~1M historical bars
# themselves, into SlcFiveMinBar (a deliberate choice - writing the full
# bootstrap window to the live rolling cache is neither needed nor
# cheap). So _cached_bars() over the bootstrap window is empty for every
# symbol until enough real trading days accumulate bars one at a time,
# and the parity gate necessarily returned False the entire time.
#
# The self-check's actual purpose - proving the live reducer still agrees
# with the frozen batch generate_signals() - does not depend on live
# data at all. A frozen, hash-verified validation corpus already exists
# for exactly this (research/slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json,
# tests/fixtures/slc_reducer_corpus/*.csv), built specifically for reducer
# parity testing. Using it here makes the daily preflight check
# deterministic, fast, and independent of how much live history has
# accumulated - never a re-download or re-persistence of the ~1M-bar
# bootstrap window, and never a silent fallback to the (possibly empty)
# live cache.

REDUCER_CORPUS_MANIFEST_PATH = REPO_ROOT / "research" / "slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json"
REQUIRED_REDUCER_CORPUS_SYMBOLS = ("AAPL", "AMD")


class EngineParityCorpusInvalid(RuntimeError):
    """The frozen validation corpus can't be trusted as-is - missing
    manifest/fixture, a hash that no longer matches, or an unparseable
    CSV. Always fails the parity check closed; never caught anywhere as
    a signal to fall back to something else."""


def _load_reducer_corpus_manifest() -> dict:
    if not REDUCER_CORPUS_MANIFEST_PATH.is_file():
        raise EngineParityCorpusInvalid(f"corpus manifest missing: {REDUCER_CORPUS_MANIFEST_PATH}")
    try:
        return json.loads(REDUCER_CORPUS_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise EngineParityCorpusInvalid(f"corpus manifest failed to parse: {exc}") from exc


def _load_and_verify_corpus_fixture(symbol: str, fixture: dict) -> pd.DataFrame:
    """Hash-verifies the raw bytes against the frozen manifest BEFORE
    parsing - a tampered or truncated fixture must never reach the
    reducer/batch engines at all."""
    path = REPO_ROOT / fixture["path"]
    if not path.is_file():
        raise EngineParityCorpusInvalid(f"{symbol} corpus fixture missing: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != fixture["sha256"]:
        raise EngineParityCorpusInvalid(
            f"{symbol} corpus fixture hash mismatch: expected={fixture['sha256']} actual={digest}"
        )
    try:
        return pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    except Exception as exc:  # noqa: BLE001
        raise EngineParityCorpusInvalid(f"{symbol} corpus fixture failed to parse: {exc}") from exc


def _run_engine_parity_self_check() -> bool:
    """Local-file-only, zero network/broker calls: loads the frozen
    AAPL/AMD corpus fixtures (hash-verified against the manifest first),
    then runs reducer.check_engine_parity() on each, scoped to the
    manifest's declared_evaluation_window (each fixture carries one extra
    real trailing bar past that window purely because generate_signals()
    structurally requires a trailing bar - never itself evaluated, per
    the manifest's own final_bar_handling note). Passes only if every
    required fixture exists and hash-verifies, every comparison reports
    matched=True, and at least one genuine signal was compared overall -
    fails closed (returns False) for anything else, including a missing/
    tampered/unparseable fixture, so a broken corpus can never silently
    look like a pass."""
    try:
        manifest = _load_reducer_corpus_manifest()
        fixtures = manifest["fixtures"]
        if any(symbol not in fixtures for symbol in REQUIRED_REDUCER_CORPUS_SYMBOLS):
            return False
        window_end = pd.Timestamp(manifest["declared_evaluation_window"]["end"])

        checked_with_signals = 0
        for symbol in REQUIRED_REDUCER_CORPUS_SYMBOLS:
            bars = _load_and_verify_corpus_fixture(symbol, fixtures[symbol])
            result = reducer.check_engine_parity(symbol, bars, window_end=window_end)
            if not result.matched:
                return False
            if result.signal_count > 0:
                checked_with_signals += 1
        return checked_with_signals > 0
    except EngineParityCorpusInvalid:
        return False
    except (KeyError, ValueError, TypeError):
        # A structurally malformed manifest (missing "fixtures"/
        # "declared_evaluation_window" keys, an unparseable window
        # timestamp) is exactly as untrustworthy as a hash mismatch -
        # fail closed rather than let a KeyError/ValueError escape and
        # crash the whole preflight run.
        return False


def run_preflight() -> dict:
    """Strictly broker-read-only (rev. 11 Step 5c/9): reconciliation here
    may only query/record/flag, never mutate - every risk-reducing action
    goes through --stage closeout or --stage cycle's own guarded path
    instead. Runs, in order: account-ID-gated operational preconditions,
    bootstrap for any never-bootstrapped symbol, split detection +
    atomic rebuild for already-bootstrapped symbols, delayed
    reconciliation of the 4 ambiguous order states, and the engine-parity
    self-check.

    Brackets the whole body in an SlcCycleRun("preflight") row, mirroring
    run_cycle()'s wrapper (found via review: this stage previously had
    zero timing telemetry at all - no way to tell a late/failed preflight
    from one that simply never got instrumented). Any uncaught exception
    records status="failed" and re-raises - Task Scheduler's own
    non-zero-exit-code detection is unaffected. Does not bump
    SlcSessionStat.failed_cycles - that field is specifically the "cycle"
    stage's own counter (run_cycle()'s wrapper), not a generic
    any-stage-failed counter; a failed preflight is fully represented by
    this row's own status alone."""
    run_id = _start_cycle_run("preflight")
    start_monotonic = time.monotonic()
    try:
        client = execution.get_alpaca_client()
        observed_account_id = str(client.get_account().id)
        operational = guardrails.assert_operational_preconditions(observed_account_id=observed_account_id)

        bootstrapped_this_run = []
        for symbol in UNIVERSE:
            state = _load_reducer_state(symbol)
            if not state.bootstrap_completed:
                start = _utc_now_naive() - timedelta(days=BOOTSTRAP_DAYS)
                end = _utc_now_naive()
                frames = bar_cache._default_fetch([symbol], start, end)
                frame = frames.get(symbol)
                if frame is not None and not frame.empty:
                    state = reducer.bootstrap(symbol, frame)
                    _save_reducer_state(state)
                    bootstrapped_this_run.append(symbol)

        split_result = _run_split_detection_and_rebuild(client, UNIVERSE)
        reconciliation_result = _reconcile_ambiguous_orders(client)
        today = _utc_now_naive().date()
        engine_parity_passed = _run_engine_parity_self_check()

        with get_live_slc_session() as session:
            stat = _get_or_create_session_stat(session, today)
            stat.engine_parity_check_passed = engine_parity_passed
            session.add(stat)

        _finish_cycle_run(run_id, duration_seconds=time.monotonic() - start_monotonic)
        return {
            "status": operational["status"],
            "bootstrapped": len(bootstrapped_this_run),
            "split_rebuilt": len(split_result["rebuilt"]),
            "split_conflicting": len(split_result["conflicting"]),
            "reconciliation_resolved": len(reconciliation_result["resolved"]),
            "reconciliation_still_ambiguous": len(reconciliation_result["still_ambiguous"]),
            "engine_parity_check_passed": engine_parity_passed,
        }
    except Exception:
        _finish_cycle_run(
            run_id, status="failed",
            duration_seconds=time.monotonic() - start_monotonic,
        )
        raise


@dataclass(frozen=True)
class LocalPositionReconciliationResult:
    closed_ids: list[str] = field(default_factory=list)
    ambiguous_ids: list[str] = field(default_factory=list)


def _read_exit_order(client, order_id: Optional[str]):
    """Read one known exit order. A genuine 404 means this candidate is
    absent; every other read failure is ambiguous and blocks the cycle."""
    if not order_id:
        return None
    try:
        return client.get_order_by_id(order_id)
    except Exception as exc:  # noqa: BLE001
        if execution.is_confirmed_not_found(exc):
            return None
        raise AccountSnapshotUnusable(
            f"broker exit-order read failed for {order_id}: {exc}"
        ) from exc


def _reconcile_local_positions_from_broker_exits(client) -> LocalPositionReconciliationResult:
    """Close local SLC positions only from strict broker-side evidence.

    A successful ``get_all_positions`` call proving a symbol absent is
    necessary but not sufficient: one and only one known SLC exit order
    must also be FILLED for the exact local quantity with a real fill
    price.  The known candidates are the persisted protective stop,
    persisted target, and the deterministic emergency/closeout client ID.
    Missing, conflicting, or quantity-mismatched evidence becomes an
    explicit ambiguous local position and blocks all new entries; it is
    never guessed closed.  All broker calls here are read-only.
    """
    with get_live_slc_session() as session:
        local_positions = list(session.exec(
            select(SlcPosition).where(SlcPosition.status.in_(("open", "protected_degraded")))
        ))
    if not local_positions:
        return LocalPositionReconciliationResult()

    try:
        broker_positions = client.get_all_positions()
    except Exception as exc:  # noqa: BLE001
        raise AccountSnapshotUnusable(f"broker positions read failed during reconciliation: {exc}") from exc

    broker_by_symbol = {
        str(getattr(p, "symbol", "")): p
        for p in (broker_positions or [])
        if str(getattr(p, "symbol", "")) and float(getattr(p, "qty", 0) or 0) != 0
    }

    closed_ids: list[str] = []
    ambiguous_ids: list[str] = []
    for position in local_positions:
        broker_position = broker_by_symbol.get(position.symbol)
        if broker_position is not None:
            raw_side = getattr(broker_position, "side", "")
            broker_side = execution.enum_value(raw_side)
            broker_qty = abs(_finite(getattr(broker_position, "qty", None), f"{position.symbol}.broker_qty"))
            expected_side = "long" if position.direction == "long" else "short"
            if broker_side != expected_side or abs(broker_qty - float(position.qty)) > 1e-6:
                _mark_position_reconciliation_ambiguous(
                    position.id,
                    f"broker/local quantity or direction mismatch: broker_side={broker_side!r}, "
                    f"broker_qty={broker_qty}, expected_side={expected_side!r}, expected_qty={position.qty}",
                )
                ambiguous_ids.append(position.id)
            continue

        candidates: list[tuple[str, object]] = []
        for reason, order_id in (
            ("protective_stop_filled", position.protective_order_id),
            ("take_profit_filled", position.target_order_id),
        ):
            order = _read_exit_order(client, order_id)
            if order is not None and execution.order_status_value(order) == "filled":
                candidates.append((reason, order))

        exit_client_id = execution.client_order_id(
            position.symbol, position.level_id, position.confirmation_time, "exit",
        )
        exit_reconciliation = execution.reconcile_by_client_id(client, exit_client_id)
        if exit_reconciliation.outcome == execution.ReconcileOutcome.AMBIGUOUS_UNREACHABLE:
            raise AccountSnapshotUnusable(
                f"broker emergency-exit reconciliation failed for {position.symbol}: "
                f"{exit_reconciliation.error}"
            )
        if (
            exit_reconciliation.outcome == execution.ReconcileOutcome.CONFIRMED_PRESENT
            and execution.order_status_value(exit_reconciliation.order) == "filled"
        ):
            candidates.append(("emergency_or_closeout_exit_filled", exit_reconciliation.order))

        # De-duplicate a candidate in case a future broker response links
        # the same physical exit order through two locally-known IDs.
        unique_candidates: dict[str, tuple[str, object]] = {}
        for reason, order in candidates:
            order_id = str(getattr(order, "id", "") or "")
            unique_candidates[order_id or f"missing-id:{reason}"] = (reason, order)
        candidates = list(unique_candidates.values())

        valid_candidates: list[tuple[str, object, float, object]] = []
        invalid_reasons: list[str] = []
        for reason, order in candidates:
            filled_qty = abs(_finite(getattr(order, "filled_qty", None), f"{position.symbol}.exit_filled_qty"))
            fill_price = _finite(getattr(order, "filled_avg_price", None), f"{position.symbol}.exit_fill_price")
            if abs(filled_qty - float(position.qty)) > 1e-6:
                invalid_reasons.append(
                    f"{reason} quantity {filled_qty} != expected {position.qty}"
                )
                continue
            if fill_price <= 0:
                invalid_reasons.append(f"{reason} has non-positive fill price {fill_price}")
                continue
            valid_candidates.append((reason, order, fill_price, getattr(order, "filled_at", None)))

        if len(valid_candidates) == 1 and not invalid_reasons:
            reason, order, fill_price, filled_at = valid_candidates[0]
            order_id = str(getattr(order, "id", "") or "") or None
            _mark_position_closed(
                position.id, exit_price=fill_price, exit_reason=reason,
                exit_order_id=order_id, exit_time=filled_at,
            )
            with get_live_slc_session() as session:
                signal = (
                    session.get(SlcSignalRecord, position.signal_id)
                    if position.signal_id else None
                )
                if signal is None:
                    signal = session.exec(
                        select(SlcSignalRecord).where(
                            SlcSignalRecord.symbol == position.symbol,
                            SlcSignalRecord.level_id == position.level_id,
                            SlcSignalRecord.confirmation_time == position.confirmation_time,
                        )
                    ).first()
                if signal is not None:
                    signal.acted_on = True
                    signal.action_result = reason
                    session.add(signal)
            closed_ids.append(position.id)
            continue

        detail = invalid_reasons or [
            f"expected exactly one filled SLC exit order, found {len(valid_candidates)}"
        ]
        _mark_position_reconciliation_ambiguous(position.id, "; ".join(detail))
        ambiguous_ids.append(position.id)

    for session_date in {p.session_date for p in local_positions if p.id in closed_ids}:
        _recalculate_session_acted_count(session_date)
    return LocalPositionReconciliationResult(closed_ids=closed_ids, ambiguous_ids=ambiguous_ids)


def _resolve_protected_degraded_positions(client, *, observed_account_id: str) -> dict:
    """Opportunistic, best-effort recovery for protected_degraded
    positions, called from run_cycle() (rev. 11 review point 6):
    execution.exit_via_marketable_replacement() existed but had no
    caller anywhere, so a degraded position sat unresolved all day until
    end-of-day closeout's more aggressive cancel-then-close finally
    touched it. Never cancels - preserves the linked stop for the whole
    attempt. A failure here is simply retried on a later cycle, never
    escalated to a same-cycle cancel; run_closeout_stage() remains the
    guaranteed backstop regardless of how many of these attempts run
    first."""
    with get_live_slc_session() as session:
        positions = list(session.exec(select(SlcPosition).where(SlcPosition.status == "protected_degraded")))
    resolved: list[str] = []
    for position in positions:
        if not position.target_order_id:
            continue  # nothing to replace against - left for closeout's guardian to discover/handle
        guardrails.assert_closeout_preconditions(observed_account_id=observed_account_id)
        result = execution.exit_via_marketable_replacement(
            client, position.symbol, position.target_order_id, position.direction,
        )
        if result.filled:
            _mark_position_closed(
                position.id, exit_price=result.fill_price, exit_reason="protected_degraded_marketable_exit",
            )
            resolved.append(position.id)
    return {"resolved": resolved}


def run_cycle() -> dict:
    """Thin wrapper: creates the SlcCycleRun row, delegates to
    _run_cycle_body() for the actual work, and on any uncaught exception
    records SlcSessionStat.failed_cycles and marks the run "failed"
    before re-raising (found via review: failed_cycles was never
    populated anywhere, and _finish_cycle_run's own status-overwrite bug
    would have silently discarded a recorded failure even if something
    had tried). Still crashes loudly (re-raises) - Task Scheduler's own
    non-zero-exit-code failure detection is unaffected by this."""
    cycle_start_monotonic = time.monotonic()
    run_id = _start_cycle_run("cycle")
    try:
        return _run_cycle_body(run_id, cycle_start_monotonic)
    except Exception:
        _finish_cycle_run(
            run_id, status="failed",
            duration_seconds=time.monotonic() - cycle_start_monotonic,
        )
        try:
            _bump_session_stat_counter(_utc_now_naive().date(), "failed_cycles")
        except Exception:  # noqa: BLE001
            pass  # best-effort telemetry - never let a stat-recording issue mask the real failure
        raise


def _run_cycle_body(run_id: str, cycle_start_monotonic: float) -> dict:
    # Account-ID verification happens here, unconditionally, before even
    # the trading-day check (rev. 11 Step 9): a dry_run cycle never
    # reaches assert_submission_preconditions() (paper_active-only), so
    # without this, a whole engineering session could run against a
    # misconfigured account undetected. The same client is reused below
    # rather than constructed twice.
    client = execution.get_alpaca_client()
    observed_account_id = str(client.get_account().id)
    operational = guardrails.assert_operational_preconditions(observed_account_id=observed_account_id)
    today = _utc_now_naive().date()
    if not is_trading_day(today):
        _finish_cycle_run(
            run_id, symbols_scanned=0,
            duration_seconds=time.monotonic() - cycle_start_monotonic,
        )
        return {"status": "not_a_trading_day"}

    # First reconcile positions that a broker-side stop/target/emergency
    # exit already closed.  This must precede protected-degraded recovery:
    # otherwise a stale local row can attempt a second exit even though
    # Alpaca is already flat (the 2026-08-19 TGT incident).
    _reconcile_local_positions_from_broker_exits(client)

    # Opportunistic recovery for any protected_degraded position, BEFORE
    # this cycle's own admission logic - a successful resolution here can
    # unblock system_wide_entry_block_reasons() for THIS same cycle's
    # candidates, not just a later one (found via review:
    # exit_via_marketable_replacement existed but was never called from
    # anywhere - every protected_degraded position sat unresolved until
    # end-of-day closeout's more aggressive cancel-then-close). Runs
    # regardless of deployment status (assert_closeout_preconditions is
    # independent of it, matching run_closeout_stage()'s own scope) -
    # resolving an already-open risk position isn't an "entry" decision.
    _resolve_protected_degraded_positions(client, observed_account_id=observed_account_id)

    now = _utc_now_naive()
    expected_bar_time = now.floor("5min") - timedelta(minutes=5)

    # A symbol with split_pending=True has stale-scale reducer state
    # (rev. 11 Step 9) - preflight already left it un-rebuilt (evidence
    # found but the atomic rebuild hasn't succeeded yet). Feeding it new,
    # current-basis live bars now would corrupt that stale-scale state
    # even further; entirely excluded from bar ingestion until a later
    # preflight's rebuild clears the flag.
    split_pending_symbols = _symbols_with_split_pending()
    active_universe = [s for s in UNIVERSE if s not in split_pending_symbols]
    for symbol in split_pending_symbols:
        with get_live_slc_session() as session:
            session.add(SlcSignalRecord(
                cycle_run_id=run_id, symbol=symbol, level_id="", direction="",
                level_state="", level_low=0.0, level_high=0.0,
                level_active_time=expected_bar_time, confirmation_time=expected_bar_time,
                entry_time=expected_bar_time, stop=0.0, stochastic_k=0.0, stochastic_d=None,
                atr14=0.0, structure="", impulse_atr=0.0,
                acted_on=False, action_result="split_pending_blocked",
            ))

    # Catch up state on any bars strictly before the current expected bar
    # first - fed through the reducer for STATE only, confirmations
    # discarded. A confirmation revealed only by this backfill already
    # missed its live entry window; the "never act on a signal after the
    # fact via backfill" rule (amendment_004) is enforced simply by never
    # collecting confirmations from this pass at all.
    backfill_through = expected_bar_time - timedelta(minutes=5)
    backfilled = bar_cache.backfill_gaps(active_universe, backfill_through)
    for symbol, written in backfilled.items():
        if written <= 0:
            continue
        with get_live_slc_session() as session:
            bars = list(session.exec(
                select(SlcFiveMinBar)
                .where(SlcFiveMinBar.symbol == symbol, SlcFiveMinBar.bar_time <= backfill_through.to_pydatetime())
                .order_by(SlcFiveMinBar.bar_time)
            ))
        state = _load_reducer_state(symbol)
        for bar in bars:
            if state.last_processed_bar_time is not None and pd.Timestamp(bar.bar_time) <= state.last_processed_bar_time:
                continue
            row = pd.Series({"open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume})
            state, _ = reducer.process_new_bar(state, row, pd.Timestamp(bar.bar_time))
        _save_reducer_state(state)

    found, missing = bar_cache.fetch_expected_bar_batch(active_universe, expected_bar_time)
    for symbol in missing:
        with get_live_slc_session() as session:
            session.add(SlcSignalRecord(
                cycle_run_id=run_id, symbol=symbol, level_id="", direction="",
                level_state="", level_low=0.0, level_high=0.0,
                level_active_time=expected_bar_time, confirmation_time=expected_bar_time,
                entry_time=expected_bar_time, stop=0.0, stochastic_k=0.0, stochastic_d=None,
                atr14=0.0, structure="", impulse_atr=0.0,
                acted_on=False, action_result="stale_or_missing_data",
            ))

    confirmations: list[reducer.Confirmation] = []
    for symbol, bar_row in found.items():
        bar_cache.persist_bars(symbol, {expected_bar_time: bar_row})
        state = _load_reducer_state(symbol)
        state, confs = reducer.process_new_bar(state, bar_row, expected_bar_time)
        _save_reducer_state(state)
        confirmations.extend(confs)

    ranked = ranking.rank_confirmations(confirmations)
    status = operational["status_record"].status

    # Structural same-day preflight/parity gate (paper_active only): a
    # cycle's own assert_operational_preconditions() call verifies account
    # ID and guardrail/rule-freeze hashes, but never checked that TODAY's
    # preflight actually ran and passed engine parity before this - a
    # failed 08:35 preflight with nobody checking by 9:36 could otherwise
    # let entries begin anyway. Keyed on today's own SlcSessionStat
    # primary key (session_date), so a prior day's success can never
    # satisfy it. split_pending is deliberately NOT re-checked here - it's
    # already handled per-symbol (active_universe exclusion above), not a
    # universe-wide condition this gate needs to duplicate.
    paper_entry_gate_ok = True
    paper_entry_gate_reason = ""
    if status == "paper_active":
        from live_slc.models import SlcSessionStat
        with get_live_slc_session() as gate_session:
            todays_stat = gate_session.get(SlcSessionStat, today)
        if todays_stat is None:
            paper_entry_gate_ok = False
            paper_entry_gate_reason = "no_same_day_preflight_session_stat"
        elif todays_stat.engine_parity_check_passed is not True:
            paper_entry_gate_ok = False
            paper_entry_gate_reason = "same_day_engine_parity_check_failed"

    with get_live_slc_session() as session:
        for c in confirmations:
            session.add(SlcSignalRecord(
                cycle_run_id=run_id, symbol=c.symbol, level_id=c.level_id, direction=c.direction,
                level_state=c.level_state, level_low=c.level_low, level_high=c.level_high,
                level_active_time=c.level_active_time.to_pydatetime(),
                confirmation_time=c.confirmation_time.to_pydatetime(),
                entry_time=c.entry_time.to_pydatetime(), stop=c.stop,
                stochastic_k=c.stochastic_k, stochastic_d=c.stochastic_d, atr14=c.atr14,
                structure=c.structure, impulse_atr=c.impulse_atr,
            ))

    orders_submitted = 0
    orders_dry_run = 0
    # Entry-cutoff filtering (Step 4) - per confirmation, never a whole-
    # cycle wall-clock gate.
    ranked = [c for c in ranking.rank_confirmations(confirmations) if confirmation_within_entry_cutoff(c, today)]
    for c in confirmations:
        if c not in ranked:
            _mark_signal_skipped(c, "after_entry_cutoff")

    if status == "paper_active" and ranked and not paper_entry_gate_ok:
        for c in ranked:
            _mark_signal_skipped(c, f"blocked_{paper_entry_gate_reason}")

    if status in ("dry_run", "paper_active") and ranked and (status != "paper_active" or paper_entry_gate_ok):
        try:
            snapshot = build_account_snapshot(client)
        except AccountSnapshotUnusable as exc:
            for c in ranked:
                _mark_signal_skipped(c, f"account_snapshot_unusable: {exc}")
            snapshot = None

        if snapshot is not None:
            remaining_daily_entries = risk.remaining_daily_entry_slots(snapshot)
            if status == "dry_run":
                remaining_daily_entries = max(
                    0,
                    remaining_daily_entries - _dry_run_proposals_for_session(today),
                )
            admitted, skipped = ranking.select_within_capacity(
                ranked,
                remaining_daily_entries=remaining_daily_entries,
                capacity_check_fn=lambda c, adm: risk.check_new_entry_capacity(snapshot, c.symbol, adm),
            )
            for confirmation, reason in skipped:
                _mark_signal_skipped(confirmation, reason)

            committed_notional_this_cycle = 0.0
            for confirmation in admitted:
                # Rechecked before EVERY candidate, not once per cycle
                # (rev. 11 point 2): if processing an earlier candidate
                # this cycle just created a blocking incident, no further
                # candidate may proceed.
                with get_live_slc_session() as block_session:
                    block_reasons = risk.system_wide_entry_block_reasons(block_session)
                if block_reasons:
                    _mark_signal_skipped(confirmation, f"system_wide_entry_block: {block_reasons}")
                    break

                outcome = _process_admitted_confirmation(
                    confirmation, status=status, client=client, run_id=run_id,
                    committed_notional_this_cycle=committed_notional_this_cycle,
                )
                if outcome.get("submitted"):
                    orders_submitted += 1
                if outcome.get("dry_run_proposed"):
                    orders_dry_run += 1
                if outcome.get("submitted") or outcome.get("dry_run_proposed"):
                    # A dry_run proposal must commit notional too (found
                    # via review: only "submitted" did, so multiple
                    # simulated proposals within the same cycle could
                    # each be sized as if they were the only trade using
                    # the account's buying power, reusing the same cash -
                    # the dry-run session gate exists precisely to prove
                    # sizing behaves correctly, which this silently
                    # defeated for any multi-candidate cycle).
                    committed_notional_this_cycle += outcome.get("notional", 0.0)
                if outcome.get("new_blocking_incident"):
                    break

    elapsed_seconds = time.monotonic() - cycle_start_monotonic
    _finish_cycle_run(
        run_id, symbols_scanned=len(active_universe), new_bars_ingested=len(found),
        confirmations_collected=len(confirmations), orders_submitted=orders_submitted,
        orders_dry_run=orders_dry_run, duration_seconds=elapsed_seconds,
    )
    with get_live_slc_session() as metric_session:
        cycle_signal_rows = list(metric_session.exec(
            select(SlcSignalRecord).where(SlcSignalRecord.cycle_run_id == run_id)
        ))
    actual_signal_rows = [row for row in cycle_signal_rows if row.direction]
    skipped_not_shortable = sum(
        1 for row in actual_signal_rows
        if (row.action_result or "").startswith("skipped_not_shortable")
    )
    skipped_stale_or_missing = sum(
        1 for row in actual_signal_rows
        if row.action_result == "stale_or_missing_data"
    )
    skipped_capacity = sum(
        1 for row in actual_signal_rows
        if (row.action_result or "").startswith("skipped_capacity")
    )
    _accumulate_session_stat(
        today, expected_symbols=len(active_universe), valid_symbols=len(found),
        stale_or_missing=len(missing), dry_run_proposals=orders_dry_run,
        signals_generated=len(actual_signal_rows),
        # Count persisted signal decisions, not only the happy-path
        # return counters. A broker submission that is immediately
        # flattened or becomes ambiguous was still genuinely acted on
        # and must not disappear from the session audit.
        signals_acted_on=sum(1 for row in actual_signal_rows if row.acted_on),
        signals_skipped_not_shortable=skipped_not_shortable,
        signals_skipped_stale_or_missing_data=skipped_stale_or_missing,
        signals_skipped_capacity=skipped_capacity,
        elapsed_seconds=elapsed_seconds,
    )
    return {"status": status, "confirmations": len(confirmations), "missing": len(missing)}


def _accumulate_session_stat(
    session_date, *, expected_symbols: int, valid_symbols: int, stale_or_missing: int,
    dry_run_proposals: int, signals_generated: int, signals_acted_on: int,
    signals_skipped_not_shortable: int,
    signals_skipped_stale_or_missing_data: int,
    signals_skipped_capacity: int, elapsed_seconds: float,
) -> None:
    """Every field here is a RUNNING TOTAL across the whole session, not
    overwritten fresh each cycle - run_cycle() is invoked many times per
    trading day. Reaching this point already proves the guardrail check
    passed (assert_operational_preconditions() would have raised
    otherwise), so guardrail_check_passed is set True unconditionally
    here rather than tracked as a separate pass/fail count. Feeds
    authorization.evaluate_dry_run_session_gate() (rev. 11 Step 12),
    which previously had no populated data to evaluate at all."""
    with get_live_slc_session() as session:
        stat = _get_or_create_session_stat(session, session_date)
        stat.cycles_run += 1
        stat.signals_generated += signals_generated
        stat.signals_acted_on += signals_acted_on
        stat.signals_skipped_not_shortable += signals_skipped_not_shortable
        stat.signals_skipped_stale_or_missing_data += signals_skipped_stale_or_missing_data
        stat.signals_skipped_capacity += signals_skipped_capacity
        stat.expected_symbol_count += expected_symbols
        stat.valid_symbol_count += valid_symbols
        if stat.expected_symbol_count > 0:
            stat.valid_bar_coverage_pct = 100.0 * stat.valid_symbol_count / stat.expected_symbol_count
        # Absolute genuine-duplicate count, not an accumulation of missing
        # market-data observations. Setting (rather than incrementing) also
        # avoids counting the same duplicate again on every later cycle.
        stat.duplicate_or_stale_signal_count = _duplicate_signal_count_for_session(session_date)
        stat.dry_run_proposal_count += dry_run_proposals
        stat.guardrail_check_passed = True
        if elapsed_seconds > MAX_CYCLE_SECONDS:
            stat.cycles_over_budget += 1
        session.add(stat)


def _process_admitted_confirmation(
    confirmation, *, status: str, client, run_id: str, committed_notional_this_cycle: float,
) -> dict:
    """The shared decision pipeline for BOTH dry_run and paper_active
    (rev. 11 point 2 fix - dry_run must exercise the identical Phase A
    *and* Phase B, or the one-session dry-run gate proves nothing about
    whether the real decision path works). Only the literal broker-
    mutating call differs, gated by `simulate`."""
    simulate = status == "dry_run"
    symbol, direction, stop_theoretical = confirmation.symbol, confirmation.direction, confirmation.stop

    # -- Phase A: shortability + provisional sizing (candidate filtering) --
    if direction == "short":
        shortability = execution.check_shortability(_get_asset_raw(client), symbol)
        if not shortability.ok:
            _mark_signal_skipped(confirmation, f"skipped_not_shortable: {shortability.reason}")
            return {}

    # -- Phase B: re-check everything immediately adjacent to the
    # broker-mutating boundary - both dry_run and paper_active run this
    # identically (rev. 11 point 2). --
    fresh_quote = _get_fresh_quote(symbol, max_age_seconds=execution.QUOTE_FRESHNESS_DEADLINE_SECONDS)
    if fresh_quote is None:
        _mark_signal_skipped(confirmation, "stale_or_missing_data")
        return {}

    try:
        snapshot = build_account_snapshot(client)
    except AccountSnapshotUnusable as exc:
        _mark_signal_skipped(confirmation, f"account_snapshot_unusable: {exc}")
        return {}

    ok, reason = risk.check_new_entry_capacity(snapshot, symbol, [])
    if not ok:
        _mark_signal_skipped(confirmation, reason)
        return {}

    stop_submitted = execution.round_stop(stop_theoretical, direction)
    provisional_target_theoretical = execution.retarget_after_fill(direction, fresh_quote, float(stop_submitted))
    target_submitted = execution.round_target(provisional_target_theoretical, direction)
    if not execution.validate_rounded_bracket(fresh_quote, stop_submitted, target_submitted, direction):
        # Pre-submission: nothing to flatten yet - skip the trade outright
        # (amendment_005 item 5, rev. 11 point 4's separation).
        _mark_signal_skipped(confirmation, "rounded_bracket_invalid_pre_submission")
        return {}

    qty = risk.size_order(
        snapshot, reference_price=fresh_quote, stop_price=float(stop_submitted),
        hypothetical_additional_notional=committed_notional_this_cycle,
    )
    if qty <= 0:
        _mark_signal_skipped(confirmation, "skipped_zero_size")
        return {}

    oc_id = execution.client_order_id(symbol, confirmation.level_id, confirmation.confirmation_time, "entry")
    reconciled = execution.reconcile_by_client_id(client, oc_id)

    if reconciled.outcome == execution.ReconcileOutcome.CONFIRMED_PRESENT:
        if simulate:
            # Never adopted/managed from dry_run - that's exclusively the
            # reconciliation/closeout path's job (rev. 11 point 3).
            _record_order(confirmation, dry_run=True, client_order_id=oc_id, status="existing_broker_order_detected")
            return {"new_blocking_incident": True}
        # paper_active: the prior attempt already reached the broker -
        # never re-submit; the fill/management pipeline picks it up via
        # the normal reconciliation path on a later cycle/preflight.
        _mark_signal_skipped(confirmation, "already_submitted_previously")
        return {}
    if reconciled.outcome == execution.ReconcileOutcome.AMBIGUOUS_UNREACHABLE:
        _record_order(confirmation, dry_run=simulate, client_order_id=oc_id, status="ambiguous_submission")
        return {"new_blocking_incident": True}

    notional = qty * fresh_quote

    if simulate:
        _record_order(
            confirmation, dry_run=True, client_order_id=oc_id, status="dry_run_proposed",
            qty=qty, expected_quote=fresh_quote, stop_submitted=float(stop_submitted),
            target_submitted=float(target_submitted), notional=notional,
            monetary_risk=qty * abs(fresh_quote - float(stop_submitted)),
            effective_reward_risk=float(execution.effective_reward_risk(fresh_quote, stop_submitted, target_submitted, direction)),
        )
        _mark_signal_result(confirmation, "dry_run_proposed", acted_on=True)
        return {"dry_run_proposed": True, "notional": notional}

    guardrails.assert_submission_preconditions(
        # freshest possible operational state at the submission boundary
        guardrails.assert_operational_preconditions(observed_account_id=snapshot.account_id),
        observed_account_id=snapshot.account_id, daily_loss_breached=risk.daily_loss_breached(snapshot),
    )
    _record_order(
        confirmation, dry_run=False, client_order_id=oc_id, status="submission_intent_pending",
        qty=qty, expected_quote=fresh_quote, stop_submitted=float(stop_submitted), target_submitted=float(target_submitted),
    )
    _mark_signal_result(confirmation, "submission_intent_pending", acted_on=True)
    result = execution.submit_bracket_entry(
        client, symbol=symbol, direction=direction, qty=qty,
        stop=stop_submitted, provisional_target=target_submitted, order_client_id=oc_id,
    )
    if not result.accepted:
        if execution._is_duplicate_client_order_id_error(RuntimeError(result.error or "")):
            _update_order_status(oc_id, "ambiguous_submission", error_message=result.error)
            return {"new_blocking_incident": True}
        if result.ambiguous:
            # A network/timeout failure, or a server-side error that
            # doesn't prove the order was never created - we genuinely
            # don't know whether the broker accepted it before the
            # connection dropped (found via review: this was previously
            # always confirmed_rejected, which could hide a real
            # accepted - possibly filled - order behind a status nothing
            # ever reconciles). Resolved the same way as a duplicate-id
            # ambiguity: blocks new entries until delayed reconciliation
            # (preflight) resolves it.
            _update_order_status(oc_id, "ambiguous_submission", error_message=result.error)
            return {"new_blocking_incident": True}
        _update_order_status(oc_id, "confirmed_rejected", error_message=result.error)
        return {}

    # Persist the broker identity immediately after acceptance, before
    # any polling or post-fill branch. The first real paper order proved
    # that waiting until a later happy-path update can leave a genuine
    # broker order with alpaca_order_id=NULL when recovery intervenes.
    _update_order_status(
        oc_id, "submitted", alpaca_order_id=result.alpaca_order_id,
        notional=notional,
    )
    final_order = execution.poll_fill_status(client, result.alpaca_order_id)
    order_status = execution.order_status_value(final_order)
    filled_qty = float(getattr(final_order, "filled_qty", 0) or 0)

    if filled_qty == 0 and order_status not in ("filled", "canceled", "expired", "rejected"):
        # Still live (new/accepted/pending_new/...) after the poll
        # deadline - must be ACTIVELY canceled and reconfirmed, never
        # just labeled canceled without ever calling cancel_order_by_id
        # (found via review: a resting order left unmanaged could still
        # fill later, invisible to SLC's own state).
        cancel_result = execution.cancel_and_confirm_unfilled_entry(client, result.alpaca_order_id)
        if cancel_result.outcome == execution.CancelUnfilledOutcome.AMBIGUOUS:
            _update_order_status(oc_id, "submitted_unresolved")
            return {"new_blocking_incident": True}
        # Re-resolve from the fresh observation, whether that's a
        # confirmed zero-fill cancellation or a race into an actual
        # fill - falls through to the SAME branching below, no
        # duplicated partial/full-fill handling.
        final_order = cancel_result.order
        order_status = execution.order_status_value(final_order)
        filled_qty = float(getattr(final_order, "filled_qty", 0) or 0)

    if filled_qty == 0:
        # Genuinely zero-filled and terminal - either the broker's own
        # terminal status, or our own confirmed cancellation above.
        _update_order_status(oc_id, "canceled_unfilled")
        _mark_signal_result(confirmation, "canceled_unfilled", acted_on=True)
        return {}

    legs = getattr(final_order, "legs", None) or []
    protective_leg_obj = execution.protective_leg(legs)
    protective_live = protective_leg_obj is not None and execution.order_status_value(protective_leg_obj) not in ("canceled", "expired", "rejected")
    protective_id = str(getattr(protective_leg_obj, "id", "")) if protective_live else None

    if order_status != "filled" or filled_qty != qty:
        # ANY nonzero-but-incomplete fill, regardless of order_status -
        # even "canceled" with a partial fill before cancellation (found
        # via review: the original condition above already matched this
        # case via `filled_qty == 0`'s sibling `or`, silently discarding
        # a real partial fill as "canceled_unfilled"). Real shares are at
        # risk here; persisted as a genuine local position BEFORE
        # attempting to flatten it (found via review: it previously
        # never was, so an ambiguous/failed flatten left the guardian
        # nothing to find and recover) so the guardian has something to
        # act on even if this flatten attempt doesn't confirm flat.
        partial_fill_price = float(getattr(final_order, "filled_avg_price", 0) or 0) or fresh_quote
        position_id = _open_position(
            confirmation, qty=filled_qty, entry_price=partial_fill_price, stop_price=float(stop_submitted),
            target_price=float(target_submitted), entry_order_id=result.alpaca_order_id,
            expected_quote=fresh_quote, protective_order_id=protective_id,
        )
        _mark_position_protected_degraded(position_id, "partial_fill_requires_flatten")
        _update_order_status(oc_id, "submitted_unresolved")
        guardrails.assert_closeout_preconditions(observed_account_id=snapshot.account_id)
        close_result = execution.cancel_then_confirm_then_close(
            client, symbol, protective_id, filled_qty,
            _closing_side(direction), close_client_order_id=execution.client_order_id(
                symbol, confirmation.level_id, confirmation.confirmation_time, "exit"),
        )
        _update_order_status(
            oc_id, f"partial_fill_{close_result.outcome.value}",
            alpaca_order_id=result.alpaca_order_id,
            filled_at=_utc_now_naive().to_pydatetime(), fill_price=partial_fill_price,
            notional=float(filled_qty) * partial_fill_price,
            monetary_risk=float(filled_qty) * abs(partial_fill_price - float(stop_submitted)),
            slippage=partial_fill_price - fresh_quote,
        )
        if close_result.outcome == execution.CloseOutcome.CONFIRMED_FLAT:
            _mark_position_closed(
                position_id, exit_price=close_result.fill_price, exit_reason="partial_fill_flattened",
                exit_order_id=close_result.close_order_id,
            )
        _mark_signal_result(
            confirmation,
            "partial_fill_flattened" if close_result.outcome == execution.CloseOutcome.CONFIRMED_FLAT
            else f"partial_fill_{close_result.outcome.value}",
            acted_on=True,
        )
        return {
            "submitted": True, "notional": notional,
            "new_blocking_incident": close_result.outcome != execution.CloseOutcome.CONFIRMED_FLAT,
        }

    # Full fill.
    actual_fill_price = float(getattr(final_order, "filled_avg_price", fresh_quote))
    _update_order_status(
        oc_id, "filled", alpaca_order_id=result.alpaca_order_id,
        filled_at=_utc_now_naive().to_pydatetime(), fill_price=actual_fill_price,
        notional=float(filled_qty) * actual_fill_price,
        monetary_risk=float(filled_qty) * abs(actual_fill_price - float(stop_submitted)),
        slippage=actual_fill_price - fresh_quote,
    )

    provisional_target_leg = execution.take_profit_leg(legs)
    provisional_target_id = str(getattr(provisional_target_leg, "id", "")) if provisional_target_leg is not None else None

    position_id = _open_position(
        confirmation, qty=filled_qty, entry_price=actual_fill_price, stop_price=float(stop_submitted),
        target_price=float(target_submitted), entry_order_id=result.alpaca_order_id,
        expected_quote=fresh_quote, protective_order_id=protective_id, target_order_id=provisional_target_id,
    )

    recalculated_target = execution.round_target(
        execution.retarget_after_fill(direction, actual_fill_price, float(stop_submitted)), direction,
    )

    # Fresh equity, not the pre-submission Phase-B snapshot (found via
    # review: equity can move - even slightly - in the seconds between
    # that read and this fill; the whole point of this check is to
    # validate risk against the account's REALIZED, current state). A
    # failed refresh must FAIL CLOSED (never assume the stale pre-fill
    # snapshot is still accurate and silently proceed - found via review:
    # the original fallback did exactly that, defeating the whole point
    # of re-checking in the first place).
    try:
        post_fill_snapshot = build_account_snapshot(client)
        equity_refresh_ok = True
    except AccountSnapshotUnusable:
        post_fill_snapshot = None
        equity_refresh_ok = False

    monetary_ok = equity_refresh_ok and execution.post_fill_monetary_risk_check(
        qty=filled_qty, actual_fill=actual_fill_price, stop=float(stop_submitted),
        equity=post_fill_snapshot.equity if post_fill_snapshot else 0.0, risk_pct=risk.SLC_RISK_PER_TRADE,
    )
    geometry_ok = execution.validate_rounded_bracket(actual_fill_price, execution.to_decimal(stop_submitted), recalculated_target, direction)
    # A fresh bracket fill should ALWAYS carry a live protective stop leg
    # (both children activate atomically with the parent) - protective_id
    # being None here is a genuine anomaly, not a normal case, and must
    # never be silently treated as healthy.
    protection_ok = protective_id is not None

    if not equity_refresh_ok or not monetary_ok or not geometry_ok or not protection_ok:
        reason = (
            "post_fill_equity_refresh_failed" if not equity_refresh_ok
            else "post_fill_monetary_risk_exceeded" if not monetary_ok
            else "post_fill_directional_validation_failed" if not geometry_ok
            else "post_fill_no_live_protective_leg"
        )
        _mark_position_protected_degraded(position_id, reason)
        guardrails.assert_closeout_preconditions(observed_account_id=snapshot.account_id)
        close_result = execution.cancel_then_confirm_then_close(
            client, symbol, protective_id, filled_qty, _closing_side(direction),
            close_client_order_id=execution.client_order_id(symbol, confirmation.level_id, confirmation.confirmation_time, "exit"),
        )
        if close_result.outcome == execution.CloseOutcome.CONFIRMED_FLAT:
            _mark_position_closed(
                position_id, exit_price=close_result.fill_price, exit_reason=reason,
                exit_order_id=close_result.close_order_id,
            )
        _mark_signal_result(
            confirmation,
            f"filled_then_flattened_{reason}"
            if close_result.outcome == execution.CloseOutcome.CONFIRMED_FLAT
            else f"filled_protected_degraded_{reason}",
            acted_on=True,
        )
        return {
            "submitted": True, "notional": notional,
            "new_blocking_incident": close_result.outcome != execution.CloseOutcome.CONFIRMED_FLAT,
        }

    try:
        from alpaca.trading.requests import ReplaceOrderRequest
        # execution.take_profit_leg() identifies the leg by TYPE (limit),
        # never by array position - Alpaca's .legs ordering is not
        # guaranteed, and picking the first leg unconditionally could
        # silently call replace_order_by_id on the STOP leg instead,
        # corrupting its price with the take-profit's recalculated value
        # (found via review, never triggered live - no order has been
        # submitted during this work).
        target_leg = execution.take_profit_leg(legs)
        target_leg_id = str(getattr(target_leg, "id", "")) if target_leg is not None else None
        if target_leg_id:
            client.replace_order_by_id(target_leg_id, ReplaceOrderRequest(limit_price=float(recalculated_target)))
            verification = execution.verify_target_replacement(
                client, result.alpaca_order_id, recalculated_target, direction, expected_qty=filled_qty,
            )
        else:
            verification = execution.TargetVerificationResult(False)
    except Exception:  # noqa: BLE001
        verification = execution.TargetVerificationResult(False)

    if not verification.confirmed:
        # Corrected recovery (rev. 8/11): never described as healthy.
        # Original bracket left untouched if it's still structurally
        # valid; resolution deferred to the guarded emergency-exit path
        # (exit_via_marketable_replacement, invoked from a later cycle -
        # see run_cycle()'s protected_degraded resolution step), never a
        # same-cycle retry loop.
        _mark_position_protected_degraded(position_id, "target_replacement_unconfirmed")
        _mark_signal_result(
            confirmation, "filled_protected_degraded_target_replacement_unconfirmed",
            acted_on=True,
        )
        return {"submitted": True, "notional": notional, "new_blocking_incident": True}

    if verification.target_order_id:
        # A successful replace_order_by_id issues a NEW order id (found
        # via review: this was computed but never saved, leaving no
        # local way to find the current target leg later).
        _update_position_order_ids(position_id, target_order_id=verification.target_order_id)
    _update_order_status(
        oc_id, "filled", target_submitted=float(recalculated_target),
        effective_reward_risk=float(execution.effective_reward_risk(
            actual_fill_price, execution.to_decimal(stop_submitted), recalculated_target, direction,
        )),
    )
    _mark_position_open(position_id)
    _mark_signal_result(confirmation, "filled_and_protected", acted_on=True)
    return {"submitted": True, "notional": notional}


def _closing_side(direction: str):
    from alpaca.trading.enums import OrderSide
    return OrderSide.SELL if direction == "long" else OrderSide.BUY


def _get_asset_raw(client) -> "Callable[[str], dict]":
    def _fetch(symbol: str) -> dict:
        return client.get(f"/assets/{symbol}")
    return _fetch


ORPHAN_PRICE_CORROBORATION_TOLERANCE = 0.02  # 2% - a sanity check, not a strict requirement


@dataclass(frozen=True)
class OrphanDiscoveryResult:
    adopted_ids: list = field(default_factory=list)
    # SLC-attributable (matched an "slc-" order for the symbol at some
    # point) but NOT confidently resolvable to one specific source order -
    # never guessed at, but still must count against broker-flat (found
    # via review: silently excluding these let broker_flat report True
    # while a real, unresolved position sat open).
    unresolved_symbols: set = field(default_factory=set)


def _candidate_entry_orders_for_symbol(session, symbol: str) -> list:
    return list(session.exec(
        select(SlcOrder).where(
            SlcOrder.symbol == symbol, SlcOrder.leg == "entry", SlcOrder.status == "filled",
        ).where(SlcOrder.client_order_id.startswith("slc-"))
    ))


def _discover_orphan_slc_positions(client) -> OrphanDiscoveryResult:
    """A crash between a real fill and _open_position() ever being called
    could leave a genuine SLC-owned broker position with NO local
    SlcPosition row at all - closeout, which previously only ever
    examined local rows, would never flatten it (found via review).
    Cross-checks the broker's actual open positions against local state.

    A broker position is only ADOPTED when it resolves to EXACTLY ONE
    unclosed SLC entry order for that symbol, with a broker avg_entry_price
    reasonably close to that order's own fill price (found via review:
    the original match used a bare, unordered .first() over ALL historical
    filled orders for the symbol, which could just as easily match a
    completely different, already-closed trade and reconstruct the
    orphan's stop/target from the wrong signal entirely - a single
    unambiguous candidate plus price corroboration is required instead;
    zero or multiple candidates are left for manual review, never
    guessed at, and reported as unresolved so callers can still count
    them against broker-flat)."""
    try:
        broker_positions = client.get_all_positions()
    except Exception:  # noqa: BLE001
        return OrphanDiscoveryResult()

    with get_live_slc_session() as session:
        tracked_symbols = {
            row.symbol for row in session.exec(
                select(SlcPosition).where(SlcPosition.status.in_(("open", "ambiguous", "protected_degraded")))
            )
        }

    adopted_ids: list[str] = []
    unresolved_symbols: set = set()
    for broker_position in broker_positions or []:
        symbol = str(getattr(broker_position, "symbol", ""))
        broker_qty = float(getattr(broker_position, "qty", 0) or 0)
        broker_avg_entry = float(getattr(broker_position, "avg_entry_price", 0) or 0)
        # broker_position.side is a real alpaca.trading.enums.PositionSide
        # on a genuine Alpaca response - str(PositionSide.LONG) is
        # "PositionSide.LONG", not "long", so a bare str() here rejected
        # every real position as a direction mismatch (found via review;
        # concealed by mocks that used plain "long"/"short" strings).
        # .value unwraps the enum to its plain string ("long"/"short");
        # getattr's default falls through unchanged for an already-plain
        # string (which has no .value attribute), so both shapes work.
        raw_side = getattr(broker_position, "side", "")
        broker_side = str(getattr(raw_side, "value", raw_side)).lower()
        if not symbol or broker_qty == 0 or symbol in tracked_symbols:
            continue

        with get_live_slc_session() as session:
            all_orders_ever = _candidate_entry_orders_for_symbol(session, symbol)
            candidates = []
            for order in all_orders_ever:
                if not order.signal_id:
                    continue
                signal = session.get(SlcSignalRecord, order.signal_id)
                if signal is None:
                    continue
                linked_position = session.exec(
                    select(SlcPosition).where(
                        SlcPosition.symbol == signal.symbol, SlcPosition.level_id == signal.level_id,
                        SlcPosition.confirmation_time == pd.Timestamp(signal.confirmation_time).to_pydatetime(),
                    )
                ).first()
                if linked_position is not None and linked_position.status == "closed":
                    continue  # already properly accounted for - not the source of THIS orphan
                # Every check below is REQUIRED, never optionally skipped
                # when a value happens to be missing (found via review:
                # the original price check was skipped entirely, silently
                # accepting the candidate, whenever either price was
                # falsy - a missing/zero value must fail closed, not be
                # treated as "nothing to disprove it").
                if not order.fill_price or not broker_avg_entry:
                    continue
                if abs(order.fill_price - broker_avg_entry) / order.fill_price > ORPHAN_PRICE_CORROBORATION_TOLERANCE:
                    continue  # doesn't corroborate - not confidently this order
                if not broker_side or signal.direction != broker_side:
                    continue  # long/short sign must match the broker's own reported side
                if not order.qty or abs(order.qty - abs(broker_qty)) > 1e-6:
                    continue  # the order's own requested qty must match the broker's current qty exactly
                candidates.append((order, signal))

        if not all_orders_ever:
            # SLC has never placed an order for this symbol at all - no
            # evidence whatsoever this is SLC's position (could easily
            # be a different strategy sharing the account). Never
            # flagged - flagging it would be a false positive, not a
            # cautious default.
            continue

        if len(candidates) != 1:
            # SLC HAS traded this symbol before, but zero or multiple
            # unclosed/price-corroborated candidates remain - genuinely
            # ambiguous which trade this is (or isn't confidently any of
            # them). Never guessed at, but flagged as an unresolved
            # SLC-attributable position so it counts against broker-flat.
            unresolved_symbols.add(symbol)
            with get_live_slc_session() as session:
                from live_slc.models import SlcAuditEvent
                session.add(SlcAuditEvent(
                    event_type="orphan_broker_position_unresolved", symbol=symbol,
                    payload_json=json.dumps({"broker_qty": broker_qty, "candidate_count": len(candidates)}),
                ))
            continue

        matching_order, signal = candidates[0]
        signal_ref = _RecoveredConfirmationRef(
            symbol=signal.symbol, level_id=signal.level_id,
            confirmation_time=pd.Timestamp(signal.confirmation_time), direction=signal.direction,
        )
        entry_price = broker_avg_entry or (matching_order.fill_price or 0.0)
        position_id = _open_position(
            signal_ref, qty=abs(broker_qty), entry_price=entry_price,
            stop_price=matching_order.stop_submitted or 0.0, target_price=matching_order.target_submitted or 0.0,
            entry_order_id=matching_order.alpaca_order_id, status="protected_degraded",
        )
        with get_live_slc_session() as session:
            from live_slc.models import SlcAuditEvent
            session.add(SlcAuditEvent(
                event_type="orphan_broker_position_adopted", symbol=symbol,
                payload_json=json.dumps({"position_id": position_id, "broker_qty": broker_qty}),
            ))
        adopted_ids.append(position_id)
    return OrphanDiscoveryResult(adopted_ids=adopted_ids, unresolved_symbols=unresolved_symbols)


def run_closeout_stage() -> dict:
    """Runs under assert_closeout_preconditions() ONLY - never
    assert_operational_preconditions() (rev. 11 point 9's fix for a real
    self-contradiction: the earlier draft would have made closeout depend
    on the same Tier-2/rule-freeze/account-ID gate that blocks new
    entries, defeating the entire point of closeout's independent minimal
    gate). Closeout must remain reachable under a disabled execution
    switch, a suspended deployment status, or Tier-2 signal-fidelity
    drift - none of those bear on the ability to safely manage or exit a
    position that's already open."""
    run_id = _start_cycle_run("closeout")
    today = _utc_now_naive().date()
    now = _utc_now_naive()
    if not is_trading_day(today) or not closeout_mod.should_begin_closeout(now, today):
        _finish_cycle_run(run_id)
        return {"status": "not_yet_or_not_a_trading_day"}

    client = execution.get_alpaca_client()
    try:
        observed_account_id = str(client.get_account().id)
    except Exception as exc:  # noqa: BLE001
        _finish_cycle_run(run_id, status="failed", errors_json=json.dumps([str(exc)]))
        return {"status": "closeout_blocked_account_unreachable"}

    guardrails.assert_closeout_preconditions(observed_account_id=observed_account_id)

    # A stop or target may fill after the final entry cycle but before
    # the closeout window. Reconcile those broker-confirmed exits before
    # the guardian decides which remaining local positions need action.
    reconciliation_errors: list[str] = []
    try:
        _reconcile_local_positions_from_broker_exits(client)
    except AccountSnapshotUnusable as exc:
        # Closeout must remain reachable even when the read-only
        # reconciliation pre-pass cannot complete. Continue into the
        # existing cancel/close guardian and report the uncertainty;
        # never let an entry-oriented read failure disable risk reduction.
        reconciliation_errors.append(str(exc))

    orphan_result = _discover_orphan_slc_positions(client)

    with get_live_slc_session() as session:
        open_positions = list(session.exec(
            select(SlcPosition).where(SlcPosition.status.in_(("open", "ambiguous", "protected_degraded")))
        ))
    from alpaca.trading.enums import OrderSide
    result = closeout_mod.run_closeout(
        client, open_positions,
        side_for=lambda s: OrderSide.SELL if s == "sell" else OrderSide.BUY,
    )
    for position, fill_price in result.flattened:
        _mark_position_closed(position.id, exit_price=fill_price, exit_reason="closeout")

    # ONE genuine broker-wide readback, covering every symbol closeout
    # actually attempted to act on PLUS any orphan broker position that
    # was SLC-attributable but couldn't be confidently resolved to adopt
    # (found via review: excluding unresolved orphans from this set let
    # broker_flat report True while a real, unaccounted-for position sat
    # open) - never trivially True just because `result.flattened`
    # happened to be empty (found via review: the original loop only ran
    # over `flattened`, so a day with zero local positions left
    # `broker_flat` at its True default without ever making a real
    # broker call), and never treated as flat on a broker read failure
    # (found via review: the original bare `except Exception: pass`
    # swallowed a network/auth failure the same as a genuine "not found").
    acted_on_symbols = (
        {p.symbol for p, _ in result.flattened} | {p.symbol for p in result.failed}
        | {p.symbol for p in result.ambiguous_close}
        | {p.symbol for p in result.skipped_ambiguous}
        | orphan_result.unresolved_symbols
    )
    broker_flat: Optional[bool]
    try:
        broker_positions_after = client.get_all_positions()
        nonzero_symbols = {
            str(getattr(p, "symbol", "")) for p in (broker_positions_after or [])
            if float(getattr(p, "qty", 0) or 0) != 0
        }
        broker_flat = not (acted_on_symbols & nonzero_symbols)
    except Exception:  # noqa: BLE001
        broker_flat = None  # readback itself failed - distinct from "confirmed not flat"

    with get_live_slc_session() as session:
        stat = _get_or_create_session_stat(session, today)
        stat.closeout_left_no_open_state = not result.skipped_ambiguous and not result.failed and not result.ambiguous_close
        stat.closeout_confirmed_flat_by_broker_readback = broker_flat
        session.add(stat)

    finish_fields = {"orders_submitted": len(result.flattened)}
    if reconciliation_errors:
        finish_fields["errors_json"] = json.dumps(reconciliation_errors)
    _finish_cycle_run(run_id, **finish_fields)
    return {
        "status": "closed_out", "flattened": len(result.flattened),
        "skipped_ambiguous": len(result.skipped_ambiguous), "failed": len(result.failed),
        "ambiguous_close": len(result.ambiguous_close), "broker_flat": broker_flat,
        "reconciliation_errors": reconciliation_errors,
    }


def _write_trade_from_closed_position(
    *, position_id: str, symbol: str, direction: str, entry_price: float, qty: float,
    session_date, exit_price, exit_reason: str, exit_order_id: Optional[str] = None,
    exit_time=None,
) -> None:
    """exit_price must be the close order's ACTUAL fill price (found via
    review: this previously always used the position's theoretical
    target_price, fabricating P&L as if every flatten hit its exact
    target - a closeout/emergency flatten exits at whatever the market
    is, essentially never the target). Falls back to entry_price (a
    reported breakeven, never a fabricated win/loss) only in the
    degenerate case where no real fill price could be determined at
    all - explicitly flagged via exit_reason, never silently treated as
    a real number."""
    if exit_price is None:
        exit_price = entry_price
        exit_reason = f"{exit_reason}_exit_price_unknown"
    wrote_trade = False
    gross = (
        (exit_price - entry_price) if direction == "long" else (entry_price - exit_price)
    ) * qty
    with get_live_slc_session() as session:
        existing = session.exec(select_trade_by_position(position_id)).first()
        if existing is not None:
            return  # a position closes exactly once (unique constraint) - idempotent no-op on retry
        from live_slc.models import SlcTrade
        session.add(SlcTrade(
            position_id=position_id, symbol=symbol, direction=direction,
            entry_price=entry_price, exit_price=exit_price,
            exit_time=exit_time or _utc_now_naive().to_pydatetime(), exit_reason=exit_reason,
            exit_order_id=exit_order_id, qty=qty, gross_pnl=gross, net_pnl=gross, pnl_r=0.0,
            session_date=session_date,
        ))
        wrote_trade = True
    if wrote_trade:
        with get_live_slc_session() as session:
            stat = _get_or_create_session_stat(session, session_date)
            stat.trades_closed += 1
            stat.net_pnl += gross
            if gross > 0:
                stat.wins += 1
            elif gross < 0:
                stat.losses += 1
            session.add(stat)


def select_trade_by_position(position_id: str):
    from live_slc.models import SlcTrade
    return select(SlcTrade).where(SlcTrade.position_id == position_id)


def _get_or_create_session_stat(session, session_date):
    from live_slc.models import SlcSessionStat
    stat = session.get(SlcSessionStat, session_date)
    if stat is None:
        stat = SlcSessionStat(session_date=session_date)
        session.add(stat)
        session.flush()
    return stat


def main() -> None:
    parser = argparse.ArgumentParser(description="SLC 4h/5m live paper-forward runner")
    parser.add_argument("--stage", choices=["preflight", "cycle", "closeout"], required=True)
    args = parser.parse_args()

    try:
        with acquire_process_lock():
            # Moved inside the lock (rev. 11 point 1/4): init_live_slc_db()
            # runs a real schema migration - two processes must never be
            # able to run one concurrently.
            init_live_slc_db()
            if args.stage == "preflight":
                result = run_preflight()
            elif args.stage == "cycle":
                result = run_cycle()
            else:
                result = run_closeout_stage()
    except LockAlreadyHeld as exc:
        print(str(exc))
        if args.stage == "cycle":
            # A scheduled cycle that couldn't even start because the
            # previous one was still running IS an overlapping cycle -
            # the process lock is exactly what makes true concurrent
            # execution impossible, so this is the one well-defined
            # place "overlap" can be observed (found via review:
            # overlapping_cycles was never populated anywhere).
            # Best-effort: never let a stat-recording issue turn this
            # already-exceptional path into a harder failure.
            try:
                _bump_session_stat_counter(_utc_now_naive().date(), "overlapping_cycles")
            except Exception:  # noqa: BLE001
                pass
        sys.exit(0)
    print(json.dumps(result, default=str))


if __name__ == "__main__":
    main()
