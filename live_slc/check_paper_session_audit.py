"""
End-of-day audit for a real (non-dry-run) live_slc paper session:
python -m live_slc.check_paper_session_audit --date YYYY-MM-DD

Unlike check_session_gate.py (built around dry-run-specific fields like
dry_run_proposal_count), this covers what a REAL paper_active session
actually produces: submitted orders, fills, protection status, derived
slippage, ambiguous states, closed trades, open positions, classified
audit events, and an independently-corroborated broker-flat check.
Read-only throughout - client.get_all_positions() is the only broker call.

Audit-event classification and the status-aware alpaca_order_id rule are
both grounded directly in the real code (see the activation plan for the
grep/verification trail), not assumed:
- SlcAuditEvent.resolved is never set True anywhere in live_slc/ for any
  event type, so "any resolved=False row" would flag every event ever
  written, including benign ones - classification is by event_type.
- Not every non-dry_run_proposed SlcOrder.status implies a broker order
  ID was ever assigned (confirmed_rejected/confirmed_no_order_resulted
  legitimately have none) - the ID requirement is status-aware.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from sqlmodel import select

from live_slc import risk
from live_slc.models import (
    SlcAuditEvent, SlcOrder, SlcPosition, SlcSessionStat, SlcTrade,
    get_live_slc_session,
)

_ID_REQUIRED_STATUSES = {
    "submitted", "filled", "canceled_unfilled",
    "ambiguous_cancel", "target_replaced", "target_replacement_pending",
}

_CRITICAL_EVENT_TYPES = {
    "protected_degraded", "ambiguous_quantity", "discovered_fill_unresolvable_identity",
    "orphan_broker_position_unresolved", "split_evidence_conflict", "split_rebuild_failed",
}


def _orders_for_date(session, target_date: date) -> list:
    return [o for o in session.exec(select(SlcOrder)) if o.submitted_at.date() == target_date]


def _audit_events_for_date(session, target_date: date) -> list:
    return [e for e in session.exec(select(SlcAuditEvent)) if e.occurred_at.date() == target_date]


def audit_orders(orders: list) -> tuple:
    """Returns (issues, entries_today_count). Blocking statuses always
    fail regardless of ID; ID-required statuses fail only if the ID is
    missing; ID-optional statuses (confirmed_rejected,
    confirmed_no_order_resulted, dry_run_proposed) are never flagged for
    a missing ID."""
    issues = []
    entries_today = 0
    for o in orders:
        if o.leg == "entry" and o.status != "dry_run_proposed":
            entries_today += 1
        if o.status in risk._BLOCKING_ORDER_STATUSES:
            issues.append(f"order {o.client_order_id}: status={o.status!r} still unresolved at audit time")
            continue
        if o.status in _ID_REQUIRED_STATUSES and not o.alpaca_order_id:
            issues.append(f"order {o.client_order_id}: status={o.status!r} missing alpaca_order_id")
        if o.dry_run:
            issues.append(f"order {o.client_order_id}: dry_run=True recorded during a real session")
    return issues, entries_today


def audit_positions(positions: list) -> list:
    issues = []
    for p in positions:
        if p.status == "ambiguous":
            issues.append(f"position {p.id} ({p.symbol}): status=ambiguous")
        elif p.status == "protected_degraded":
            issues.append(f"position {p.id} ({p.symbol}): status=protected_degraded")
        elif p.status == "open" and not p.protective_order_id:
            issues.append(f"position {p.id} ({p.symbol}): open with no protective_order_id")
    return issues


def derive_order_slippage(order):
    if order.fill_price is None or order.expected_quote is None:
        return None
    return order.fill_price - order.expected_quote


def check_broker_flat(client, symbols_touched: set) -> tuple:
    """Returns (independently_flat, still_nonzero_symbols). The only
    broker call in this module - read-only."""
    broker_positions = client.get_all_positions()
    nonzero = {
        str(getattr(p, "symbol", "")) for p in (broker_positions or [])
        if float(getattr(p, "qty", 0) or 0) != 0
    }
    still_open = symbols_touched & nonzero
    return not still_open, still_open


def run_audit(client, target_date: date) -> dict:
    with get_live_slc_session() as session:
        orders = _orders_for_date(session, target_date)
        positions = list(session.exec(select(SlcPosition).where(SlcPosition.session_date == target_date)))
        trades = list(session.exec(select(SlcTrade).where(SlcTrade.session_date == target_date)))
        events_today = _audit_events_for_date(session, target_date)
        block_reasons = risk.system_wide_entry_block_reasons(session)
        stat = session.get(SlcSessionStat, target_date)

    order_issues, entries_today = audit_orders(orders)
    position_issues = audit_positions(positions)
    critical_events = [e for e in events_today if e.event_type in _CRITICAL_EVENT_TYPES]

    symbols_touched = {o.symbol for o in orders} | {p.symbol for p in positions}
    broker_flat, nonflat_symbols = check_broker_flat(client, symbols_touched)

    passed = (
        not order_issues and not position_issues and not critical_events
        and not block_reasons and broker_flat
    )

    return {
        "date": target_date.isoformat(),
        "passed": passed,
        "orders": {
            "count": len(orders),
            "entries_today": entries_today,
            "entries_exceed_daily_cap": entries_today > 2,  # informational only - the cap is enforced upstream
            "issues": order_issues,
            "slippage_derived": {
                o.client_order_id: derive_order_slippage(o) for o in orders if o.status == "filled"
            },
        },
        "positions": {"count": len(positions), "issues": position_issues},
        "trades": [
            {
                "symbol": t.symbol, "entry_price": t.entry_price, "exit_price": t.exit_price,
                "exit_reason": t.exit_reason, "exit_order_id": t.exit_order_id,
                "gross_pnl": t.gross_pnl, "net_pnl": t.net_pnl, "pnl_r": t.pnl_r,
                "fees": "not_reported",
            }
            for t in trades
        ],
        "ambiguous_states": block_reasons,
        "audit_events": {
            "all": [{"event_type": e.event_type, "symbol": e.symbol, "resolved": e.resolved} for e in events_today],
            "critical": [e.event_type for e in critical_events],
        },
        "broker_flat": {
            "closeout_confirmed": bool(stat.closeout_confirmed_flat_by_broker_readback) if stat else None,
            "independently_corroborated": broker_flat,
            "still_nonzero_symbols": sorted(nonflat_symbols),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="Session date, YYYY-MM-DD")
    args = parser.parse_args()
    target_date = date.fromisoformat(args.date)

    from live_slc.execution import get_alpaca_client
    client = get_alpaca_client()

    result = run_audit(client, target_date)

    print(f"live_slc paper session audit - {result['date']}")
    print(f"  orders: {result['orders']['count']} (entries today: {result['orders']['entries_today']})")
    if result["orders"]["entries_exceed_daily_cap"]:
        print(f"    NOTE: entries_today ({result['orders']['entries_today']}) exceeds the 2/day cap - "
              f"the cap is enforced upstream at admission time, this is informational only")
    for issue in result["orders"]["issues"]:
        print(f"    ISSUE: {issue}")
    print(f"  positions: {result['positions']['count']}")
    for issue in result["positions"]["issues"]:
        print(f"    ISSUE: {issue}")
    print(f"  trades: {len(result['trades'])}")
    print(f"  ambiguous states: {result['ambiguous_states'] or 'none'}")
    print(f"  audit events: {len(result['audit_events']['all'])} total, "
          f"{len(result['audit_events']['critical'])} critical")
    for et in result["audit_events"]["critical"]:
        print(f"    CRITICAL: {et}")
    print(f"  broker-flat: closeout_confirmed={result['broker_flat']['closeout_confirmed']} "
          f"independently_corroborated={result['broker_flat']['independently_corroborated']}")
    if result["broker_flat"]["still_nonzero_symbols"]:
        print(f"    ISSUE: still nonzero at broker: {result['broker_flat']['still_nonzero_symbols']}")

    if result["passed"]:
        print("PASS - paper session audit clean.")
        sys.exit(0)
    print("FAIL - see ISSUE/CRITICAL lines above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
