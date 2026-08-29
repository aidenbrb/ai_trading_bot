"""
Monitor Node - Phase 7 of the pipeline.

Runs daily (independently of the entry pipeline) to manage all open positions.
Pure Python - no LLM calls.

Two responsibilities:

A) Auto-reconcile closed positions
   - Check each DB "open" position against the live broker's positions
     (Alpaca, or Robinhood instead when ROBINHOOD_ENABLED=true - the two are
     mutually exclusive live brokers for the same stock universe).
   - If the broker no longer holds the position (stop hit or target hit),
     record a Trade with PnL and mark the position closed.
   - If the broker connection cannot be verified, reconciliation and exit
     management are skipped entirely for the run rather than guessed - an
     unreachable broker must never be mistaken for "zero open positions."

B) Rule-based position review (for positions still open at the broker)
   - Deterministic HOLD / TIGHTEN_STOP / CLOSE decision from live price,
     trend, RSI, MACD, and ATR (see _evaluate_position).
       HOLD          - thesis intact, keep holding
       TIGHTEN_STOP  - price moved in our favor; raise stop to lock in gains
       CLOSE         - trend/momentum reversal; exit now at market

Safety rules enforced in Python:
   - TIGHTEN_STOP: new_stop must be strictly > current stop (never loosen)
   - TIGHTEN_STOP: new_stop must be < current price (still valid stop)
   - CLOSE: always a market sell - no limit exits
   - Only one open sell order per position at a time (broker constraint)

Run standalone:
    python -m nodes.monitor_node
    python -m nodes.monitor_node --tickers NVDA AAPL

This node is designed to run every morning before the entry pipeline,
or at any point during the day to check positions.
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date, datetime
from utils.timeutil import utcnow
from typing import Optional

from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET
from db.connection import get_session, init_db
from db.models import Indicator, Order, Position, RunLog, Ticker, Trade

_NODE = "monitor_node"

# -- Rule-based exit thresholds --------------------------------------------------
_RSI_BREAKDOWN       = 45.0  # RSI below this + negative MACD hist => CLOSE
_FAVORABLE_ATR_MULT  = 1.0   # min favorable move (in ATR) before trailing the stop


def _canonical_symbol(symbol: str) -> str:
    """Normalize Alpaca crypto symbols (BTC/USD or BTCUSD) to BTC-USD."""
    upper = symbol.upper()
    if "/" in upper:
        return upper.replace("/", "-")
    compact = upper.replace("-", "")
    for crypto_symbol in CRYPTO_SET:
        if compact == crypto_symbol.replace("-", ""):
            return crypto_symbol
    return upper


def _alpaca_symbol(symbol: str) -> str:
    upper = symbol.upper()
    return upper[:-4] + "/USD" if upper in CRYPTO_SET else upper


def _find_alpaca_position(client, symbol: str):
    """Find a live position despite Alpaca's SOLUSD/SOL/USD variants."""
    return next(
        (
            position
            for position in client.get_all_positions()
            if _canonical_symbol(str(getattr(position, "symbol", ""))) == symbol
        ),
        None,
    )


def _round_price(symbol: str, value: float) -> float:
    return round(value, 8 if symbol.upper() in CRYPTO_SET else 2)


# -- Public entry point --------------------------------------------------------

def run(tickers: list[str] | None = None) -> dict:
    """
    Review and manage all open positions.

    Parameters
    ----------
    tickers : list[str] | None
        Limit review to these symbols. Defaults to all open positions.

    Returns
    -------
    dict with keys: held, tightened, closed, reconciled, failed, run_id
    """
    settings.validate()

    monitor_run_id = str(uuid.uuid4())
    started        = utcnow()

    print(f"\n{'='*55}")
    print(f"  MONITOR NODE   run_id={monitor_run_id[:8]}")
    print(f"  Mode: rule-based (ATR trail + indicator deterioration)")
    print(f"{'='*55}")

    init_db()

    # Fetch live broker positions once; reused for pending-fill promotion and
    # reconciliation. Stock positions come from Robinhood instead of Alpaca
    # when ROBINHOOD_ENABLED - the two are mutually exclusive live brokers for
    # the same stock universe, so exactly one of them holds a given symbol.
    broker_positions, broker_connected = (
        _get_robinhood_positions() if settings.ROBINHOOD_ENABLED else _get_alpaca_positions()
    )
    if not broker_connected:
        print("\n  *** BROKER UNREACHABLE - skipping reconciliation and exit management "
              "this run to avoid mis-marking real positions as closed. Resting broker "
              "stop/target orders remain the only protection until the next successful "
              "run. ***")
        duration = (utcnow() - started).total_seconds()
        _write_log(monitor_run_id, {}, 0, duration, status="broker_unreachable")
        return {"run_id": monitor_run_id, "held": [], "tightened": [], "closed": [],
                "reconciled": [], "failed": [], "broker_connected": False}

    # Resolve pending entries that filled and exited entirely between monitor
    # runs.  Looking only at current positions cannot distinguish those from a
    # genuinely unfilled GTC entry; the nested bracket order can.
    pending_resolution = _reconcile_pending_positions(
        broker_positions, monitor_run_id
    )
    pending_reconciled = pending_resolution["reconciled"]
    pending_failed = pending_resolution["failed"]

    # Promote any 'pending' limit entries that have since filled so they begin
    # being managed. Pending (unfilled) entries are deliberately NOT loaded as
    # open positions below, so they can never be "reconciled" into a phantom
    # stop-out Trade just because the broker has no position yet.
    promoted = _promote_pending_positions(broker_positions)
    if promoted:
        print(f"  Promoted {len(promoted)} pending entry(ies) to open: {', '.join(promoted)}")

    open_positions = _load_open_positions(tickers)
    if not open_positions:
        print("  No open positions to monitor.")
        summary = {"reconciled": pending_reconciled, "failed": pending_failed}
        _write_log(
            monitor_run_id,
            summary,
            len(pending_reconciled) + len(pending_failed),
            (utcnow() - started).total_seconds(),
            status="partial" if pending_failed else "success",
        )
        return {"run_id": monitor_run_id, "held": [], "tightened": [],
                "closed": [], "reconciled": pending_reconciled,
                "failed": pending_failed}

    print(f"  Open positions in DB: {len(open_positions)}")

    # -- A. Reconcile with the live broker --------------------------------------
    reconciled = pending_reconciled + _reconcile(
        open_positions, broker_positions, monitor_run_id
    )

    # Filter to positions still open after reconciliation
    still_open = [p for p in open_positions
                  if _symbol_for(p.ticker_id) not in reconciled]

    print(f"  Reconciled closed: {len(reconciled)}  Still open: {len(still_open)}")

    if not still_open:
        duration = (utcnow() - started).total_seconds()
        _write_log(monitor_run_id, {"reconciled": reconciled}, len(open_positions), duration)
        return {"run_id": monitor_run_id, "held": [], "tightened": [],
                "closed": [], "reconciled": reconciled, "failed": []}

    # -- B. Rule-based review ---------------------------------------------------
    held_list, tightened_list, closed_list = [], [], []
    failed_list = list(pending_failed)

    for pos in still_open:
        symbol = _symbol_for(pos.ticker_id)
        broker_pos = broker_positions.get(symbol)
        current_price = float(broker_pos["current_price"]) if broker_pos else None

        print(f"\n  {symbol}  "
              f"entry=${pos.entry_price:.2f}  "
              f"current=${current_price or '?'}  "
              f"stop=${pos.stop_price:.2f}  "
              f"target=${pos.target_price:.2f}  "
              f"held={_days_held(pos.entry_date)}d")

        try:
            ind = _get_latest_indicator(pos.ticker_id)
            decision = _evaluate_position(symbol, pos, current_price, ind)
        except Exception as exc:
            failed_list.append({"symbol": symbol, "reason": str(exc)})
            print(f"    ERR {exc}")
            continue

        action = decision["action"]
        reason = decision.get("reason", "")

        if action == "HOLD":
            held_list.append(symbol)
            print(f"    -> HOLD  ({reason})")

        elif action == "TIGHTEN_STOP":
            new_stop = decision.get("new_stop")
            executed_stop = _tighten_stop(symbol, pos, float(new_stop)) if new_stop else None
            if executed_stop:
                _update_position_stop(pos.id, executed_stop)
                tightened_list.append(symbol)
                print(f"    -> TIGHTEN_STOP  ${pos.stop_price:.2f} -> ${executed_stop:.2f}  "
                      f"({reason})")
            else:
                held_list.append(symbol)
                print(f"    -> TIGHTEN_STOP rejected by safety rails - HOLD instead")

        elif action == "CLOSE":
            exit_price = (
                _close_robinhood_position(symbol) if settings.ROBINHOOD_ENABLED
                else _close_alpaca_position(symbol)
            )
            if exit_price is not None:
                _close_position_in_db(pos, exit_price, date.today(), "monitor", monitor_run_id)
                closed_list.append(symbol)
                print(f"    -> CLOSE  reason: {reason}")
            else:
                failed_list.append({
                    "symbol": symbol,
                    "reason": f"broker close could not be confirmed: {reason}",
                })
                print(f"    -> CLOSE FAILED  reason: {reason}")

    duration = (utcnow() - started).total_seconds()
    summary = {
        "held":        held_list,
        "tightened":   tightened_list,
        "closed":      closed_list,
        "reconciled":  reconciled,
        "failed":      failed_list,
    }
    _write_log(monitor_run_id, summary, len(open_positions), duration)

    print(f"\n  Done in {duration:.1f}s - "
          f"held={len(held_list)}  tightened={len(tightened_list)}  "
          f"closed={len(closed_list)}  reconciled={len(reconciled)}  "
          f"failed={len(failed_list)}")

    return {"run_id": monitor_run_id, **summary}


# -- A. Reconciliation ---------------------------------------------------------

def _promote_pending_positions(broker_positions: dict[str, dict]) -> list[str]:
    """
    A 'pending' position is one whose limit entry order had not filled when the
    execution node ran. If the broker now holds the symbol, the entry filled --
    promote it to 'open' and record the broker's real average entry price.
    Positions that have not filled are left pending and are NOT reconciled as
    closed (which would otherwise fabricate a stop-out loss for an order that
    simply has not executed yet).
    """
    promoted: list[str] = []
    with get_session() as session:
        pending = session.exec(
            select(Position).where(Position.status == "pending")
        ).all()
        for pos in pending:
            symbol = _symbol_for(pos.ticker_id)
            ap = broker_positions.get(symbol)
            if ap and ap.get("qty", 0) > 0:
                pos.status      = "open"
                pos.entry_price = ap.get("avg_entry") or pos.entry_price
                pos.current_price = ap.get("current_price")
                pos.updated_at  = utcnow()
                promoted.append(symbol)
    return promoted


def _enum_text(value) -> str:
    return str(value or "").lower().split(".")[-1]


def _filled_bracket_exit(broker_order) -> Optional[dict]:
    """Return the filled sell leg from a nested Alpaca bracket, if present."""
    filled = []
    for leg in getattr(broker_order, "legs", None) or []:
        if _enum_text(getattr(leg, "side", None)) != "sell":
            continue
        if _enum_text(getattr(leg, "status", None)) != "filled":
            continue
        price = getattr(leg, "filled_avg_price", None)
        if price is None or float(price) <= 0:
            continue
        filled.append({
            "price": float(price),
            "filled_at": getattr(leg, "filled_at", None),
            "reason": (
                "broker_stop" if _enum_text(
                    getattr(leg, "type", None) or getattr(leg, "order_type", None)
                ) in {"stop", "stop_limit", "trailing_stop"}
                else "broker_target"
            ),
        })
    if not filled:
        return None
    return max(
        filled,
        key=lambda item: (
            item["filled_at"].timestamp()
            if isinstance(item["filled_at"], datetime)
            else float("-inf")
        ),
    )


def _reconcile_pending_positions(
    broker_positions: dict[str, dict],
    run_id: str,
    client=None,
) -> dict:
    """Resolve stock entries that filled and exited between monitor runs.

    The broker is queried read-only.  An absent current position is never
    treated as a loss by itself: a local pending row is closed only when a
    nested bracket contains a confirmed filled sell leg.  Unfilled live GTC
    entries remain pending, while ambiguous states fail closed for review.
    """
    result = {"reconciled": [], "canceled": [], "failed": []}
    if settings.ROBINHOOD_ENABLED:
        return result

    with get_session() as session:
        pending = session.exec(
            select(Position).where(Position.status == "pending")
        ).all()
        items = []
        for pos in pending:
            symbol = _symbol_for(pos.ticker_id)
            if symbol in broker_positions or symbol in CRYPTO_SET:
                continue
            order = session.exec(
                select(Order).where(Order.id == pos.order_id)
            ).first()
            items.append((pos, symbol, order))

    if not items:
        return result

    if client is None:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(
                api_key=settings.ALPACA_API_KEY,
                secret_key=settings.ALPACA_SECRET_KEY,
                paper=True,
            )
        except Exception as exc:
            return {
                **result,
                "failed": [{"symbol": symbol, "reason": f"broker lookup unavailable: {exc}"}
                           for _, symbol, _ in items],
            }

    from alpaca.trading.requests import GetOrderByIdRequest
    for pos, symbol, order in items:
        if not order or not order.alpaca_order_id:
            result["failed"].append({
                "symbol": symbol,
                "reason": "pending DB position has no broker order ID",
            })
            continue
        try:
            broker_order = client.get_order_by_id(
                order.alpaca_order_id,
                GetOrderByIdRequest(nested=True),
            )
        except Exception as exc:
            result["failed"].append({
                "symbol": symbol,
                "reason": f"pending broker order lookup failed: {exc}",
            })
            continue

        exit_fill = _filled_bracket_exit(broker_order)
        parent_status = _enum_text(getattr(broker_order, "status", None))
        parent_fill = getattr(broker_order, "filled_avg_price", None)
        parent_qty = getattr(broker_order, "filled_qty", None)
        if exit_fill is not None:
            real_entry = float(parent_fill) if parent_fill else pos.entry_price
            real_qty = float(parent_qty) if parent_qty else pos.shares
            pos.entry_price = real_entry
            pos.shares = real_qty
            with get_session() as session:
                db_pos = session.exec(
                    select(Position).where(Position.id == pos.id)
                ).first()
                db_order = session.exec(
                    select(Order).where(Order.id == order.id)
                ).first()
                if db_pos:
                    db_pos.entry_price = real_entry
                    db_pos.shares = real_qty
                    db_pos.updated_at = utcnow()
                if db_order:
                    db_order.status = "filled_closed"
                    db_order.filled_price = real_entry
                    db_order.filled_at = getattr(broker_order, "filled_at", None)
            exit_at = exit_fill["filled_at"]
            exit_date = exit_at.date() if isinstance(exit_at, datetime) else date.today()
            _close_position_in_db(
                pos,
                exit_fill["price"],
                exit_date,
                exit_fill["reason"],
                run_id,
            )
            result["reconciled"].append(symbol)
            print(
                f"  Reconciled pending {symbol}: confirmed {exit_fill['reason']} "
                f"fill at ${exit_fill['price']:.8g}"
            )
            continue

        filled_qty = float(parent_qty or 0)
        if parent_status in {"canceled", "cancelled", "expired", "rejected"} and filled_qty <= 0:
            with get_session() as session:
                db_pos = session.exec(
                    select(Position).where(Position.id == pos.id)
                ).first()
                db_order = session.exec(
                    select(Order).where(Order.id == order.id)
                ).first()
                if db_pos:
                    db_pos.status = "canceled"
                    db_pos.updated_at = utcnow()
                if db_order:
                    db_order.status = parent_status
            result["canceled"].append(symbol)
        elif parent_status == "filled":
            result["failed"].append({
                "symbol": symbol,
                "reason": "entry filled and broker position absent, but no filled bracket exit was found",
            })
        # Otherwise the entry remains a legitimate resting GTC order.

    return result


def _get_alpaca_positions() -> tuple[dict[str, dict], bool]:
    """
    Return ({symbol: {current_price, qty, unrealized_pnl}}, connected) from Alpaca
    paper. connected=False (missing creds or any API failure) must NOT be treated
    as "zero open positions" by the caller - see run()'s broker_connected check.
    """
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        return {}, False
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,   # HARDCODED
        )
        positions = {
            _canonical_symbol(p.symbol): {
                "current_price":  float(p.current_price) if p.current_price else 0.0,
                "qty":            float(p.qty),
                "unrealized_pnl": float(p.unrealized_pl) if p.unrealized_pl else 0.0,
                "avg_entry":      float(p.avg_entry_price),
            }
            for p in client.get_all_positions()
        }
        return positions, True
    except Exception as exc:
        print(f"  [Monitor] Could not fetch Alpaca positions: {exc}")
        return {}, False


def _get_robinhood_positions() -> tuple[dict[str, dict], bool]:
    """
    Return ({symbol: {current_price, qty, unrealized_pnl, avg_entry}}, connected)
    from Robinhood. connected=False must NOT be treated as "zero open positions"
    by the caller - see run()'s broker_connected check.
    """
    if not settings.ROBINHOOD_USERNAME or not settings.ROBINHOOD_PASSWORD:
        return {}, False
    try:
        from utils.robinhood_account import login
        r = login()
        holdings = r.account.build_holdings() or {}
        positions = {
            symbol: {
                "current_price":  float(h.get("price", 0) or 0),
                "qty":            int(float(h.get("quantity", 0) or 0)),
                "unrealized_pnl": float(h.get("equity_change", 0) or 0),
                "avg_entry":      float(h.get("average_buy_price", 0) or 0),
            }
            for symbol, h in holdings.items()
            if float(h.get("quantity", 0) or 0) > 0
        }
        return positions, True
    except Exception as exc:
        print(f"  [Monitor] Could not fetch Robinhood positions: {exc}")
        return {}, False


def _reconcile(
    db_positions:     list[Position],
    broker_positions: dict[str, dict],
    run_id:           str,
) -> list[str]:
    """
    Close out any DB positions that the live broker no longer holds.

    Handles multiple DB lots for the same symbol (e.g. two MRNA entries):
    if the broker's share count for a symbol is less than the DB total, the
    most-recently-entered lots are closed first until counts match.

    Returns list of symbols that were reconciled (auto-closed by the broker).
    """
    reconciled = []
    today      = date.today()

    # Group DB positions by symbol, sorted oldest-first within each group
    from collections import defaultdict
    by_symbol: dict[str, list[Position]] = defaultdict(list)
    for pos in db_positions:
        by_symbol[_symbol_for(pos.ticker_id)].append(pos)
    for lots in by_symbol.values():
        lots.sort(key=lambda p: p.entry_date)

    for symbol, lots in by_symbol.items():
        if symbol not in broker_positions:
            # Entire symbol gone - close all lots
            for pos in lots:
                exit_price = _estimate_exit_price(pos, symbol)
                _close_position_in_db(pos, exit_price, today, "stop_or_target", run_id)
                reconciled.append(symbol)
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                sign = "+" if pnl >= 0 else ""
                print(f"  Reconciled {symbol}: exit~${exit_price:.2f}  "
                      f"PnL={sign}${pnl:.0f} ({sign}{pnl_pct:.1f}%)")
        else:
            ap = broker_positions[symbol]
            broker_shares = ap["qty"]
            db_total      = sum(p.shares for p in lots)

            # Close most-recent lots until DB total matches the broker
            remaining = db_total
            for pos in reversed(lots):  # newest first
                tolerance = max(1e-8, abs(broker_shares) * 1e-8)
                if remaining <= broker_shares + tolerance:
                    break
                exit_price = _estimate_exit_price(pos, symbol)
                _close_position_in_db(pos, exit_price, today, "stop_or_target", run_id)
                reconciled.append(symbol)
                pnl = (exit_price - pos.entry_price) * pos.shares
                pnl_pct = (exit_price - pos.entry_price) / pos.entry_price * 100
                sign = "+" if pnl >= 0 else ""
                print(f"  Reconciled {symbol} lot ({pos.shares}sh): exit~${exit_price:.2f}  "
                      f"PnL={sign}${pnl:.0f} ({sign}{pnl_pct:.1f}%)")
                remaining -= pos.shares

            # Update price/PnL on remaining open lots
            for pos in lots:
                if pos.status == "open":
                    with get_session() as session:
                        db_pos = session.exec(
                            select(Position).where(Position.id == pos.id)
                        ).first()
                        if db_pos:
                            db_pos.current_price  = ap["current_price"]
                            db_pos.unrealized_pnl = ap["unrealized_pnl"]
                            db_pos.updated_at     = utcnow()

    return reconciled


def _alpaca_crypto_exit_fill(pos: Position, symbol: str) -> Optional[float]:
    """Read the actual filled crypto stop, following any replacement chain."""
    if symbol not in CRYPTO_SET:
        return None
    try:
        with get_session() as session:
            entry_order = session.exec(
                select(Order).where(Order.id == pos.order_id)
            ).first()
        if not entry_order or not entry_order.risk_approval_id:
            return None

        from alpaca.trading.client import TradingClient
        client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,
        )
        stop_client_id = f"appr-{entry_order.risk_approval_id}-sl"
        exit_order = client.get_order_by_client_id(stop_client_id)

        # Alpaca assigns a new order ID when monitor_node tightens a stop.
        # Follow that chain until the current/final order is reached.
        seen = set()
        for _ in range(10):
            order_id = str(getattr(exit_order, "id", "") or "")
            if not order_id or order_id in seen:
                break
            seen.add(order_id)
            replacement_id = getattr(exit_order, "replaced_by", None)
            if not replacement_id:
                break
            exit_order = client.get_order_by_id(str(replacement_id))

        status = str(getattr(exit_order, "status", "")).lower().split(".")[-1]
        fill = getattr(exit_order, "filled_avg_price", None)
        if status == "filled" and fill and float(fill) > 0:
            return float(fill)
    except Exception as exc:
        print(f"    [Warn] Could not read actual crypto exit fill for {symbol}: {exc}")
    return None


def _alpaca_stock_exit_fill(pos: Position, symbol: str) -> Optional[float]:
    """Read the actual filled stop/target leg from an Alpaca stock bracket."""
    if symbol in CRYPTO_SET or settings.ROBINHOOD_ENABLED:
        return None
    try:
        with get_session() as session:
            entry_order = session.exec(
                select(Order).where(Order.id == pos.order_id)
            ).first()
        if not entry_order or not entry_order.alpaca_order_id:
            return None

        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrderByIdRequest
        client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,
        )
        broker_order = client.get_order_by_id(
            entry_order.alpaca_order_id,
            GetOrderByIdRequest(nested=True),
        )
        exit_fill = _filled_bracket_exit(broker_order)
        return exit_fill["price"] if exit_fill is not None else None
    except Exception as exc:
        print(f"    [Warn] Could not read actual stock bracket exit for {symbol}: {exc}")
        return None


def _estimate_exit_price(pos: Position, symbol: str = "") -> float:
    """
    Best-effort exit price when we don't have Alpaca fill data.
    Uses stop or target depending on which side price likely exited.
    """
    broker_fill = None
    if symbol:
        broker_fill = (
            _alpaca_crypto_exit_fill(pos, symbol)
            if symbol in CRYPTO_SET
            else _alpaca_stock_exit_fill(pos, symbol)
        )
    if broker_fill is not None:
        return broker_fill
    if pos.current_price:
        return pos.current_price
    return pos.stop_price   # conservative default (assume stop was hit)


def _close_position_in_db(
    pos:         Position,
    exit_price:  float,
    exit_date:   date,
    exit_reason: str,
    run_id:      str,
) -> None:
    """Mark position closed and record the Trade in DB."""
    pnl         = (exit_price - pos.entry_price) * pos.shares
    pnl_pct     = (exit_price - pos.entry_price) / pos.entry_price
    holding_days = (exit_date - pos.entry_date).days

    with get_session() as session:
        # Update position status
        db_pos = session.exec(
            select(Position).where(Position.id == pos.id)
        ).first()
        if db_pos:
            db_pos.status        = "closed"
            db_pos.current_price = exit_price
            db_pos.updated_at    = utcnow()

        # Record trade
        session.add(Trade(
            ticker_id=pos.ticker_id,
            position_id=pos.id,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=exit_date,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            holding_days=holding_days,
        ))


# -- B. Rule-based position review ----------------------------------------------

def _evaluate_position(
    symbol:        str,
    pos:           Position,
    current_price: Optional[float],
    ind:           Optional[Indicator],
) -> dict:
    """
    Deterministic HOLD / TIGHTEN_STOP / CLOSE decision for an open position,
    mirroring stock_strategy_node's entry-gate vocabulary in reverse.

    CLOSE        - trend has flipped to DOWNTREND, or MACD histogram is negative
                   while RSI has broken down below _RSI_BREAKDOWN (momentum
                   reversal - the entry thesis is broken)
    TIGHTEN_STOP - price has moved at least _FAVORABLE_ATR_MULT x ATR-14 in our
                   favor; trail the stop to MONITOR_ATR_TRAIL_MULT x ATR behind
                   the current price
    HOLD         - otherwise
    """
    if current_price is not None and current_price >= pos.target_price:
        return {"action": "CLOSE", "new_stop": None,
                "reason": f"target reached (${current_price:.8g} >= "
                          f"${pos.target_price:.8g})"}

    if ind and ind.trend == "DOWNTREND":
        return {"action": "CLOSE", "new_stop": None,
                "reason": "trend flipped to DOWNTREND - thesis broken"}

    if (ind and ind.macd_hist is not None and ind.macd_hist < 0
            and ind.rsi_14 is not None and ind.rsi_14 < _RSI_BREAKDOWN):
        return {"action": "CLOSE", "new_stop": None,
                "reason": f"MACD hist {ind.macd_hist:.4f} negative and RSI "
                          f"{ind.rsi_14:.1f} < {_RSI_BREAKDOWN:.0f} - momentum reversal"}

    if current_price is not None:
        atr = (ind.atr_14 if ind and ind.atr_14 else None) or (pos.entry_price * 0.02)
        favorable_move = current_price - pos.entry_price
        if favorable_move >= _FAVORABLE_ATR_MULT * atr:
            candidate = _round_price(
                symbol,
                current_price - settings.MONITOR_ATR_TRAIL_MULT * atr,
            )
            if candidate > pos.stop_price and candidate < current_price:
                return {"action": "TIGHTEN_STOP", "new_stop": candidate,
                        "reason": f"price +{favorable_move:.2f} "
                                  f"({favorable_move / atr:.1f}x ATR) above entry - trailing stop"}

    return {"action": "HOLD", "new_stop": None, "reason": "thesis intact"}


# -- Action execution ----------------------------------------------------------

def _tighten_stop(symbol: str, pos: Position, new_stop: float) -> Optional[float]:
    """
    Validate and execute a stop tightening.
    Returns the new stop price if successful, None if validation failed.
    """
    # Python enforces: can only move stop UP, never down
    if new_stop <= pos.stop_price:
        print(f"    [Risk] Rejected: new_stop ${new_stop:.2f} <= "
              f"current ${pos.stop_price:.2f} - cannot loosen stop")
        return None

    if pos.current_price and new_stop >= pos.current_price:
        print(f"    [Risk] Rejected: new_stop ${new_stop:.2f} >= "
              f"current price ${pos.current_price:.2f}")
        return None

    if not settings.EXECUTION_ENABLED:
        print(f"    [DryRun] Would tighten stop: ${pos.stop_price:.2f} -> ${new_stop:.2f}")
        return new_stop

    try:
        if settings.ROBINHOOD_ENABLED:
            _replace_stop_order_robinhood(symbol, new_stop)
        else:
            _replace_stop_order(symbol, new_stop)
        return new_stop
    except Exception as exc:
        print(f"    [Error] Could not update stop order: {exc}")
        return None


def _replace_stop_order(symbol: str, new_stop: float) -> None:
    """Find the open stop loss order for symbol and replace it with new_stop."""
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest, ReplaceOrderRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderType

    client = TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=True,
    )

    orders = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    stop_order = next(
        (o for o in orders
         if _canonical_symbol(o.symbol) == symbol
         and o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT)),
        None,
    )
    if not stop_order:
        raise RuntimeError(f"No open stop order found for {symbol}")

    replacement = {"stop_price": _round_price(symbol, new_stop)}
    if symbol.upper() in CRYPTO_SET:
        replacement["limit_price"] = _round_price(symbol, new_stop * 0.98)
    client.replace_order_by_id(str(stop_order.id), ReplaceOrderRequest(**replacement))


def _replace_stop_order_robinhood(symbol: str, new_stop: float) -> None:
    """Find the open stop loss order for symbol and replace it with new_stop.

    robin_stocks has no in-place order replace, so this cancels the existing
    stop order and submits a new one - same net effect as Alpaca's
    replace_order_by_id for this purpose.
    """
    from utils.robinhood_account import login
    r = login()

    orders = r.orders.get_all_open_stock_orders() or []
    stop_order = next(
        (o for o in orders
         if o.get("side") == "sell" and _order_symbol(r, o) == symbol
         and o.get("stop_price")),
        None,
    )
    if not stop_order:
        raise RuntimeError(f"No open stop order found for {symbol}")

    qty = float(stop_order.get("quantity") or 0)
    r.orders.cancel_stock_order(stop_order["id"])
    r.orders.order_sell_stop_loss(symbol, qty, round(new_stop, 2))


def _order_symbol(r, order: dict) -> str:
    """robin_stocks order dicts reference the instrument by URL, not symbol."""
    try:
        return r.stocks.get_symbol_by_url(order.get("instrument", ""))
    except Exception:
        return ""


def _close_robinhood_position(symbol: str) -> Optional[float]:
    """
    Cancel any resting orders for the symbol (the stop/target legs) and submit
    a market sell to close the position on Robinhood.

    Cancelling the protective legs FIRST is important: closing a position with
    a manual market sell does not remove the resting stop/target sell orders,
    so a leftover leg could later fire against a flat account and open an
    unintended short (same rationale as _close_alpaca_position below).
    """
    if not settings.EXECUTION_ENABLED:
        print(f"    [DryRun] Would close position: {symbol}")
        return None

    try:
        from utils.robinhood_account import login
        r = login()

        holdings = r.account.build_holdings() or {}
        holding = holdings.get(symbol)
        exit_price = float(holding["price"]) if holding and holding.get("price") else None
        qty = float(holding["quantity"]) if holding and holding.get("quantity") else 0.0

        # Cancel resting orders for this symbol (stop/target legs) so a stale
        # protective order cannot execute after we are flat.
        try:
            for o in (r.orders.get_all_open_stock_orders() or []):
                if _order_symbol(r, o) == symbol:
                    try:
                        r.orders.cancel_stock_order(o["id"])
                    except Exception:
                        pass
        except Exception as exc:
            print(f"    [Warn] Could not cancel resting orders for {symbol}: {exc}")

        if qty > 0:
            r.orders.order_sell_market(symbol, qty)
        return exit_price
    except Exception as exc:
        print(f"    [Error] Could not close {symbol}: {exc}")
        return None


def _close_alpaca_position(symbol: str) -> Optional[float]:
    """
    Cancel any resting orders for the symbol (the bracket stop/target legs) and
    submit a market sell to close the position in Alpaca.

    Cancelling the protective legs FIRST is important: closing a position with a
    manual market sell does not remove the original bracket's stop/target sell
    orders, so a leftover leg could later fire against a flat account and open an
    unintended short. Returns the last price captured before closing, or None.
    """
    if not settings.EXECUTION_ENABLED:
        print(f"    [DryRun] Would close position: {symbol}")
        return None

    try:
        import time
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
        client = TradingClient(
            api_key=settings.ALPACA_API_KEY,
            secret_key=settings.ALPACA_SECRET_KEY,
            paper=True,
        )

        # Alpaca's positions endpoint currently returns legacy crypto symbols
        # such as SOLUSD while orders use SOL/USD. Resolve the live position
        # first and close by its asset UUID so pair spelling cannot cause a 404.
        snap = _find_alpaca_position(client, symbol)
        if not snap:
            raise RuntimeError(f"live Alpaca position not found for {symbol}")
        exit_price = float(snap.current_price) if snap.current_price else None
        position_ref = str(getattr(snap, "asset_id", None) or snap.symbol)

        # Cancel resting orders for this symbol (bracket stop/target legs) so a
        # stale protective order cannot execute after we are flat.
        request = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        matching_orders = [
            order for order in client.get_orders(request)
            if _canonical_symbol(str(order.symbol)) == symbol
        ]
        for order in matching_orders:
            client.cancel_order_by_id(str(order.id))

        # Wait until cancellations are visible before submitting the close;
        # crypto stop orders reserve the asset balance while still open.
        if matching_orders:
            for _ in range(20):
                remaining_orders = [
                    order for order in client.get_orders(request)
                    if _canonical_symbol(str(order.symbol)) == symbol
                ]
                if not remaining_orders:
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"resting orders for {symbol} did not cancel; close aborted"
                )

        close_order = client.close_position(position_ref)
        close_order_id = str(getattr(close_order, "id", "") or "")
        for _ in range(20):
            if _find_alpaca_position(client, symbol) is None:
                return exit_price
            if close_order_id:
                latest = client.get_order_by_id(close_order_id)
                status = str(getattr(latest, "status", "")).lower().split(".")[-1]
                if status in {"canceled", "rejected", "expired"}:
                    break
            time.sleep(0.25)
        raise RuntimeError(f"position still present after close attempt for {symbol}")
    except Exception as exc:
        print(f"    [Error] Could not close {symbol}: {exc}")
        return None


def _update_position_stop(position_id: str, new_stop: float) -> None:
    with get_session() as session:
        pos = session.exec(
            select(Position).where(Position.id == position_id)
        ).first()
        if pos:
            pos.stop_price = new_stop
            pos.updated_at = utcnow()


# -- DB helpers ----------------------------------------------------------------

def _load_open_positions(tickers: Optional[list[str]]) -> list[Position]:
    with get_session() as session:
        rows = session.exec(
            select(Position).where(Position.status == "open")
        ).all()
    if tickers:
        upper = {t.upper() for t in tickers}
        rows = [r for r in rows if _symbol_for(r.ticker_id) in upper]
    return list(rows)


def _get_latest_indicator(ticker_id: str) -> Optional[Indicator]:
    with get_session() as session:
        return session.exec(
            select(Indicator)
            .where(Indicator.ticker_id == ticker_id)
            .order_by(Indicator.bar_date.desc())  # type: ignore[attr-defined]
        ).first()


_ticker_cache: dict[str, str] = {}

def _symbol_for(ticker_id: str) -> str:
    if ticker_id not in _ticker_cache:
        with get_session() as session:
            t = session.exec(select(Ticker).where(Ticker.id == ticker_id)).first()
            _ticker_cache[ticker_id] = t.symbol if t else "UNKNOWN"
    return _ticker_cache[ticker_id]


def _days_held(entry_date: date) -> int:
    return (date.today() - entry_date).days


# -- Logging -------------------------------------------------------------------

def _write_log(run_id: str, summary: dict, total: int, duration: float,
               status: str = "success") -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status=status,
            tickers_processed=total,
            records_written=(
                len(summary.get("tightened", [])) +
                len(summary.get("closed", [])) +
                len(summary.get("reconciled", []))
            ),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor Node - manage open positions")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(tickers=args.tickers)
