"""
Execution Node - Phase 6 of the pipeline.

Submits protected orders to Alpaca paper trading for every strategy approved
by the risk node. Stocks use atomic brackets. Crypto uses a fractional market
entry followed by a mandatory stop-limit because Alpaca does not support
bracket order classes for crypto; target exits are handled by monitor_node.

Safety layers (both must be true before any order is sent):
  1. EXECUTION_ENABLED=true in .env  (default: false)
  2. paper=True hardcoded in TradingClient - physically cannot reach live broker

When EXECUTION_ENABLED=false the node runs in DRY RUN mode:
  - Logs the full order details to the DB as dry_run=True
  - Prints exactly what would have been submitted
  - No API call is made - safe for testing the full pipeline end-to-end

Responsibilities:
  1. Read RiskApproval rows with approved=True for the current run
  2. For each approval, submit (or dry-run) a bracket order
  3. Write one Order row to the DB (dry_run flag records the mode)
  4. Write one Position row to the DB when a real order is submitted
  5. Log to run_log

Run standalone:
    python -m nodes.execution_node --run-id <risk_run_id>
    python -m nodes.execution_node   (uses most recent risk run)

Bracket order structure:
    Market BUY (fills immediately at open)
      --- Stop loss  - GTC sell stop at strategy.stop
      --- Take profit - GTC sell limit at strategy.target
"""
from __future__ import annotations

import argparse
import uuid
from datetime import date, datetime
from utils.timeutil import utcnow
from typing import Optional

from sqlmodel import select

from config.settings import settings
from db.connection import get_session, init_db
from db.models import (
    EvaluationLedger,
    OHLCV,
    Order,
    Position,
    RiskApproval,
    RunLog,
    Strategy,
    Ticker,
)
from utils.strategy_registry import execution_block_reason

_NODE = "execution_node"


class AmbiguousOrderState(RuntimeError):
    """A broker position may be open and unprotected; automatic retry is unsafe."""


class ConfirmedOrderFailure(RuntimeError):
    """The broker confirms the entry is terminal and has zero filled quantity."""


def _is_crypto(symbol: str) -> bool:
    from config.universe import CRYPTO_SET
    return symbol.upper() in CRYPTO_SET


# -- Public entry point --------------------------------------------------------

def run(
    risk_run_id: Optional[str]   = None,
    tickers:     list[str] | None = None,
) -> dict:
    """
    Submit bracket orders for all approved strategies.

    Parameters
    ----------
    risk_run_id : str | None
        run_id from risk_node. Defaults to most recent risk run.
    tickers : list[str] | None
        Limit to these symbols.

    Returns
    -------
    dict  with keys: submitted, dry_run, failed, run_id
    """
    exec_run_id = str(uuid.uuid4())
    started     = utcnow()
    dry_run     = not settings.EXECUTION_ENABLED

    print(f"\n{'='*55}")
    print(f"  EXECUTION NODE   run_id={exec_run_id[:8]}")
    print(f"  Mode: PAPER  |  "
          f"Execution: {'ENABLED' if not dry_run else 'DRY RUN (EXECUTION_ENABLED=false)'}")
    print(f"{'='*55}")

    if dry_run:
        print("  *** DRY RUN - orders will be logged but NOT submitted ***")

    init_db()

    # Broker clients are created lazily only after the selected strategy has
    # passed the evidence registry. A rejected strategy therefore cannot even
    # reach client construction, let alone order submission.
    alpaca_broker     = None
    robinhood_broker  = None

    resolved_run_id = risk_run_id or _latest_risk_run_id()
    if not resolved_run_id:
        print("  No risk approval run found. Run risk_node first.")
        return {"run_id": exec_run_id, "submitted": [], "dry_run": [], "failed": []}

    approvals = _load_approved(resolved_run_id, tickers)
    print(f"\n  Approved trades to execute: {len(approvals)}")

    if not approvals:
        _write_log(exec_run_id, [], [], [], 0, (utcnow() - started).total_seconds())
        return {"run_id": exec_run_id, "submitted": [], "dry_run": [], "failed": []}

    submitted_list, dry_run_list, failed_list = [], [], []
    today = date.today()

    for i, (approval, strat, symbol) in enumerate(approvals, 1):
        print(f"\n  [{i}/{len(approvals)}] {symbol}  "
              f"{approval.shares or 0:g} units  "
              f"entry~${strat.entry or 0:.2f}  "
              f"stop=${strat.stop or 0:.2f}  "
              f"target=${strat.target or 0:.2f}  "
              f"R:R={strat.rr or 0:.2f}")

        # Idempotency: never submit a second live order for an approval that
        # already has one (guards against manual re-runs, retries, crash
        # recovery against the same risk_run_id).
        is_crypto = _is_crypto(symbol)
        order_dry_run = dry_run or (
            is_crypto and not settings.CRYPTO_EXECUTION_ENABLED
        )

        if not order_dry_run:
            deployment_reason = execution_block_reason(strat.model_used)
            if deployment_reason:
                failed_list.append({
                    "symbol": symbol,
                    "reason": deployment_reason,
                })
                print(f"    ERR Blocked    {deployment_reason}")
                continue

        if not order_dry_run and _already_executed(approval.id):
            print(f"    -- skipped: already executed for this approval (idempotency guard)")
            continue

        is_robinhood = (not is_crypto) and settings.ROBINHOOD_ENABLED
        if not order_dry_run:
            if is_robinhood and robinhood_broker is None:
                robinhood_broker = _get_robinhood_client()
            elif not is_robinhood and alpaca_broker is None:
                alpaca_broker = _get_alpaca_client()
        if is_crypto:
            broker = alpaca_broker
        elif is_robinhood:
            broker = robinhood_broker
        else:
            broker = alpaca_broker
        result = _execute_one(
            symbol=symbol,
            approval=approval,
            strat=strat,
            run_id=exec_run_id,
            today=today,
            broker=broker,
            dry_run=order_dry_run,
            crypto=is_crypto,
            robinhood=is_robinhood,
        )

        if result["status"] == "submitted":
            submitted_list.append(symbol)
            broker_name = (
                "Alpaca crypto paper"
                if is_crypto else
                "Robinhood"
                if is_robinhood else
                "Alpaca stock paper"
            )
            order_id    = result.get("order_id", "")
            print(f"    OK Submitted [{broker_name}]  id={order_id}  "
                  f"status={result.get('order_status', '')}")
        elif result["status"] == "dry_run":
            dry_run_list.append(symbol)
            print(f"    ~ Dry run    (would submit {approval.shares} units)")
        else:
            failed_list.append({"symbol": symbol, "reason": result["reason"]})
            print(f"    ERR Failed     {result['reason']}")

    duration = (utcnow() - started).total_seconds()
    _write_log(exec_run_id, submitted_list, dry_run_list, failed_list, len(approvals), duration)

    mode = "dry-run" if dry_run else "live"
    print(f"\n  Done in {duration:.1f}s [{mode}] - "
          f"submitted={len(submitted_list)}  "
          f"dry_run={len(dry_run_list)}  "
          f"failed={len(failed_list)}")

    return {
        "run_id":    exec_run_id,
        "submitted": submitted_list,
        "dry_run":   dry_run_list,
        "failed":    failed_list,
    }


# -- Broker clients ------------------------------------------------------------

def _get_alpaca_client():
    from alpaca.trading.client import TradingClient
    return TradingClient(
        api_key=settings.ALPACA_API_KEY,
        secret_key=settings.ALPACA_SECRET_KEY,
        paper=True,   # HARDCODED - never live
    )


def _get_robinhood_client():
    from utils.robinhood_account import login
    return login()   # returns the robin_stocks.robinhood module


# -- Order submission ----------------------------------------------------------

def _validate_bracket(entry: float, stop: float, target: float) -> None:
    """Raise ValueError unless the three prices form a valid long bracket."""
    if not (entry > 0 and stop > 0 and target > 0):
        raise ValueError(f"non-positive bracket price entry={entry} stop={stop} target={target}")
    if not (stop < entry < target):
        raise ValueError(
            f"invalid long bracket: require stop < entry < target, got "
            f"stop={stop} entry={entry} target={target}"
        )


def _submit_alpaca_bracket(
    broker,
    symbol:          str,
    shares:          int,
    entry:           float,
    stop:            float,
    target:          float,
    client_order_id: str,
) -> dict:
    """
    Submit a GTC limit bracket order so the stop-loss and take-profit legs
    persist across multiple trading days (swing trades).

    The entry limit is anchored to the risk-sized entry price (strat.entry) that
    the position size was calculated against. A limit buy can only fill at the
    limit or BETTER (lower), never higher, so the realized risk-per-share can
    never exceed (entry - stop) -- the value the risk node sized for. If price
    has gapped above entry the order simply does not fill (we do not chase),
    which is the safe outcome for a real-money account.

    A deterministic client_order_id makes the submission idempotent at the
    broker: a retry with the same id will not create a second order.
    """
    from alpaca.trading.requests import LimitOrderRequest, StopLossRequest, TakeProfitRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    limit_price = round(float(entry), 2)
    stop_r      = round(float(stop), 2)
    target_r    = round(float(target), 2)

    _validate_bracket(limit_price, stop_r, target_r)

    order = broker.submit_order(LimitOrderRequest(
        symbol=symbol,
        qty=shares,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        limit_price=limit_price,
        order_class=OrderClass.BRACKET,
        stop_loss=StopLossRequest(stop_price=stop_r),
        take_profit=TakeProfitRequest(limit_price=target_r),
        client_order_id=client_order_id,
    ))

    status     = str(getattr(order, "status", "")).lower()
    fill_price = None
    try:
        if getattr(order, "filled_avg_price", None):
            fill_price = float(order.filled_avg_price)
    except (TypeError, ValueError):
        fill_price = None
    filled = status == "filled" or (fill_price is not None and fill_price > 0)

    return {
        "order_id":     str(order.id),
        "order_status": str(order.status),
        "filled":       filled,
        "fill_price":   fill_price,
    }


def _alpaca_crypto_symbol(symbol: str) -> str:
    """Translate the bot's BTC-USD spelling to Alpaca's BTC/USD spelling."""
    upper = symbol.upper()
    return upper[:-4] + "/USD" if upper.endswith("-USD") else upper


def _decimal_down(value: float, increment: float) -> str:
    """Round a positive quantity/price down to an exchange increment."""
    from decimal import Decimal, ROUND_DOWN

    number = Decimal(str(value))
    step = Decimal(str(increment))
    if number <= 0 or step <= 0:
        raise ValueError(f"invalid value/increment: {value}/{increment}")
    units = (number / step).to_integral_value(rounding=ROUND_DOWN)
    rounded = units * step
    return format(rounded, "f")


def _crypto_symbol_key(symbol: str) -> str:
    """Normalize SOL-USD, SOL/USD, and Alpaca's legacy SOLUSD identically."""
    return str(symbol or "").upper().replace("/", "").replace("-", "")


def _find_alpaca_crypto_position(client, broker_symbol: str):
    """Return Alpaca's live crypto Position regardless of pair spelling."""
    target = _crypto_symbol_key(broker_symbol)
    return next(
        (
            position
            for position in client.get_all_positions()
            if _crypto_symbol_key(getattr(position, "symbol", "")) == target
        ),
        None,
    )


def _alpaca_emergency_flatten(client, broker_symbol: str) -> bool:
    """Flatten a crypto position by asset ID and verify the account is flat."""
    import time

    try:
        position = _find_alpaca_crypto_position(client, broker_symbol)
    except Exception as exc:
        print(
            f"    [Alpaca crypto] *** EMERGENCY FLATTEN FAILED for "
            f"{broker_symbol}: could not read the live position ({exc}) ***"
        )
        return False

    if not position:
        print(
            f"    [Alpaca crypto] *** EMERGENCY FLATTEN FAILED for "
            f"{broker_symbol}: live position not found ***"
        )
        return False

    # The positions endpoint currently returns legacy symbols such as SOLUSD,
    # while the order endpoint uses SOL/USD. Closing by asset UUID avoids that
    # spelling mismatch entirely.
    position_ref = str(
        getattr(position, "asset_id", None)
        or getattr(position, "symbol", broker_symbol)
    )
    try:
        close_order = client.close_position(position_ref)
    except Exception as exc:
        # A response can be lost after Alpaca accepted the close. Re-read the
        # live position before declaring the outcome unsafe.
        try:
            if _find_alpaca_crypto_position(client, broker_symbol) is None:
                return True
        except Exception:
            pass
        print(
            f"    [Alpaca crypto] *** EMERGENCY FLATTEN FAILED for "
            f"{broker_symbol}: {exc} ***"
        )
        return False

    close_order_id = str(getattr(close_order, "id", "") or "")
    for _ in range(20):
        try:
            if _find_alpaca_crypto_position(client, broker_symbol) is None:
                return True
        except Exception:
            pass

        if close_order_id:
            try:
                latest = client.get_order_by_id(close_order_id)
                status = str(getattr(latest, "status", "")).lower().split(".")[-1]
                if status in {"canceled", "rejected", "expired"}:
                    break
            except Exception:
                pass
        time.sleep(0.25)

    print(
        f"    [Alpaca crypto] *** EMERGENCY FLATTEN FAILED for "
        f"{broker_symbol}: position still present after close attempt ***"
    )
    return False


def _submit_alpaca_crypto_protected(
    broker,
    symbol:          str,
    qty:             float,
    entry:           float,
    stop:            float,
    target:          float,
    client_order_id: str,
) -> dict:
    """
    Submit a fractional Alpaca paper crypto entry and mandatory stop-limit.

    Alpaca crypto supports market/limit/stop-limit orders, but not bracket/OCO
    classes. The downside stop is therefore placed immediately after confirming
    the market fill. The strategy target remains in the DB and monitor_node
    closes the position when it is reached. If protection cannot be confirmed,
    the new paper position is flattened and the execution is reported as failed.
    """
    import time
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.trading.requests import MarketOrderRequest, StopLimitOrderRequest

    _validate_bracket(entry, stop, target)
    broker_symbol = _alpaca_crypto_symbol(symbol)

    asset = broker.get_asset(broker_symbol)
    if not bool(getattr(asset, "tradable", False)):
        raise ValueError(f"{broker_symbol} is not tradable on Alpaca paper")

    # Defense in depth: never add to a crypto position even if an upstream
    # symbol-normalization or DB guard regresses.
    existing_position = _find_alpaca_crypto_position(broker, broker_symbol)
    if existing_position and float(getattr(existing_position, "qty", 0) or 0) > 0:
        raise ValueError(f"already holding {broker_symbol} at Alpaca paper")

    qty_increment = float(getattr(asset, "min_trade_increment", 0) or 1e-9)
    min_qty = float(getattr(asset, "min_order_size", 0) or 0)
    order_qty = _decimal_down(qty, qty_increment)
    if float(order_qty) < min_qty:
        raise ValueError(
            f"{broker_symbol} quantity {order_qty} is below minimum {min_qty}"
        )

    request = MarketOrderRequest(
        symbol=broker_symbol,
        qty=order_qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        client_order_id=client_order_id,
    )
    try:
        buy = broker.submit_order(request)
    except Exception:
        # Resolve a dropped response without risking a duplicate entry.
        buy = broker.get_order_by_client_id(client_order_id)
        if not buy:
            raise

    buy_order_id = str(buy.id)
    filled_qty = None
    fill_price = None
    fully_filled = False
    for _ in range(20):
        try:
            latest = broker.get_order_by_id(buy_order_id)
        except Exception:
            time.sleep(0.5)
            continue
        raw_filled = getattr(latest, "filled_qty", None)
        if raw_filled and float(raw_filled) > 0:
            filled_qty = str(raw_filled)
            raw_price = getattr(latest, "filled_avg_price", None)
            fill_price = float(raw_price) if raw_price else None
            status = str(getattr(latest, "status", "")).lower()
            if status.split(".")[-1] == "filled":
                fully_filled = True
                break
        status = str(getattr(latest, "status", "")).lower()
        if any(s in status for s in ("canceled", "rejected", "expired")):
            break
        time.sleep(0.5)

    # A partially filled entry must not remain open while the filled portion is
    # protected; cancel any remainder before sizing the stop to filled_qty.
    if filled_qty and not fully_filled:
        try:
            broker.cancel_order_by_id(buy_order_id)
        except Exception:
            pass

    if not filled_qty:
        try:
            broker.cancel_order_by_id(buy_order_id)
        except Exception:
            pass

        # A market entry can remain NEW longer than the initial confirmation
        # window on a thin crypto book. After requesting cancellation, resolve
        # the final broker state. A terminal order with zero fill is a known
        # safe failure—not an ambiguous, possibly-open position.
        terminal_no_fill = None
        for _ in range(20):
            try:
                latest = broker.get_order_by_id(buy_order_id)
            except Exception:
                time.sleep(0.25)
                continue

            raw_filled = getattr(latest, "filled_qty", None)
            if raw_filled and float(raw_filled) > 0:
                filled_qty = str(raw_filled)
                raw_price = getattr(latest, "filled_avg_price", None)
                fill_price = float(raw_price) if raw_price else None

            status = str(getattr(latest, "status", "")).lower().split(".")[-1]
            if status == "filled":
                fully_filled = True
                break
            if status in {"canceled", "cancelled", "rejected", "expired"}:
                if not filled_qty:
                    terminal_no_fill = status
                break
            time.sleep(0.25)

        if terminal_no_fill:
            raise ConfirmedOrderFailure(
                f"Alpaca crypto entry for {symbol} was {terminal_no_fill} "
                "with zero fill"
            )

    if not filled_qty:
        flattened = _alpaca_emergency_flatten(broker, broker_symbol)
        message = (
            f"Alpaca crypto fill for {symbol} could not be confirmed; "
            f"{'flattened for safety' if flattened else 'MANUAL PAPER-ACCOUNT CHECK REQUIRED'}"
        )
        if not flattened:
            raise AmbiguousOrderState(message)
        raise RuntimeError(message)

    # Alpaca charges a crypto BUY fee in the asset received. Consequently the
    # order's gross filled_qty is larger than the balance available to a SELL
    # stop (e.g. 295.172570199 SOL filled but only 294.434638773 SOL credited).
    # Wait for the live position and protect the actual post-fee balance.
    position = None
    position_lookup_error = None
    for _ in range(20):
        try:
            position = _find_alpaca_crypto_position(broker, broker_symbol)
            if position:
                break
        except Exception as exc:
            position_lookup_error = exc
        time.sleep(0.25)

    if not position:
        flattened = _alpaca_emergency_flatten(broker, broker_symbol)
        detail = f": {position_lookup_error}" if position_lookup_error else ""
        message = (
            f"post-fee crypto balance for {symbol} could not be confirmed{detail}; "
            f"{'flattened for safety' if flattened else 'MANUAL PAPER-ACCOUNT CHECK REQUIRED'}"
        )
        if not flattened:
            raise AmbiguousOrderState(message)
        raise RuntimeError(message)

    available_qty = float(getattr(position, "qty", 0) or 0)
    protected_qty = _decimal_down(
        min(float(filled_qty), available_qty),
        qty_increment,
    )
    if float(protected_qty) < min_qty:
        flattened = _alpaca_emergency_flatten(broker, broker_symbol)
        message = (
            f"post-fee balance {protected_qty} for {symbol} is below minimum {min_qty}; "
            f"{'flattened for safety' if flattened else 'MANUAL PAPER-ACCOUNT CHECK REQUIRED'}"
        )
        if not flattened:
            raise AmbiguousOrderState(message)
        raise RuntimeError(message)

    if float(protected_qty) < float(filled_qty):
        print(
            f"    [Alpaca crypto] fee-adjusted protection quantity: "
            f"gross={filled_qty}  available={protected_qty}"
        )

    # Crypto stop orders must be stop-limit. Put the limit 2% below the trigger
    # so ordinary slippage can fill while still bounding a simulated crash.
    price_increment = float(getattr(asset, "price_increment", 0) or 1e-9)
    stop_price = _decimal_down(stop, price_increment)
    limit_price = _decimal_down(stop * 0.98, price_increment)
    stop_client_id = f"{client_order_id}-sl"
    try:
        protective = broker.submit_order(StopLimitOrderRequest(
            symbol=broker_symbol,
            qty=protected_qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=stop_price,
            limit_price=limit_price,
            client_order_id=stop_client_id,
        ))
    except Exception as exc:
        # As with the entry, first resolve an ambiguous response. Only flatten
        # if the protective order truly cannot be found.
        try:
            protective = broker.get_order_by_client_id(stop_client_id)
        except Exception:
            protective = None
        if not protective:
            flattened = _alpaca_emergency_flatten(broker, broker_symbol)
            message = (
                f"protective stop failed for {symbol} ({exc}); "
                f"{'flattened for safety' if flattened else 'MANUAL PAPER-ACCOUNT CHECK REQUIRED'}"
            )
            if not flattened:
                raise AmbiguousOrderState(message)
            raise RuntimeError(message)

    return {
        "order_id":     buy_order_id,
        "order_status": f"filled_protected:{getattr(protective, 'status', 'accepted')}",
        "filled":       True,
        "fill_price":   fill_price,
        "position_qty": float(protected_qty),
        "position_id":  str(getattr(position, "asset_id", "") or buy_order_id),
        "gross_filled_qty": float(filled_qty),
        "estimated_fee": max(0.0, float(filled_qty) - float(protected_qty))
        * float(fill_price or entry),
    }


def _robinhood_emergency_flatten(r, symbol: str, shares) -> bool:
    """
    Best-effort market sell to flatten a just-opened Robinhood position when
    its protective stop-loss could not be placed. Returns True if the sell
    order was submitted.
    """
    if not shares:
        return False
    try:
        r.orders.order_sell_market(symbol, shares)
        return True
    except Exception as exc:
        print(f"    [Robinhood] *** EMERGENCY FLATTEN FAILED for {symbol}: {exc} ***")
        return False


def _submit_robinhood_bracket(
    r,
    symbol:          str,
    shares:          int,
    stop:            float,
    target:          float,
) -> dict:
    """
    Place a market buy on Robinhood, then attach a REQUIRED stop-loss and a
    best-effort take-profit.

    robin_stocks has no atomic bracket/OCO order, so this uses three steps:
      - The exact filled quantity/price is read back from the broker (not
        guessed); if the fill cannot be confirmed the position is flattened
        and an exception is raised.
      - The stop-loss is mandatory: if it cannot be placed, the position is
        immediately flattened with a market sell and an exception is raised.
      - The take-profit is best-effort (a missing TP only forgoes upside and
        adds no risk), so its failure is logged but not fatal.
    """
    import time

    _validate_bracket((stop + target) / 2.0, stop, target)

    buy = r.orders.order_buy_market(symbol, shares)
    order_id = (buy or {}).get("id")
    if not order_id:
        raise RuntimeError(f"Robinhood buy order for {symbol} did not return an order id: {buy}")

    # Confirm the fill and read back the EXACT filled quantity/price. Protective
    # orders are sized off the real fill, never an estimate.
    filled_qty, fill_price = None, None
    for _ in range(20):                       # up to ~20s for the fill to settle
        time.sleep(1)
        try:
            info = r.orders.get_stock_order_info(order_id)
            if info and (info.get("state") == "filled"):
                filled_qty = float(info.get("cumulative_quantity") or shares)
                avg_price  = info.get("average_price")
                fill_price = float(avg_price) if avg_price else None
                break
        except Exception:
            pass

    if not filled_qty:
        flattened = _robinhood_emergency_flatten(r, symbol, shares)
        message = (
            f"Robinhood fill for {symbol} could not be confirmed within timeout; "
            f"{'flattened for safety' if flattened else 'COULD NOT BE FLATTENED - MANUAL INTERVENTION REQUIRED'}"
        )
        if not flattened:
            raise AmbiguousOrderState(message)
        raise RuntimeError(message)

    # Stop-loss is MANDATORY. On failure, flatten and raise so the caller
    # records an error and does NOT persist a (falsely) protected Position.
    try:
        r.orders.order_sell_stop_loss(symbol, filled_qty, round(stop, 2))
    except Exception as exc:
        flattened = _robinhood_emergency_flatten(r, symbol, filled_qty)
        message = (
            f"stop-loss placement failed for {symbol} ({exc}); position "
            f"{'flattened for safety' if flattened else 'COULD NOT BE FLATTENED - MANUAL INTERVENTION REQUIRED'}"
        )
        if not flattened:
            raise AmbiguousOrderState(message)
        raise RuntimeError(message)

    # Take-profit (best-effort - missing TP forgoes upside, adds no risk)
    tp_ok = True
    try:
        r.orders.order_sell_limit(symbol, filled_qty, round(target, 2))
    except Exception as exc:
        tp_ok = False
        print(f"    [Robinhood] take-profit placement failed (non-fatal): {exc}")

    return {
        "order_id":     order_id,
        "order_status": "submitted" if tp_ok else "submitted_no_tp",
        "filled":       True,
        "fill_price":   fill_price,
    }


def _try_reconcile_by_client_order_id(
    broker,
    client_order_id: str,
    crypto:          bool,
    robinhood:       bool,
) -> Optional[dict]:
    """
    After an exception from _submit_alpaca_bracket, try to resolve whether the
    order actually reached Alpaca (e.g. the connection dropped after Alpaca
    accepted it) before assuming the submission failed outright. Crypto and
    Robinhood already confirm-or-flatten internally before raising, so this
    only applies to the Alpaca stock-bracket path.
    """
    if crypto or robinhood or broker is None:
        return None
    try:
        order = broker.get_order_by_client_id(client_order_id)
        if not order:
            return None
        status = str(getattr(order, "status", "")).lower()
        fill_price = None
        try:
            if getattr(order, "filled_avg_price", None):
                fill_price = float(order.filled_avg_price)
        except (TypeError, ValueError):
            fill_price = None
        filled = status == "filled" or (fill_price is not None and fill_price > 0)
        return {
            "order_id":     str(order.id),
            "order_status": str(order.status),
            "filled":       filled,
            "fill_price":   fill_price,
        }
    except Exception:
        return None


def _looks_ambiguous(exc: Exception) -> bool:
    """
    Classify a submission exception as ambiguous (broker-side outcome
    unknown - must not be silently retried) vs a confirmed pre-submission
    rejection (safe to retry). Defaults to the SAFER assumption - ambiguous -
    unless the message clearly indicates the order never reached the book.
    """
    msg = str(exc).lower()
    _SAFE_TO_RETRY_MARKERS = (
        "422", "invalid", "insufficient", "not tradable", "not found",
        "unprocessable", "forbidden", "403", "401",
    )
    if any(m in msg for m in _SAFE_TO_RETRY_MARKERS):
        return False
    return True


def _execute_one(
    symbol:    str,
    approval:  RiskApproval,
    strat:     Strategy,
    run_id:    str,
    today:     date,
    broker,
    dry_run:   bool,
    crypto:    bool = False,
    robinhood: bool = False,
) -> dict:
    if dry_run:
        with get_session() as session:
            session.add(Order(
                run_id=run_id,
                risk_approval_id=approval.id,
                symbol=symbol,
                side="buy",
                qty=approval.shares or 0,
                order_type="market",
                status=(
                    "dry_run (Alpaca crypto paper)"
                    if crypto else
                    f"dry_run ({'Robinhood' if robinhood else 'Alpaca stock paper'})"
                ),
                stop_price=strat.stop,
                target_price=strat.target,
                dry_run=True,
            ))
        return {"status": "dry_run"}

    client_order_id = f"appr-{approval.id}"
    order_type      = "limit" if (not crypto and not robinhood) else "market"

    # Live submission
    try:
        if crypto:
            resp = _submit_alpaca_crypto_protected(
                broker=broker,
                symbol=symbol,
                qty=approval.shares or 0,
                entry=strat.entry,
                stop=strat.stop,
                target=strat.target,
                client_order_id=client_order_id,
            )
        elif robinhood:
            resp = _submit_robinhood_bracket(
                r=broker,
                symbol=symbol,
                shares=approval.shares,
                stop=strat.stop,
                target=strat.target,
            )
        else:
            resp = _submit_alpaca_bracket(
                broker=broker,
                symbol=symbol,
                shares=approval.shares,
                entry=strat.entry,
                stop=strat.stop,
                target=strat.target,
                client_order_id=client_order_id,
            )
    except Exception as exc:
        # The exception may have happened AFTER the broker already accepted
        # the order (e.g. a dropped connection) - try to resolve the true
        # state via the deterministic client_order_id before assuming failure.
        resp = _try_reconcile_by_client_order_id(broker, client_order_id, crypto, robinhood)
        if resp is None:
            error_msg = str(exc)
            if isinstance(exc, AmbiguousOrderState):
                status = "error_ambiguous"
            elif isinstance(exc, ConfirmedOrderFailure):
                status = "error"
            else:
                status = "error_ambiguous" if _looks_ambiguous(exc) else "error"
            with get_session() as session:
                order = Order(
                    run_id=run_id,
                    risk_approval_id=approval.id,
                    symbol=symbol,
                    side="buy",
                    qty=approval.shares or 0,
                    order_type=order_type,
                    status=status,
                    stop_price=strat.stop,
                    target_price=strat.target,
                    dry_run=False,
                    error_message=error_msg,
                )
                session.add(order)
                session.flush()
                raw_version = getattr(strat, "model_used", None)
                session.add(EvaluationLedger(
                    order_id=order.id,
                    strategy_id=str(getattr(strat, "id", "unknown")),
                    strategy_version=(raw_version if isinstance(raw_version, str) else "unknown"),
                    market="crypto" if crypto else "stock",
                    symbol=symbol,
                    decision_time_utc=utcnow(),
                    signal_bar_end_utc=_latest_signal_bar_time(strat.ticker_id, today),
                    expected_entry=float(strat.entry or 0.0),
                    expected_stop=float(strat.stop or 0.0),
                    expected_target=float(strat.target or 0.0),
                    expected_qty=float(approval.shares or 0.0),
                    reconciliation_status=status,
                    discrepancy=(error_msg if status == "error_ambiguous" else None),
                    fee_source=("estimated_tier1_taker" if crypto else "stock_commission_zero"),
                ))
                # Re-adding an already-persistent object is a SQLAlchemy no-op;
                # keeping Order as the final audit object also preserves the
                # execution node's long-standing test/instrumentation contract.
                session.add(order)
            return {"status": status, "reason": error_msg}
        print(f"    [Recover] {symbol}: submission raised {exc!r}, but the order "
              f"was found at the broker via client_order_id - treating as submitted")

    # Alpaca stock entry is a GTC limit that may not have filled yet. Crypto and
    # Robinhood market buys confirm fill synchronously in their submit
    # functions above, so resp["filled"] is always True for those. Record the
    # real fill price when known, and mark the Position 'pending' until the
    # entry actually fills (the monitor promotes pending -> open on fill). This
    # stops a not-yet-filled order from being mistaken for a live position and
    # from being "reconciled" as a phantom stop-out.
    filled      = bool(resp.get("filled"))
    fill_price  = resp.get("fill_price")
    entry_price = fill_price if (fill_price and fill_price > 0) else (strat.entry or 0.0)
    pos_status  = "open" if filled else "pending"
    position_qty = resp.get("position_qty") or approval.shares or 0
    position_id = resp.get("position_id") or resp["order_id"]

    signal_bar_end = _latest_signal_bar_time(strat.ticker_id, today)

    # Write Order + Position to DB
    with get_session() as session:
        order = Order(
            run_id=run_id,
            risk_approval_id=approval.id,
            alpaca_order_id=resp["order_id"],
            symbol=symbol,
            side="buy",
            qty=approval.shares or 0,
            order_type=order_type,
            status=resp["order_status"],
            filled_price=float(fill_price) if fill_price else None,
            stop_price=strat.stop,
            target_price=strat.target,
            dry_run=False,
            filled_at=utcnow() if filled else None,
        )
        session.add(order)
        session.flush()

        session.add(Position(
            ticker_id=strat.ticker_id,
            order_id=order.id,
            entry_date=today,
            entry_price=entry_price,
            shares=position_qty,
            stop_price=strat.stop,
            target_price=strat.target,
            alpaca_position_id=position_id,
            status=pos_status,
        ))

        raw_decision_time = getattr(strat, "created_at", None)
        decision_time = raw_decision_time if isinstance(raw_decision_time, datetime) else utcnow()
        raw_version = getattr(strat, "model_used", None)
        strategy_version = raw_version if isinstance(raw_version, str) else "unknown"
        slippage_bps = None
        if fill_price and strat.entry:
            slippage_bps = (float(fill_price) / float(strat.entry) - 1.0) * 10_000.0
        estimated_fee = float(resp.get("estimated_fee") or 0.0)
        session.add(EvaluationLedger(
            order_id=order.id,
            strategy_id=str(getattr(strat, "id", "unknown")),
            strategy_version=strategy_version,
            market="crypto" if crypto else "stock",
            symbol=symbol,
            decision_time_utc=decision_time,
            signal_bar_end_utc=signal_bar_end,
            expected_entry=float(strat.entry or 0.0),
            expected_stop=float(strat.stop or 0.0),
            expected_target=float(strat.target or 0.0),
            expected_qty=float(approval.shares or 0.0),
            broker_order_id=resp["order_id"],
            broker_order_status=str(resp["order_status"]),
            reconciliation_status="filled" if filled else "submitted",
            filled_at_utc=utcnow() if filled else None,
            actual_fill_price=float(fill_price) if fill_price else None,
            entry_slippage_bps=slippage_bps,
            fee_amount=estimated_fee,
            fee_source=("estimated_from_fee_adjusted_quantity" if crypto else "stock_commission_zero"),
        ))

    return {
        "status":       "submitted",
        "order_id":     resp["order_id"],
        "order_status": resp["order_status"],
        "filled":       filled,
    }


def _latest_signal_bar_time(ticker_id: str, on_date: date) -> Optional[datetime]:
    """Best-effort audit metadata; failure must never affect order handling."""
    if not isinstance(on_date, date):
        return None
    try:
        with get_session() as session:
            row = session.exec(
                select(OHLCV)
                .where(OHLCV.ticker_id == ticker_id)
                .where(OHLCV.bar_time <= datetime.combine(on_date, datetime.max.time()))
                .order_by(OHLCV.bar_time.desc())  # type: ignore[attr-defined]
            ).first()
            value = getattr(row, "bar_time", None) if row else None
            return value if isinstance(value, datetime) else None
    except Exception:
        return None


# -- DB helpers ----------------------------------------------------------------

def _already_executed(approval_id: str) -> bool:
    """
    True if a live (non-dry-run, non-error) Order already exists for this approval.

    Deliberately does NOT exclude "error_ambiguous" - an ambiguous submission
    (broker-side outcome unknown, see _try_reconcile_by_client_order_id) must
    keep blocking a retry until a human resolves it, unlike a confirmed "error".
    """
    with get_session() as session:
        row = session.exec(
            select(Order)
            .where(Order.risk_approval_id == approval_id)
            .where(Order.dry_run == False)        # noqa: E712
            .where(Order.status != "error")
        ).first()
        return row is not None


def _latest_risk_run_id() -> Optional[str]:
    with get_session() as session:
        row = session.exec(
            select(RiskApproval).order_by(RiskApproval.created_at.desc())  # type: ignore[attr-defined]
        ).first()
        return row.run_id if row else None


def _load_approved(
    risk_run_id: str,
    tickers:     Optional[list[str]],
) -> list[tuple[RiskApproval, Strategy, str]]:
    with get_session() as session:
        approvals = session.exec(
            select(RiskApproval)
            .where(RiskApproval.run_id  == risk_run_id)
            .where(RiskApproval.approved == True)       # noqa: E712
        ).all()

        results = []
        for appr in approvals:
            strat = session.exec(
                select(Strategy).where(Strategy.id == appr.strategy_id)
            ).first()
            if not strat:
                continue

            ticker = session.exec(
                select(Ticker).where(Ticker.id == strat.ticker_id)
            ).first()
            symbol = ticker.symbol if ticker else "UNKNOWN"

            if tickers and symbol not in {t.upper() for t in tickers}:
                continue

            results.append((appr, strat, symbol))

    return results


def _write_log(run_id, submitted, dry_run_list, failed, total, duration) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success",
            tickers_processed=total,
            records_written=len(submitted) + len(dry_run_list),
            error_message=(
                "; ".join(f"{f['symbol']}: {f['reason']}" for f in failed)
                if failed else None
            ),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execution Node - submit bracket orders")
    parser.add_argument("--run-id",  type=str,  help="risk run_id (default: latest)")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(risk_run_id=args.run_id, tickers=args.tickers)
