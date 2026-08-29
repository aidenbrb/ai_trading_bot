"""
Risk Node - Phase 5 of the pipeline.

PURE PYTHON - no Claude, no broker writes. The risk node is the final
safety gate before any order reaches the execution node.

Every rule here is intentionally hardcoded. No AI can override them.
If a strategy doesn't pass every check, it is rejected with a clear reason.

Responsibilities:
  1. Read live account state (equity, open positions, trades today)
  2. Apply the daily loss limit check - if triggered, reject ALL strategies
  3. For each BUY strategy from strategy_node, run the full rule stack:
       OK R:R >= MIN_RISK_REWARD
       OK conviction_score >= MIN_CONVICTION_SCORE
       OK entry price > MIN_PRICE
       OK open positions < MAX_OPEN_POSITIONS
       OK trades today < MAX_NEW_TRADES_PER_DAY
       OK entry > stop (valid stop placement)
       OK position concentration <= 20% of equity
  4. Compute position size (shares) based on fixed-fractional risk
  5. Write one RiskApproval row per strategy (approved + rejection_reason)
  6. Print a clean summary table
  7. Log to run_log

Run standalone:
    python -m nodes.risk_node --run-id <strategies_run_id>
    python -m nodes.risk_node   (uses most recent strategy run)

Position sizing formula:
    risk_amount    = equity * MAX_RISK_PER_TRADE     (e.g. 1% of $100k = $1,000)
    shares         = floor(risk_amount / (entry - stop))
    position_value = shares * entry
    rejected if position_value > 20% of equity
"""
from __future__ import annotations

import argparse
import math
import uuid
from datetime import date, datetime
from utils.timeutil import utcnow
from typing import Optional

from sqlmodel import select

from config.settings import settings
from config.universe import CRYPTO_SET
from db.connection import get_session, init_db
from db.models import RiskApproval, RunLog, Strategy, TradingAgentsSignal
from utils.account import get_account_state
from utils.market_intelligence import news_gate_status
from utils.robinhood_account import get_robinhood_account_state
from utils.strategy_registry import execution_block_reason

_NODE             = "risk_node"
MAX_POSITION_PCT  = 0.20  # single position cannot exceed 20% of equity
EARNINGS_BUFFER   = 3     # block trades within this many days of earnings
_TA_BEARISH       = {"Sell", "Underweight"}


# -- Public entry point --------------------------------------------------------

def run(
    strategies_run_id: Optional[str] = None,
    tickers:           list[str] | None = None,
) -> dict:
    """
    Evaluate all BUY strategies and issue risk approvals.

    Parameters
    ----------
    strategies_run_id : str | None
        run_id from strategy_node. If None, uses the most recent strategy run.
    tickers : list[str] | None
        Limit to these symbols.

    Returns
    -------
    dict  with keys: approved, rejected, run_id
    """
    risk_run_id = str(uuid.uuid4())
    started     = utcnow()

    print(f"\n{'='*55}")
    print(f"  RISK NODE   run_id={risk_run_id[:8]}")
    print(f"{'='*55}")

    init_db()

    # -- 1. Live account state -------------------------------------------------
    print("  Reading account state ...")
    acct = get_robinhood_account_state() if settings.ROBINHOOD_ENABLED else get_account_state()
    equity         = acct["equity"]
    # Robinhood's dict has no available_cash/buying_power (Alpaca-only
    # concepts) - preserve its existing behavior unchanged. Alpaca sizes
    # from the conservative, fail-closed available_cash, never raw cash.
    available_cash = acct["cash"] if settings.ROBINHOOD_ENABLED else acct["available_cash"]
    open_positions = acct["open_positions"]
    trades_today   = acct["trades_today"]
    daily_pnl_pct  = acct["daily_pnl_pct"]

    # Duplicate-position guard set: union of Alpaca's live holdings AND the bot's
    # own in-flight intent recorded in the DB (open positions + submitted buy
    # orders whose entry limit may not have filled yet). Relying on Alpaca alone
    # missed positions during the pending-fill window and when the broker call
    # fell back to safe defaults, which allowed a second entry into a symbol the
    # bot was already entering.
    held_symbols = {p["symbol"].upper() for p in acct.get("positions", [])}
    inflight_check_failed = False
    try:
        held_symbols |= _db_inflight_symbols()
    except Exception as exc:
        inflight_check_failed = True
        print(
            "\n  *** DUPLICATE-POSITION CHECK FAILED - rejecting ALL "
            f"strategies this run: {exc} ***"
        )

    _print_account(acct)

    # -- 2. Daily loss limit + broker connectivity + news gate checks ---------
    daily_loss_breached = daily_pnl_pct <= -settings.DAILY_LOSS_LIMIT
    if daily_loss_breached:
        pct = daily_pnl_pct * 100
        print(f"\n  *** DAILY LOSS LIMIT BREACHED ({pct:.2f}%) - "
              f"rejecting ALL strategies ***")

    broker_disconnected = (
        settings.RISK_REQUIRE_BROKER_CONNECTED and not acct.get("connected", False)
    )
    if broker_disconnected:
        print("\n  *** BROKER DISCONNECTED - rejecting ALL new-trade strategies "
              "this run (RISK_REQUIRE_BROKER_CONNECTED=true) ***")

    _, news_gate_blocked, news_gate_reason = news_gate_status()
    if news_gate_blocked:
        print(f"\n  *** NEWS GATE: {news_gate_reason} - rejecting ALL strategies ***")

    # -- 3. Load BUY strategies ------------------------------------------------
    resolved_run_id = strategies_run_id or _latest_strategy_run_id()
    if not resolved_run_id:
        print("  No strategy run found. Run strategy_node first.")
        return {"run_id": risk_run_id, "approved": [], "rejected": []}

    strategies = _load_buy_strategies(resolved_run_id, tickers)
    print(f"\n  BUY strategies to evaluate: {len(strategies)}")

    if not strategies:
        _write_log(risk_run_id, [], [], 0, (utcnow() - started).total_seconds())
        return {"run_id": risk_run_id, "approved": [], "rejected": []}

    # -- 4. Evaluate each strategy ---------------------------------------------
    approved_list, rejected_list = [], []
    approved_count  = 0   # track within this run to enforce per-run limits
    current_positions = open_positions

    print(f"\n  {'Symbol':<10}  {'R:R':>5}  {'Conv':>5}  {'Qty':>12}  "
          f"{'Value':>9}  {'Decision'}")
    print(f"  {'-'*10}  {'-'*5}  {'-'*5}  {'-'*12}  {'-'*9}  {'-'*20}")

    for strat in strategies:
        symbol = _symbol_for(strat.ticker_id)

        if _already_evaluated(strat.id, risk_run_id):
            continue

        verdict = _evaluate(
            strat=strat,
            equity=equity,
            open_positions=current_positions,
            trades_today=trades_today + approved_count,
            daily_loss_breached=daily_loss_breached,
            symbol=symbol,
            ta_signal=_get_ta_signal(strat.ticker_id, strat.bar_date, symbol=symbol),
            held_symbols=held_symbols,
            cash=available_cash,
            broker_disconnected=broker_disconnected,
            inflight_check_failed=inflight_check_failed,
            news_gate_blocked=news_gate_blocked,
            news_gate_reason=news_gate_reason,
            crypto_tradable=(
                _alpaca_crypto_is_tradable(symbol)
                if symbol.upper() in CRYPTO_SET else None
            ),
        )

        # The operational rules above remain useful for research sizing, but a
        # setup that would otherwise pass must still be rejected when its
        # version failed the locked evidence comparison. Unknown/unversioned
        # strategies fail closed as well.
        if verdict["approved"]:
            deployment_reason = execution_block_reason(strat.model_used)
            if deployment_reason:
                verdict = {"approved": False, "reason": deployment_reason}

        approved   = verdict["approved"]
        shares     = verdict.get("shares", 0)
        risk_amt   = verdict.get("risk_amount", 0.0)
        pos_value  = verdict.get("position_value", 0.0)
        reason     = verdict.get("reason", "")

        decision = "APPROVED" if approved else f"REJECTED: {reason}"
        print(f"  {symbol:<10}  {strat.rr or 0:>5.2f}  "
              f"{strat.conviction_score or 0:>5}  "
              f"{shares:>12.8g}  ${pos_value:>8,.0f}  {decision}")

        with get_session() as session:
            session.add(RiskApproval(
                run_id=risk_run_id,
                strategy_id=strat.id,
                approved=approved,
                rejection_reason=reason if not approved else None,
                shares=shares if approved else None,
                risk_amount=risk_amt if approved else None,
                position_value=pos_value if approved else None,
            ))

        if approved:
            approved_list.append(symbol)
            approved_count  += 1
            current_positions += 1
            available_cash = max(0.0, available_cash - pos_value)
            # Block a second approval for the same symbol within this run
            held_symbols.add(symbol.upper())
        else:
            rejected_list.append({"symbol": symbol, "reason": reason})

    duration = (utcnow() - started).total_seconds()
    _write_log(risk_run_id, approved_list, rejected_list, len(strategies), duration)

    print(f"\n  Risk gate: {len(approved_list)} approved, "
          f"{len(rejected_list)} rejected  ({duration:.1f}s)")

    return {
        "run_id":   risk_run_id,
        "approved": approved_list,
        "rejected": rejected_list,
    }


# -- Safety helpers ------------------------------------------------------------

def _days_to_earnings(symbol: str) -> Optional[int]:
    """Return days until next earnings, or None if the date is unknown."""
    try:
        import yfinance as yf
        from datetime import date as _date

        cal = yf.Ticker(symbol).calendar
        if not cal:
            return None

        # yfinance >=1.1 returns a dict: {'Earnings Date': [Timestamp, ...], ...}
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if not isinstance(dates, list):
                dates = [dates]
        else:
            # Older yfinance: DataFrame indexed by field name
            try:
                dates = cal.loc["Earnings Date"].dropna().tolist()
            except (KeyError, AttributeError):
                return None

        today = _date.today()
        for d in dates:
            try:
                if hasattr(d, "date"):
                    d = d.date()
                if d >= today:
                    return (d - today).days
            except Exception:
                continue
        return None
    except Exception as exc:
        print(f"  [Risk] earnings-date lookup failed for {symbol}: {exc} - "
              f"proceeding WITHOUT earnings-date protection")
        return None


def _get_ta_signal(
    ticker_id: str,
    bar_date: "date",
    symbol: Optional[str] = None,
) -> Optional[str]:
    """Return the most recent TradingAgents signal for this ticker on bar_date."""
    try:
        with get_session() as session:
            row = session.exec(
                select(TradingAgentsSignal)
                .where(TradingAgentsSignal.ticker_id == ticker_id)
                .where(TradingAgentsSignal.bar_date  == bar_date)
                .order_by(TradingAgentsSignal.created_at.desc())  # type: ignore[attr-defined]
            ).first()
            return row.signal if row else None
    except Exception as exc:
        print(
            f"  [Risk] TradingAgents signal lookup failed for "
            f"{symbol or ticker_id}: {exc} - proceeding WITHOUT "
            "TradingAgents veto protection"
        )
        return None


# -- Evaluation logic ----------------------------------------------------------

def _evaluate(
    strat:               "Strategy",
    equity:              float,
    open_positions:      int,
    trades_today:        int,
    daily_loss_breached: bool,
    symbol:              str = "",
    ta_signal:           Optional[str] = None,
    held_symbols:        Optional[set] = None,
    cash:                float = float("inf"),
    broker_disconnected: bool = False,
    inflight_check_failed: bool = False,
    news_gate_blocked:   bool = False,
    news_gate_reason:    str = "",
    crypto_tradable:     Optional[bool] = None,
) -> dict:
    """Run every rule in order. Return at first failure."""

    # Daily loss gate
    if daily_loss_breached:
        return _reject("daily loss limit exceeded")

    # Broker connectivity gate - never approve a new trade against an
    # unverifiable account/position state (see RISK_REQUIRE_BROKER_CONNECTED)
    if broker_disconnected:
        return _reject("broker disconnected - cannot verify live account/positions")

    # DB-backed open/pending positions and live entry orders are part of the
    # duplicate-position check. If that state cannot be read completely, no
    # new entry is safe to approve this run.
    if inflight_check_failed:
        return _reject(
            "duplicate-position check failed - rejecting all strategies this run"
        )

    # Morning news gate - refuse ALL new trades if today's report is missing
    # or stale (defense-in-depth; the strategy nodes already enforce this too)
    if news_gate_blocked:
        return _reject(f"news gate: {news_gate_reason}")

    # Duplicate position gate - never add to an already-held symbol
    if held_symbols and symbol.upper() in held_symbols:
        return _reject(f"already holding {symbol}")

    is_crypto = symbol.upper() in CRYPTO_SET
    if is_crypto and crypto_tradable is False:
        return _reject(f"{symbol} is not currently tradable on Alpaca paper")

    # Earnings proximity gate - never hold a position through an announcement
    if symbol and not is_crypto:
        dte = _days_to_earnings(symbol)
        if dte is not None and dte <= EARNINGS_BUFFER:
            return _reject(f"earnings in {dte}d (buffer={EARNINGS_BUFFER}d)")

    # TradingAgents veto - reject if the multi-agent team is bearish
    if ta_signal and ta_signal in _TA_BEARISH:
        return _reject(f"TradingAgents signal is {ta_signal}")

    entry  = strat.entry  or 0.0
    stop   = strat.stop   or 0.0
    target = strat.target or 0.0
    rr     = strat.rr     or 0.0
    conv   = strat.conviction_score or 0

    # Basic validity
    if entry <= 0 or stop <= 0 or target <= 0:
        return _reject("entry, stop, or target price is missing")
    if entry <= stop:
        return _reject("entry must be above stop loss")
    if target <= entry:
        return _reject("target must be above entry (invalid long bracket)")

    # R:R gate
    if rr < settings.MIN_RISK_REWARD:
        return _reject(f"R:R {rr:.2f} < minimum {settings.MIN_RISK_REWARD}")

    # Conviction gate
    if conv < settings.MIN_CONVICTION_SCORE:
        return _reject(f"conviction {conv} < minimum {settings.MIN_CONVICTION_SCORE}")

    # Price floor
    if not is_crypto and entry < settings.MIN_PRICE:
        return _reject(f"price ${entry:.2f} below minimum ${settings.MIN_PRICE:.2f}")

    # Position count gate
    if open_positions >= settings.MAX_OPEN_POSITIONS:
        return _reject(
            f"already at max positions ({open_positions}/{settings.MAX_OPEN_POSITIONS})"
        )

    # Daily trade count gate
    if trades_today >= settings.MAX_NEW_TRADES_PER_DAY:
        return _reject(
            f"daily trade limit reached ({trades_today}/{settings.MAX_NEW_TRADES_PER_DAY})"
        )

    # Position sizing (fixed-fractional risk)
    risk_budget    = equity * settings.MAX_RISK_PER_TRADE
    risk_per_share = entry - stop
    raw_qty = risk_budget / risk_per_share
    shares = (
        math.floor(raw_qty * 1_000_000_000) / 1_000_000_000
        if is_crypto else
        int(math.floor(raw_qty))
    )

    if shares <= 0 or (not is_crypto and shares < 1):
        return _reject(
            f"position too small: risk ${risk_budget:.0f} / "
            f"${risk_per_share:.8f}/unit = {shares} units"
        )

    position_value = shares * entry
    max_position   = equity * MAX_POSITION_PCT

    # Concentration cap: a single position cannot exceed 20% of equity
    if position_value > max_position:
        cap_qty = max_position / entry
        shares = (
            math.floor(cap_qty * 1_000_000_000) / 1_000_000_000
            if is_crypto else
            int(math.floor(cap_qty))
        )
        position_value = shares * entry
        if shares <= 0 or (not is_crypto and shares < 1):
            return _reject("position value exceeds 20% concentration limit")

    # Buying-power gate: never size a position larger than available cash.
    # Without this the broker may reject the order, or (on a margin account)
    # the bot could take on unintended leverage. `cash` here is the caller's
    # available_cash (Alpaca: fail-closed, margin-free) or, under
    # Robinhood, its own raw cash - never Alpaca's raw ledger cash directly.
    if position_value > cash:
        cash_qty = cash / entry if entry > 0 else 0
        shares = (
            math.floor(cash_qty * 1_000_000_000) / 1_000_000_000
            if is_crypto else
            int(math.floor(cash_qty))
        )
        position_value = shares * entry
        if shares <= 0 or (not is_crypto and shares < 1):
            return _reject(
                f"insufficient buying power: ${cash:,.2f}"
            )

    if is_crypto and position_value < 1.0:
        return _reject("crypto position value is below Alpaca's $1 minimum")

    # Report the ACTUAL dollar risk for the final (possibly scaled-down) share
    # count, not the pre-scaling risk budget.
    risk_amount = shares * risk_per_share

    return {
        "approved":       True,
        "shares":         shares,
        "risk_amount":    risk_amount,
        "position_value": position_value,
    }


def _reject(reason: str) -> dict:
    return {"approved": False, "reason": reason, "shares": 0,
            "risk_amount": 0.0, "position_value": 0.0}


def _alpaca_crypto_is_tradable(symbol: str) -> bool:
    """Fail closed unless Alpaca confirms this crypto pair is paper-tradable."""
    if not settings.ALPACA_API_KEY or not settings.ALPACA_SECRET_KEY:
        return False
    broker_symbol = symbol.upper().replace("-USD", "/USD")
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            settings.ALPACA_API_KEY,
            settings.ALPACA_SECRET_KEY,
            paper=True,
        )
        asset = client.get_asset(broker_symbol)
        return bool(getattr(asset, "tradable", False))
    except Exception as exc:
        print(f"  [Risk] Alpaca asset check failed for {symbol}: {exc}")
        return False


# -- DB helpers ----------------------------------------------------------------

def _db_inflight_symbols() -> set:
    """
    Symbols the bot has already committed to according to the DB: any open or
    pending Position, plus any live (non-dry-run) BUY order that has been
    submitted but may not have filled yet. Used to augment the broker's live
    holdings so the duplicate-position gate also covers the pending-fill window.
    """
    inflight: set = set()
    from db.models import Order, Position
    # Deliberately does NOT include "error_ambiguous" - an ambiguous
    # submission (broker-side outcome unknown, see execution_node's
    # _try_reconcile_by_client_order_id) must keep blocking a fresh entry
    # into the same symbol until a human resolves it.
    _DONE = {
        "filled", "canceled", "cancelled", "rejected", "expired", "error",
        "resolved_flat",
    }
    with get_session() as session:
        # An Order row keeps the broker's submission-time status, while the
        # linked Position/Trade lifecycle is reconciled separately.  Therefore
        # an old ``PENDING_NEW`` string must not keep blocking a symbol after
        # its Position has closed.  Load every position so the linked lifecycle
        # is authoritative; standalone orders still fail closed below.
        positions = session.exec(select(Position)).all()
        positions_by_order = {p.order_id: p for p in positions if p.order_id}
        for p in positions:
            if p.status in {"open", "pending"}:
                inflight.add(_symbol_for(p.ticker_id).upper())
        for o in session.exec(
            select(Order)
            .where(Order.side == "buy")
            .where(Order.dry_run == False)  # noqa: E712
        ).all():
            linked_position = positions_by_order.get(o.id)
            if linked_position is not None:
                if linked_position.status == "closed":
                    continue
                # Open/pending positions were already added above.  Unknown
                # lifecycle values remain blocking as the conservative choice.
                if linked_position.status in {"open", "pending"}:
                    continue
            if (o.status or "").lower() not in _DONE:
                inflight.add((o.symbol or "").upper())
    return inflight


def _latest_strategy_run_id() -> Optional[str]:
    with get_session() as session:
        row = session.exec(
            select(Strategy).order_by(Strategy.created_at.desc())  # type: ignore[attr-defined]
        ).first()
        return row.run_id if row else None


def _load_buy_strategies(run_id: str, tickers: Optional[list[str]]) -> list[Strategy]:
    with get_session() as session:
        rows = session.exec(
            select(Strategy)
            .where(Strategy.run_id == run_id)
            .where(Strategy.signal == "BUY")
            .order_by(Strategy.conviction_score.desc())  # type: ignore[attr-defined]
        ).all()

    if tickers:
        upper = {t.upper() for t in tickers}
        rows = [r for r in rows if _symbol_for(r.ticker_id) in upper]

    return list(rows)


def _already_evaluated(strategy_id: str, run_id: str) -> bool:
    with get_session() as session:
        row = session.exec(
            select(RiskApproval)
            .where(RiskApproval.strategy_id == strategy_id)
            .where(RiskApproval.run_id      == run_id)
        ).first()
        return row is not None


_ticker_cache: dict[str, str] = {}

def _symbol_for(ticker_id: str) -> str:
    if ticker_id not in _ticker_cache:
        from db.models import Ticker
        with get_session() as session:
            t = session.exec(select(Ticker).where(Ticker.id == ticker_id)).first()
            _ticker_cache[ticker_id] = t.symbol if t else "UNKNOWN"
    return _ticker_cache[ticker_id]


# -- Logging -------------------------------------------------------------------

def _print_account(acct: dict) -> None:
    connected = "paper" if acct["connected"] else "offline (defaults)"
    print(f"  Equity:         ${acct['equity']:>12,.2f}  [{connected}]")
    print(f"  Cash:           ${acct['cash']:>12,.2f}")
    if "available_cash" in acct:  # Alpaca only - absent for Robinhood
        if acct.get("available_cash_valid", True):
            print(f"  Available cash: ${acct['available_cash']:>12,.2f}  (sizing budget)")
        else:
            reason = acct.get("available_cash_reason") or "unknown"
            print(f"  Available cash: $0.00  *** CAPITAL DATA INVALID - sizing disabled ({reason}) ***")
        print(f"  Buying power:   ${acct.get('buying_power', 0.0):>12,.2f}  (margin, audit-only)")
    print(f"  Open positions: {acct['open_positions']}")
    print(f"  Trades today:   {acct['trades_today']}")
    daily_pnl_pct = acct['daily_pnl_pct'] * 100
    sign = "+" if daily_pnl_pct >= 0 else ""
    print(f"  Daily P&L:      {sign}{daily_pnl_pct:.2f}%  "
          f"(limit: -{settings.DAILY_LOSS_LIMIT * 100:.1f}%)")


def _write_log(run_id, approved, rejected, total, duration) -> None:
    with get_session() as session:
        session.add(RunLog(
            run_id=run_id,
            node_name=_NODE,
            status="success",
            tickers_processed=total,
            records_written=len(approved),
            error_message=(
                "; ".join(f"{r['symbol']}: {r['reason']}" for r in rejected)
                if rejected else None
            ),
            duration_seconds=duration,
            finished_at=utcnow(),
        ))


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk Node - safety gate")
    parser.add_argument("--run-id",  type=str,  help="strategy run_id (default: latest)")
    parser.add_argument("--tickers", nargs="+", help="limit to these symbols")
    args = parser.parse_args()
    run(strategies_run_id=args.run_id, tickers=args.tickers)
