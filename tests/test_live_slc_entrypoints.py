"""
Entrypoint tests (rev. 11 Step 14): python -m live_slc.run_slc_live
--stage {preflight,cycle,closeout}, end to end, everything below the
broker boundary mocked. Covers not_authorized preflight, a dry_run cycle,
a fully-mocked paper_active cycle that actually submits, and proves
closeout stays reachable when entry-only gates would fail.
"""
import sys
from datetime import timedelta
from decimal import Decimal

import pandas as pd
import pytest

import live_slc.authorization as authorization
import live_slc.guardrails as guardrails
import live_slc.models as models
import live_slc.run_slc_live as run_slc_live
from live_slc import reauth_signature
from live_slc.models import SlcOrder, SlcPosition, SlcReducerState, SlcSessionStat, get_live_slc_session
from live_slc.reducer import Confirmation
from sqlmodel import select

from tests._slc_reauth_helpers import make_test_key, signed_reauth_kwargs


def _seed_session_stat(session_date, *, engine_parity_check_passed=True):
    with get_live_slc_session() as session:
        session.add(SlcSessionStat(
            session_date=session_date, engine_parity_check_passed=engine_parity_check_passed,
        ))


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def _fake_operational(status: str):
    status_record = type("SR", (), {"status": status, "observed_account_id": None, "live_baseline_sha256": None,
                                      "activation_proposal_sha256": None})()
    return {"status": status, "baseline": {"baseline_sha256": "x"}, "status_record": status_record}


MUTATING = {"submit_order", "cancel_order_by_id", "replace_order_by_id", "cancel_orders", "close_position"}


class _AccountOnlyClient:
    """A client that answers get_account() and nothing else - any other
    call is a test failure, proving preflight/dry_run never go further."""
    def __init__(self):
        self.calls = []

    def get_account(self):
        self.calls.append("get_account")
        return type("A", (), {
            "id": "acct-1", "equity": 100000.0, "cash": 100000.0,
            "non_marginable_buying_power": 100000.0, "last_equity": 100000.0,
        })()

    def __getattr__(self, name):
        def _unexpected(*a, **k):
            self.calls.append(name)
            raise AssertionError(f"unexpected client call in this scenario: {name}")
        return _unexpected


# -- real Alpaca order-enum/account-snapshot boundary -----------------------

def test_account_snapshot_ignores_terminal_exit_legs_without_requesting_a_quote():
    """Regression from the first paper-active session.

    QueryOrderStatus.ALL returned the canceled stop child of the closed
    NVDA bracket.  It had neither a limit nor a fill price, and the old
    snapshot builder tried to obtain an after-hours fresh quote for it,
    making the entire submission gate unusable.  Real alpaca-py enums are
    used here so a second bare-str(enum) bug cannot hide behind mocks.
    """
    from alpaca.trading.enums import OrderClass, OrderStatus, OrderType, PositionIntent

    def _order(*, oid, cid, status, order_type, intent, qty=98, filled_qty=0,
               filled_avg_price=None, limit_price=None):
        return type("O", (), {
            "id": oid, "client_order_id": cid, "symbol": "NVDA",
            "status": status, "type": order_type, "order_class": OrderClass.BRACKET,
            "position_intent": intent, "qty": qty, "filled_qty": filled_qty,
            "filled_avg_price": filled_avg_price, "limit_price": limit_price,
            "parent_id": None, "parent_order_id": None, "legs": None,
        })()

    orders = [
        _order(
            oid="entry", cid="slc-signal-entry", status=OrderStatus.FILLED,
            order_type=OrderType.MARKET, intent=PositionIntent.BUY_TO_OPEN,
            filled_qty=98, filled_avg_price=221.399184,
        ),
        _order(
            oid="target", cid="broker-target", status=OrderStatus.CANCELED,
            order_type=OrderType.LIMIT, intent=PositionIntent.SELL_TO_CLOSE,
            limit_price=223.48,
        ),
        _order(
            oid="stop", cid="broker-stop", status=OrderStatus.CANCELED,
            order_type=OrderType.STOP, intent=PositionIntent.SELL_TO_CLOSE,
        ),
        _order(
            oid="exit", cid="slc-signal-exit", status=OrderStatus.FILLED,
            order_type=OrderType.MARKET, intent=PositionIntent.SELL_TO_CLOSE,
            filled_qty=98, filled_avg_price=221.26,
        ),
        # A definitively canceled, zero-fill entry is not a trade and is
        # not pending exposure; it must not consume a daily entry slot.
        _order(
            oid="never-filled", cid="slc-never-filled-entry", status=OrderStatus.CANCELED,
            order_type=OrderType.MARKET, intent=PositionIntent.BUY_TO_OPEN,
        ),
    ]

    class _Client:
        def get_account(self):
            return type("A", (), {
                "id": "acct-1", "equity": 100000.0, "cash": 100000.0,
                "non_marginable_buying_power": 100000.0, "last_equity": 100000.0,
            })()

        def get_all_positions(self):
            return []

        def get_orders(self, request):
            return orders

    snapshot = run_slc_live.build_account_snapshot(
        _Client(), quote_fn=lambda symbol: (_ for _ in ()).throw(
            AssertionError(f"terminal order unexpectedly requested quote for {symbol}")
        ),
    )

    assert run_slc_live.risk.account_wide_entries_today(snapshot) == 1
    assert run_slc_live.risk.pending_opening_notional(snapshot) == 0.0
    assert [o.classification for o in snapshot.today_orders].count("entry") == 1
    assert [o.classification for o in snapshot.today_orders].count("exit_leg") == 3


def test_account_snapshot_requires_quote_for_a_genuinely_pending_market_entry():
    from alpaca.trading.enums import OrderClass, OrderStatus, OrderType, PositionIntent

    pending = type("O", (), {
        "client_order_id": "slc-pending-entry", "symbol": "AAPL",
        "status": OrderStatus.NEW, "type": OrderType.MARKET,
        "order_class": OrderClass.BRACKET, "position_intent": PositionIntent.BUY_TO_OPEN,
        "qty": 10, "filled_qty": 0, "filled_avg_price": None, "limit_price": None,
        "parent_id": None, "parent_order_id": None, "legs": None,
    })()

    class _Client:
        def get_account(self):
            return type("A", (), {
                "id": "acct-1", "equity": 100000.0, "cash": 100000.0,
                "non_marginable_buying_power": 100000.0, "last_equity": 100000.0,
            })()

        def get_all_positions(self):
            return []

        def get_orders(self, request):
            return [pending]

    snapshot = run_slc_live.build_account_snapshot(_Client(), quote_fn=lambda symbol: 200.0)
    assert run_slc_live.risk.account_wide_entries_today(snapshot) == 1
    assert run_slc_live.risk.pending_opening_notional(snapshot) == 2000.0


# -- main() CLI dispatch ------------------------------------------------------

def test_main_dispatches_to_run_preflight_and_inits_db_inside_the_lock(monkeypatch):
    order = []
    monkeypatch.setattr(sys, "argv", ["run_slc_live.py", "--stage", "preflight"])

    class _FakeLock:
        def __enter__(self):
            order.append("lock_acquired")
            return self
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(run_slc_live, "acquire_process_lock", lambda: _FakeLock())
    monkeypatch.setattr(run_slc_live, "init_live_slc_db", lambda: order.append("db_init"))
    monkeypatch.setattr(run_slc_live, "run_preflight", lambda: order.append("preflight") or {"status": "ok"})
    monkeypatch.setattr(run_slc_live, "run_cycle", lambda: (_ for _ in ()).throw(AssertionError("wrong stage")))
    monkeypatch.setattr(run_slc_live, "run_closeout_stage", lambda: (_ for _ in ()).throw(AssertionError("wrong stage")))

    run_slc_live.main()
    # Lock acquired BEFORE db_init (rev. 11 concurrency fix), and preflight
    # dispatched correctly for --stage preflight.
    assert order == ["lock_acquired", "db_init", "preflight"]


def test_main_dispatches_cycle_and_closeout(monkeypatch):
    class _FakeLock:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(run_slc_live, "acquire_process_lock", lambda: _FakeLock())
    monkeypatch.setattr(run_slc_live, "init_live_slc_db", lambda: None)

    monkeypatch.setattr(sys, "argv", ["run_slc_live.py", "--stage", "cycle"])
    called = {}
    monkeypatch.setattr(run_slc_live, "run_cycle", lambda: called.setdefault("cycle", True) or {"status": "ok"})
    run_slc_live.main()
    assert called == {"cycle": True}

    monkeypatch.setattr(sys, "argv", ["run_slc_live.py", "--stage", "closeout"])
    called2 = {}
    monkeypatch.setattr(run_slc_live, "run_closeout_stage", lambda: called2.setdefault("closeout", True) or {"status": "ok"})
    run_slc_live.main()
    assert called2 == {"closeout": True}


def test_main_handles_lock_already_held_without_crashing(monkeypatch, capsys):
    from live_slc.process_lock import LockAlreadyHeld

    class _RefusingLock:
        def __enter__(self):
            raise LockAlreadyHeld("already running")
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(run_slc_live, "acquire_process_lock", lambda: _RefusingLock())
    monkeypatch.setattr(sys, "argv", ["run_slc_live.py", "--stage", "cycle"])

    with pytest.raises(SystemExit) as exc_info:
        run_slc_live.main()
    assert exc_info.value.code == 0


# -- not_authorized preflight -------------------------------------------------

def test_not_authorized_preflight_end_to_end_makes_zero_mutating_calls(monkeypatch):
    client = _AccountOnlyClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", lambda symbols, start, end: {})
    monkeypatch.setattr(guardrails, "assert_operational_preconditions", lambda **kw: _fake_operational("not_authorized"))

    result = run_slc_live.run_preflight()
    assert result["status"] == "not_authorized"
    assert client.calls == ["get_account"]


def test_preflight_passes_engine_parity_with_142_bootstrapped_states_and_empty_bar_table(monkeypatch):
    """Regression for the exact real-run failure: overnight preflight
    completes, all 142 SlcReducerState rows are bootstrap_completed=True,
    SlcFiveMinBar has zero rows (bootstrap() never persists the
    historical bars it fetched), and the parity gate must still pass -
    via the frozen validation corpus, not the (correctly empty) live bar
    cache. No symbol gets re-bootstrapped."""
    import live_slc.split_detection as split_detection
    from config.universe import UNIVERSE
    from live_slc.models import SlcFiveMinBar

    with get_live_slc_session() as session:
        for symbol in UNIVERSE:
            session.add(SlcReducerState(symbol=symbol, bootstrap_completed=True))

    client = _AccountOnlyClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)
    monkeypatch.setattr(guardrails, "assert_operational_preconditions", lambda **kw: _fake_operational("dry_run"))
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", lambda symbols, start, end: {})
    monkeypatch.setattr(split_detection, "corporate_action_split_evidence", lambda *a, **k: {})

    result = run_slc_live.run_preflight()

    assert result["status"] == "dry_run"
    assert result["bootstrapped"] == 0
    assert result["engine_parity_check_passed"] is True
    with get_live_slc_session() as session:
        assert session.exec(select(SlcFiveMinBar)).first() is None
    assert client.calls == ["get_account"]


# -- dry_run cycle end to end --------------------------------------------------

def _seed_paper_active_or_dry_run(target_status: str, tmp_path=None):
    authorization.record_transition("not_authorized", "dry_run", "start")
    if target_status == "paper_active":
        import tempfile
        from pathlib import Path
        key_dir = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp())
        key_path, allowed_signers_path, identity = make_test_key(key_dir)
        reauth_signature.ALLOWED_SIGNERS_PATH = allowed_signers_path
        sig_kwargs = signed_reauth_kwargs(
            key_path=key_path, identity=identity, from_status="dry_run", to_status="paper_active",
            guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
            observed_account_id="acct-1",
        )
        authorization.record_transition(
            "dry_run", "paper_active", "activate",
            guardrail_baseline_sha256="a" * 64, activation_proposal_sha256="b" * 64,
            observed_account_id="acct-1", **sig_kwargs,
        )


def _confirmation(symbol="AAPL", direction="long"):
    ts = pd.Timestamp("2026-08-13 14:00:00")
    return Confirmation(
        "slc_4h_5m_stock_v1", symbol, "demand:x", direction, "fresh",
        95.0, 99.0, ts - pd.Timedelta(minutes=20), ts, ts + pd.Timedelta(minutes=5),
        98.0, 15.0, 12.0, 1.0, "uptrend" if direction == "long" else "downtrend", 1.5,
    )


def _wire_one_confirmation(monkeypatch, confirmation, *, status: str):
    """Mocks the bar-fetch/reducer boundary run_cycle() sits on top of, so
    the test drives the REAL admission/ranking/order-recording pipeline
    against a single, controlled Confirmation - proving the orchestration
    wiring end to end without needing hand-crafted OHLCV data that
    actually triggers the frozen engine. assert_operational_preconditions
    is mocked here (not the guardrail internals, which have their own
    dedicated test file) since it depends on the REAL, currently-stale
    deployment baseline hash - regenerated as part of Step 15's final
    verification, not something this orchestration test should re-prove."""
    monkeypatch.setattr(guardrails, "assert_operational_preconditions", lambda **kw: _fake_operational(status))
    monkeypatch.setattr(run_slc_live, "is_trading_day", lambda day: True)
    monkeypatch.setattr(run_slc_live, "confirmation_within_entry_cutoff", lambda c, day: True)
    monkeypatch.setattr(run_slc_live.bar_cache, "backfill_gaps", lambda symbols, through: {})
    fake_bar = pd.Series({"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 1000.0})
    monkeypatch.setattr(
        run_slc_live.bar_cache, "fetch_expected_bar_batch",
        lambda symbols, expected_bar_time, **kw: ({confirmation.symbol: fake_bar}, []),
    )
    monkeypatch.setattr(
        run_slc_live.reducer, "process_new_bar",
        lambda state, row, ts: (state, [confirmation] if row is fake_bar else []),
    )
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 100.0)


def test_dry_run_cycle_end_to_end_records_a_proposal_with_zero_mutating_calls(monkeypatch):
    _seed_paper_active_or_dry_run("dry_run")
    confirmation = _confirmation()
    _wire_one_confirmation(monkeypatch, confirmation, status="dry_run")

    class _DryRunClient(_AccountOnlyClient):
        def get_order_by_client_id(self, oc_id):
            from alpaca.common.exceptions import APIError
            resp = type("R", (), {"status_code": 404})()
            err = type("E", (), {"response": resp})()
            raise APIError("not found", http_error=err)
        def get_all_positions(self):
            return []
        def get_orders(self, request):
            return []

    client = _DryRunClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "dry_run"
    assert not (set(client.calls) & MUTATING)

    today = pd.Timestamp.now(tz="UTC").tz_localize(None).date()
    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
        signal = session.exec(select(models.SlcSignalRecord).where(models.SlcSignalRecord.direction != "")).first()
        stat = session.get(models.SlcSessionStat, today)
    assert order is not None
    assert order.status == "dry_run_proposed"
    assert order.qty > 0
    assert signal.acted_on is True
    assert signal.action_result == "dry_run_proposed"
    assert stat.signals_generated == 1
    assert stat.signals_acted_on == 1
    assert stat.dry_run_proposal_count == 1
    assert stat.duplicate_or_stale_signal_count == 0


def test_dry_run_daily_entry_limit_counts_prior_simulated_proposals(monkeypatch):
    """The first real dry-run session exposed this cross-process bug:
    Alpaca never sees simulated orders, so each new five-minute process
    incorrectly believed both daily slots were unused. Two prior DB
    proposals must exhaust the entire day's simulated entry allowance."""
    _seed_paper_active_or_dry_run("dry_run")
    confirmation = _confirmation(symbol="AAPL")
    _wire_one_confirmation(monkeypatch, confirmation, status="dry_run")

    with get_live_slc_session() as session:
        for index, symbol in enumerate(("MSFT", "NVDA"), start=1):
            session.add(SlcOrder(
                client_order_id=f"prior-dry-{index}", symbol=symbol, leg="entry",
                side="buy", position_intent="buy_to_open", qty=1.0,
                order_class="bracket", status="dry_run_proposed", dry_run=True,
            ))

    class _DryRunClient(_AccountOnlyClient):
        def get_order_by_client_id(self, oc_id):
            raise AssertionError("capacity must reject before reconciliation")
        def get_all_positions(self):
            return []
        def get_orders(self, request):
            return []

    client = _DryRunClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["confirmations"] == 1
    with get_live_slc_session() as session:
        orders = list(session.exec(select(SlcOrder)))
        signal = session.exec(
            select(models.SlcSignalRecord).where(models.SlcSignalRecord.symbol == "AAPL")
        ).first()
    assert len(orders) == 2
    assert signal.action_result == "skipped_capacity_daily_trades"


def test_missing_bar_observations_are_not_duplicate_signals():
    """Coverage misses belong in expected-vs-valid bar counts. They must
    not populate the duplicate-signal gate field."""
    today = pd.Timestamp.now(tz="UTC").tz_localize(None).date()
    run_slc_live._accumulate_session_stat(
        today, expected_symbols=5, valid_symbols=0, stale_or_missing=5,
        dry_run_proposals=0, signals_generated=0, signals_acted_on=0,
        signals_skipped_not_shortable=0,
        signals_skipped_stale_or_missing_data=0,
        signals_skipped_capacity=0, elapsed_seconds=1.0,
    )
    with get_live_slc_session() as session:
        stat = session.get(models.SlcSessionStat, today)
    assert stat.expected_symbol_count == 5
    assert stat.valid_symbol_count == 0
    assert stat.duplicate_or_stale_signal_count == 0
    assert stat.signals_skipped_stale_or_missing_data == 0


def test_dry_run_cycle_accumulates_committed_notional_across_multiple_proposals(monkeypatch):
    """The exact bug found via review: only "submitted" outcomes bumped
    committed_notional_this_cycle, never "dry_run_proposed" ones - so
    within one dry_run cycle, a second (or third) admitted confirmation
    was sized as if it were the ONLY trade using the account's buying
    power, reusing the same cash the first proposal already committed."""
    _seed_paper_active_or_dry_run("dry_run")
    confirmation_a = _confirmation(symbol="AAPL")
    confirmation_b = _confirmation(symbol="MSFT")

    monkeypatch.setattr(guardrails, "assert_operational_preconditions", lambda **kw: _fake_operational("dry_run"))
    monkeypatch.setattr(run_slc_live, "is_trading_day", lambda day: True)
    monkeypatch.setattr(run_slc_live, "confirmation_within_entry_cutoff", lambda c, day: True)
    monkeypatch.setattr(run_slc_live.bar_cache, "backfill_gaps", lambda symbols, through: {})
    fake_bar_a = pd.Series({"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 1000.0})
    fake_bar_b = pd.Series({"open": 98.0, "high": 99.0, "low": 97.5, "close": 98.5, "volume": 1000.0})
    monkeypatch.setattr(
        run_slc_live.bar_cache, "fetch_expected_bar_batch",
        lambda symbols, expected_bar_time, **kw: ({"AAPL": fake_bar_a, "MSFT": fake_bar_b}, []),
    )

    def _fake_process_new_bar(state, row, ts):
        if row is fake_bar_a:
            return state, [confirmation_a]
        if row is fake_bar_b:
            return state, [confirmation_b]
        return state, []
    monkeypatch.setattr(run_slc_live.reducer, "process_new_bar", _fake_process_new_bar)
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 100.0)

    # Cash reduced so available buying power (not the risk-per-share cap)
    # is the binding constraint on sizing - otherwise the fixed 0.25%
    # risk formula alone would size both proposals identically regardless
    # of whether notional correctly accumulates, and this test would
    # prove nothing.
    from live_slc.risk import AccountSnapshot

    def _reduced_cash_snapshot(client, **kw):
        return AccountSnapshot(
            account_id="acct-1", equity=100000.0, cash=14000.0, non_marginable_buying_power=14000.0,
            start_of_day_equity=100000.0, daily_realized_pnl=0.0, daily_unrealized_pnl=0.0,
            positions=[], today_orders=[],
        )
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", _reduced_cash_snapshot)

    class _DryRunClient(_AccountOnlyClient):
        def get_order_by_client_id(self, oc_id):
            from alpaca.common.exceptions import APIError
            resp = type("R", (), {"status_code": 404})()
            err = type("E", (), {"response": resp})()
            raise APIError("not found", http_error=err)
        def get_all_positions(self):
            return []
        def get_orders(self, request):
            return []

    client = _DryRunClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    run_slc_live.run_cycle()

    with get_live_slc_session() as session:
        orders = list(session.exec(select(SlcOrder).order_by(SlcOrder.submitted_at)))
    assert len(orders) == 2
    assert all(o.status == "dry_run_proposed" for o in orders)
    # The second proposal must be sized SMALLER than the first, since the
    # first one's notional should have already reduced the available
    # buying power for the second (found via review: without the fix,
    # both would be sized identically, each computed against the FULL
    # equity as if the other proposal never happened).
    assert orders[1].qty < orders[0].qty


# -- paper_active cycle end to end: a real (mocked) submission ---------------

class _PaperActiveClient:
    def __init__(self):
        self.mutating_calls = []
        self._submitted_qty = None
        self._target_price = None

    def get_account(self):
        return type("A", (), {
            "id": "acct-1", "equity": 100000.0, "cash": 100000.0,
            "non_marginable_buying_power": 100000.0, "last_equity": 100000.0,
        })()

    def get_all_positions(self):
        return []

    def get_orders(self, request):
        return []

    def get_order_by_client_id(self, oc_id):
        from alpaca.common.exceptions import APIError
        resp = type("R", (), {"status_code": 404})()
        err = type("E", (), {"response": resp})()
        raise APIError("not found", http_error=err)

    def submit_order(self, request):
        self.mutating_calls.append("submit_order")
        self._submitted_qty = request.qty
        self._target_price = request.take_profit.limit_price
        return type("O", (), {"id": "entry-order-1"})()

    def get_order_by_id(self, order_id, filter=None):
        # The stop leg is listed FIRST on purpose - proving
        # execution.take_profit_leg()'s type-based selection (not array
        # position) is what run_slc_live.py actually uses.
        stop_leg = type("Leg", (), {"id": "stop-1", "type": "stop", "status": "held"})()
        target_leg = type("Leg", (), {
            "id": "target-1", "type": "limit", "status": "new",
            "limit_price": self._target_price, "qty": self._submitted_qty,
        })()
        return type("O", (), {
            "id": order_id, "status": "filled",
            "filled_qty": self._submitted_qty, "filled_avg_price": 100.0,
            "legs": [stop_leg, target_leg],
        })()

    def replace_order_by_id(self, leg_id, request):
        self.mutating_calls.append("replace_order_by_id")
        assert leg_id == "target-1"  # never the stop leg
        self._target_price = float(request.limit_price)
        return type("O", (), {"id": "target-2"})()

    def cancel_order_by_id(self, *a, **k):
        self.mutating_calls.append("cancel_order_by_id")
        raise AssertionError("must not cancel on the happy path")

    def close_position(self, *a, **k):
        self.mutating_calls.append("close_position")
        raise AssertionError("must not close on the happy path")


def test_paper_active_cycle_end_to_end_submits_fills_and_opens_a_protected_position(monkeypatch):
    _seed_paper_active_or_dry_run("paper_active")
    _seed_session_stat(run_slc_live._utc_now_naive().date(), engine_parity_check_passed=True)
    confirmation = _confirmation(direction="long")
    _wire_one_confirmation(monkeypatch, confirmation, status="paper_active")
    monkeypatch.setattr(guardrails, "assert_submission_preconditions", lambda *a, **k: None)

    client = _PaperActiveClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "paper_active"
    assert client.mutating_calls == ["submit_order", "replace_order_by_id"]

    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
        position = session.exec(select(SlcPosition)).first()
        cycle_stat = session.get(models.SlcSessionStat, run_slc_live._utc_now_naive().date())
        trade_stat = session.get(models.SlcSessionStat, confirmation.confirmation_time.date())
    assert order.status == "filled"
    assert order.alpaca_order_id == "entry-order-1"
    assert position.status == "open"
    assert position.qty == client._submitted_qty
    assert cycle_stat.signals_acted_on == 1
    assert trade_stat.trades_opened == 1


# -- structural same-day preflight/parity gate (paper_active only) ----------

def test_paper_active_cycle_blocks_entries_with_no_same_day_session_stat(monkeypatch):
    """No SlcSessionStat row for today at all - a preflight that never
    ran (or crashed before writing one) must block entries, not silently
    allow them."""
    _seed_paper_active_or_dry_run("paper_active")
    confirmation = _confirmation(direction="long")
    _wire_one_confirmation(monkeypatch, confirmation, status="paper_active")
    monkeypatch.setattr(guardrails, "assert_submission_preconditions", lambda *a, **k: None)

    client = _PaperActiveClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "paper_active"
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        signal = session.exec(
            select(models.SlcSignalRecord).where(models.SlcSignalRecord.symbol == "AAPL")
        ).first()
    assert signal.action_result == "blocked_no_same_day_preflight_session_stat"


def test_paper_active_cycle_blocks_entries_when_same_day_parity_failed(monkeypatch):
    _seed_paper_active_or_dry_run("paper_active")
    _seed_session_stat(run_slc_live._utc_now_naive().date(), engine_parity_check_passed=False)
    confirmation = _confirmation(direction="long")
    _wire_one_confirmation(monkeypatch, confirmation, status="paper_active")
    monkeypatch.setattr(guardrails, "assert_submission_preconditions", lambda *a, **k: None)

    client = _PaperActiveClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "paper_active"
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        signal = session.exec(
            select(models.SlcSignalRecord).where(models.SlcSignalRecord.symbol == "AAPL")
        ).first()
    assert signal.action_result == "blocked_same_day_engine_parity_check_failed"


def test_paper_active_cycle_ignores_a_prior_days_passing_session_stat(monkeypatch):
    """A stale success from yesterday must never satisfy today's gate -
    proves the check is genuinely same-day, not 'most recent success'."""
    _seed_paper_active_or_dry_run("paper_active")
    yesterday = run_slc_live._utc_now_naive().date() - timedelta(days=1)
    _seed_session_stat(yesterday, engine_parity_check_passed=True)
    confirmation = _confirmation(direction="long")
    _wire_one_confirmation(monkeypatch, confirmation, status="paper_active")
    monkeypatch.setattr(guardrails, "assert_submission_preconditions", lambda *a, **k: None)

    client = _PaperActiveClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "paper_active"
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        signal = session.exec(
            select(models.SlcSignalRecord).where(models.SlcSignalRecord.symbol == "AAPL")
        ).first()
    assert signal.action_result == "blocked_no_same_day_preflight_session_stat"


def test_dry_run_cycle_is_unaffected_by_the_paper_active_only_gate(monkeypatch):
    """The gate is scoped to paper_active only - a dry_run cycle with no
    SlcSessionStat row at all must proceed exactly as before."""
    _seed_paper_active_or_dry_run("dry_run")
    confirmation = _confirmation(direction="long")
    _wire_one_confirmation(monkeypatch, confirmation, status="dry_run")

    class _DryRunClient(_AccountOnlyClient):
        def get_order_by_client_id(self, oc_id):
            from alpaca.common.exceptions import APIError
            resp = type("R", (), {"status_code": 404})()
            err = type("E", (), {"response": resp})()
            raise APIError("not found", http_error=err)
        def get_all_positions(self):
            return []
        def get_orders(self, request):
            return []

    client = _DryRunClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_cycle()
    assert result["status"] == "dry_run"

    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
    assert order is not None
    assert order.status == "dry_run_proposed"


# -- closeout stays reachable when entry-only preconditions would fail -------

def test_closeout_remains_reachable_when_operational_preconditions_would_fail(monkeypatch):
    """run_closeout_stage() must never call assert_operational_preconditions()
    at all (rev. 11 Step 7) - simulate it raising to prove closeout
    doesn't depend on it."""
    def _always_fails(**kw):
        raise RuntimeError("Tier-2 drift - blocks new entries")
    monkeypatch.setattr(guardrails, "assert_operational_preconditions", _always_fails)
    monkeypatch.setattr(guardrails, "assert_closeout_preconditions", lambda **kw: None)
    monkeypatch.setattr(run_slc_live, "is_trading_day", lambda day: True)
    monkeypatch.setattr(run_slc_live.closeout_mod, "should_begin_closeout", lambda now, day: True)

    class _CloseoutClient(_AccountOnlyClient):
        def get_open_position(self, symbol):
            return None

    client = _CloseoutClient()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_closeout_stage()
    assert result["status"] == "closed_out"  # never raised, despite the operational gate being broken


# -- exit_via_marketable_replacement is actually called (rev. 11 review point 6) --

def test_resolve_protected_degraded_positions_calls_marketable_replacement_and_closes(monkeypatch):
    """The exact gap found via review: execution.exit_via_marketable_replacement
    existed but had no caller anywhere - a protected_degraded position
    sat unresolved until end-of-day closeout."""
    from live_slc.reducer import Confirmation as _Confirmation

    confirmation = _confirmation()
    with get_live_slc_session() as session:
        session.add(models.SlcSignalRecord(
            cycle_run_id="run-1", symbol="AAPL", level_id="demand:x", direction="long",
            level_state="fresh", level_low=95.0, level_high=99.0,
            level_active_time=pd.Timestamp("2026-08-13 13:40:00").to_pydatetime(),
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            entry_time=pd.Timestamp("2026-08-13 14:05:00").to_pydatetime(),
            stop=94.0, stochastic_k=15.0, stochastic_d=12.0, atr14=1.0, structure="uptrend", impulse_atr=1.5,
        ))
    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol="AAPL", level_id="demand:x",
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-13").date(), status="protected_degraded",
            qty=10.0, entry_price=100.0, stop_price=98.0, target_price=104.0,
            target_order_id="target-old-1",
        ))

    calls = []

    class _Client:
        def get_latest_quote(self, symbol):
            raise AssertionError("not needed - exit_via_marketable_replacement is mocked directly")

    monkeypatch.setattr(guardrails, "assert_closeout_preconditions", lambda **kw: None)
    from live_slc.execution import MarketableExitResult

    def _fake_exit(client, symbol, target_order_id, direction, **kw):
        calls.append((symbol, target_order_id, direction))
        return MarketableExitResult(True, fill_price=103.5)
    monkeypatch.setattr(run_slc_live.execution, "exit_via_marketable_replacement", _fake_exit)

    result = run_slc_live._resolve_protected_degraded_positions(_Client(), observed_account_id="acct-1")
    assert calls == [("AAPL", "target-old-1", "long")]
    assert len(result["resolved"]) == 1

    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
        from live_slc.models import SlcTrade
        trade = session.exec(select(SlcTrade)).first()
    assert position.status == "closed"
    assert trade.exit_price == 103.5


def test_resolve_protected_degraded_positions_leaves_it_unresolved_on_failure(monkeypatch):
    """Never escalated to a same-cycle cancel on failure - simply left
    for a later cycle (or end-of-day closeout) to retry."""
    with get_live_slc_session() as session:
        session.add(models.SlcSignalRecord(
            cycle_run_id="run-1", symbol="AAPL", level_id="demand:x", direction="long",
            level_state="fresh", level_low=95.0, level_high=99.0,
            level_active_time=pd.Timestamp("2026-08-13 13:40:00").to_pydatetime(),
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            entry_time=pd.Timestamp("2026-08-13 14:05:00").to_pydatetime(),
            stop=94.0, stochastic_k=15.0, stochastic_d=12.0, atr14=1.0, structure="uptrend", impulse_atr=1.5,
        ))
    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol="AAPL", level_id="demand:x",
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-13").date(), status="protected_degraded",
            qty=10.0, entry_price=100.0, stop_price=98.0, target_price=104.0,
            target_order_id="target-old-1",
        ))
    monkeypatch.setattr(guardrails, "assert_closeout_preconditions", lambda **kw: None)
    from live_slc.execution import MarketableExitResult
    monkeypatch.setattr(run_slc_live.execution, "exit_via_marketable_replacement", lambda *a, **k: MarketableExitResult(False))

    result = run_slc_live._resolve_protected_degraded_positions(object(), observed_account_id="acct-1")
    assert result["resolved"] == []
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
    assert position.status == "protected_degraded"  # left as-is, not escalated


# -- orphan broker positions adopted by closeout (rev. 11 review point 8) ---

def test_discover_orphan_slc_positions_adopts_an_untracked_broker_position(monkeypatch):
    """The exact gap found via review: closeout only ever examined local
    SlcPosition rows - a crash between a real fill and _open_position()
    ever being called left a real broker position closeout would never
    have found."""
    with get_live_slc_session() as session:
        signal = models.SlcSignalRecord(
            cycle_run_id="run-1", symbol="AAPL", level_id="demand:x", direction="long",
            level_state="fresh", level_low=95.0, level_high=99.0,
            level_active_time=pd.Timestamp("2026-08-13 13:40:00").to_pydatetime(),
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            entry_time=pd.Timestamp("2026-08-13 14:05:00").to_pydatetime(),
            stop=94.0, stochastic_k=15.0, stochastic_d=12.0, atr14=1.0, structure="uptrend", impulse_atr=1.5,
        )
        session.add(signal)
        session.flush()
        session.refresh(signal)
        session.add(SlcOrder(
            client_order_id="slc-orphan-entry", symbol="AAPL", leg="entry", side="buy",
            position_intent="buy_to_open", order_class="bracket", dry_run=False, qty=10.0,
            status="filled", signal_id=signal.id, stop_submitted=98.0, target_submitted=104.0,
            alpaca_order_id="entry-order-orphan", fill_price=100.0,
        ))

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert len(result.adopted_ids) == 1
    assert result.unresolved_symbols == set()
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
    assert position is not None
    assert position.symbol == "AAPL"
    assert position.status == "protected_degraded"
    assert position.qty == 10.0


def test_discover_orphan_slc_positions_never_adopts_an_unmatched_position():
    """A broker position with no confidently-matching SLC entry order at
    all must never be guessed at - left for manual review, and not even
    flagged as unresolved (there's no evidence at all it's SLC's)."""
    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "MSFT", "qty": 5.0, "avg_entry_price": 300.0})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == set()
    with get_live_slc_session() as session:
        positions = list(session.exec(select(SlcPosition)))
    assert positions == []


def test_discover_orphan_slc_positions_never_raises_on_broker_read_failure():
    class _Client:
        def get_all_positions(self):
            raise TimeoutError("timed out")

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == set()


def _seed_slc_entry_order(*, symbol, level_id, confirmation_time, fill_price, client_order_id, direction="long"):
    with get_live_slc_session() as session:
        signal = models.SlcSignalRecord(
            cycle_run_id="run-1", symbol=symbol, level_id=level_id, direction=direction,
            level_state="fresh", level_low=95.0, level_high=99.0,
            level_active_time=confirmation_time - pd.Timedelta(minutes=20),
            confirmation_time=confirmation_time.to_pydatetime(),
            entry_time=(confirmation_time + pd.Timedelta(minutes=5)).to_pydatetime(),
            stop=94.0, stochastic_k=15.0, stochastic_d=12.0, atr14=1.0,
            structure="uptrend" if direction == "long" else "downtrend", impulse_atr=1.5,
        )
        session.add(signal)
        session.flush()
        session.refresh(signal)
        session.add(SlcOrder(
            client_order_id=client_order_id, symbol=symbol, leg="entry",
            side="buy" if direction == "long" else "sell",
            position_intent="buy_to_open" if direction == "long" else "sell_to_open",
            order_class="bracket", dry_run=False, qty=10.0,
            status="filled", signal_id=signal.id, stop_submitted=98.0, target_submitted=104.0,
            alpaca_order_id=f"{client_order_id}-alpaca", fill_price=fill_price,
        ))


def test_discover_orphan_slc_positions_never_guesses_among_multiple_unclosed_candidates():
    """The exact bug found via review: a bare, unordered .first() over
    ALL historical filled orders for the symbol could match a completely
    unrelated trade and reconstruct the orphan's stop/target from the
    wrong signal entirely. Two genuinely unclosed candidates for the same
    symbol must never be guessed between."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-10 14:00:00"),
        fill_price=100.0, client_order_id="slc-candidate-1",
    )
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:y", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-candidate-2",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}
    with get_live_slc_session() as session:
        positions = list(session.exec(select(SlcPosition)))
    assert positions == []  # never guessed at which candidate is the source


def test_discover_orphan_slc_positions_excludes_already_closed_trades_from_candidates():
    """An order whose resulting position was already properly closed is
    NOT a candidate for a NEW orphan - only unclosed history counts."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:old", confirmation_time=pd.Timestamp("2026-08-01 14:00:00"),
        fill_price=90.0, client_order_id="slc-old-closed",
    )
    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol="AAPL", level_id="demand:old",
            confirmation_time=pd.Timestamp("2026-08-01 14:00:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-01").date(), status="closed",
            qty=10.0, entry_price=90.0, stop_price=88.0, target_price=94.0,
        ))
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:new", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-new-unclosed",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert len(result.adopted_ids) == 1
    with get_live_slc_session() as session:
        adopted = session.get(SlcPosition, result.adopted_ids[0])
    assert adopted.level_id == "demand:new"  # the unclosed candidate, never the already-closed one


def test_discover_orphan_slc_positions_rejects_a_price_mismatched_candidate():
    """The lone candidate's fill price must reasonably corroborate the
    broker's own avg_entry_price - a wild mismatch is suspicious even
    with only one candidate present."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=50.0, client_order_id="slc-mismatched",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}


def test_discover_orphan_slc_positions_requires_price_corroboration_not_optional():
    """rev. 11 review: price corroboration must be REQUIRED, not skipped
    whenever either price happens to be missing/zero - a candidate with
    no fill_price on record must never be silently adopted."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-no-broker-price",
    )
    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
        order.fill_price = None  # simulate a record with no usable fill price
        session.add(order)

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}


def test_discover_orphan_slc_positions_rejects_a_direction_mismatch():
    """rev. 11 review: a long SLC order must never be adopted against a
    broker SHORT position, or vice versa."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-long-order",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "short"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}


# -- broker_position.side normalization: real PositionSide enum vs plain --
# -- strings (SDK-compatibility fix, found via review) --------------------

def test_discover_orphan_slc_positions_corroborates_a_real_position_side_long_enum():
    """str(PositionSide.LONG) is "PositionSide.LONG", not "long" - a bare
    str() on the raw enum rejected every real Alpaca long position as a
    direction mismatch. Uses the actual installed alpaca-py enum, not a
    plain-string mock, so this can't be concealed the same way again."""
    from alpaca.trading.enums import PositionSide

    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-long-enum", direction="long",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {
                "symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": PositionSide.LONG,
            })()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert len(result.adopted_ids) == 1
    assert result.unresolved_symbols == set()


def test_discover_orphan_slc_positions_corroborates_a_real_position_side_short_enum():
    from alpaca.trading.enums import PositionSide

    _seed_slc_entry_order(
        symbol="TSLA", level_id="supply:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-short-enum", direction="short",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {
                "symbol": "TSLA", "qty": -10.0, "avg_entry_price": 100.0, "side": PositionSide.SHORT,
            })()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert len(result.adopted_ids) == 1
    assert result.unresolved_symbols == set()


def test_discover_orphan_slc_positions_rejects_opposite_real_position_side_enum():
    """A long SLC order against a broker PositionSide.SHORT position must
    still be rejected as a direction mismatch, using the real enum."""
    from alpaca.trading.enums import PositionSide

    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-long-vs-short-enum", direction="long",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {
                "symbol": "AAPL", "qty": -10.0, "avg_entry_price": 100.0, "side": PositionSide.SHORT,
            })()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}


def test_discover_orphan_slc_positions_missing_or_invalid_side_stays_unresolved():
    """Fails closed: no side attribute at all, and an empty-string side,
    must both leave the symbol unresolved rather than adopted."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-no-side", direction="long",
    )

    class _ClientNoSide:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0})()]  # no .side at all

    result = run_slc_live._discover_orphan_slc_positions(_ClientNoSide())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}

    class _ClientEmptySide:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": ""})()]

    result2 = run_slc_live._discover_orphan_slc_positions(_ClientEmptySide())
    assert result2.adopted_ids == []
    assert result2.unresolved_symbols == {"AAPL"}


def test_discover_orphan_slc_positions_still_accepts_plain_string_side():
    """Backward/mock compatibility: a plain "long"/"short" string (no
    .value attribute) must continue to work exactly as before."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-plain-string-side", direction="long",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert len(result.adopted_ids) == 1
    assert result.unresolved_symbols == set()


def test_discover_orphan_slc_positions_rejects_a_quantity_mismatch():
    """rev. 11 review: the order's own requested qty must match the
    broker's current qty exactly - a mismatch must never be silently
    adopted using the broker's quantity as if it were confirmed."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-qty-10",
    )

    class _Client:
        def get_all_positions(self):
            # order was for qty=10.0 (per _seed_slc_entry_order), broker
            # shows a completely different quantity.
            return [type("P", (), {"symbol": "AAPL", "qty": 15.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.adopted_ids == []
    assert result.unresolved_symbols == {"AAPL"}


# -- unresolved orphans actually block new entries (rev. 11 review) ---------

def test_unresolved_orphan_blocks_new_entries_on_a_later_cycle():
    """The exact gap found via review: an unresolved orphan was only ever
    recorded as an SlcAuditEvent - a historical log entry nothing
    re-reads - so system_wide_entry_block_reasons() (what run_cycle()'s
    admission loop actually checks) never saw it, and a later cycle could
    resume entries while the broker position stayed genuinely unresolved."""
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:x", confirmation_time=pd.Timestamp("2026-08-10 14:00:00"),
        fill_price=100.0, client_order_id="slc-candidate-1",
    )
    _seed_slc_entry_order(
        symbol="AAPL", level_id="demand:y", confirmation_time=pd.Timestamp("2026-08-12 14:00:00"),
        fill_price=100.0, client_order_id="slc-candidate-2",
    )

    class _Client:
        def get_all_positions(self):
            return [type("P", (), {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 100.0, "side": "long"})()]

    result = run_slc_live._discover_orphan_slc_positions(_Client())
    assert result.unresolved_symbols == {"AAPL"}  # genuinely ambiguous - two unclosed candidates

    with get_live_slc_session() as session:
        from live_slc.risk import system_wide_entry_block_reasons
        reasons = system_wide_entry_block_reasons(session)
    assert "unresolved_orphan_broker_position" in reasons


# -- broker-flat readback never trivially True, never assumed on failure ----
# (rev. 11 review point 10)

def _seed_open_position(symbol="AAPL", *, protective_order_id="prot-1"):
    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol=symbol, level_id="demand:x",
            confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-13").date(), status="open",
            qty=10.0, entry_price=100.0, stop_price=98.0, target_price=104.0,
            protective_order_id=protective_order_id,
        ))


def _api_not_found():
    from alpaca.common.exceptions import APIError

    class _HttpErr:
        response = type("R", (), {"status_code": 404})()

    return APIError("not found", http_error=_HttpErr())


def test_reconcile_local_position_closed_by_filled_protective_stop():
    """Exact 2026-08-19 MSFT regression: the broker stop filled and the
    account was flat, but the local SlcPosition remained open forever."""
    from alpaca.trading.enums import OrderStatus
    from live_slc.models import SlcTrade

    _seed_open_position("MSFT", protective_order_id="msft-stop")

    class _Client:
        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id):
            assert order_id == "msft-stop"
            return type("O", (), {
                "id": order_id, "status": OrderStatus.FILLED,
                "qty": 10, "filled_qty": 10, "filled_avg_price": 99.25,
                "filled_at": pd.Timestamp("2026-08-19 13:45:10", tz="UTC"),
            })()

        def get_order_by_client_id(self, client_order_id):
            raise _api_not_found()

        def __getattr__(self, name):
            if name in MUTATING:
                raise AssertionError(f"reconciliation must be read-only: {name}")
            raise AttributeError(name)

    result = run_slc_live._reconcile_local_positions_from_broker_exits(_Client())
    assert len(result.closed_ids) == 1
    assert result.ambiguous_ids == []
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
        trade = session.exec(select(SlcTrade)).first()
    assert position.status == "closed"
    assert trade.exit_reason == "protective_stop_filled"
    assert trade.exit_price == 99.25
    assert trade.exit_order_id == "msft-stop"
    assert pd.Timestamp(trade.exit_time) == pd.Timestamp("2026-08-19 13:45:10")


def test_reconcile_local_position_closed_by_deterministic_emergency_exit():
    """Exact 2026-08-19 TGT regression: the emergency market exit filled,
    then get_open_position returned 404 before local closure was saved."""
    from alpaca.trading.enums import OrderStatus
    from live_slc.models import SlcTrade

    with get_live_slc_session() as session:
        session.add(SlcPosition(
            symbol="TGT", level_id="demand:tgt",
            confirmation_time=pd.Timestamp("2026-08-19 13:35:00").to_pydatetime(),
            direction="long", session_date=pd.Timestamp("2026-08-19").date(),
            status="protected_degraded", qty=55, entry_price=158.09,
            stop_price=153.0, target_price=168.0,
            protective_order_id="tgt-stop", target_order_id="tgt-target",
        ))

    class _Client:
        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id):
            return type("O", (), {
                "id": order_id, "status": OrderStatus.CANCELED,
                "qty": 55, "filled_qty": 0, "filled_avg_price": None,
            })()

        def get_order_by_client_id(self, client_order_id):
            assert client_order_id.endswith("-exit")
            return type("O", (), {
                "id": "tgt-emergency-exit", "status": OrderStatus.FILLED,
                "qty": 55, "filled_qty": 55, "filled_avg_price": 157.790545,
                "filled_at": pd.Timestamp("2026-08-19 13:36:57", tz="UTC"),
            })()

    result = run_slc_live._reconcile_local_positions_from_broker_exits(_Client())
    assert len(result.closed_ids) == 1
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
        trade = session.exec(select(SlcTrade)).first()
    assert position.status == "closed"
    assert trade.exit_reason == "emergency_or_closeout_exit_filled"
    assert trade.exit_price == 157.790545
    assert trade.exit_order_id == "tgt-emergency-exit"


def test_broker_flat_without_one_exact_filled_exit_becomes_ambiguous():
    """Broker absence alone is never enough to fabricate a local close or
    its P&L; without exact exit evidence the position blocks new entries."""
    from alpaca.trading.enums import OrderStatus
    from live_slc.models import SlcTrade

    _seed_open_position("AAPL", protective_order_id="aapl-stop")

    class _Client:
        def get_all_positions(self):
            return []

        def get_order_by_id(self, order_id):
            return type("O", (), {
                "id": order_id, "status": OrderStatus.CANCELED,
                "qty": 10, "filled_qty": 0, "filled_avg_price": None,
            })()

        def get_order_by_client_id(self, client_order_id):
            raise _api_not_found()

    result = run_slc_live._reconcile_local_positions_from_broker_exits(_Client())
    assert result.closed_ids == []
    assert len(result.ambiguous_ids) == 1
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
        trade = session.exec(select(SlcTrade)).first()
    assert position.status == "ambiguous"
    assert trade is None


class _ClosingClient(_AccountOnlyClient):
    def __init__(self, *, readback_positions=None, readback_raises=False):
        super().__init__()
        self._readback_positions = readback_positions if readback_positions is not None else []
        self._readback_raises = readback_raises
        self._known_close_orders = {}

    def cancel_order_by_id(self, oid):
        pass

    def get_order_by_id(self, oid, filter=None):
        if oid in self._known_close_orders:
            return self._known_close_orders[oid]
        return type("O", (), {"id": oid, "status": "canceled", "filled_avg_price": 100.0, "legs": []})()

    def get_order_by_client_id(self, oc_id):
        if oc_id in self._known_close_orders:
            return self._known_close_orders[oc_id]
        from alpaca.common.exceptions import APIError
        resp = type("R", (), {"status_code": 404})()
        err = type("E", (), {"response": resp})()
        raise APIError("not found", http_error=err)

    def submit_order(self, request):
        order = type("O", (), {"id": "close-order-1", "status": "filled", "filled_avg_price": 99.5})()
        self._known_close_orders["close-order-1"] = order
        self._known_close_orders[request.client_order_id] = order
        return order

    def get_open_position(self, symbol):
        return None

    def get_all_positions(self):
        if self._readback_raises:
            raise TimeoutError("timed out")
        return self._readback_positions


def _closeout_common_mocks(monkeypatch):
    monkeypatch.setattr(guardrails, "assert_closeout_preconditions", lambda **kw: None)
    monkeypatch.setattr(run_slc_live, "is_trading_day", lambda day: True)
    monkeypatch.setattr(run_slc_live.closeout_mod, "should_begin_closeout", lambda now, day: True)


def test_broker_flat_readback_fails_closed_not_true_on_exception(monkeypatch):
    _seed_open_position()
    _closeout_common_mocks(monkeypatch)
    client = _ClosingClient(readback_raises=True)
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_closeout_stage()
    assert result["broker_flat"] is None  # never silently True on a readback failure


def test_broker_flat_readback_false_when_broker_still_shows_a_position(monkeypatch):
    _seed_open_position()
    _closeout_common_mocks(monkeypatch)
    client = _ClosingClient(readback_positions=[type("P", (), {"symbol": "AAPL", "qty": 3.0})()])
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_closeout_stage()
    assert result["broker_flat"] is False


def test_broker_flat_readback_true_when_genuinely_flat(monkeypatch):
    _seed_open_position()
    _closeout_common_mocks(monkeypatch)
    client = _ClosingClient(readback_positions=[])
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: client)

    result = run_slc_live.run_closeout_stage()
    assert result["broker_flat"] is True


# -- _finish_cycle_run status + failed_cycles/overlapping_cycles ------------
# (rev. 11 review point 11)

def test_finish_cycle_run_preserves_an_explicitly_passed_status():
    """The exact bug found via review: status was unconditionally
    overwritten to "completed", silently discarding e.g. status="failed"."""
    run_id = run_slc_live._start_cycle_run("cycle")
    run_slc_live._finish_cycle_run(run_id, status="failed")
    with get_live_slc_session() as session:
        from live_slc.models import SlcCycleRun
        run = session.get(SlcCycleRun, run_id)
    assert run.status == "failed"


def test_finish_cycle_run_still_defaults_to_completed_when_status_not_passed():
    run_id = run_slc_live._start_cycle_run("cycle")
    run_slc_live._finish_cycle_run(run_id, symbols_scanned=5)
    with get_live_slc_session() as session:
        from live_slc.models import SlcCycleRun
        run = session.get(SlcCycleRun, run_id)
    assert run.status == "completed"


def test_run_cycle_records_failed_cycles_and_reraises_on_an_uncaught_exception(monkeypatch):
    def _boom(run_id, cycle_start_monotonic):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(run_slc_live, "_run_cycle_body", _boom)

    with pytest.raises(RuntimeError, match="simulated crash"):
        run_slc_live.run_cycle()

    with get_live_slc_session() as session:
        from live_slc.models import SlcCycleRun, SlcSessionStat
        run = session.exec(select(SlcCycleRun)).first()
        stat = session.exec(select(SlcSessionStat)).first()
    assert run.status == "failed"
    assert stat.failed_cycles == 1


def test_main_records_overlapping_cycles_when_lock_already_held(monkeypatch):
    from live_slc.process_lock import LockAlreadyHeld

    class _RefusingLock:
        def __enter__(self):
            raise LockAlreadyHeld("already running")
        def __exit__(self, *a):
            return False
    monkeypatch.setattr(run_slc_live, "acquire_process_lock", lambda: _RefusingLock())
    monkeypatch.setattr(sys, "argv", ["run_slc_live.py", "--stage", "cycle"])
    # the isolated-db autouse fixture already initialized the DB

    with pytest.raises(SystemExit):
        run_slc_live.main()

    with get_live_slc_session() as session:
        from live_slc.models import SlcSessionStat
        stat = session.exec(select(SlcSessionStat)).first()
    assert stat.overlapping_cycles == 1
