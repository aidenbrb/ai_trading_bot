"""
Direct tests of _process_admitted_confirmation() - the shared decision
pipeline for both dry_run and paper_active (rev. 11 point 2's fix).
Uses a spy client to prove dry_run makes zero broker-mutating calls under
every branch, including discovering a real existing order.
"""
from datetime import date
from decimal import Decimal

import pandas as pd
import pytest

import live_slc.models as models
import live_slc.run_slc_live as run_slc_live
from live_slc.execution import ReconcileOutcome, ReconcileResult
from live_slc.reducer import Confirmation
from live_slc.risk import AccountSnapshot


class _SpyClient:
    """Records every method call - the test asserts which ones (if any)
    were mutating."""

    MUTATING = {"submit_order", "cancel_order_by_id", "replace_order_by_id", "cancel_orders", "close_position"}

    def __init__(self):
        self.calls = []
        self.mutating_calls = []

    def __getattr__(self, name):
        def _recorder(*args, **kwargs):
            self.calls.append(name)
            if name in self.MUTATING:
                self.mutating_calls.append(name)
            raise AssertionError(f"unstubbed spy method called: {name}")
        return _recorder


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def _confirmation(symbol="AAPL", direction="long", level_id="demand:x"):
    ts = pd.Timestamp("2026-08-13 14:00:00")
    return Confirmation(
        "slc_4h_5m_stock_v1", symbol, level_id, direction, "fresh",
        99.0, 101.0, ts - pd.Timedelta(minutes=20), ts, ts + pd.Timedelta(minutes=5),
        98.5, 15.0, 12.0, 1.0, "uptrend" if direction == "long" else "downtrend", 1.5,
    )


def _snapshot(**overrides):
    defaults = dict(
        account_id="acct-1", equity=100000.0, cash=100000.0, non_marginable_buying_power=100000.0,
        start_of_day_equity=100000.0, daily_realized_pnl=0.0, daily_unrealized_pnl=0.0,
        positions=[], today_orders=[],
    )
    defaults.update(overrides)
    return AccountSnapshot(**defaults)


def test_dry_run_never_calls_any_mutating_broker_method_on_the_happy_path(monkeypatch):
    confirmation = _confirmation()
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 100.0)
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", lambda client, **kw: _snapshot())

    class ReadOnlySpy(_SpyClient):
        def get_order_by_client_id(self, oc_id):
            from alpaca.common.exceptions import APIError

            class _Resp:
                status_code = 404

            class _Err:
                response = _Resp()
            raise APIError("not found", http_error=_Err())

    client = ReadOnlySpy()
    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="dry_run", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result.get("dry_run_proposed") is True
    assert client.mutating_calls == []

    order = list(models.get_live_slc_session().__enter__().exec(
        __import__("sqlmodel").select(models.SlcOrder)
    ))
    assert order[0].status == "dry_run_proposed"
    assert order[0].qty > 0  # NOT the old qty=0.0 stub


def test_dry_run_discovering_a_real_order_never_adopts_it(monkeypatch):
    """rev. 11 point 3: dry-run discovering an existing broker order must
    record existing_broker_order_detected, block, and NEVER call
    replace/cancel/close/submit."""
    confirmation = _confirmation()
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 100.0)
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", lambda client, **kw: _snapshot())

    class RealOrderFound(_SpyClient):
        def get_order_by_client_id(self, oc_id):
            class _Order:
                id = "existing-order-1"
            return _Order()

    client = RealOrderFound()
    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="dry_run", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result.get("new_blocking_incident") is True
    assert client.mutating_calls == []

    with models.get_live_slc_session() as session:
        from sqlmodel import select
        orders = list(session.exec(select(models.SlcOrder)))
    assert orders[0].status == "existing_broker_order_detected"

    with models.get_live_slc_session() as session:
        from live_slc.risk import system_wide_entry_block_reasons
        assert "unresolved_existing_broker_order_detected" in system_wide_entry_block_reasons(session)


def test_pre_submission_invalid_bracket_skips_with_nothing_to_flatten(monkeypatch):
    """rev. 11 point 4/amendment_005 item 5: a pre-submission invalid
    rounded bracket just skips the trade - there is no position to
    flatten, so no emergency-exit path is invoked."""
    confirmation = _confirmation(direction="long")
    # A quote that's already below the stop makes the bracket invalid
    # before any submission - nothing to flatten.
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 50.0)
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", lambda client, **kw: _snapshot())

    client = _SpyClient()
    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="dry_run", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result == {}
    assert client.calls == []  # never even reached the broker


def test_shortability_checked_for_short_confirmations_in_dry_run_too(monkeypatch):
    confirmation = _confirmation(direction="short", level_id="supply:x")
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: 100.0)
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", lambda client, **kw: _snapshot())

    calls = []

    def fake_asset_raw(client):
        def _fetch(symbol):
            calls.append(symbol)
            return {"tradable": True, "shortable": False}
        return _fetch

    monkeypatch.setattr(run_slc_live, "_get_asset_raw", fake_asset_raw)
    client = _SpyClient()
    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="dry_run", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result == {}
    assert calls == ["AAPL"]
    assert client.calls == []  # never reached the broker order path


# -- paper_active submission-path tests (rev. 11 review points 1-4, 7) ------

class _ConfigurableClient(_SpyClient):
    """A fake for paper_active submission-path tests - unlike the bare
    _SpyClient (which raises on anything unstubbed), the methods here are
    real and driven by test-supplied callables, while everything else
    still raises via the inherited spy behavior."""

    def __init__(self):
        super().__init__()
        self.canceled_ids = []
        self.replaced = []
        self.submit_response = None       # callable(request) -> order, or raises
        self.get_order_by_id_response = None  # callable(order_id) -> order

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
        self.calls.append("submit_order")
        self.mutating_calls.append("submit_order")
        return self.submit_response(request)

    def get_order_by_id(self, order_id, filter=None):
        self.calls.append("get_order_by_id")
        return self.get_order_by_id_response(order_id)

    def cancel_order_by_id(self, order_id):
        self.calls.append("cancel_order_by_id")
        self.mutating_calls.append("cancel_order_by_id")
        self.canceled_ids.append(order_id)

    def replace_order_by_id(self, order_id, request):
        self.calls.append("replace_order_by_id")
        self.mutating_calls.append("replace_order_by_id")
        self.replaced.append((order_id, request))
        return type("O", (), {"id": order_id + "-replaced"})()


def _fake_order(*, status, filled_qty=0, filled_avg_price=None, legs=None, qty=None, id="entry-order-1"):
    return type("O", (), {
        "id": id, "status": status, "filled_qty": filled_qty,
        "filled_avg_price": filled_avg_price, "legs": legs or [], "qty": qty,
    })()


def _fake_leg(*, id, type, status="new", limit_price=None, qty=None):
    import types as _types
    return _types.SimpleNamespace(id=id, type=type, status=status, limit_price=limit_price, qty=qty)


def _paper_active_setup(monkeypatch, *, fresh_quote=100.0):
    monkeypatch.setattr(run_slc_live, "_get_fresh_quote", lambda symbol, max_age_seconds: fresh_quote)
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", lambda client, **kw: _snapshot())
    monkeypatch.setattr(run_slc_live.guardrails, "assert_operational_preconditions", lambda **kw: {})
    monkeypatch.setattr(run_slc_live.guardrails, "assert_submission_preconditions", lambda *a, **k: None)
    monkeypatch.setattr(run_slc_live.guardrails, "assert_closeout_preconditions", lambda **kw: None)


def test_zero_fill_still_live_order_is_actively_canceled_not_just_labeled(monkeypatch):
    """rev. 11 review point 1: a still-live (new/accepted) zero-fill
    order at the poll deadline must be ACTIVELY canceled via
    cancel_order_by_id, never just labeled canceled_unfilled."""
    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    client = _ConfigurableClient()
    client.submit_response = lambda request: _fake_order(status="new", id="entry-order-1")

    def _get_order(order_id):
        # cancel_and_confirm_unfilled_entry's re-fetch, after
        # cancel_order_by_id was actually called, reports canceled.
        if client.canceled_ids:
            return _fake_order(status="canceled", filled_qty=0)
        return _fake_order(status="new", filled_qty=0)
    client.get_order_by_id_response = _get_order

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result == {}
    assert "cancel_order_by_id" in client.mutating_calls
    assert client.canceled_ids == ["entry-order-1"]

    with models.get_live_slc_session() as session:
        from sqlmodel import select
        order = session.exec(select(models.SlcOrder)).first()
    assert order.status == "canceled_unfilled"


def test_canceled_order_with_a_partial_fill_is_routed_to_partial_fill_handling_not_discarded(monkeypatch):
    """rev. 11 review point 1: a broker-canceled order that still carries
    a nonzero filled_qty is a REAL partial fill, not "unfilled" - must
    never be silently abandoned."""
    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    client = _ConfigurableClient()
    client.submit_response = lambda request: _fake_order(status="new", id="entry-order-1")
    client.get_order_by_id_response = lambda order_id: _fake_order(
        status="canceled", filled_qty=4, filled_avg_price=100.0,
        legs=[_fake_leg(id="stop-1", type="stop", status="canceled")],
    )
    # The ENTRY-side pre-submission reconcile must still see a clean 404
    # (CONFIRMED_ABSENT) so submission proceeds; only the CLOSE-side
    # reconcile (during the flatten attempt) goes ambiguous - this test
    # only needs to prove the position was persisted BEFORE that flatten
    # attempt even runs, regardless of how it resolves.
    def _get_order_by_client_id(oc_id):
        if oc_id.endswith("-exit"):
            raise TimeoutError("timed out")
        from alpaca.common.exceptions import APIError
        resp = type("R", (), {"status_code": 404})()
        err = type("E", (), {"response": resp})()
        raise APIError("not found", http_error=err)
    client.get_order_by_client_id = _get_order_by_client_id

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    # A real position was persisted for the partial fill BEFORE the
    # flatten attempt (review point 3) - never only an order-status string.
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        positions = list(session.exec(select(models.SlcPosition)))
    assert len(positions) == 1
    assert positions[0].qty == 4
    assert result.get("submitted") is True
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        event = session.exec(select(models.SlcAuditEvent)).first()
        stat = session.get(models.SlcSessionStat, confirmation.confirmation_time.date())
    assert event.event_type == "protected_degraded"
    assert stat.trades_opened == 1
    assert stat.unprotected_position_incident_count == 1


def test_real_alpaca_filled_enum_is_a_full_fill_not_an_emergency_partial_fill(monkeypatch):
    """Regression for the first paper-active incident on 2026-08-18.

    Alpaca returned OrderStatus.FILLED with filled_qty == requested qty;
    bare str(status) produced ``OrderStatus.FILLED`` and routed the full
    fill through the partial-fill emergency-flatten branch.
    """
    from alpaca.trading.enums import OrderStatus

    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    client = _ConfigurableClient()
    requested_qty = {"value": 0.0}

    def _submit(request):
        requested_qty["value"] = float(request.qty)
        return _fake_order(status=OrderStatus.NEW, id="entry-order-1")

    client.submit_response = _submit
    client.get_order_by_id_response = lambda order_id: _fake_order(
        status=OrderStatus.FILLED,
        filled_qty=requested_qty["value"],
        filled_avg_price=100.0,
        qty=requested_qty["value"],
        legs=[
            _fake_leg(id="stop-1", type="stop", status=OrderStatus.HELD),
            _fake_leg(
                id="target-1", type="limit", status=OrderStatus.NEW,
                limit_price=103.0, qty=requested_qty["value"],
            ),
        ],
    )
    monkeypatch.setattr(
        run_slc_live.execution,
        "verify_target_replacement",
        lambda *a, **k: run_slc_live.execution.TargetVerificationResult(
            True, target_order_id="target-1-replaced",
        ),
    )

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client,
        run_id="run-1", committed_notional_this_cycle=0.0,
    )

    assert result.get("submitted") is True
    assert "cancel_order_by_id" not in client.mutating_calls
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        order = session.exec(select(models.SlcOrder)).first()
        position = session.exec(select(models.SlcPosition)).first()
        stat = session.get(models.SlcSessionStat, confirmation.confirmation_time.date())
    assert order.status == "filled"
    assert order.alpaca_order_id == "entry-order-1"
    assert order.fill_price == 100.0
    assert order.notional == pytest.approx(requested_qty["value"] * 100.0)
    assert order.slippage == pytest.approx(0.0)
    assert position.status == "open"
    assert position.notional == pytest.approx(requested_qty["value"] * 100.0)
    assert position.slippage == pytest.approx(0.0)
    assert stat.trades_opened == 1


def test_position_and_trade_session_counters_are_idempotent():
    confirmation = _confirmation()
    position_id = run_slc_live._open_position(
        confirmation, qty=10, entry_price=100.0, stop_price=98.5,
        target_price=103.0, entry_order_id="entry-1",
    )
    # Replaying the same natural-key position must not count a second open.
    assert run_slc_live._open_position(
        confirmation, qty=10, entry_price=100.0, stop_price=98.5,
        target_price=103.0, entry_order_id="entry-1",
    ) == position_id

    run_slc_live._mark_position_closed(
        position_id, exit_price=99.5, exit_reason="test_flatten",
        exit_order_id="exit-1",
    )
    # Closing/reconciling the same position again is idempotent at the
    # trade unique constraint and must not double-count session totals.
    run_slc_live._mark_position_closed(
        position_id, exit_price=99.5, exit_reason="test_flatten",
        exit_order_id="exit-1",
    )

    with models.get_live_slc_session() as session:
        from sqlmodel import select
        stat = session.get(models.SlcSessionStat, confirmation.confirmation_time.date())
        trades = list(session.exec(select(models.SlcTrade)))
    assert stat.trades_opened == 1
    assert stat.trades_closed == 1
    assert stat.losses == 1
    assert stat.wins == 0
    assert stat.net_pnl == pytest.approx(-5.0)
    assert len(trades) == 1


# -- Phase 6 Step 4: pnl_r must be computed, not hardcoded to 0.0 -----------

def test_mark_position_closed_computes_pnl_r_for_a_long():
    confirmation = _confirmation(direction="long")
    position_id = run_slc_live._open_position(
        confirmation, qty=10, entry_price=100.0, stop_price=98.5,
        target_price=103.0, entry_order_id="entry-long-r",
    )
    run_slc_live._mark_position_closed(
        position_id, exit_price=99.5, exit_reason="test_flatten", exit_order_id="exit-long-r",
    )
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        trade = session.exec(select(models.SlcTrade).where(models.SlcTrade.position_id == position_id)).first()
    # gross_per_share = 99.5 - 100.0 = -0.5; initial_risk = |100.0 - 98.5| = 1.5
    assert trade.pnl_r == pytest.approx(-0.5 / 1.5)
    assert not trade.exit_reason.endswith("_pnl_r_unknown")


def test_mark_position_closed_computes_pnl_r_for_a_short():
    confirmation = _confirmation(symbol="MSFT", direction="short", level_id="supply:x")
    position_id = run_slc_live._open_position(
        confirmation, qty=5, entry_price=200.0, stop_price=204.0,
        target_price=192.0, entry_order_id="entry-short-r",
    )
    run_slc_live._mark_position_closed(
        position_id, exit_price=194.0, exit_reason="test_flatten", exit_order_id="exit-short-r",
    )
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        trade = session.exec(select(models.SlcTrade).where(models.SlcTrade.position_id == position_id)).first()
    # gross_per_share = 200.0 - 194.0 = 6.0; initial_risk = |200.0 - 204.0| = 4.0
    assert trade.pnl_r == pytest.approx(6.0 / 4.0)


def test_mark_position_closed_flags_pnl_r_unknown_on_non_positive_risk():
    """Degenerate case the pre-registration says should never happen for a
    position that was actually opened (entry requires positive risk) -
    must fail closed to a flagged 0.0, never crash or fabricate a
    plausible-looking nonzero R."""
    confirmation = _confirmation(symbol="ZERO", direction="long", level_id="demand:zero")
    position_id = run_slc_live._open_position(
        confirmation, qty=10, entry_price=100.0, stop_price=100.0,  # zero risk
        target_price=103.0, entry_order_id="entry-zero-risk",
    )
    run_slc_live._mark_position_closed(
        position_id, exit_price=99.5, exit_reason="test_flatten", exit_order_id="exit-zero-risk",
    )
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        trade = session.exec(select(models.SlcTrade).where(models.SlcTrade.position_id == position_id)).first()
    assert trade.pnl_r == 0.0
    assert trade.exit_reason == "test_flatten_pnl_r_unknown"


def test_ambiguous_submission_failure_never_becomes_confirmed_rejected(monkeypatch):
    """rev. 11 review point 2: a network/timeout failure during
    submit_order() must be ambiguous_submission, never confirmed_rejected
    (which nothing ever reconciles/blocks on)."""
    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    client = _ConfigurableClient()

    def _submit(request):
        raise TimeoutError("connection timed out")
    client.submit_response = _submit

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result.get("new_blocking_incident") is True
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        order = session.exec(select(models.SlcOrder)).first()
        from live_slc.risk import system_wide_entry_block_reasons
        assert "unresolved_ambiguous_submission" in system_wide_entry_block_reasons(session)
    assert order.status == "ambiguous_submission"


def test_post_fill_risk_check_uses_a_freshly_refetched_equity(monkeypatch):
    """rev. 11 review point 4: post-fill risk must read equity again,
    not reuse the pre-submission Phase-B snapshot."""
    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    request_qty = {"value": 0}
    client = _ConfigurableClient()

    def _submit(request):
        request_qty["value"] = request.qty
        return _fake_order(status="new", id="entry-order-1")
    client.submit_response = _submit
    client.get_order_by_id_response = lambda order_id: _fake_order(
        status="filled", filled_qty=request_qty["value"], filled_avg_price=100.0,
        legs=[_fake_leg(id="stop-1", type="stop"),
              _fake_leg(id="target-1", type="limit", limit_price=104.0, qty=request_qty["value"])],
    )

    snapshot_calls = {"n": 0}
    real_snapshot = _snapshot()

    def _fake_build_snapshot(client, **kw):
        snapshot_calls["n"] += 1
        return real_snapshot
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", _fake_build_snapshot)

    run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    # Called at least twice: once in Phase B before submission, once more
    # after the fill for the post-fill risk check.
    assert snapshot_calls["n"] >= 2


def test_post_fill_equity_refresh_failure_fails_closed_and_flattens(monkeypatch):
    """The exact bug found via review: a failed fresh-equity re-fetch
    silently fell back to the stale pre-submission snapshot and
    proceeded as if nothing were wrong - defeating the entire point of
    re-checking. Must instead fail closed: protected_degraded plus the
    guarded flatten, never silently trusting stale data."""
    confirmation = _confirmation()
    _paper_active_setup(monkeypatch)
    client = _ConfigurableClient()
    client.submit_response = lambda request: _fake_order(status="new", id="entry-order-1")
    client.get_order_by_id_response = lambda order_id: _fake_order(
        status="filled", filled_qty=10, filled_avg_price=100.0,
        legs=[_fake_leg(id="stop-1", type="stop"), _fake_leg(id="target-1", type="limit", limit_price=104.0, qty=10)],
    )

    call_count = {"n": 0}
    real_snapshot = _snapshot()

    def _flaky_build_snapshot(client, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return real_snapshot  # Phase B - succeeds
        raise run_slc_live.AccountSnapshotUnusable("broker unreachable")  # post-fill refresh - fails
    monkeypatch.setattr(run_slc_live, "build_account_snapshot", _flaky_build_snapshot)

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client, run_id="run-1", committed_notional_this_cycle=0.0,
    )
    assert result.get("new_blocking_incident") is True
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        position = session.exec(select(models.SlcPosition)).first()
    # protected_degraded at minimum - closeout's guarded flatten was
    # attempted (this fake's cancel/close plumbing isn't wired for a
    # full CONFIRMED_FLAT here, so "closed" is also an acceptable
    # terminal outcome of that attempt).
    assert position.status in ("protected_degraded", "closed")


def test_post_fill_emergency_exit_treats_real_alpaca_position_404_as_flat(monkeypatch):
    """Exact 2026-08-19 TGT regression: excess post-fill risk correctly
    triggers an emergency exit; after that exit fills, Alpaca's 404
    ``position does not exist`` read-back must close the local position
    instead of crashing the cycle and leaving it protected_degraded."""
    from alpaca.common.exceptions import APIError
    from alpaca.trading.enums import OrderStatus

    confirmation = _confirmation()
    _paper_active_setup(monkeypatch, fresh_quote=100.0)
    client = _ConfigurableClient()
    requested_qty = {"value": 0.0}

    def _submit(request):
        if str(request.client_order_id).endswith("-exit"):
            return _fake_order(
                id="emergency-exit-1", status=OrderStatus.FILLED,
                qty=requested_qty["value"], filled_qty=requested_qty["value"],
                filled_avg_price=109.5,
            )
        requested_qty["value"] = float(request.qty)
        return _fake_order(id="entry-order-1", status=OrderStatus.NEW)

    client.submit_response = _submit

    def _get_order(order_id):
        if order_id == "entry-order-1":
            return _fake_order(
                id=order_id, status=OrderStatus.FILLED,
                qty=requested_qty["value"], filled_qty=requested_qty["value"],
                filled_avg_price=110.0,
                legs=[
                    _fake_leg(id="stop-1", type="stop", status=OrderStatus.HELD),
                    _fake_leg(id="target-1", type="limit", status=OrderStatus.NEW,
                              limit_price=103.0, qty=requested_qty["value"]),
                ],
            )
        if order_id == "stop-1":
            return _fake_order(id=order_id, status=OrderStatus.CANCELED)
        if order_id == "emergency-exit-1":
            return _fake_order(
                id=order_id, status=OrderStatus.FILLED,
                qty=requested_qty["value"], filled_qty=requested_qty["value"],
                filled_avg_price=109.5,
            )
        raise AssertionError(order_id)

    client.get_order_by_id_response = _get_order

    class _HttpErr:
        response = type("R", (), {"status_code": 404})()

    def _position_absent(symbol):
        raise APIError("position does not exist", http_error=_HttpErr())

    client.get_open_position = _position_absent

    result = run_slc_live._process_admitted_confirmation(
        confirmation, status="paper_active", client=client,
        run_id="run-1", committed_notional_this_cycle=0.0,
    )

    assert result.get("submitted") is True
    assert result.get("new_blocking_incident") is False
    with models.get_live_slc_session() as session:
        from sqlmodel import select
        position = session.exec(select(models.SlcPosition)).first()
        trade = session.exec(select(models.SlcTrade)).first()
    assert position.status == "closed"
    assert trade.exit_order_id == "emergency-exit-1"
    assert trade.exit_price == 109.5
