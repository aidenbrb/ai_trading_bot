"""
Preflight (rev. 11 Step 9): delayed reconciliation of ambiguous order
states (strictly broker-read-only), split detection + atomic staged
rebuild, and the engine-parity self-check.
"""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from alpaca.common.exceptions import APIError

import live_slc.models as models
import live_slc.run_slc_live as run_slc_live
from live_slc.models import SlcFiveMinBar, SlcOrder, SlcPosition, SlcReducerState, get_live_slc_session
from live_slc.split_detection import SplitEvidence
from decimal import Decimal
from sqlmodel import select


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


MUTATING = {"submit_order", "cancel_order_by_id", "replace_order_by_id", "cancel_orders", "close_position"}


class _SpyClient:
    def __init__(self):
        self.mutating_calls = []

    def _record(self, name):
        self.mutating_calls.append(name)

    def submit_order(self, *a, **k):
        self._record("submit_order")
        raise AssertionError("preflight must never call submit_order")

    def cancel_order_by_id(self, *a, **k):
        self._record("cancel_order_by_id")
        raise AssertionError("preflight must never call cancel_order_by_id")

    def replace_order_by_id(self, *a, **k):
        self._record("replace_order_by_id")
        raise AssertionError("preflight must never call replace_order_by_id")

    def cancel_orders(self, *a, **k):
        self._record("cancel_orders")
        raise AssertionError("preflight must never call cancel_orders")

    def close_position(self, *a, **k):
        self._record("close_position")
        raise AssertionError("preflight must never call close_position")


class _FakeLeg:
    def __init__(self, status, type="stop"):
        self.status = status
        self.type = type


class _FakeOrder:
    def __init__(self, *, status, filled_qty, id="broker-order-1", filled_avg_price=100.0,
                 qty=10, legs=None):
        self.status = status
        self.filled_qty = filled_qty
        self.id = id
        self.filled_avg_price = filled_avg_price
        self.qty = qty
        self.legs = legs or []


def _make_ambiguous_order(*, status="submission_intent_pending", age_minutes=10, signal=True):
    with get_live_slc_session() as session:
        signal_id = None
        if signal:
            from live_slc.models import SlcSignalRecord
            rec = SlcSignalRecord(
                cycle_run_id="run-1", symbol="AAPL", level_id="demand:x", direction="long",
                level_state="fresh", level_low=95.0, level_high=99.0,
                level_active_time=pd.Timestamp("2026-08-13 13:00:00").to_pydatetime(),
                confirmation_time=pd.Timestamp("2026-08-13 14:00:00").to_pydatetime(),
                entry_time=pd.Timestamp("2026-08-13 14:05:00").to_pydatetime(),
                stop=94.0, stochastic_k=15.0, stochastic_d=12.0, atr14=1.0, structure="uptrend", impulse_atr=1.5,
            )
            session.add(rec)
            session.flush()
            session.refresh(rec)
            signal_id = rec.id
        row = SlcOrder(
            client_order_id="slc-test-entry", symbol="AAPL", leg="entry", side="buy",
            position_intent="buy_to_open", order_class="bracket", dry_run=False, qty=10.0,
            status=status, signal_id=signal_id,
            stop_submitted=93.0, target_submitted=107.0, expected_quote=100.0,
        )
        session.add(row)
        session.flush()
        session.refresh(row)
        row.submitted_at = pd.Timestamp.utcnow().tz_localize(None).to_pydatetime() - timedelta(minutes=age_minutes)
        session.add(row)
    return "slc-test-entry"


# -- Delayed reconciliation --------------------------------------------------

def test_order_younger_than_5_minutes_stays_ambiguous_and_makes_zero_broker_calls():
    _make_ambiguous_order(age_minutes=1)
    client = _SpyClient()

    def _fail(*a, **k):
        raise AssertionError("must not even attempt reconciliation on a too-young order")
    client.get_order_by_client_id = _fail
    client.get_orders = _fail

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["still_ambiguous"] == ["slc-test-entry"]
    assert client.mutating_calls == []


def test_filled_and_protected_order_becomes_a_normal_open_position():
    _make_ambiguous_order()
    client = _SpyClient()
    client.get_order_by_client_id = lambda oc_id: _FakeOrder(
        status="filled", filled_qty=10, qty=10, legs=[_FakeLeg("held")],
    )

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
        position = session.exec(select(SlcPosition)).first()
    assert order.status == "filled"
    assert position.status == "open"


def test_filled_and_unprotected_order_becomes_protected_degraded():
    _make_ambiguous_order()
    client = _SpyClient()
    client.get_order_by_client_id = lambda oc_id: _FakeOrder(
        status="filled", filled_qty=10, qty=10, legs=[_FakeLeg("canceled")],
    )

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
    assert position.status == "protected_degraded"


def test_partially_filled_order_becomes_ambiguous_quantity():
    _make_ambiguous_order()
    client = _SpyClient()
    client.get_order_by_client_id = lambda oc_id: _FakeOrder(
        status="partially_filled", filled_qty=4, qty=10, legs=[_FakeLeg("held")],
    )

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
    assert position.status == "ambiguous"


def test_confirmed_absent_requires_agreement_from_both_direct_lookup_and_history_scan():
    _make_ambiguous_order()
    client = _SpyClient()

    def _not_found(oc_id):
        raise APIError("not found", http_error=type("E", (), {"response": type("R", (), {"status_code": 404})()})())
    client.get_order_by_client_id = _not_found
    client.get_orders = lambda request: []

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []
    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
    assert order.status == "confirmed_no_order_resulted"


def test_confirmed_absent_but_history_scan_finds_it_is_not_treated_as_absent():
    """A single fresh 404 alone is insufficient evidence - Alpaca's own
    eventual-consistency lag can make a real order briefly invisible to
    the direct lookup."""
    _make_ambiguous_order()
    client = _SpyClient()

    def _not_found(oc_id):
        raise APIError("not found", http_error=type("E", (), {"response": type("R", (), {"status_code": 404})()})())
    client.get_order_by_client_id = _not_found
    found = _FakeOrder(status="filled", filled_qty=10, qty=10, legs=[_FakeLeg("held")])
    found.client_order_id = "slc-test-entry"
    client.get_orders = lambda request: [found]

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []
    with get_live_slc_session() as session:
        position = session.exec(select(SlcPosition)).first()
    assert position.status == "open"


def test_ambiguous_unreachable_direct_lookup_stays_ambiguous_never_scans_history():
    _make_ambiguous_order()
    client = _SpyClient()
    client.get_order_by_client_id = lambda oc_id: (_ for _ in ()).throw(TimeoutError("timed out"))

    def _fail_scan(request):
        raise AssertionError("must not scan history when the direct lookup was itself unreachable")
    client.get_orders = _fail_scan

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["still_ambiguous"] == ["slc-test-entry"]
    assert client.mutating_calls == []


def test_history_scan_failure_stays_ambiguous_not_treated_as_absence():
    _make_ambiguous_order()
    client = _SpyClient()

    def _not_found(oc_id):
        raise APIError("not found", http_error=type("E", (), {"response": type("R", (), {"status_code": 404})()})())
    client.get_order_by_client_id = _not_found

    def _scan_fails(request):
        raise TimeoutError("scan timed out")
    client.get_orders = _scan_fails

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["still_ambiguous"] == ["slc-test-entry"]
    assert client.mutating_calls == []
    with get_live_slc_session() as session:
        order = session.exec(select(SlcOrder)).first()
    assert order.status == "submission_intent_pending"  # untouched


def test_orphaned_order_with_no_signal_id_is_flagged_not_guessed():
    _make_ambiguous_order(signal=False)
    client = _SpyClient()
    client.get_order_by_client_id = lambda oc_id: _FakeOrder(status="filled", filled_qty=10, qty=10)

    result = run_slc_live._reconcile_ambiguous_orders(client)
    assert result["resolved"] == ["slc-test-entry"]
    assert client.mutating_calls == []
    with get_live_slc_session() as session:
        positions = list(session.exec(select(SlcPosition)))
    assert positions == []  # never guessed at a position identity


# -- Atomic split rebuild ----------------------------------------------------

def _seed_bootstrapped_symbol(symbol="AAPL"):
    with get_live_slc_session() as session:
        session.add(SlcReducerState(symbol=symbol, bootstrap_completed=True))
        session.add(SlcFiveMinBar(
            symbol=symbol, bar_time=pd.Timestamp("2026-08-13 13:30:00").to_pydatetime(),
            open=100.0, high=101.0, low=99.0, close=100.0, volume=1000.0,
        ))


def test_atomic_split_rebuild_failure_leaves_old_state_completely_untouched(monkeypatch):
    _seed_bootstrapped_symbol("AAPL")

    def _fetch_fails(symbols, start, end):
        raise RuntimeError("simulated broker outage")
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", _fetch_fails)

    evidence = SplitEvidence("AAPL", Decimal("0.25"), "corporate_actions", "4-for-1")
    ok = run_slc_live._atomic_split_rebuild("AAPL", evidence)
    assert ok is False

    with get_live_slc_session() as session:
        bars = list(session.exec(select(SlcFiveMinBar).where(SlcFiveMinBar.symbol == "AAPL")))
        state = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == "AAPL")).first()
    assert len(bars) == 1  # the original bar, never deleted
    assert state.bootstrap_completed is True


def test_atomic_split_rebuild_insufficient_bars_returns_false_and_touches_nothing(monkeypatch):
    _seed_bootstrapped_symbol("AAPL")
    idx = pd.date_range("2026-08-13 13:30:00", periods=5, freq="5min")
    small_frame = pd.DataFrame({"open": 25.0, "high": 25.0, "low": 25.0, "close": 25.0, "volume": 1.0}, index=idx)
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", lambda symbols, start, end: {"AAPL": small_frame})

    evidence = SplitEvidence("AAPL", Decimal("0.25"), "corporate_actions", "4-for-1")
    ok = run_slc_live._atomic_split_rebuild("AAPL", evidence)
    assert ok is False
    with get_live_slc_session() as session:
        bars = list(session.exec(select(SlcFiveMinBar).where(SlcFiveMinBar.symbol == "AAPL")))
    assert len(bars) == 1


def test_atomic_split_rebuild_success_replaces_bars_and_clears_split_pending(monkeypatch):
    _seed_bootstrapped_symbol("AAPL")
    with get_live_slc_session() as session:
        row = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == "AAPL")).first()
        row.split_pending = True
        session.add(row)

    idx = pd.date_range("2026-08-13 13:30:00", periods=run_slc_live.MIN_REBUILD_BARS + 10, freq="5min")
    big_frame = pd.DataFrame({"open": 25.0, "high": 25.0, "low": 25.0, "close": 25.0, "volume": 1.0}, index=idx)
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", lambda symbols, start, end: {"AAPL": big_frame})

    evidence = SplitEvidence("AAPL", Decimal("0.25"), "corporate_actions", "4-for-1")
    ok = run_slc_live._atomic_split_rebuild("AAPL", evidence)
    assert ok is True

    with get_live_slc_session() as session:
        bars = list(session.exec(select(SlcFiveMinBar).where(SlcFiveMinBar.symbol == "AAPL")))
        state = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == "AAPL")).first()
    assert len(bars) == len(idx)  # old single bar replaced entirely
    assert all(b.close == 25.0 for b in bars)
    assert state.split_pending is False


# -- Split detection orchestration -------------------------------------------

def test_split_pending_set_before_rebuild_attempted_and_stays_set_on_rebuild_failure(monkeypatch):
    """SlcReducerState.split_pending's own contract: True the instant
    validated evidence is found, cleared only on a successful rebuild -
    never silently left stale-scale-but-unblocked after a failed attempt."""
    _seed_bootstrapped_symbol("AAPL")
    import live_slc.split_detection as split_detection
    monkeypatch.setattr(split_detection, "corporate_action_split_evidence", lambda *a, **k: {})

    cached = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
                           index=pd.date_range("2026-08-13 13:30:00", periods=6, freq="5min"))
    fresh = cached / 4.0

    # First call (overlap window) returns a valid 4:1 ratio; the SECOND
    # call, made by _atomic_split_rebuild for the full 120-day rebuild,
    # fails - proving split_pending survives a failed rebuild attempt.
    calls = {"n": 0}

    def _fetch(symbols, start, end):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"AAPL": fresh}
        raise RuntimeError("simulated broker outage on the rebuild fetch")
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", _fetch)
    monkeypatch.setattr(run_slc_live, "_cached_bars", lambda symbol, start, end: cached)

    client = _SpyClient()
    result = run_slc_live._run_split_detection_and_rebuild(client, ["AAPL"])
    assert result["failed"] == ["AAPL"]
    assert client.mutating_calls == []

    with get_live_slc_session() as session:
        state = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == "AAPL")).first()
    assert state.split_pending is True  # still blocked - correctly not silently cleared
    assert state.last_split_check_at is not None


def test_not_yet_bootstrapped_symbols_are_never_split_candidates():
    with get_live_slc_session() as session:
        session.add(SlcReducerState(symbol="MSFT", bootstrap_completed=False))
    client = _SpyClient()
    result = run_slc_live._run_split_detection_and_rebuild(client, ["MSFT"])
    assert result == {"rebuilt": [], "conflicting": [], "failed": []}
    assert client.mutating_calls == []


def test_conflicting_evidence_flags_but_never_rebuilds(monkeypatch):
    _seed_bootstrapped_symbol("AAPL")
    import live_slc.split_detection as split_detection
    monkeypatch.setattr(
        split_detection, "corporate_action_split_evidence",
        lambda *a, **k: {"AAPL": SplitEvidence("AAPL", Decimal("0.5"), "corporate_actions", "2-for-1")},
    )
    cached = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
                           index=pd.date_range("2026-08-13 13:30:00", periods=6, freq="5min"))
    fresh = cached / 4.0  # 4:1 by price - conflicts with corp actions' 2:1
    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", lambda symbols, start, end: {"AAPL": fresh})
    monkeypatch.setattr(run_slc_live, "_cached_bars", lambda symbol, start, end: cached)

    client = _SpyClient()
    result = run_slc_live._run_split_detection_and_rebuild(client, ["AAPL"])
    assert result["conflicting"] == ["AAPL"]
    assert client.mutating_calls == []
    with get_live_slc_session() as session:
        state = session.exec(select(SlcReducerState).where(SlcReducerState.symbol == "AAPL")).first()
        bars = list(session.exec(select(SlcFiveMinBar).where(SlcFiveMinBar.symbol == "AAPL")))
    assert state.split_pending is False  # never set on a conflict - nothing validated
    assert len(bars) == 1  # untouched


# -- Engine-parity self-check: frozen validation-corpus source --------------
#
# The self-check used to sample the live universe via _cached_bars()
# (SlcFiveMinBar) - broken by construction immediately after a real
# bootstrap, since bootstrap() never persists historical bars into
# SlcFiveMinBar (see run_slc_live.py's rev. 12 comment above
# _run_engine_parity_self_check). It now reads exclusively from the
# frozen, hash-verified reducer validation corpus (research/
# slc_4h_5m_stock_v1_reducer_validation_corpus_manifest.json), completely
# independent of SlcFiveMinBar/_cached_bars.

import json as _json


def test_real_corpus_passes_using_the_actual_installed_code():
    """No mocking of check_engine_parity or its inputs at all - proves the
    real, currently-installed reducer genuinely agrees with the real,
    currently-installed frozen batch generate_signals() over the real
    corpus files, exactly like the daily preflight run will exercise."""
    assert run_slc_live._run_engine_parity_self_check() is True


def test_engine_parity_self_check_makes_zero_network_or_broker_calls(monkeypatch):
    """The frozen corpus lives entirely on local disk - loading and
    comparing it must never touch a broker/network client. No client is
    even constructed by this code path; asserting that bar_cache/
    execution are never reached is the strongest available proxy."""
    def _forbidden(*a, **k):
        raise AssertionError("engine-parity self-check must never fetch/call the broker")

    monkeypatch.setattr(run_slc_live.bar_cache, "_default_fetch", _forbidden)
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", _forbidden)
    assert run_slc_live._run_engine_parity_self_check() is True


def test_engine_parity_self_check_fails_closed_on_missing_manifest(monkeypatch):
    monkeypatch.setattr(run_slc_live, "REDUCER_CORPUS_MANIFEST_PATH", Path("/nonexistent/manifest.json"))
    assert run_slc_live._run_engine_parity_self_check() is False


def test_engine_parity_self_check_fails_closed_on_missing_fixture_file(tmp_path, monkeypatch):
    real_manifest = run_slc_live._load_reducer_corpus_manifest()
    tampered = _json.loads(_json.dumps(real_manifest))
    tampered["fixtures"]["AAPL"]["path"] = "tests/fixtures/slc_reducer_corpus/does_not_exist.csv"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(tampered), encoding="utf-8")
    monkeypatch.setattr(run_slc_live, "REDUCER_CORPUS_MANIFEST_PATH", manifest_path)
    assert run_slc_live._run_engine_parity_self_check() is False


def test_engine_parity_self_check_fails_closed_on_tampered_fixture_hash(tmp_path, monkeypatch):
    """A fixture whose bytes no longer match the frozen manifest hash
    (corrupted or maliciously edited) must never be loaded and compared -
    the hash check must happen before any parsing/comparison."""
    real_manifest = run_slc_live._load_reducer_corpus_manifest()
    fixture = real_manifest["fixtures"]["AAPL"]
    original_path = run_slc_live.REPO_ROOT / fixture["path"]

    tampered_csv_dir = tmp_path / "slc_reducer_corpus"
    tampered_csv_dir.mkdir()
    tampered_csv_path = tampered_csv_dir / "AAPL_tampered.csv"
    original_bytes = original_path.read_bytes()
    tampered_csv_path.write_bytes(original_bytes + b"\n# tampered")

    tampered_manifest = _json.loads(_json.dumps(real_manifest))
    # pytest's tmp_path is not guaranteed to live under the repository
    # (normal Windows runs use %TEMP%). Path joining deliberately accepts
    # an absolute right-hand operand, so this exercises the same hash-
    # mismatch path without assuming a workspace-local temp policy.
    tampered_manifest["fixtures"]["AAPL"]["path"] = str(tampered_csv_path)
    # sha256 left as the ORIGINAL (now-stale) hash - simulating drift/tampering.
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(_json.dumps(tampered_manifest), encoding="utf-8")
    monkeypatch.setattr(run_slc_live, "REDUCER_CORPUS_MANIFEST_PATH", manifest_path)
    assert run_slc_live._run_engine_parity_self_check() is False


def test_engine_parity_self_check_fails_closed_on_zero_compared_signals(monkeypatch):
    """matched=True with signal_count=0 for every fixture is a vacuous
    pass (both engines agreeing on nothing) - must not count as a
    meaningfully exercised check."""
    from live_slc.reducer import EngineParityResult
    monkeypatch.setattr(
        run_slc_live.reducer, "check_engine_parity",
        lambda symbol, bars, **kw: EngineParityResult(matched=True, signal_count=0),
    )
    assert run_slc_live._run_engine_parity_self_check() is False


def test_engine_parity_self_check_fails_closed_on_a_genuine_mismatch(monkeypatch):
    from live_slc.reducer import EngineParityResult
    monkeypatch.setattr(
        run_slc_live.reducer, "check_engine_parity",
        lambda symbol, bars, **kw: EngineParityResult(matched=False, signal_count=1),
    )
    assert run_slc_live._run_engine_parity_self_check() is False


# -- run_cycle() respects split_pending --------------------------------------

def test_run_cycle_excludes_split_pending_symbols_from_bar_ingestion(monkeypatch):
    with get_live_slc_session() as session:
        session.add(SlcReducerState(symbol="AAPL", bootstrap_completed=True, split_pending=True))

    class _AccountClient(_SpyClient):
        def get_account(self):
            return type("A", (), {"id": "acct-1"})()

    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: _AccountClient())
    monkeypatch.setattr(run_slc_live.guardrails, "assert_operational_preconditions",
                         lambda **kw: {"status": "not_authorized", "status_record": type("S", (), {"status": "not_authorized"})()})
    monkeypatch.setattr(run_slc_live, "is_trading_day", lambda day: True)

    seen_symbols = []

    def _fake_backfill(symbols, through):
        seen_symbols.extend(symbols)
        return {}
    monkeypatch.setattr(run_slc_live.bar_cache, "backfill_gaps", _fake_backfill)

    def _fake_fetch_expected(symbols, expected_bar_time, **kw):
        seen_symbols.extend(symbols)
        return {}, list(symbols)
    monkeypatch.setattr(run_slc_live.bar_cache, "fetch_expected_bar_batch", _fake_fetch_expected)

    run_slc_live.run_cycle()
    assert "AAPL" not in seen_symbols

    with get_live_slc_session() as session:
        from live_slc.models import SlcSignalRecord
        records = list(session.exec(select(SlcSignalRecord).where(SlcSignalRecord.symbol == "AAPL")))
    assert any(r.action_result == "split_pending_blocked" for r in records)


# -- Preflight timing telemetry (sleep-fix Part 2) ---------------------------
#
# Found during the sleep-related-coverage-loss review: preflight had zero
# SlcCycleRun rows at all, so there was no way to tell a late/failed
# preflight from one that simply predates instrumentation. run_preflight()
# now brackets its whole body the same way run_cycle()/run_closeout_stage()
# already do.

def _account_client(monkeypatch):
    class _AccountClient(_SpyClient):
        def get_account(self):
            return type("A", (), {"id": "acct-1"})()
    monkeypatch.setattr(run_slc_live.execution, "get_alpaca_client", lambda: _AccountClient())


def test_run_preflight_records_a_completed_cycle_run_on_success(monkeypatch):
    _account_client(monkeypatch)
    monkeypatch.setattr(
        run_slc_live.guardrails, "assert_operational_preconditions",
        lambda **kw: {"status": "paper_active"},
    )
    monkeypatch.setattr(run_slc_live, "UNIVERSE", [])
    monkeypatch.setattr(run_slc_live, "_run_split_detection_and_rebuild", lambda client, universe: {"rebuilt": [], "conflicting": [], "failed": []})
    monkeypatch.setattr(run_slc_live, "_reconcile_ambiguous_orders", lambda client: {"resolved": [], "still_ambiguous": []})
    monkeypatch.setattr(run_slc_live, "_run_engine_parity_self_check", lambda: True)

    run_slc_live.run_preflight()

    with get_live_slc_session() as session:
        runs = list(session.exec(select(models.SlcCycleRun)))
    assert len(runs) == 1
    assert runs[0].stage == "preflight"
    assert runs[0].status == "completed"
    assert runs[0].duration_seconds is not None


def test_run_preflight_records_failed_and_reraises_on_exception(monkeypatch):
    _account_client(monkeypatch)

    def _boom(**kw):
        raise RuntimeError("simulated guardrail failure")
    monkeypatch.setattr(run_slc_live.guardrails, "assert_operational_preconditions", _boom)

    with pytest.raises(RuntimeError, match="simulated guardrail failure"):
        run_slc_live.run_preflight()

    with get_live_slc_session() as session:
        runs = list(session.exec(select(models.SlcCycleRun)))
    assert len(runs) == 1
    assert runs[0].stage == "preflight"
    assert runs[0].status == "failed"
