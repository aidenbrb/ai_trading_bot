"""
Tests for live_slc.check_paper_session_audit - the end-of-day audit for a
real (non-dry-run) paper_active session.
"""
from datetime import date, datetime

import pytest

import live_slc.check_paper_session_audit as audit
import live_slc.models as models
from live_slc.models import SlcAuditEvent, SlcOrder, SlcPosition, SlcSessionStat, SlcTrade, get_live_slc_session

TARGET_DATE = date(2026, 8, 18)
TS = datetime(2026, 8, 18, 14, 0, 0)


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


class _SpyClient:
    """Records every method called. get_all_positions() returns the
    configured fixture; every other method raises, proving the audit
    never reaches for a broker mutation."""

    def __init__(self, *, positions=None, raise_on_positions=False):
        self.calls = []
        self._positions = positions if positions is not None else []
        self._raise_on_positions = raise_on_positions

    def get_all_positions(self):
        self.calls.append("get_all_positions")
        if self._raise_on_positions:
            raise RuntimeError("broker unreachable")
        return self._positions

    def __getattr__(self, name):
        def _forbidden(*a, **k):
            self.calls.append(name)
            raise AssertionError(f"check_paper_session_audit must never call {name}")
        return _forbidden


class _FakeBrokerPosition:
    def __init__(self, symbol, qty):
        self.symbol = symbol
        self.qty = qty


def _seed_order(**overrides):
    defaults = dict(
        client_order_id=f"slc-{overrides.get('symbol', 'AAPL')}-{overrides.get('status', 'filled')}",
        symbol="AAPL", leg="entry", side="buy", position_intent="buy_to_open",
        qty=10.0, order_class="bracket", status="filled", dry_run=False,
        submitted_at=TS, alpaca_order_id="broker-order-1",
        fill_price=100.5, expected_quote=100.0,
    )
    defaults.update(overrides)
    with get_live_slc_session() as session:
        session.add(SlcOrder(**defaults))


def _seed_position(**overrides):
    defaults = dict(
        symbol="AAPL", direction="long", qty=10.0, entry_price=100.5,
        stop_price=98.0, target_price=105.0, status="closed",
        session_date=TARGET_DATE, protective_order_id="protect-1",
    )
    defaults.update(overrides)
    with get_live_slc_session() as session:
        session.add(SlcPosition(**defaults))


def _seed_trade(**overrides):
    defaults = dict(
        position_id="pos-1", symbol="AAPL", direction="long",
        entry_price=100.5, exit_price=103.5, exit_time=TS, exit_reason="target",
        qty=10.0, gross_pnl=30.0, net_pnl=30.0, pnl_r=2.0, session_date=TARGET_DATE,
    )
    defaults.update(overrides)
    with get_live_slc_session() as session:
        session.add(SlcTrade(**defaults))


def _seed_audit_event(event_type, **overrides):
    defaults = dict(event_type=event_type, symbol="AAPL", occurred_at=TS)
    defaults.update(overrides)
    with get_live_slc_session() as session:
        session.add(SlcAuditEvent(**defaults))


def _seed_session_stat(**overrides):
    defaults = dict(session_date=TARGET_DATE, closeout_confirmed_flat_by_broker_readback=True)
    defaults.update(overrides)
    with get_live_slc_session() as session:
        session.add(SlcSessionStat(**defaults))


# -- clean scenarios ----------------------------------------------------------

def test_clean_no_trade_day_passes():
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is True
    assert result["orders"]["count"] == 0
    assert result["positions"]["count"] == 0


def test_clean_filled_and_closed_trade_passes_with_correct_derived_slippage():
    _seed_order(status="filled", fill_price=100.75, expected_quote=100.50)
    _seed_position(status="closed", protective_order_id="protect-1")
    _seed_trade()
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is True
    slippage_values = list(result["orders"]["slippage_derived"].values())
    assert slippage_values == [pytest.approx(0.25)]


# -- entries-per-day, informational only --------------------------------------

def test_more_than_two_entries_flagged_informationally_not_a_failure():
    for i in range(3):
        _seed_order(symbol=f"SYM{i}", client_order_id=f"slc-entry-{i}", leg="entry", status="filled")
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["orders"]["entries_today"] == 3
    assert result["orders"]["entries_exceed_daily_cap"] is True
    assert result["passed"] is True  # informational only - does not fail the audit


# -- dry_run rows during paper_active -----------------------------------------

def test_dry_run_order_during_paper_active_day_is_flagged():
    _seed_order(status="dry_run_proposed", dry_run=True, alpaca_order_id=None)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert any("dry_run=True" in issue for issue in result["orders"]["issues"])


# -- status-aware alpaca_order_id requirement ---------------------------------

def test_missing_order_id_on_filled_status_is_flagged():
    _seed_order(status="filled", alpaca_order_id=None)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert any("missing alpaca_order_id" in issue for issue in result["orders"]["issues"])


@pytest.mark.parametrize("status", ["confirmed_rejected", "confirmed_no_order_resulted"])
def test_missing_order_id_on_id_optional_status_is_not_flagged(status):
    _seed_order(status=status, alpaca_order_id=None, fill_price=None, expected_quote=None)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert not any("missing alpaca_order_id" in issue for issue in result["orders"]["issues"])


def test_blocking_status_order_always_flagged_regardless_of_id():
    _seed_order(status="ambiguous_submission", alpaca_order_id="broker-order-9")
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert any("still unresolved" in issue for issue in result["orders"]["issues"])


# -- position status ------------------------------------------------------------

@pytest.mark.parametrize("status", ["ambiguous", "protected_degraded"])
def test_ambiguous_or_protected_degraded_position_is_flagged(status):
    _seed_position(status=status)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert any(status in issue for issue in result["positions"]["issues"])


def test_open_position_with_no_protective_order_id_is_flagged():
    _seed_position(status="open", protective_order_id=None)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert any("no protective_order_id" in issue for issue in result["positions"]["issues"])


# -- audit-event classification: critical vs benign ---------------------------

_ALL_EVENT_TYPES = [
    "protected_degraded", "ambiguous_quantity", "discovered_fill_unresolvable_identity",
    "orphan_broker_position_unresolved", "split_evidence_conflict", "split_rebuild_failed",
    "split_rebuild_applied", "orphan_broker_position_adopted",
]
_CRITICAL = {
    "protected_degraded", "ambiguous_quantity", "discovered_fill_unresolvable_identity",
    "orphan_broker_position_unresolved", "split_evidence_conflict", "split_rebuild_failed",
}
_BENIGN = {"split_rebuild_applied", "orphan_broker_position_adopted"}


def test_critical_events_fail_benign_events_do_not_and_all_are_reported():
    for event_type in _ALL_EVENT_TYPES:
        _seed_audit_event(event_type)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)

    assert result["passed"] is False  # the critical events alone must fail it
    assert set(result["audit_events"]["critical"]) == _CRITICAL
    reported_types = {e["event_type"] for e in result["audit_events"]["all"]}
    assert reported_types == set(_ALL_EVENT_TYPES)  # every event reported, benign included


def test_benign_only_events_never_fail_the_audit():
    for event_type in _BENIGN:
        _seed_audit_event(event_type)
    _seed_session_stat()
    client = _SpyClient(positions=[])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is True
    assert set(result["audit_events"]["critical"]) == set()


# -- broker read failure / non-flat -------------------------------------------

def test_broker_read_failure_fails_closed():
    _seed_session_stat()
    client = _SpyClient(raise_on_positions=True)
    with pytest.raises(RuntimeError, match="broker unreachable"):
        audit.run_audit(client, TARGET_DATE)


def test_broker_still_shows_nonzero_position_is_flagged_even_if_closeout_confirmed_flat():
    _seed_order(symbol="AAPL", status="filled")
    _seed_session_stat(closeout_confirmed_flat_by_broker_readback=True)
    client = _SpyClient(positions=[_FakeBrokerPosition("AAPL", 5.0)])
    result = audit.run_audit(client, TARGET_DATE)
    assert result["passed"] is False
    assert result["broker_flat"]["closeout_confirmed"] is True
    assert result["broker_flat"]["independently_corroborated"] is False
    assert "AAPL" in result["broker_flat"]["still_nonzero_symbols"]


# -- spy-client proof: zero mutating calls -------------------------------------

def test_only_get_all_positions_is_ever_called_across_every_scenario():
    _seed_order(status="filled")
    _seed_position(status="open", protective_order_id=None)
    _seed_audit_event("protected_degraded")
    _seed_session_stat()
    client = _SpyClient(positions=[_FakeBrokerPosition("AAPL", 3.0)])
    audit.run_audit(client, TARGET_DATE)
    assert client.calls == ["get_all_positions"]
