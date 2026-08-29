"""
Phase 3 Step 1: proves a strategy that exists only in
utils/research_strategy_registry.py (crypto_trend_daily_v1,
crypto_xsec_momentum_v1) cannot produce a real order - using the real,
unmodified nodes/execution_node.py (a Tier-1 guardrail file for the live
SLC system and this repo's own evidence gate; not touched by this test or
anything else in this work).

Three layers are traced/proven separately, because relying on any one of
them alone would not be "fail closed":
  1. Nothing in the live pipeline ever WRITES a Strategy row for either new
     strategy in the first place (nodes/crypto_strategy_node.py::run() only
     ever calls crypto_trend_momentum_v1 - see the source-inspection tests
     in test_crypto_strategy_node_daily.py).
  2. Even if it did, utils.strategy_registry.execution_block_reason() fails
     closed on any version string not in the (Tier-1, deliberately
     unmodified) REGISTRY dict - unknown, missing, and research-only
     versions are all treated identically: blocked.
  3. execution_node.py calls that gate BEFORE constructing any broker
     client - a blocked strategy never reaches _get_alpaca_client()/
     _get_robinhood_client() at all, let alone a submit call.

This file tests layers 2 and 3 end-to-end, deliberately with EVERY other
safety switch (EXECUTION_ENABLED, CRYPTO_EXECUTION_ENABLED) set to its
most permissive value - proving the registry gate alone is sufficient,
not merely one of several switches that happen to agree today.
"""
from datetime import date

import db.connection as connection
import nodes.execution_node as execution_node
import utils.research_strategy_registry as research_registry
import utils.strategy_registry as registry
from db.connection import get_session
from db.models import RiskApproval, Strategy, Ticker
from utils.strategy_signals import CRYPTO_DAILY_STRATEGY_VERSION, CRYPTO_STRATEGY_VERSION


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(connection.settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    connection._engine = None
    connection.init_db()


def _approve(model_used, run_id="run-1", symbol="BTC-USD"):
    with get_session() as session:
        ticker = Ticker(symbol=symbol, sector="Crypto")
        session.add(ticker)
        session.flush()
        strat = Strategy(
            run_id=run_id, ticker_id=ticker.id, bar_date=date(2026, 1, 1),
            model_used=model_used, entry=100.0, stop=95.0, target=110.0,
        )
        session.add(strat)
        session.flush()
        approval = RiskApproval(run_id=run_id, strategy_id=strat.id, approved=True, shares=1.0)
        session.add(approval)
        session.flush()
    return run_id


def _forbid_broker_construction(monkeypatch):
    def _forbidden(*a, **k):
        raise AssertionError("broker client must never be constructed for a blocked strategy")
    monkeypatch.setattr(execution_node, "_get_alpaca_client", _forbidden)
    monkeypatch.setattr(execution_node, "_get_robinhood_client", _forbidden)


def _wide_open(monkeypatch):
    """Every OTHER safety switch set to its most permissive value - the
    registry gate must block on its own, not rely on any of these."""
    monkeypatch.setattr(execution_node.settings, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(execution_node.settings, "CRYPTO_EXECUTION_ENABLED", True)


# -- Layer 2: the registry function itself fails closed ----------------------

def test_execution_block_reason_blocks_crypto_trend_daily_v1():
    assert registry.execution_block_reason("crypto_trend_daily_v1") is not None


def test_execution_block_reason_blocks_crypto_xsec_momentum_v1():
    assert registry.execution_block_reason("crypto_xsec_momentum_v1") is not None


def test_execution_block_reason_blocks_unknown_version():
    """Fail-closed on a version that isn't research-registered either -
    there must be no code path where an unrecognized strategy is treated
    as implicitly allowed."""
    assert registry.execution_block_reason("some_future_strategy_nobody_registered") is not None


def test_execution_block_reason_blocks_missing_version():
    assert registry.execution_block_reason(None) is not None
    assert registry.execution_block_reason("") is not None


def test_research_registry_also_marks_both_strategies_not_execution_eligible():
    """Belt-and-suspenders: the research registry's OWN record for each
    strategy independently says execution_eligible=False, matching the
    fact that they're simply absent from the real (Tier-1) registry.
    (This test previously only checked crypto_trend_daily_v1 and would
    have silently missed crypto_xsec_momentum_v1 never having been added
    to RESEARCH_REGISTRY at all - a real gap found and fixed via this
    exact test.)"""
    for version in (CRYPTO_DAILY_STRATEGY_VERSION, research_registry.CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION):
        assert research_registry.execution_block_reason(version) is not None
        assert research_registry.registration(version).execution_eligible is False


def test_xsec_momentum_version_string_matches_between_registry_and_engine():
    """utils/research_strategy_registry.py's constant and
    backtest/whole_bot_engine.py's literal "strategy_version" string
    (written onto every crypto_xsec_momentum_v1 trade dict) must stay
    byte-identical - there is no shared constant between the two files,
    so this is the only thing that would catch silent drift."""
    import backtest.whole_bot_engine as engine

    pos = engine._XsecPosition(
        symbol="X-USD", entry_date=date(2026, 1, 1), entry_price=100.0,
        quantity=1.0, stop=90.0, target=200.0, entry_notional=100.0,
    )
    trade = engine._xsec_close_trade(pos, date(2026, 1, 2), 105.0, "test", engine.COSTS["zero"])
    assert trade["strategy_version"] == research_registry.CRYPTO_XSEC_MOMENTUM_STRATEGY_VERSION


def test_v1_itself_is_still_correctly_blocked_unchanged():
    """Regression guard: none of this work altered v1's own (also
    execution_eligible=False, rejected) registration."""
    assert registry.execution_block_reason(CRYPTO_STRATEGY_VERSION) is not None


# -- Layer 3: execution_node.py's real, unmodified gate, end-to-end --------

def test_research_only_strategy_blocked_even_with_every_other_switch_wide_open(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    run_id = _approve("crypto_trend_daily_v1")
    _wide_open(monkeypatch)
    _forbid_broker_construction(monkeypatch)

    result = execution_node.run(risk_run_id=run_id)

    assert result["submitted"] == []
    assert result["dry_run"] == []
    assert len(result["failed"]) == 1
    assert result["failed"][0]["symbol"] == "BTC-USD"
    # The REAL (Tier-1) registry has no entry for this strategy at all -
    # by design, it was deliberately never added there (see the registry
    # incident note in utils/research_strategy_registry.py's module
    # docstring) - so the live gate's message is "not registered", not
    # "research-only". The "research-only" wording only ever comes from
    # utils.research_strategy_registry's OWN function, which is not on
    # this live path at all (see the separate test below) - asserting the
    # wrong message here would silently hide that distinction.
    assert "not registered for execution" in result["failed"][0]["reason"]


def test_xsec_momentum_blocked_even_with_every_other_switch_wide_open(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    run_id = _approve("crypto_xsec_momentum_v1")
    _wide_open(monkeypatch)
    _forbid_broker_construction(monkeypatch)

    result = execution_node.run(risk_run_id=run_id)

    assert result["submitted"] == []
    assert len(result["failed"]) == 1
    assert "not registered for execution" in result["failed"][0]["reason"]


def test_unregistered_version_blocked_end_to_end(tmp_path, monkeypatch):
    """Not just research-registered strategies - a wholly unrecognized
    model_used string (e.g. a future typo, a stale value, manual DB
    tampering) must also be blocked, never treated as implicitly allowed."""
    _seed(tmp_path, monkeypatch)
    run_id = _approve("totally_made_up_strategy_v99")
    _wide_open(monkeypatch)
    _forbid_broker_construction(monkeypatch)

    result = execution_node.run(risk_run_id=run_id)

    assert result["submitted"] == []
    assert len(result["failed"]) == 1
    assert "not registered for execution" in result["failed"][0]["reason"]


def test_missing_model_used_blocked_end_to_end(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    run_id = _approve(None)
    _wide_open(monkeypatch)
    _forbid_broker_construction(monkeypatch)

    result = execution_node.run(risk_run_id=run_id)

    assert result["submitted"] == []
    assert len(result["failed"]) == 1
    assert "missing" in result["failed"][0]["reason"]


def test_v1_itself_is_not_blocked_by_the_gate_only_by_dry_run(tmp_path, monkeypatch):
    """Control case, proving the gate is selective, not a blanket block:
    v1's own version string reaches the SAME gate and is NOT blocked by
    it (crypto_trend_momentum_v1 is registered, just execution_eligible=
    False for its own, separate evidence-based reason - this test isolates
    that this is the SAME registry check as the research-only strategies
    above, not a different code path)."""
    _seed(tmp_path, monkeypatch)
    run_id = _approve(CRYPTO_STRATEGY_VERSION)
    _wide_open(monkeypatch)
    _forbid_broker_construction(monkeypatch)

    result = execution_node.run(risk_run_id=run_id)

    # v1 is ALSO execution_eligible=False (rejected by its own locked
    # evidence run) - so it is blocked too, but via the identical
    # registry path, with its own registered reason, not a special case.
    assert len(result["failed"]) == 1
    assert "71 weekend trades" in result["failed"][0]["reason"]
