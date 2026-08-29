"""Tests for nodes/risk_node.py - the _evaluate() gate logic."""
import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch


def _make_strat(entry=100.0, stop=95.0, target=110.0, rr=2.0, conviction=80):
    s = MagicMock()
    s.entry = entry
    s.stop = stop
    s.target = target
    s.rr = rr
    s.conviction_score = conviction
    s.bar_date = None
    return s


def _evaluate(strat, equity=100_000, open_positions=0, trades_today=0,
              daily_loss_breached=False, symbol="AAPL",
              ta_signal=None, held_symbols=None,
              broker_disconnected=False, inflight_check_failed=False,
              news_gate_blocked=False, news_gate_reason=""):
    from nodes.risk_node import _evaluate as ev
    # Gate-unit tests must not depend on today's live earnings calendar.
    with patch("nodes.risk_node._days_to_earnings", return_value=None):
        return ev(
            strat=strat,
            equity=equity,
            open_positions=open_positions,
            trades_today=trades_today,
            daily_loss_breached=daily_loss_breached,
            symbol=symbol,
            ta_signal=ta_signal,
            held_symbols=held_symbols,
            broker_disconnected=broker_disconnected,
            inflight_check_failed=inflight_check_failed,
            news_gate_blocked=news_gate_blocked,
            news_gate_reason=news_gate_reason,
        )


class TestDailyLossGate:
    def test_blocks_when_breached(self):
        result = _evaluate(_make_strat(), daily_loss_breached=True)
        assert not result["approved"]
        assert "daily loss" in result["reason"]

    def test_passes_when_not_breached(self):
        result = _evaluate(_make_strat(), daily_loss_breached=False)
        assert result["approved"]


class TestDuplicateSymbolGate:
    def test_blocks_already_held(self):
        result = _evaluate(_make_strat(), symbol="AAPL", held_symbols={"AAPL"})
        assert not result["approved"]
        assert "already holding" in result["reason"]

    def test_allows_different_symbol(self):
        result = _evaluate(_make_strat(), symbol="MSFT", held_symbols={"AAPL"})
        assert result["approved"]

    def test_case_insensitive(self):
        result = _evaluate(_make_strat(), symbol="aapl", held_symbols={"AAPL"})
        assert not result["approved"]

    def test_no_held_symbols_passes(self):
        result = _evaluate(_make_strat(), symbol="AAPL", held_symbols=None)
        assert result["approved"]

    def test_inflight_lookup_failure_rejects_every_strategy(self):
        result = _evaluate(_make_strat(), inflight_check_failed=True)
        assert not result["approved"]
        assert result["reason"] == (
            "duplicate-position check failed - rejecting all strategies this run"
        )


def test_ta_signal_lookup_failure_warns_with_symbol(capsys):
    from nodes.risk_node import _get_ta_signal

    with patch("nodes.risk_node.get_session", side_effect=RuntimeError("database offline")):
        assert _get_ta_signal("ticker-id", None, symbol="AAPL") is None

    output = capsys.readouterr().out
    assert "TradingAgents signal lookup failed for AAPL" in output
    assert "database offline" in output
    assert "WITHOUT TradingAgents veto protection" in output


def test_run_turns_inflight_db_failure_into_rejections(capsys):
    import nodes.risk_node as rn

    strat = _make_strat()
    strat.id = "strategy-1"
    strat.ticker_id = "ticker-1"
    account = {
        "connected": True,
        "equity": 100_000.0,
        "cash": 100_000.0,
        "available_cash": 100_000.0,
        "available_cash_valid": True,
        "available_cash_reason": None,
        "buying_power": 100_000.0,
        "non_marginable_buying_power": 100_000.0,
        "open_positions": 0,
        "trades_today": 0,
        "daily_pnl_pct": 0.0,
        "positions": [],
    }
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "init_db"), \
         patch.object(rn.settings, "ROBINHOOD_ENABLED", False), \
         patch.object(rn, "get_account_state", return_value=account), \
         patch.object(rn, "_db_inflight_symbols", side_effect=RuntimeError("DB locked")), \
         patch.object(rn, "news_gate_status", return_value=(None, False, "")), \
         patch.object(rn, "_load_buy_strategies", return_value=[strat]), \
         patch.object(rn, "_symbol_for", return_value="AAPL"), \
         patch.object(rn, "_already_evaluated", return_value=False), \
         patch.object(rn, "_get_ta_signal", return_value=None), \
         patch.object(rn, "_write_log"), \
         patch.object(rn, "get_session", return_value=mock_cm):
        result = rn.run(strategies_run_id="strategy-run-1")

    assert result["approved"] == []
    assert result["rejected"] == [{
        "symbol": "AAPL",
        "reason": "duplicate-position check failed - rejecting all strategies this run",
    }]
    assert "DUPLICATE-POSITION CHECK FAILED" in capsys.readouterr().out


def test_run_rejects_strategy_version_that_failed_evidence_gate():
    import nodes.risk_node as rn

    strat = _make_strat()
    strat.id = "strategy-locked"
    strat.ticker_id = "ticker-aapl"
    strat.model_used = "stock_trend_momentum_v1"
    account = {
        "connected": True,
        "equity": 100_000.0,
        "cash": 100_000.0,
        "available_cash": 100_000.0,
        "available_cash_valid": True,
        "available_cash_reason": None,
        "buying_power": 100_000.0,
        "non_marginable_buying_power": 100_000.0,
        "open_positions": 0,
        "trades_today": 0,
        "daily_pnl_pct": 0.0,
        "positions": [],
    }
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    otherwise_approved = {
        "approved": True,
        "shares": 10,
        "risk_amount": 1_000.0,
        "position_value": 1_000.0,
        "reason": "",
    }
    with patch.object(rn, "init_db"), \
         patch.object(rn.settings, "ROBINHOOD_ENABLED", False), \
         patch.object(rn, "get_account_state", return_value=account), \
         patch.object(rn, "_db_inflight_symbols", return_value=set()), \
         patch.object(rn, "news_gate_status", return_value=(None, False, "")), \
         patch.object(rn, "_load_buy_strategies", return_value=[strat]), \
         patch.object(rn, "_symbol_for", return_value="AAPL"), \
         patch.object(rn, "_already_evaluated", return_value=False), \
         patch.object(rn, "_get_ta_signal", return_value=None), \
         patch.object(rn, "_evaluate", return_value=otherwise_approved), \
         patch.object(rn, "_write_log"), \
         patch.object(rn, "get_session", return_value=mock_cm):
        result = rn.run(strategies_run_id="strategy-run-locked")

    assert result["approved"] == []
    assert result["rejected"][0]["symbol"] == "AAPL"
    assert "research-only" in result["rejected"][0]["reason"]


def test_run_sizes_from_available_cash_not_raw_cash():
    """risk_node must size from acct['available_cash'] (the fail-closed,
    margin-free budget), never acct['cash'] (raw Alpaca ledger cash) - a
    regression that reverted to acct['cash'] would silently re-permit
    sizing off an untrustworthy or overly-permissive number."""
    import nodes.risk_node as rn

    strat = _make_strat()
    strat.id = "strategy-avail-cash"
    strat.ticker_id = "ticker-1"
    account = {
        "connected": True,
        "equity": 100_000.0,
        "cash": 999_999.0,             # must NOT be used for sizing
        "available_cash": 12_345.0,    # must be used for sizing
        "available_cash_valid": True,
        "available_cash_reason": None,
        "buying_power": 400_000.0,
        "non_marginable_buying_power": 12_345.0,
        "open_positions": 0,
        "trades_today": 0,
        "daily_pnl_pct": 0.0,
        "positions": [],
    }
    otherwise_approved = {
        "approved": True, "shares": 10, "risk_amount": 1_000.0,
        "position_value": 1_000.0, "reason": "",
    }
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "init_db"), \
         patch.object(rn.settings, "ROBINHOOD_ENABLED", False), \
         patch.object(rn, "get_account_state", return_value=account), \
         patch.object(rn, "_db_inflight_symbols", return_value=set()), \
         patch.object(rn, "news_gate_status", return_value=(None, False, "")), \
         patch.object(rn, "_load_buy_strategies", return_value=[strat]), \
         patch.object(rn, "_symbol_for", return_value="AAPL"), \
         patch.object(rn, "_already_evaluated", return_value=False), \
         patch.object(rn, "_get_ta_signal", return_value=None), \
         patch.object(rn, "_evaluate", return_value=otherwise_approved) as mock_evaluate, \
         patch.object(rn, "_write_log"), \
         patch.object(rn, "get_session", return_value=mock_cm):
        rn.run(strategies_run_id="strategy-run-avail-cash")

    assert mock_evaluate.call_args.kwargs["cash"] == 12_345.0


def test_run_with_robinhood_enabled_sizes_from_cash_key_unchanged():
    """Robinhood's account dict (utils/robinhood_account.py) has no
    available_cash/buying_power keys - this must keep working exactly as
    before the Alpaca fail-closed sizing fix, with no KeyError."""
    import nodes.risk_node as rn

    strat = _make_strat()
    strat.id = "strategy-robinhood"
    strat.ticker_id = "ticker-1"
    robinhood_account = {
        "connected": True,
        "equity": 50_000.0,
        "cash": 7_777.0,
        "open_positions": 0,
        "trades_today": 0,
        "daily_pnl_pct": 0.0,
        "positions": [],
    }
    otherwise_approved = {
        "approved": True, "shares": 10, "risk_amount": 1_000.0,
        "position_value": 1_000.0, "reason": "",
    }
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "init_db"), \
         patch.object(rn.settings, "ROBINHOOD_ENABLED", True), \
         patch.object(rn, "get_robinhood_account_state", return_value=robinhood_account), \
         patch.object(rn, "_db_inflight_symbols", return_value=set()), \
         patch.object(rn, "news_gate_status", return_value=(None, False, "")), \
         patch.object(rn, "_load_buy_strategies", return_value=[strat]), \
         patch.object(rn, "_symbol_for", return_value="AAPL"), \
         patch.object(rn, "_already_evaluated", return_value=False), \
         patch.object(rn, "_get_ta_signal", return_value=None), \
         patch.object(rn, "_evaluate", return_value=otherwise_approved) as mock_evaluate, \
         patch.object(rn, "execution_block_reason", return_value=None), \
         patch.object(rn, "_write_log"), \
         patch.object(rn, "get_session", return_value=mock_cm):
        result = rn.run(strategies_run_id="strategy-run-robinhood")

    assert mock_evaluate.call_args.kwargs["cash"] == 7_777.0
    assert result["approved"] == ["AAPL"]


def test_print_account_tolerates_robinhood_shape(capsys):
    """_print_account() must not raise KeyError on Robinhood's dict shape,
    which has no available_cash/buying_power keys, and must not print the
    Alpaca-only lines for it."""
    import nodes.risk_node as rn

    robinhood_account = {
        "connected": True, "equity": 50_000.0, "cash": 7_777.0,
        "open_positions": 0, "trades_today": 0, "daily_pnl_pct": 0.0,
    }
    with patch.object(rn.settings, "DAILY_LOSS_LIMIT", 0.02):
        rn._print_account(robinhood_account)
    output = capsys.readouterr().out
    assert "Available cash" not in output
    assert "Buying power" not in output


def test_print_account_flags_invalid_available_cash(capsys):
    import nodes.risk_node as rn

    account = {
        "connected": True, "equity": 100_000.0, "cash": 100_000.0,
        "available_cash": 0.0, "available_cash_valid": False,
        "available_cash_reason": "cash or non_marginable_buying_power is missing or non-numeric",
        "buying_power": 0.0, "open_positions": 0, "trades_today": 0,
        "daily_pnl_pct": 0.0,
    }
    with patch.object(rn.settings, "DAILY_LOSS_LIMIT", 0.02):
        rn._print_account(account)
    output = capsys.readouterr().out
    assert "CAPITAL DATA INVALID" in output
    assert "sizing disabled" in output


def test_resolved_flat_order_no_longer_blocks_symbol():
    import nodes.risk_node as rn

    position_result = MagicMock()
    position_result.all.return_value = []
    order_result = MagicMock()
    order_result.all.return_value = [
        MagicMock(status="resolved_flat", symbol="SOL-USD")
    ]
    mock_session = MagicMock()
    mock_session.exec.side_effect = [position_result, order_result]
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "get_session", return_value=mock_cm):
        assert rn._db_inflight_symbols() == set()


def test_stale_pending_new_order_with_closed_position_no_longer_blocks_symbol():
    import nodes.risk_node as rn

    closed_position = MagicMock(
        order_id="order-1", ticker_id="ticker-crm", status="closed"
    )
    stale_order = MagicMock(
        id="order-1", status="OrderStatus.PENDING_NEW", symbol="CRM"
    )
    position_result = MagicMock()
    position_result.all.return_value = [closed_position]
    order_result = MagicMock()
    order_result.all.return_value = [stale_order]
    mock_session = MagicMock()
    mock_session.exec.side_effect = [position_result, order_result]
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "get_session", return_value=mock_cm):
        assert rn._db_inflight_symbols() == set()


def test_standalone_pending_new_order_still_blocks_symbol():
    import nodes.risk_node as rn

    position_result = MagicMock()
    position_result.all.return_value = []
    order_result = MagicMock()
    order_result.all.return_value = [
        MagicMock(id="order-1", status="OrderStatus.PENDING_NEW", symbol="CRM")
    ]
    mock_session = MagicMock()
    mock_session.exec.side_effect = [position_result, order_result]
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    with patch.object(rn, "get_session", return_value=mock_cm):
        assert rn._db_inflight_symbols() == {"CRM"}


class TestRRGate:
    def test_blocks_low_rr(self):
        result = _evaluate(_make_strat(rr=1.5))
        assert not result["approved"]
        assert "R:R" in result["reason"]

    def test_passes_exact_minimum(self):
        # MIN_RISK_REWARD is 2.0 from settings
        result = _evaluate(_make_strat(rr=2.0))
        assert result["approved"]


class TestConvictionGate:
    def test_blocks_low_conviction(self):
        result = _evaluate(_make_strat(conviction=30))
        assert not result["approved"]
        assert "conviction" in result["reason"]


class TestPositionCountGate:
    def test_blocks_at_max(self):
        result = _evaluate(_make_strat(), open_positions=5)
        assert not result["approved"]
        assert "max positions" in result["reason"]

    def test_passes_below_max(self):
        result = _evaluate(_make_strat(), open_positions=4)
        assert result["approved"]


class TestEntryStopValidity:
    def test_blocks_entry_below_stop(self):
        result = _evaluate(_make_strat(entry=90.0, stop=95.0))
        assert not result["approved"]
        assert "entry must be above stop" in result["reason"]

    def test_blocks_zero_entry(self):
        result = _evaluate(_make_strat(entry=0.0, stop=0.0))
        assert not result["approved"]


class TestTargetValidityGate:
    def test_blocks_missing_target(self):
        result = _evaluate(_make_strat(target=None))
        assert not result["approved"]
        assert "target" in result["reason"]

    def test_blocks_target_below_entry(self):
        result = _evaluate(_make_strat(entry=100.0, stop=95.0, target=100.0))
        assert not result["approved"]
        assert "target must be above entry" in result["reason"]

    def test_passes_valid_target(self):
        result = _evaluate(_make_strat(entry=100.0, stop=95.0, target=110.0))
        assert result["approved"]


class TestBrokerDisconnectedGate:
    def test_blocks_when_disconnected(self):
        result = _evaluate(_make_strat(), broker_disconnected=True)
        assert not result["approved"]
        assert "broker disconnected" in result["reason"]

    def test_passes_when_connected(self):
        result = _evaluate(_make_strat(), broker_disconnected=False)
        assert result["approved"]


class TestNewsGateBlock:
    def test_blocks_when_gate_blocked(self):
        result = _evaluate(_make_strat(), news_gate_blocked=True,
                           news_gate_reason="morning news report is missing for today")
        assert not result["approved"]
        assert "news gate" in result["reason"]
        assert "missing" in result["reason"]

    def test_passes_when_gate_not_blocked(self):
        result = _evaluate(_make_strat(), news_gate_blocked=False)
        assert result["approved"]


class TestPositionSizing:
    def test_shares_calculated(self):
        # equity=$100k, risk=1%, entry=100, stop=95 -> risk=$1000 / $5 = 200 shares
        result = _evaluate(_make_strat(entry=100.0, stop=95.0), equity=100_000)
        assert result["approved"]
        assert result["shares"] == 200

    def test_concentration_cap(self):
        # $200k equity, 1% risk, entry=100, stop=99 -> 2000 shares x $100 = $200k (>20% cap)
        # Should be capped to 20% = $40k / $100 = 400 shares
        result = _evaluate(_make_strat(entry=100.0, stop=99.0), equity=200_000)
        assert result["approved"]
        assert result["shares"] <= 400

    def test_crypto_uses_fractional_quantity_and_skips_stock_price_floor(self):
        result = _evaluate(
            _make_strat(
                entry=0.000010,
                stop=0.000009,
                target=0.000012,
                rr=2.0,
            ),
            symbol="PEPE-USD",
            equity=100_000,
        )
        assert result["approved"]
        assert result["shares"] > 1
        assert result["position_value"] <= 20_000

    def test_untradable_crypto_is_rejected(self):
        from nodes.risk_node import _evaluate as ev
        result = ev(
            strat=_make_strat(entry=100, stop=95, target=110),
            equity=100_000,
            open_positions=0,
            trades_today=0,
            daily_loss_breached=False,
            symbol="BTC-USD",
            crypto_tradable=False,
        )
        assert not result["approved"]
        assert "not currently tradable" in result["reason"]


class TestTASignalVeto:
    def test_bearish_signal_blocks(self):
        result = _evaluate(_make_strat(), ta_signal="Sell")
        assert not result["approved"]

    def test_underweight_blocks(self):
        result = _evaluate(_make_strat(), ta_signal="Underweight")
        assert not result["approved"]

    def test_hold_doesnt_block(self):
        result = _evaluate(_make_strat(), ta_signal="Hold")
        assert result["approved"]

    def test_buy_signal_passes(self):
        result = _evaluate(_make_strat(), ta_signal="Buy")
        assert result["approved"]


class TestBuyingPowerGate:
    """Position size must never exceed available cash (symbol='' skips the
    network earnings lookup so these stay hermetic)."""

    def _ev(self, **kw):
        from nodes.risk_node import _evaluate as ev
        base = dict(strat=_make_strat(entry=100.0, stop=95.0, rr=2.0, conviction=80),
                    equity=100_000, open_positions=0, trades_today=0,
                    daily_loss_breached=False, symbol="")
        base.update(kw)
        return ev(**base)

    def test_scales_down_to_available_cash(self):
        # Risk would buy 200 shares ($20k), but only $10k cash -> 100 shares
        result = self._ev(cash=10_000)
        assert result["approved"]
        assert result["shares"] == 100
        assert result["position_value"] <= 10_000

    def test_rejects_when_cash_below_one_share(self):
        result = self._ev(cash=50)
        assert not result["approved"]
        assert "buying power" in result["reason"]

    def test_risk_amount_reflects_scaled_size(self):
        # 100 shares * $5 risk/share = $500, not the $1,000 pre-scale budget
        result = self._ev(cash=10_000)
        assert result["risk_amount"] == 500

    def test_unlimited_cash_keeps_full_size(self):
        result = self._ev()  # cash defaults to inf
        assert result["approved"]
        assert result["shares"] == 200
