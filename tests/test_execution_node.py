"""Tests for nodes/execution_node.py - the missing-price crash fix and the
ambiguous-order-outcome reconciliation added for Alpaca disconnection safety."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch
import pytest

import nodes.execution_node as en
from nodes.execution_node import (
    AmbiguousOrderState,
    ConfirmedOrderFailure,
    _alpaca_emergency_flatten,
    _execute_one,
    _alpaca_crypto_symbol,
    _decimal_down,
    _looks_ambiguous,
    _submit_alpaca_crypto_protected,
    _try_reconcile_by_client_order_id,
)


def _alpaca_crypto_position(qty="0.01", symbol="BTCUSD"):
    return MagicMock(
        symbol=symbol,
        qty=qty,
        asset_id="crypto-asset-1",
    )


def test_crypto_symbol_translation():
    assert _alpaca_crypto_symbol("BTC-USD") == "BTC/USD"


def test_decimal_rounds_down_to_exchange_increment():
    assert _decimal_down(1.234567, 0.001) == "1.234"


def test_crypto_entry_is_followed_by_protective_stop():
    broker = MagicMock()
    broker.get_asset.return_value = MagicMock(
        tradable=True,
        min_trade_increment=1e-9,
        min_order_size=0.00001,
        price_increment=1e-9,
    )
    buy = MagicMock(id="buy-1")
    stop = MagicMock(id="stop-1", status="accepted")
    broker.submit_order.side_effect = [buy, stop]
    broker.get_order_by_id.return_value = MagicMock(
        status="filled",
        filled_qty="0.01",
        filled_avg_price="64000",
    )
    broker.get_all_positions.side_effect = [[], [_alpaca_crypto_position()]]

    result = _submit_alpaca_crypto_protected(
        broker=broker,
        symbol="BTC-USD",
        qty=0.01,
        entry=64000,
        stop=62000,
        target=68000,
        client_order_id="appr-test",
    )

    assert result["filled"] is True
    assert result["fill_price"] == 64000
    assert broker.submit_order.call_count == 2
    assert broker.submit_order.call_args_list[0].args[0].symbol == "BTC/USD"
    assert broker.submit_order.call_args_list[1].args[0].side.value == "sell"
    assert float(broker.submit_order.call_args_list[1].args[0].qty) == 0.01
    assert result["position_qty"] == 0.01


def test_crypto_stop_uses_post_fee_available_balance():
    broker = MagicMock()
    broker.get_asset.return_value = MagicMock(
        tradable=True,
        min_trade_increment=1e-9,
        min_order_size=0.00001,
        price_increment=0.001,
    )
    broker.submit_order.side_effect = [
        MagicMock(id="buy-sol"),
        MagicMock(id="stop-sol", status="accepted"),
    ]
    broker.get_order_by_id.return_value = MagicMock(
        status="filled",
        filled_qty="295.172570199",
        filled_avg_price="74.872141287",
    )
    net_position = _alpaca_crypto_position(
        qty="294.434638773",
        symbol="SOLUSD",
    )
    broker.get_all_positions.side_effect = [[], [net_position]]

    result = _submit_alpaca_crypto_protected(
        broker=broker,
        symbol="SOL-USD",
        qty=295.172570199,
        entry=74.62,
        stop=73.64,
        target=76.57,
        client_order_id="appr-sol-fee-test",
    )

    stop_request = broker.submit_order.call_args_list[1].args[0]
    assert stop_request.symbol == "SOL/USD"
    assert float(stop_request.qty) == 294.434638773
    assert result["position_qty"] == 294.434638773
    assert result["position_id"] == "crypto-asset-1"


def test_crypto_unprotected_and_unflattened_raises_known_ambiguous_state():
    broker = MagicMock()
    broker.get_asset.return_value = MagicMock(
        tradable=True,
        min_trade_increment=1e-9,
        min_order_size=0.00001,
        price_increment=1e-9,
    )
    broker.submit_order.side_effect = [
        MagicMock(id="buy-1"),
        RuntimeError("insufficient buying power for protective stop"),
    ]
    broker.get_order_by_id.return_value = MagicMock(
        status="filled",
        filled_qty="0.01",
        filled_avg_price="64000",
    )
    broker.get_order_by_client_id.return_value = None
    broker.close_position.side_effect = RuntimeError("flatten rejected")
    position = _alpaca_crypto_position()
    broker.get_all_positions.side_effect = [[], [position], [position], [position]]

    with pytest.raises(AmbiguousOrderState, match="insufficient buying power"):
        _submit_alpaca_crypto_protected(
            broker=broker,
            symbol="BTC-USD",
            qty=0.01,
            entry=64000,
            stop=62000,
            target=68000,
            client_order_id="appr-test",
        )

    broker.close_position.assert_called_once_with("crypto-asset-1")


def test_emergency_flatten_uses_asset_id_and_confirms_flat():
    broker = MagicMock()
    position = _alpaca_crypto_position(qty="2.5", symbol="SOLUSD")
    broker.get_all_positions.side_effect = [[position], []]
    broker.close_position.return_value = MagicMock(id="close-1")

    assert _alpaca_emergency_flatten(broker, "SOL/USD") is True
    broker.close_position.assert_called_once_with("crypto-asset-1")


def test_crypto_canceled_with_zero_fill_is_confirmed_failure():
    broker = MagicMock()
    broker.get_asset.return_value = MagicMock(
        tradable=True,
        min_trade_increment=1e-9,
        min_order_size=1.0,
        price_increment=1e-8,
    )
    broker.get_all_positions.return_value = []
    broker.submit_order.return_value = MagicMock(id="buy-wif")
    broker.get_order_by_id.return_value = MagicMock(
        status="canceled",
        filled_qty="0",
        filled_avg_price=None,
    )

    with pytest.raises(ConfirmedOrderFailure, match="canceled with zero fill"):
        _submit_alpaca_crypto_protected(
            broker=broker,
            symbol="WIF-USD",
            qty=151202,
            entry=0.146,
            stop=0.1432,
            target=0.1515,
            client_order_id="appr-wif-test",
        )

    broker.cancel_order_by_id.assert_called_once_with("buy-wif")
    broker.close_position.assert_not_called()


class TestLooksAmbiguous:
    def test_network_style_error_is_ambiguous(self):
        assert _looks_ambiguous(Exception("connection reset by peer")) is True

    def test_timeout_is_ambiguous(self):
        assert _looks_ambiguous(Exception("Read timed out")) is True

    def test_422_validation_error_is_not_ambiguous(self):
        assert _looks_ambiguous(Exception("422 Unprocessable: invalid qty")) is False

    def test_insufficient_funds_is_not_ambiguous(self):
        assert _looks_ambiguous(Exception("insufficient buying power")) is False


def test_known_ambiguous_state_overrides_safe_retry_text():
    approval = MagicMock(id="approval-1", shares=0.01)
    strat = MagicMock(entry=64_000, stop=62_000, target=68_000)
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    failure = AmbiguousOrderState(
        "protective stop failed: insufficient buying power; "
        "MANUAL PAPER-ACCOUNT CHECK REQUIRED"
    )
    with patch.object(en, "_submit_alpaca_crypto_protected", side_effect=failure), \
         patch.object(en, "_try_reconcile_by_client_order_id", return_value=None), \
         patch.object(en, "get_session", return_value=mock_cm):
        result = _execute_one(
            symbol="BTC-USD",
            approval=approval,
            strat=strat,
            run_id="run-1",
            today=MagicMock(),
            broker=MagicMock(),
            dry_run=False,
            crypto=True,
        )

    stored_order = mock_session.add.call_args.args[0]
    assert stored_order.status == "error_ambiguous"
    assert result["status"] == "error_ambiguous"


def test_confirmed_zero_fill_is_stored_as_retry_safe_error():
    approval = MagicMock(id="approval-wif", shares=151202)
    strat = MagicMock(entry=0.146, stop=0.1432, target=0.1515)
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False

    failure = ConfirmedOrderFailure(
        "Alpaca crypto entry for WIF-USD was canceled with zero fill"
    )
    with patch.object(en, "_submit_alpaca_crypto_protected", side_effect=failure), \
         patch.object(en, "_try_reconcile_by_client_order_id", return_value=None), \
         patch.object(en, "get_session", return_value=mock_cm):
        result = _execute_one(
            symbol="WIF-USD",
            approval=approval,
            strat=strat,
            run_id="run-wif",
            today=MagicMock(),
            broker=MagicMock(),
            dry_run=False,
            crypto=True,
        )

    stored_order = mock_session.add.call_args.args[0]
    assert stored_order.status == "error"
    assert result["status"] == "error"


def test_execute_one_persists_net_crypto_position_quantity():
    approval = MagicMock(id="approval-1", shares=295.172570199)
    strat = MagicMock(
        ticker_id="ticker-sol",
        entry=74.62,
        stop=73.64,
        target=76.57,
    )
    mock_session = MagicMock()
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_session
    mock_cm.__exit__.return_value = False
    response = {
        "order_id": "buy-sol",
        "order_status": "filled_protected:accepted",
        "filled": True,
        "fill_price": 74.872141287,
        "position_qty": 294.434638773,
        "position_id": "crypto-asset-sol",
    }

    with patch.object(en, "_submit_alpaca_crypto_protected", return_value=response), \
         patch.object(en, "get_session", return_value=mock_cm):
        result = _execute_one(
            symbol="SOL-USD",
            approval=approval,
            strat=strat,
            run_id="run-1",
            today=MagicMock(),
            broker=MagicMock(),
            dry_run=False,
            crypto=True,
        )

    stored = [call.args[0] for call in mock_session.add.call_args_list]
    position = next(row for row in stored if isinstance(row, en.Position))
    assert position.shares == 294.434638773
    assert position.alpaca_position_id == "crypto-asset-sol"
    assert result["status"] == "submitted"


class TestTryReconcileByClientOrderId:
    def test_crypto_never_reconciles(self):
        broker = MagicMock()
        result = _try_reconcile_by_client_order_id(broker, "id-1", crypto=True, robinhood=False)
        assert result is None
        broker.get_order_by_client_id.assert_not_called()

    def test_robinhood_never_reconciles(self):
        broker = MagicMock()
        result = _try_reconcile_by_client_order_id(broker, "id-1", crypto=False, robinhood=True)
        assert result is None
        broker.get_order_by_client_id.assert_not_called()

    def test_none_broker_returns_none(self):
        assert _try_reconcile_by_client_order_id(None, "id-1", crypto=False, robinhood=False) is None

    def test_order_found_and_filled(self):
        broker = MagicMock()
        order = MagicMock(id="o-1", status="filled", filled_avg_price="101.50")
        broker.get_order_by_client_id.return_value = order
        result = _try_reconcile_by_client_order_id(broker, "id-1", crypto=False, robinhood=False)
        assert result["filled"] is True
        assert result["fill_price"] == 101.50
        assert result["order_id"] == "o-1"

    def test_order_not_found_returns_none(self):
        broker = MagicMock()
        broker.get_order_by_client_id.side_effect = Exception("404 not found")
        result = _try_reconcile_by_client_order_id(broker, "id-1", crypto=False, robinhood=False)
        assert result is None


class TestRunDoesNotCrashOnNoneTarget:
    """Regression test for the missing-price crash: a Strategy row with
    target=None must not raise a TypeError when execution_node prints it."""

    def test_dry_run_with_none_target_and_rr(self):
        approval = MagicMock(id="appr-1", shares=10)
        strat = MagicMock(entry=100.0, stop=95.0, target=None, rr=None, ticker_id="t1")

        mock_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_session
        mock_cm.__exit__.return_value = False

        with patch.object(en, "_load_approved", return_value=[(approval, strat, "AAPL")]), \
             patch.object(en, "_latest_risk_run_id", return_value="run-1"), \
             patch.object(en, "init_db"), \
             patch.object(en, "_write_log"), \
             patch.object(en, "get_session", return_value=mock_cm), \
             patch.object(en.settings, "EXECUTION_ENABLED", False):
            result = en.run(risk_run_id="run-1")

        assert result["dry_run"] == ["AAPL"]
        assert result["failed"] == []


def test_live_run_blocks_rejected_strategy_before_broker_client_creation():
    approval = MagicMock(id="appr-locked", shares=10)
    strat = MagicMock(
        entry=100.0,
        stop=95.0,
        target=110.0,
        rr=2.0,
        ticker_id="t1",
        model_used="stock_trend_momentum_v1",
    )

    with patch.object(en, "_load_approved", return_value=[(approval, strat, "AAPL")]), \
         patch.object(en, "init_db"), \
         patch.object(en, "_write_log"), \
         patch.object(en, "_get_alpaca_client") as get_client, \
         patch.object(en.settings, "EXECUTION_ENABLED", True), \
         patch.object(en.settings, "ROBINHOOD_ENABLED", False):
        result = en.run(risk_run_id="locked-risk-run")

    get_client.assert_not_called()
    assert result["submitted"] == []
    assert result["failed"][0]["symbol"] == "AAPL"
    assert "research-only" in result["failed"][0]["reason"]
