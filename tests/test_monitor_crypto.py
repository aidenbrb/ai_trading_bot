"""Crypto symbol and target-exit behavior in the position monitor."""
from unittest.mock import MagicMock, patch

import nodes.monitor_node as mn
from nodes.monitor_node import (
    _alpaca_symbol,
    _canonical_symbol,
    _close_alpaca_position,
    _estimate_exit_price,
    _evaluate_position,
)


def test_alpaca_crypto_symbol_round_trip():
    assert _alpaca_symbol("BTC-USD") == "BTC/USD"
    assert _canonical_symbol("BTC/USD") == "BTC-USD"
    assert _canonical_symbol("BTCUSD") == "BTC-USD"


def test_crypto_target_reached_closes_position():
    position = MagicMock(
        entry_price=100.0,
        stop_price=95.0,
        target_price=110.0,
    )
    result = _evaluate_position(
        "BTC-USD",
        position,
        current_price=111.0,
        ind=None,
    )
    assert result["action"] == "CLOSE"
    assert "target reached" in result["reason"]


def test_crypto_close_uses_asset_id_and_confirms_flat():
    broker = MagicMock()
    position = MagicMock(
        symbol="SOLUSD",
        asset_id="sol-asset-id",
        current_price="76.10",
    )
    stop_order = MagicMock(id="stop-1", symbol="SOL/USD")
    broker.get_all_positions.side_effect = [[position], []]
    broker.get_orders.side_effect = [[stop_order], []]
    broker.close_position.return_value = MagicMock(id="close-1")

    with patch("alpaca.trading.client.TradingClient", return_value=broker), \
         patch.object(mn.settings, "EXECUTION_ENABLED", True):
        exit_price = _close_alpaca_position("SOL-USD")

    assert exit_price == 76.10
    broker.cancel_order_by_id.assert_called_once_with("stop-1")
    broker.close_position.assert_called_once_with("sol-asset-id")


def test_crypto_reconciliation_prefers_actual_broker_exit_fill():
    position = MagicMock(current_price=77.7185, stop_price=76.36)
    with patch.object(mn, "_alpaca_crypto_exit_fill", return_value=77.018001494):
        price = _estimate_exit_price(position, "SOL-USD")
    assert price == 77.018001494
