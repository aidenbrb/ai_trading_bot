"""Tests for Alpaca account-state symbol normalization and the fail-closed
available-cash sizing gate."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import utils.account as account_mod
from utils.account import (
    AvailableCash,
    _available_cash,
    _canonical_alpaca_symbol,
    _lenient_float,
    get_account_state,
)


def test_alpaca_crypto_position_symbols_use_bot_spelling():
    assert _canonical_alpaca_symbol("SOLUSD") == "SOL-USD"
    assert _canonical_alpaca_symbol("SOL/USD") == "SOL-USD"
    assert _canonical_alpaca_symbol("SOL-USD") == "SOL-USD"


def test_stock_symbol_is_unchanged():
    assert _canonical_alpaca_symbol("AAPL") == "AAPL"


# -- _available_cash(): fail-closed sizing budget --------------------------

class TestAvailableCash:
    def test_non_marginable_below_cash_returns_it(self):
        result = _available_cash(100_000.0, 98_702.11)
        assert result == AvailableCash(98_702.11, True, None)

    def test_non_marginable_above_cash_caps_at_cash(self):
        result = _available_cash(100_000.0, 436_594.28)
        assert result == AvailableCash(100_000.0, True, None)

    @pytest.mark.parametrize("bad", [None, "not-a-number", [1, 2], object()])
    def test_non_marginable_missing_or_nonnumeric_fails_closed(self, bad):
        result = _available_cash(100_000.0, bad)
        assert result.value == 0.0
        assert result.valid is False

    def test_non_marginable_nan_fails_closed(self):
        result = _available_cash(100_000.0, float("nan"))
        assert result.value == 0.0
        assert result.valid is False

    @pytest.mark.parametrize("inf_value", [float("inf"), float("-inf")])
    def test_non_marginable_infinite_fails_closed(self, inf_value):
        result = _available_cash(100_000.0, inf_value)
        assert result.value == 0.0
        assert result.valid is False

    def test_non_marginable_overflow_fails_closed(self):
        result = _available_cash(100_000.0, 10 ** 400)  # int too large for float()
        assert result.value == 0.0
        assert result.valid is False

    @pytest.mark.parametrize("bad", [
        None, "not-a-number", [1, 2], float("nan"), float("inf"), float("-inf"),
    ])
    def test_cash_missing_or_malformed_fails_closed(self, bad):
        # cash must be validated too, not just non_marginable_buying_power -
        # this function must not trust that its caller already checked it.
        result = _available_cash(bad, 50_000.0)
        assert result.value == 0.0
        assert result.valid is False

    def test_cash_overflow_fails_closed(self):
        result = _available_cash(10 ** 400, 50_000.0)
        assert result.value == 0.0
        assert result.valid is False

    def test_non_marginable_negative_floors_to_zero_but_stays_valid(self):
        result = _available_cash(100_000.0, -500.0)
        assert result.value == 0.0
        assert result.valid is True
        assert result.reason is not None

    def test_cash_negative_floors_to_zero_but_stays_valid(self):
        result = _available_cash(-500.0, 50_000.0)
        assert result.value == 0.0
        assert result.valid is True


class TestLenientFloat:
    def test_valid_number_passes_through(self):
        assert _lenient_float("98702.11") == pytest.approx(98702.11)

    @pytest.mark.parametrize("bad", [None, "abc", float("nan"), float("inf"), object()])
    def test_malformed_returns_default(self, bad):
        assert _lenient_float(bad) == 0.0

    def test_custom_default(self):
        assert _lenient_float(None, default=-1.0) == -1.0

    def test_overflow_returns_default(self):
        assert _lenient_float(10 ** 400) == 0.0


# -- get_account_state() end-to-end: preservation and fail-closed behavior -

def _fake_account(**overrides):
    defaults = dict(
        equity="109148.57",
        cash="109148.57",
        buying_power="436594.28",
        non_marginable_buying_power="98702.11",
        last_equity="108000.00",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_position(symbol="AAPL", qty="10", avg_entry_price="150.0",
                    current_price="155.0", unrealized_pl="50.0"):
    return SimpleNamespace(
        symbol=symbol, qty=qty, avg_entry_price=avg_entry_price,
        current_price=current_price, unrealized_pl=unrealized_pl,
    )


def _run_with_mock_client(account, positions=None, client_side_effect=None):
    mock_client = MagicMock()
    mock_client.get_account.return_value = account
    mock_client.get_all_positions.return_value = positions or []
    mock_client.get_orders.return_value = []
    with patch.object(account_mod.settings, "ALPACA_API_KEY", "key"), \
         patch.object(account_mod.settings, "ALPACA_SECRET_KEY", "secret"), \
         patch(
             "alpaca.trading.client.TradingClient",
             side_effect=client_side_effect,
             return_value=None if client_side_effect else mock_client,
         ):
        return get_account_state()


class TestGetAccountStateIntegration:
    def test_malformed_non_marginable_preserves_raw_cash(self):
        account = _fake_account(non_marginable_buying_power=None)
        result = _run_with_mock_client(account)
        assert result["cash"] == pytest.approx(109148.57)
        assert result["available_cash"] == 0.0
        assert result["available_cash_valid"] is False

    def test_malformed_raw_cash_preserves_real_account_data(self):
        # This is the v4 fix: a malformed raw `cash` must not raise before
        # _available_cash() sees it, or the whole fetch falls back to
        # _DEFAULTS and silently blanks out real positions/open_positions/
        # connected - not just the cash figure.
        account = _fake_account(cash=None)
        positions = [_fake_position()]
        result = _run_with_mock_client(account, positions=positions)

        assert result["connected"] is True
        assert result["equity"] == pytest.approx(109148.57)
        assert result["open_positions"] == 1
        assert result["positions"][0]["symbol"] == "AAPL"
        assert result["cash"] == 0.0                        # safe lenient-parse default
        assert result["available_cash"] == 0.0
        assert result["available_cash_valid"] is False

    def test_valid_account_reports_conservative_available_cash(self):
        account = _fake_account()
        result = _run_with_mock_client(account)
        assert result["cash"] == pytest.approx(109148.57)
        assert result["available_cash"] == pytest.approx(98702.11)
        assert result["available_cash_valid"] is True
        assert result["buying_power"] == pytest.approx(436594.28)

    def test_missing_buying_power_attribute_does_not_raise_or_affect_sizing(self):
        account = _fake_account()
        del account.buying_power
        result = _run_with_mock_client(account)
        assert result["buying_power"] == 0.0
        assert result["available_cash"] == pytest.approx(98702.11)
        assert result["available_cash_valid"] is True

    def test_no_credentials_fails_closed(self):
        with patch.object(account_mod.settings, "ALPACA_API_KEY", None):
            result = get_account_state()
        assert result["available_cash"] == 0.0
        assert result["available_cash_valid"] is False
        assert result["connected"] is False

    def test_connection_exception_falls_back_to_defaults_fail_closed(self):
        result = _run_with_mock_client(_fake_account(), client_side_effect=RuntimeError("boom"))
        assert result["available_cash"] == 0.0
        assert result["available_cash_valid"] is False
        assert result["connected"] is False
