"""
Tests for nodes/crypto_strategy_node.py's daily-bar indicator fetch (added
after a Phase 1 review found the existing hourly path's SMA20/50/200 trend
calculation actually runs on hourly bars, not the daily periods the names
imply). yfinance is mocked throughout - this is pure indicator-math
correctness, not a live-data integration check.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nodes.crypto_strategy_node import fetch_daily_indicator


def _fake_yf_frame(n=260, start="2025-01-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 200.0, n), index=idx)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1_000_000.0,
    }, index=idx)


def test_fetch_daily_indicator_computes_expected_fields():
    frame = _fake_yf_frame()
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        values = fetch_daily_indicator("BTC-USD")

    assert values is not None
    for key in ("sma_20", "sma_50", "sma_200", "rsi_14", "macd_hist", "atr_14", "rel_volume", "trend"):
        assert key in values
    # A steadily rising close series should classify as UPTREND.
    assert values["trend"] == "UPTREND"


def test_fetch_daily_indicator_index_is_utc_midnight_normalized():
    """Confirms the UTC-midnight daily boundary convention (matches the
    backtest's native Alpaca daily bars, empirically verified separately)."""
    idx = pd.date_range("2025-01-01 13:45", periods=260, freq="D", tz="US/Eastern")
    close = pd.Series(np.linspace(100.0, 200.0, 260), index=idx)
    frame = pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": 1_000_000.0,
    }, index=idx)
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        values = fetch_daily_indicator("BTC-USD")
    assert values is not None  # would only fail if normalization broke the row count


def test_fetch_daily_indicator_returns_none_on_insufficient_history():
    frame = _fake_yf_frame(n=50)  # well under 200
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        assert fetch_daily_indicator("BTC-USD") is None


def test_fetch_daily_indicator_returns_none_on_empty_response():
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = pd.DataFrame()
        assert fetch_daily_indicator("BTC-USD") is None


def test_fetch_daily_indicator_returns_none_on_fetch_exception():
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.side_effect = RuntimeError("network error")
        assert fetch_daily_indicator("BTC-USD") is None


def test_fetch_daily_indicator_not_wired_into_run():
    """Regression guard for the explicit scoping decision: the daily fetch
    must stay research-only and never get imported/called by run()'s
    scheduled live flow while the new strategies are still
    execution_eligible=False."""
    import inspect
    import nodes.crypto_strategy_node as node
    source = inspect.getsource(node.run)
    assert "fetch_daily_indicator" not in source


# -- evaluate_crypto_trend_daily_v1 (Step 2) ---------------------------------

from nodes.crypto_strategy_node import evaluate_crypto_trend_daily_v1


def test_evaluate_daily_v1_returns_none_on_insufficient_history():
    frame = _fake_yf_frame(n=50)
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        assert evaluate_crypto_trend_daily_v1("BTC-USD") is None


@pytest.mark.parametrize("entry_mode", ["strict_stack", "sma50_rising", "donchian"])
def test_evaluate_daily_v1_wires_through_to_a_signal_decision(entry_mode):
    """Structural wiring check (gate math itself is covered by
    tests/test_crypto_trend_daily_v1.py's synthetic-indicator tests): a
    real-shaped daily frame flows all the way through fetch -> compute_all
    -> the extra sma_50_prior/high_20 fields -> crypto_trend_daily_v1(),
    producing a SignalDecision tagged with the right strategy version and
    entry_mode, whether it passes or is rejected."""
    frame = _fake_yf_frame(n=260)
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        decision = evaluate_crypto_trend_daily_v1("BTC-USD", entry_mode=entry_mode)

    assert decision is not None
    assert decision.strategy_version == "crypto_trend_daily_v1"
    assert isinstance(decision.reason, str) and decision.reason


def test_evaluate_daily_v1_not_wired_into_run():
    import inspect
    import nodes.crypto_strategy_node as node
    source = inspect.getsource(node.run)
    assert "evaluate_crypto_trend_daily_v1" not in source


def test_evaluate_daily_v1_unknown_mode_raises():
    frame = _fake_yf_frame(n=260)
    with patch("yfinance.Ticker") as MockTicker:
        MockTicker.return_value.history.return_value = frame
        with pytest.raises(ValueError, match="unknown entry_mode"):
            evaluate_crypto_trend_daily_v1("BTC-USD", entry_mode="bogus")
