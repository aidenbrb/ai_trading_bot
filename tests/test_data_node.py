"""Data-source routing regressions."""
from datetime import date
from unittest.mock import patch

import pandas as pd

from nodes.data_node import _download


def _frame():
    return pd.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [10.0]},
        index=pd.DatetimeIndex(["2026-07-25"], name="bar_time"),
    )


def test_crypto_prefers_fast_yahoo_history():
    yahoo = _frame()
    with patch("nodes.data_node._download_yfinance", return_value=yahoo) as yf, \
         patch("nodes.data_node._download_coinbase") as cb:
        result = _download("BTC-USD", date(2026, 1, 1), date(2026, 1, 2))
    assert result is yahoo
    yf.assert_called_once()
    cb.assert_not_called()


def test_crypto_falls_back_to_public_coinbase_when_yahoo_empty():
    coinbase = _frame()
    with patch("nodes.data_node._download_yfinance", return_value=pd.DataFrame()), \
         patch("nodes.data_node._download_coinbase", return_value=coinbase) as cb:
        result = _download("UNI-USD", date(2026, 1, 1), date(2026, 1, 2))
    assert result is coinbase
    cb.assert_called_once()
