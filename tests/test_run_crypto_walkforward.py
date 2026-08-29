"""
Tests for backtest/run_crypto_walkforward.py (Phase 3 Step 5) - focused on
_half_year_pnl(), the new bucketing logic used to check whether OOS decay
is monotonic or concentrated in one stretch.
"""
from datetime import date, datetime

from backtest.run_crypto_walkforward import _half_year_pnl


def _trade(exit_time, net_pnl):
    return {"exit_time": exit_time, "net_pnl": net_pnl}


def test_buckets_by_correct_half_year():
    trades = [
        _trade(date(2024, 1, 15), 100.0),   # 2024H1
        _trade(date(2024, 6, 30), 50.0),    # 2024H1 (boundary day)
        _trade(date(2024, 7, 1), -20.0),    # 2024H2 (boundary day)
        _trade(date(2024, 12, 31), 10.0),   # 2024H2
        _trade(date(2025, 3, 1), 5.0),      # 2025H1
    ]
    result = _half_year_pnl(trades)
    assert result["2024H1"] == {"net_pnl": 150.0, "n": 2}
    assert result["2024H2"] == {"net_pnl": -10.0, "n": 2}
    assert result["2025H1"] == {"net_pnl": 5.0, "n": 1}


def test_accepts_datetime_exit_time_not_just_date():
    trades = [_trade(datetime(2024, 8, 15, 13, 30), 42.0)]
    result = _half_year_pnl(trades)
    assert result["2024H2"] == {"net_pnl": 42.0, "n": 1}


def test_skips_trades_with_no_net_pnl():
    trades = [
        _trade(date(2024, 1, 1), None),
        _trade(date(2024, 1, 2), 10.0),
    ]
    result = _half_year_pnl(trades)
    assert result["2024H1"] == {"net_pnl": 10.0, "n": 1}


def test_result_is_sorted_chronologically():
    trades = [
        _trade(date(2025, 1, 1), 1.0),
        _trade(date(2023, 1, 1), 1.0),
        _trade(date(2024, 7, 1), 1.0),
    ]
    result = _half_year_pnl(trades)
    assert list(result.keys()) == ["2023H1", "2024H2", "2025H1"]


def test_empty_trades_produces_empty_result():
    assert _half_year_pnl([]) == {}
