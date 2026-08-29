"""
Tests for backtest/whole_bot_engine.py::simulate_daily_crypto_order_outcome()
- the daily-cadence counterpart to simulate_order_outcome(), added for
Phase 2 Step 4 (exit timeframe consistency) so crypto_trend_daily_v1
candidates get monitored on daily bars instead of the hourly frame.
"""
from datetime import date, datetime, time, timedelta

import pandas as pd
import pytest

from backtest.whole_bot_engine import (
    Candidate,
    daily_decision_time_utc,
    simulate_daily_crypto_order_outcome,
)


def _daily_bar_frame(rows):
    """rows: list of (date_str, open, high, low, close, trend, macd_hist, rsi_14, atr_14)."""
    idx = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame({
        "open":      [r[1] for r in rows],
        "high":      [r[2] for r in rows],
        "low":       [r[3] for r in rows],
        "close":     [r[4] for r in rows],
        "trend":     [r[5] for r in rows],
        "macd_hist": [r[6] for r in rows],
        "rsi_14":    [r[7] for r in rows],
        "atr_14":    [r[8] for r in rows],
    }, index=idx)


def _candidate(decision_day, entry=100.0, stop=95.0, target=120.0):
    return Candidate(
        symbol="BTC-USD", market="crypto", strategy_version="crypto_trend_daily_v1",
        decision_time=daily_decision_time_utc(decision_day),
        signal_bar_end=datetime.combine(decision_day, time.min) + timedelta(days=1),
        entry=entry, stop=stop, target=target, conviction=80, atr=2.0,
        timeframe="daily",
    )


# -- Fill behavior ------------------------------------------------------------

def test_fills_at_decision_days_open():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
    ])
    candidate = _candidate(date(2026, 1, 5))
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 5), {"BTC-USD": frame})
    assert result["filled_at"] == datetime(2026, 1, 5)
    assert result["fill_price"] == 101.0


def test_unfilled_when_decision_day_bar_missing():
    frame = _daily_bar_frame([("2026-01-04", 100.0, 101.0, 99.0, 100.0, "UPTREND", 1.0, 60.0, 2.0)])
    candidate = _candidate(date(2026, 1, 5))
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 5), {"BTC-USD": frame})
    assert result["status"] == "unfilled_end"
    assert result["filled_at"] is None


def test_missing_open_is_outcome_data_missing():
    import numpy as np
    frame = _daily_bar_frame([("2026-01-05", np.nan, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0)])
    candidate = _candidate(date(2026, 1, 5))
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 5), {"BTC-USD": frame})
    assert result["outcome_data_missing"] is True


def test_no_frame_for_symbol_is_outcome_data_missing():
    candidate = _candidate(date(2026, 1, 5))
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 5), {})
    assert result["outcome_data_missing"] is True


# -- Stop / target ------------------------------------------------------------

def test_stop_hit_on_a_later_day():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 101.0, 90.0, 92.0, "UPTREND", 1.0, 60.0, 2.0),  # low breaches stop=95
    ])
    candidate = _candidate(date(2026, 1, 5), stop=95.0, target=200.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "stop"
    assert result["exit_price"] == 95.0
    assert result["exit_time"] == datetime(2026, 1, 6)


def test_target_hit_on_a_later_day():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 105.0, 125.0, 104.0, 120.0, "UPTREND", 1.0, 60.0, 2.0),  # high breaches target=120
    ])
    candidate = _candidate(date(2026, 1, 5), stop=90.0, target=120.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "target"
    assert result["exit_price"] == 120.0


def test_downside_checked_before_upside_on_the_same_bar():
    """Matches simulate_order_outcome()'s own convention: the required
    adverse-ambiguity assumption when both stop and target could have
    occurred within the same daily bar."""
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 130.0, 90.0, 100.0, "UPTREND", 1.0, 60.0, 2.0),  # both stop and target breached
    ])
    candidate = _candidate(date(2026, 1, 5), stop=95.0, target=120.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "stop"


# -- Reversal exit --------------------------------------------------------

def test_downtrend_forces_exit():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 101.0, 98.0, 99.0, "DOWNTREND", 1.0, 60.0, 2.0),
    ])
    candidate = _candidate(date(2026, 1, 5), stop=50.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "monitor_reversal"
    assert result["exit_price"] == 99.0


def test_macd_negative_and_rsi_breakdown_forces_exit():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 101.0, 98.0, 99.0, "UPTREND", -0.5, 40.0, 2.0),
    ])
    candidate = _candidate(date(2026, 1, 5), stop=50.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "monitor_reversal"


def test_macd_negative_but_rsi_above_breakdown_does_not_exit():
    frame = _daily_bar_frame([
        ("2026-01-05", 101.0, 102.0, 99.0, 100.5, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 101.0, 98.0, 99.0, "UPTREND", -0.5, 50.0, 2.0),
    ])
    candidate = _candidate(date(2026, 1, 5), stop=50.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] != "monitor_reversal"


# -- Trailing stop ----------------------------------------------------------

def test_trailing_stop_tightens_after_favorable_move():
    frame = _daily_bar_frame([
        ("2026-01-05", 100.0, 101.0, 99.0, 100.0, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 105.0, 99.5, 103.0, "UPTREND", 1.0, 60.0, 2.0),  # +3 >= 1x ATR(2.0)
        ("2026-01-07", 103.0, 104.0, 102.0, 103.0, "UPTREND", 1.0, 60.0, 2.0),  # neutral day, no new stop hit
    ])
    candidate = _candidate(date(2026, 1, 5), stop=90.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 7), {"BTC-USD": frame})
    # Trailed to 103 - 1.5*2.0 = 100.0, well above the original 90.0 stop.
    assert result["final_stop"] == pytest.approx(100.0)
    assert result["final_stop"] > 90.0


def test_trailing_stop_never_loosens():
    frame = _daily_bar_frame([
        ("2026-01-05", 100.0, 101.0, 99.0, 100.0, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 106.0, 99.5, 105.0, "UPTREND", 1.0, 60.0, 2.0),  # big favorable move, trail up
        ("2026-01-07", 105.0, 105.5, 96.0, 97.0, "UPTREND", 1.0, 60.0, 2.0),   # pulls back - must not loosen
    ])
    candidate = _candidate(date(2026, 1, 5), stop=90.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 7), {"BTC-USD": frame})
    # Trailed to 105 - 1.5*2 = 102.0 on day 2; day 3's low (96.0) breaches
    # that trailed stop, so this should exit at the STOP, not just end_of_test.
    assert result["exit_reason"] == "stop"
    assert result["exit_price"] == pytest.approx(102.0)


# -- End of test --------------------------------------------------------

def test_end_of_test_when_nothing_triggers():
    frame = _daily_bar_frame([
        ("2026-01-05", 100.0, 101.0, 99.0, 100.0, "UPTREND", 1.0, 60.0, 2.0),
        ("2026-01-06", 100.0, 101.0, 99.5, 100.5, "UPTREND", 1.0, 60.0, 2.0),
    ])
    candidate = _candidate(date(2026, 1, 5), stop=50.0, target=500.0)
    result = simulate_daily_crypto_order_outcome(candidate, date(2026, 1, 6), {"BTC-USD": frame})
    assert result["exit_reason"] == "end_of_test"
    assert result["exit_price"] == 100.5
