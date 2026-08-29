"""Regression tests for the frozen SLC research signal engine."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import utils.slc_signals as slc
from utils.market_calendar import session_for


def _bars(index, *, open_=10.0, high=10.5, low=9.5, close=10.0):
    size = len(index)
    def values(value):
        return [value] * size if np.isscalar(value) else value
    return pd.DataFrame({
        "open": values(open_), "high": values(high), "low": values(low),
        "close": values(close), "volume": [1000.0] * size,
    }, index=pd.DatetimeIndex(index))


def _regular_day(day: date, price: float = 10.0):
    session = session_for(day)
    count = int((session["close"] - session["open"]).total_seconds() // 300)
    index = pd.date_range(session["open"], periods=count, freq="5min")
    return _bars(index, open_=price, high=price + 0.5, low=price - 0.5, close=price)


def _signal(direction="long", entry=10.0, stop=9.0, target=12.0):
    return slc.SlcSignal(
        slc.VERSION, "AAPL", direction, "level", "fresh", 9.5, 10.5,
        pd.Timestamp("2025-06-02 14:00"), pd.Timestamp("2025-06-02 15:00"),
        pd.Timestamp("2025-06-02 15:00"), entry, stop, target, 1.0, 2.0,
        25.0, 20.0, 1.0, "uptrend" if direction == "long" else "downtrend", 1.5,
    )


def test_stochastic_is_exact_5_3_3_and_requires_full_windows():
    index = pd.date_range("2025-01-01", periods=9, freq="5min")
    bars = _bars(index, open_=np.arange(9) + 4.0, high=np.arange(9) + 10.0,
                 low=np.arange(9), close=np.arange(9) + 5.0)
    result = slc.stochastic_5_3_3(bars)
    assert result["raw_k"].iloc[-1] == pytest.approx(100 * 9 / 14)
    assert result["k"].iloc[-1] == pytest.approx(100 * 9 / 14)
    assert result["d"].iloc[-1] == pytest.approx(100 * 9 / 14)
    assert result["k"].iloc[:6].isna().all()
    assert result["d"].iloc[:8].isna().all()


def test_stochastic_zero_range_fails_closed():
    bars = _bars(pd.date_range("2025-01-01", periods=12, freq="5min"),
                 open_=10, high=10, low=10, close=10)
    assert slc.stochastic_5_3_3(bars)[["k", "d"]].isna().all().all()


def test_atr14_is_simple_mean_and_requires_all_values():
    index = pd.date_range("2025-01-01", periods=16, freq="5min")
    bars = _bars(index)
    result = slc.atr14(bars)
    assert result.iloc[:14].isna().all()
    assert result.iloc[14] == pytest.approx(1.0)


def test_session_anchored_bars_handle_dst():
    bars = pd.concat([_regular_day(date(2025, 3, 7)), _regular_day(date(2025, 3, 10))])
    result = slc.session_anchored_4h_bars(bars)
    assert list(result.index) == [
        pd.Timestamp("2025-03-07 18:30"), pd.Timestamp("2025-03-07 21:00"),
        pd.Timestamp("2025-03-10 17:30"), pd.Timestamp("2025-03-10 20:00"),
    ]


def test_session_anchored_bars_handle_early_close():
    result = slc.session_anchored_4h_bars(_regular_day(date(2025, 11, 28)))
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2025-11-28 18:00")


def test_incomplete_four_hour_bucket_is_omitted():
    bars = _regular_day(date(2025, 6, 2)).iloc[:-1]
    result = slc.session_anchored_4h_bars(bars)
    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2025-06-02 17:30")


def _uptrend_four_hour_bars():
    highs = [8, 9, 10, 9, 8, 10, 11, 12, 11, 10, 12, 13, 14, 13, 12]
    lows = [7, 8, 9, 8, 6, 8, 9, 10, 9, 7, 10, 11, 12, 11, 9]
    index = pd.date_range("2025-01-01", periods=len(highs), freq="4h")
    return _bars(index, open_=np.array(lows) + .5, high=highs, low=lows,
                 close=np.array(lows) + .75)


def test_pivots_are_not_known_until_two_right_bars_close():
    bars = _uptrend_four_hour_bars()
    pivots = slc.confirmed_pivots(bars)
    pivot = next(p for p in pivots if p["pivot_at"] == bars.index[7] and p["kind"] == "high")
    assert pivot["confirmed_at"] == bars.index[9]
    assert slc.classify_structure(bars, bars.index[10]) == "consolidation"
    assert slc.classify_structure(bars, bars.index[11]) == "uptrend"


def test_equal_high_is_not_a_pivot():
    bars = _uptrend_four_hour_bars()
    bars.iloc[1, bars.columns.get_loc("high")] = 10
    assert not any(p["pivot_at"] == bars.index[2] and p["kind"] == "high"
                   for p in slc.confirmed_pivots(bars))


def test_detects_full_base_candle_demand_after_completed_impulse():
    index = pd.date_range("2025-06-02 13:30", periods=18, freq="5min")
    bars = _bars(index)
    bars.iloc[15] = [10.0, 10.6, 10.0, 10.5, 1000]
    bars.iloc[16] = [10.5, 10.9, 10.4, 10.8, 1000]
    bars.iloc[17] = [10.8, 11.2, 10.7, 11.1, 1000]
    levels = slc.detect_levels(bars)
    demand = [level for level in levels if level.base_time == index[14] and level.direction == "long"]
    assert len(demand) == 1
    assert demand[0].low == 9.5
    assert demand[0].high == 10.5
    assert demand[0].active_time == index[17] + timedelta(minutes=5)


def test_impulse_cannot_cross_an_overnight_gap():
    first = pd.date_range("2025-06-02 19:50", periods=2, freq="5min")
    second = pd.date_range("2025-06-03 13:30", periods=3, freq="5min")
    bars = _bars(first.append(second), open_=[10, 10, 11, 12, 13],
                 high=[10.5, 10.5, 11.5, 12.5, 13.5], low=[9.5, 9.5, 10.5, 11.5, 12.5],
                 close=[10, 10, 11.4, 12.4, 13.4])
    assert slc.detect_levels(bars) == []


def test_long_same_minute_stop_and_target_uses_stop_adversely():
    signal = _signal()
    bars = _bars([signal.entry_time], open_=10, high=12.1, low=8.9, close=11)
    result = slc.simulate_intraday_outcome(
        signal, bars, session_close=pd.Timestamp("2025-06-02 20:00")
    )
    assert result["exit_reason"] == "stop"
    assert result["exit_price"] == 9
    assert result["ambiguous"] is True


def test_short_gap_through_stop_fills_at_open():
    signal = _signal(direction="short", entry=10, stop=11, target=8)
    bars = _bars([signal.entry_time], open_=11.5, high=11.7, low=10.8, close=11)
    result = slc.simulate_intraday_outcome(
        signal, bars, session_close=pd.Timestamp("2025-06-02 20:00")
    )
    assert result["exit_reason"] == "stop_gap"
    assert result["exit_price"] == 11.5


def test_open_trade_flattens_at_last_available_minute():
    signal = _signal()
    index = pd.date_range(signal.entry_time, periods=2, freq="1min")
    bars = _bars(index, open_=[10, 10.1], high=[10.2, 10.3], low=[9.8, 9.9], close=[10.1, 10.2])
    result = slc.simulate_intraday_outcome(
        signal, bars, session_close=pd.Timestamp("2025-06-02 20:00")
    )
    assert result["exit_reason"] == "eod"
    assert result["exit_price"] == 10.2


def test_missing_outcome_data_is_not_a_trade_result():
    result = slc.simulate_intraday_outcome(
        _signal(), None, session_close=pd.Timestamp("2025-06-02 20:00")
    )
    assert result == {"status": "outcome_data_missing", "reason": "coverage_missing"}


def _stub_generate_inputs(monkeypatch, *, direction="long", start="2025-06-02 14:00"):
    index = pd.date_range(start, periods=12, freq="5min")
    bars = _bars(index)
    level = slc.SlcLevel(
        "test-level", direction, 9.0, 11.0, index[1], index[5], 1.5,
        pd.Timestamp(index[5]).tz_localize("UTC").tz_convert(slc.EASTERN).date(),
    )
    values = [np.nan] * len(index)
    values[6], values[7] = ((10.0, 25.0) if direction == "long" else (90.0, 75.0))
    stochastic = pd.DataFrame({"raw_k": values, "k": values, "d": values}, index=index)
    monkeypatch.setattr(slc, "detect_levels", lambda frame: [level])
    monkeypatch.setattr(slc, "stochastic_5_3_3", lambda frame: stochastic)
    monkeypatch.setattr(slc, "atr14", lambda frame: pd.Series(1.0, index=index))
    monkeypatch.setattr(
        slc, "classify_structure",
        lambda frame, asof: "uptrend" if direction == "long" else "downtrend",
    )
    return bars, level


def test_confirmation_enters_next_bar_with_buffer_and_exact_2r(monkeypatch):
    bars, _ = _stub_generate_inputs(monkeypatch)
    signals = slc.generate_signals("aapl", bars, four_hour_bars=bars)
    assert len(signals) == 1
    signal = signals[0]
    assert signal.entry_time == bars.index[8]
    assert signal.confirmation_time == bars.index[8]
    assert signal.stop == pytest.approx(8.9)
    assert signal.target == pytest.approx(signal.entry + 2 * signal.initial_risk)
    assert signal.reward_risk == 2.0
    assert signal.symbol == "AAPL"


def test_long_rule_is_mirrored_for_short(monkeypatch):
    bars, _ = _stub_generate_inputs(monkeypatch, direction="short")
    signal = slc.generate_signals("AAPL", bars, four_hour_bars=bars)[0]
    assert signal.direction == "short"
    assert signal.stop == pytest.approx(11.1)
    assert signal.target == pytest.approx(signal.entry - 2 * signal.initial_risk)


def test_structure_mismatch_fails_closed(monkeypatch):
    bars, _ = _stub_generate_inputs(monkeypatch)
    monkeypatch.setattr(slc, "classify_structure", lambda frame, asof: "consolidation")
    assert slc.generate_signals("AAPL", bars, four_hour_bars=bars) == []


def test_broken_once_level_requires_reclaim_before_retest(monkeypatch):
    bars, _ = _stub_generate_inputs(monkeypatch)
    # Break demand below its far edge, reclaim above the far edge, then arm
    # and confirm on the one allowed retest.
    bars.iloc[5, bars.columns.get_loc("low")] = 8.0
    bars.iloc[5, bars.columns.get_loc("close")] = 8.5
    bars.iloc[6, bars.columns.get_loc("high")] = 12.5
    bars.iloc[6, bars.columns.get_loc("close")] = 12.0
    values = [np.nan] * len(bars)
    values[7], values[8] = 10.0, 25.0
    stochastic = pd.DataFrame({"raw_k": values, "k": values, "d": values}, index=bars.index)
    monkeypatch.setattr(slc, "stochastic_5_3_3", lambda frame: stochastic)
    signal = slc.generate_signals("AAPL", bars, four_hour_bars=bars)[0]
    assert signal.level_state == "reclaimed"
    assert signal.entry_time == bars.index[9]


def test_entry_after_1530_eastern_is_rejected(monkeypatch):
    bars, _ = _stub_generate_inputs(monkeypatch, start="2025-06-02 19:00")
    assert slc.generate_signals("AAPL", bars, four_hour_bars=bars) == []
