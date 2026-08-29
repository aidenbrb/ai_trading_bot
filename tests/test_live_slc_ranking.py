import pandas as pd
import pytest

from live_slc.reducer import Confirmation
from live_slc.ranking import rank_confirmations, select_within_capacity


def _mk(symbol, entry_time, level_active_time, impulse_atr, direction="long"):
    return Confirmation(
        "slc_4h_5m_stock_v1", symbol, f"demand:{symbol}", direction, "fresh",
        10.0, 11.0, pd.Timestamp(level_active_time), pd.Timestamp(entry_time),
        pd.Timestamp(entry_time), 9.5, 15.0, 12.0, 1.0, "uptrend", impulse_atr,
    )


def test_ranking_matches_exact_frozen_tie_break_tuple():
    """Verified directly against backtest/run_slc_backtest.py:238-241's
    sorted(day_signals, key=lambda s: (s.entry_time, -s.level_active_time.value,
    -s.impulse_atr, s.symbol))."""
    earliest = _mk("ZZZZ", "2026-08-13 09:55", "2026-08-13 08:00", 1.0)
    later_less_recent_level = _mk("MSFT", "2026-08-13 10:00", "2026-08-13 09:00", 2.0)
    later_more_recent_level = _mk("AAPL", "2026-08-13 10:00", "2026-08-13 09:30", 2.0)
    ranked = rank_confirmations([later_less_recent_level, later_more_recent_level, earliest])
    assert [c.symbol for c in ranked] == ["ZZZZ", "AAPL", "MSFT"]


def test_ranking_tie_break_falls_through_to_impulse_atr_then_symbol():
    same_time_and_level = _mk("BBBB", "2026-08-13 10:00", "2026-08-13 09:00", 1.0)
    same_time_and_level_higher_impulse = _mk("AAAA", "2026-08-13 10:00", "2026-08-13 09:00", 5.0)
    ranked = rank_confirmations([same_time_and_level, same_time_and_level_higher_impulse])
    assert [c.symbol for c in ranked] == ["AAAA", "BBBB"]  # higher impulse wins the tie


def test_ranking_is_independent_of_input_order():
    a = _mk("AAPL", "2026-08-13 10:00", "2026-08-13 09:30", 2.0)
    b = _mk("MSFT", "2026-08-13 09:55", "2026-08-13 09:00", 1.0)
    c = _mk("ZZZZ", "2026-08-13 10:05", "2026-08-13 09:00", 1.0)
    order1 = [x.symbol for x in rank_confirmations([a, b, c])]
    order2 = [x.symbol for x in rank_confirmations([c, a, b])]
    order3 = [x.symbol for x in rank_confirmations([b, c, a])]
    assert order1 == order2 == order3


def test_select_within_capacity_never_submits_symbol_by_symbol_before_full_scan():
    """The whole ranked list must be walked in ranked order - never
    dependent on processing/API response order."""
    ranked = [
        _mk("ZZZZ", "2026-08-13 09:55", "2026-08-13 08:00", 1.0),
        _mk("AAPL", "2026-08-13 10:00", "2026-08-13 09:30", 2.0),
        _mk("MSFT", "2026-08-13 10:00", "2026-08-13 09:00", 2.0),
    ]

    def always_ok(c, admitted):
        return True, None

    admitted, skipped = select_within_capacity(ranked, remaining_daily_entries=2, capacity_check_fn=always_ok)
    assert [c.symbol for c in admitted] == ["ZZZZ", "AAPL"]
    assert [c.symbol for c, _ in skipped] == ["MSFT"]
    assert skipped[0][1] == "skipped_capacity_daily_trades"


def test_select_within_capacity_passes_admitted_so_far_to_capacity_check():
    ranked = [
        _mk("A", "2026-08-13 09:55", "2026-08-13 08:00", 1.0),
        _mk("B", "2026-08-13 10:00", "2026-08-13 09:00", 1.0),
    ]
    seen_admitted_lengths = []

    def capacity_check(c, admitted):
        seen_admitted_lengths.append(len(admitted))
        return True, None

    select_within_capacity(ranked, remaining_daily_entries=2, capacity_check_fn=capacity_check)
    assert seen_admitted_lengths == [0, 1]
