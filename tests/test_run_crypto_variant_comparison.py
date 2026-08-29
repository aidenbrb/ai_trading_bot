"""
Tests for backtest/run_crypto_variant_comparison.py (Phase 2 Step 5). Focus
is _universe_coverage() - a real bug was found and fixed there during this
work: an entirely-empty frame (one of the 8 dead-on-Alpaca symbols from
Step 0) was silently excluded from the denominator entirely, inflating
coverage from the correct ~52% to a misleading ~97%.
"""
from datetime import date

import numpy as np
import pandas as pd

import backtest.whole_bot_engine as engine
from backtest.run_crypto_variant_comparison import _universe_coverage


def _frame(rows, start="2026-01-01"):
    idx = pd.date_range(start, periods=len(rows), freq="D")
    close = pd.Series(rows, index=idx, dtype=float)
    df = pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1000.0,
    }, index=idx)
    return engine.build_daily_crypto_indicator_frames({"X": df})["X"]


def test_dead_symbol_counts_as_attempted_never_usable():
    """The exact bug: an entirely-empty frame must still count toward the
    denominator every day - not be skipped as if it were merely
    pre-inception."""
    good = _frame([100.0 + i * 0.1 for i in range(230)], start="2025-06-01")
    dead = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    daily_ind = {"GOOD-USD": good, "DEAD-USD": dead}

    start, end = date(2026, 1, 5), date(2026, 1, 10)
    coverage = _universe_coverage(daily_ind, start, end)
    # 2 symbols x 6 days = 12 attempted; DEAD-USD is never usable (6),
    # GOOD-USD is usable all 6 days (assuming it's warmed up by then) ->
    # coverage should land at roughly 50%, not ~100%.
    assert 0.4 < coverage < 0.6


def test_genuine_pre_inception_is_excluded_from_denominator():
    """A symbol whose history starts partway through the range (not
    empty, just late) is correctly excluded before its own first bar -
    this is NOT the bug; only fully-empty frames were affected."""
    late = _frame([100.0] * 5, start="2026-01-08")  # starts mid-range
    daily_ind = {"LATE-USD": late}
    start, end = date(2026, 1, 1), date(2026, 1, 5)  # entirely before LATE-USD exists
    coverage = _universe_coverage(daily_ind, start, end)
    assert coverage == 0.0  # attempted=0 (all pre-inception) -> defined as 0.0, not divide-by-zero


def test_fully_covered_universe_is_100pct():
    good = _frame([100.0 + i * 0.1 for i in range(260)], start="2025-05-01")
    daily_ind = {"A-USD": good, "B-USD": good.copy()}
    start, end = date(2026, 1, 5), date(2026, 1, 10)
    coverage = _universe_coverage(daily_ind, start, end)
    assert coverage == 1.0
