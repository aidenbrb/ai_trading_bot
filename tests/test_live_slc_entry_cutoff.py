"""
Proves the rev. 11 point 1 fix: the entry cutoff is evaluated per-
confirmation against `entry_time`, never against the cycle's own
wall-clock invocation time. The bug this replaces: comparing `now` (e.g.
3:31 PM, when the cycle that processes the 3:25-3:30 bar actually runs)
against a 3:30 PM cutoff would incorrectly reject the last legitimately
permitted signal of every session.
"""
from dataclasses import dataclass
from datetime import date

import pandas as pd
import pytest

from live_slc.run_slc_live import compute_entry_cutoff, confirmation_within_entry_cutoff


@dataclass
class _FakeConfirmation:
    entry_time: pd.Timestamp


def test_regular_session_cutoff_is_330pm_et():
    cutoff = compute_entry_cutoff(date(2024, 10, 2))
    assert cutoff.hour == 15 and cutoff.minute == 30


def test_early_close_cutoff_is_30min_before_official_close():
    # 2024-11-29 (day after Thanksgiving) closes at 1:00 PM ET
    cutoff = compute_entry_cutoff(date(2024, 11, 29))
    assert cutoff.hour == 12 and cutoff.minute == 30


def test_non_trading_day_returns_none():
    assert compute_entry_cutoff(date(2024, 10, 5)) is None  # Saturday


def test_confirmation_exactly_at_cutoff_is_permitted_regardless_of_wall_clock():
    """The exact bug scenario: a cycle invoked at 3:31 PM processes the
    confirmation whose entry_time is exactly 3:30 PM - this must be
    permitted. There is no `now` parameter to this function at all -
    wall-clock time is structurally incapable of influencing the result."""
    day = date(2024, 10, 2)
    confirmation = _FakeConfirmation(entry_time=pd.Timestamp("2024-10-02 19:30:00"))  # 3:30 PM EDT = 19:30 UTC
    assert confirmation_within_entry_cutoff(confirmation, day) is True


def test_confirmation_after_cutoff_is_rejected():
    day = date(2024, 10, 2)
    confirmation = _FakeConfirmation(entry_time=pd.Timestamp("2024-10-02 19:35:00"))  # 3:35 PM EDT
    assert confirmation_within_entry_cutoff(confirmation, day) is False


def test_early_close_confirmation_exactly_at_cutoff_is_permitted():
    day = date(2024, 11, 29)
    confirmation = _FakeConfirmation(entry_time=pd.Timestamp("2024-11-29 17:30:00"))  # 12:30 PM EST = 17:30 UTC
    assert confirmation_within_entry_cutoff(confirmation, day) is True


def test_early_close_confirmation_that_would_be_fine_on_a_regular_day_is_rejected():
    """A confirmation at, say, 2:00 PM ET would be fine on a regular
    session (well before 3:30) but must be rejected on this early-close
    day (cutoff is 12:30 PM)."""
    day = date(2024, 11, 29)
    confirmation = _FakeConfirmation(entry_time=pd.Timestamp("2024-11-29 19:00:00"))  # 2:00 PM EST
    assert confirmation_within_entry_cutoff(confirmation, day) is False


def test_confirmation_on_non_trading_day_is_rejected():
    confirmation = _FakeConfirmation(entry_time=pd.Timestamp("2024-10-05 19:00:00"))
    assert confirmation_within_entry_cutoff(confirmation, date(2024, 10, 5)) is False
