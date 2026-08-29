"""
Tests for live_slc/check_schedule_health.py - the time-aware diagnostic
written after a performance review found 3 of the last 4 trading days
with real, lost cycle coverage (58/72, 60/72, 1/72), caused by the
machine sleeping overnight. See the approved plan (sleep-fix) for full
context; this file exercises every scenario it enumerates.
"""
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import live_slc.check_schedule_health as csh
import live_slc.models as models
from live_slc.models import SlcCycleRun, get_live_slc_session

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "LIVE_SLC_DB_PATH", tmp_path / "test.db")
    models.init_live_slc_db()


def _et(day: date, hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=EASTERN)


def _to_naive_utc(dt_aware: datetime) -> datetime:
    return dt_aware.astimezone(UTC).replace(tzinfo=None)


def _seed(stage: str, cycle_time_et: datetime, status: str = "completed", heartbeat_et=None):
    heartbeat_et = heartbeat_et or cycle_time_et
    with get_live_slc_session() as session:
        session.add(SlcCycleRun(
            stage=stage,
            status=status,
            cycle_time_utc=_to_naive_utc(cycle_time_et),
            heartbeat_at=_to_naive_utc(heartbeat_et),
        ))


def _all_slots(day: date):
    return csh._session_slots_et(day)


# -- On-time day: everything present ------------------------------------

def test_on_time_day_is_pass():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE  # 2026-08-31, a Monday trading day
    _seed("preflight", _et(day, 8, 35))
    for slot in _all_slots(day):
        _seed("cycle", slot)
    _seed("closeout", _et(day, 16, 5))

    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.status == "PASS"
    assert result.preflight.status == "PASS"
    assert result.cycle.status == "PASS"
    assert result.closeout.status == "PASS"


# -- Preflight: legacy vs. instrumented dates ----------------------------

def test_late_preflight_on_effective_date_is_warning_with_delay_minutes():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    _seed("preflight", _et(day, 11, 35))  # 3 hours (180 min) late
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.preflight.status == "WARNING"
    assert "180" in result.preflight.detail
    assert result.status == "WARNING"


def test_missing_preflight_before_effective_date_is_legacy_not_warning():
    day = date(2026, 8, 28)  # Friday, predates instrumentation
    assert day < csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    # Fully cover cycle/closeout so only preflight's own contribution is at
    # play - proves NO_TELEMETRY_LEGACY truly never taints the overall status.
    for slot in _all_slots(day):
        _seed("cycle", slot)
    _seed("closeout", _et(day, 16, 5))

    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.preflight.status == "NO_TELEMETRY_LEGACY"
    assert result.status == "PASS"  # must never contribute to the exit code


def test_missing_preflight_on_effective_date_is_warning():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.preflight.status == "WARNING"


# -- Missing cycle slots --------------------------------------------------

def test_missing_cycle_slots_reports_correct_count_and_warns():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    _seed("preflight", _et(day, 8, 35))
    slots = _all_slots(day)
    skipped = slots[10:13]  # skip 3 slots
    for slot in slots:
        if slot not in skipped:
            _seed("cycle", slot)

    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.cycle.status == "WARNING"
    assert len(result.cycle.missing_slots) == 3
    assert set(result.cycle.missing_slots) == {s.strftime("%H:%M") for s in skipped}


def test_duplicate_rows_in_one_slot_do_not_inflate_coverage():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    slots = _all_slots(day)
    for slot in slots:
        _seed("cycle", slot)
    _seed("cycle", slots[0] + timedelta(minutes=1))  # duplicate, same bucket as slots[0]

    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.cycle.missing_slots == []
    assert result.cycle.present_count == len(slots)


# -- Mid-session grace period, exact -------------------------------------

def test_mid_session_grace_period_exact_five_minutes():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    for hh, mm in [(9, 36), (9, 41), (9, 46), (9, 51), (9, 56)]:
        _seed("cycle", _et(day, hh, mm))
    # 10:01 deliberately left unseeded (still "in progress")

    result_pass = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 10, 5))
    assert result_pass.cycle.status == "PASS"
    assert result_pass.cycle.missing_slots == []
    assert result_pass.closeout.status == "PENDING"

    result_warn = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 10, 7))
    assert result_warn.cycle.status == "WARNING"
    assert result_warn.cycle.missing_slots == ["10:01"]
    assert result_warn.closeout.status == "PENDING"


# -- Holiday / non-trading day --------------------------------------------

def test_non_trading_day_is_not_applicable():
    day = date(2026, 8, 29)  # Saturday
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 0))
    assert result.status == "NOT_APPLICABLE"
    assert result.preflight is None
    assert result.cycle is None


# -- Early-close golden test, via the real confirmation_within_entry_cutoff --

@dataclass
class _FakeConfirmation:
    entry_time: pd.Timestamp


def test_early_close_eligibility_matches_real_entry_cutoff_function():
    from live_slc.run_slc_live import confirmation_within_entry_cutoff

    day = date(2024, 11, 29)  # day after Thanksgiving, closes 1:00 PM ET
    slot_eligible = _et(day, 12, 31)
    slot_not_eligible = _et(day, 12, 36)

    conf_eligible = _FakeConfirmation(entry_time=pd.Timestamp(_to_naive_utc(slot_eligible - timedelta(minutes=1))))
    conf_not_eligible = _FakeConfirmation(entry_time=pd.Timestamp(_to_naive_utc(slot_not_eligible - timedelta(minutes=1))))
    assert confirmation_within_entry_cutoff(conf_eligible, day) is True
    assert confirmation_within_entry_cutoff(conf_not_eligible, day) is False

    all_slots = _all_slots(day)
    eligible_slots = csh._entry_eligible_slots_et(day, all_slots)
    assert slot_eligible in eligible_slots
    assert slot_not_eligible not in eligible_slots
    # The whole point of the split: raw Windows cadence is unaffected by
    # the early close, only the entry-eligible subset shrinks.
    assert len(eligible_slots) < len(all_slots)


# -- DST boundaries ---------------------------------------------------------

@pytest.mark.parametrize("day", [date(2026, 3, 9), date(2026, 11, 2)])
def test_slot_grid_stays_correct_across_dst_boundaries(day):
    slots = _all_slots(day)
    assert slots[0].hour == 9 and slots[0].minute == 36
    assert slots[-1].hour == 15 and slots[-1].minute == 31
    assert len(slots) == 72


# -- Failed / stale-running rows: independent, both-reported signals -----

def test_failed_cycle_row_warns_independently_even_when_slot_already_missing():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    slots = _all_slots(day)
    failed_slot = slots[5]
    for slot in slots:
        if slot != failed_slot:
            _seed("cycle", slot)
    _seed("cycle", failed_slot, status="failed")

    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.cycle.status == "WARNING"
    assert failed_slot.strftime("%H:%M") in result.cycle.missing_slots  # still counted missing
    assert result.cycle.failed_rows == 1  # AND independently reported


def test_stale_running_cycle_row_warns():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    slots = _all_slots(day)
    for slot in slots[:5]:
        _seed("cycle", slot)
    stuck_time = _et(day, 9, 56)
    with get_live_slc_session() as session:
        session.add(SlcCycleRun(
            stage="cycle", status="running",
            cycle_time_utc=_to_naive_utc(stuck_time),
            heartbeat_at=_to_naive_utc(stuck_time),  # never updated again -> stale
        ))
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    assert result.cycle.status == "WARNING"
    assert result.cycle.stale_running_rows == 1


# -- Closeout ---------------------------------------------------------------

def test_closeout_missing_after_window_opens_warns():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 13, 0))
    assert result.closeout.status == "WARNING"


def test_closeout_failed_warns():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    _seed("closeout", _et(day, 13, 0), status="failed")
    result = csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 13, 5))
    assert result.closeout.status == "WARNING"


# -- Structural read-only guarantee -----------------------------------------

def test_db_connection_is_structurally_read_only():
    conn = csh._read_only_connection(models.LIVE_SLC_DB_PATH)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute(
                "INSERT INTO slc_cycle_runs (id, stage, status, errors_json) "
                "VALUES ('x', 'cycle', 'completed', '[]')"
            )
    finally:
        conn.close()


def test_evaluate_schedule_health_never_modifies_the_database_file():
    day = csh.PREFLIGHT_TELEMETRY_EFFECTIVE_DATE
    _seed("preflight", _et(day, 8, 35))
    for slot in _all_slots(day):
        _seed("cycle", slot)
    _seed("closeout", _et(day, 16, 5))

    before = models.LIVE_SLC_DB_PATH.read_bytes()
    csh.evaluate_schedule_health(day, models.LIVE_SLC_DB_PATH, now_et=_et(day, 16, 10))
    after = models.LIVE_SLC_DB_PATH.read_bytes()
    assert before == after
