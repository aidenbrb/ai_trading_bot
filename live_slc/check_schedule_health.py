"""
Time-aware schedule-health check for live_slc, meant to be run unattended
by its own Scheduled Task (~10:00 AM ET, Mon-Fri): python -m
live_slc.check_schedule_health [--date YYYY-MM-DD]

Written after a performance review found 3 of the last 4 trading days
with real, lost cycle coverage (58/72, 60/72, 1/72 scheduled slots),
traced to the machine being put to sleep at night. This script answers,
every morning, "did preflight/cycle/closeout actually run on schedule
yesterday/today" without a human having to reconstruct it by hand from
Windows Event Log forensics again.

Structurally read-only: opens its own dedicated sqlite3 connection in
`mode=ro`, issuing plain SELECTs directly against live_slc.db - never
live_slc.models.get_live_slc_session() (a read-write ORM session), never
init_live_slc_db() or anything migration-related. SQLite itself rejects
any write attempt at the connection level under mode=ro, so this is a
structural guarantee, not just a convention.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from utils.market_calendar import is_trading_day, session_for

EASTERN = ZoneInfo("America/New_York")

# The first session whose 8:35 AM preflight actually executes the
# _start_cycle_run("preflight")/_finish_cycle_run(...) instrumentation
# added alongside this script (see run_slc_live.py:run_preflight()).
# There is no retroactive telemetry - any trading day before this date
# will never have a "preflight" SlcCycleRun row, by construction, and
# that must never be misreported as a missed/failed preflight.
PREFLIGHT_TELEMETRY_EFFECTIVE_DATE = date(2026, 8, 31)

# run_slc_preflight.bat targets ~8:35 AM ET. A few minutes of scheduler
# jitter is normal; the incidents this script exists to catch were
# preflight starting 2-5 hours late, so this leaves comfortable margin
# while still catching a genuine problem.
PREFLIGHT_EXPECTED_TIME_ET = dt_time(8, 35)
PREFLIGHT_LATE_THRESHOLD_MINUTES = 15

# A "running" SlcCycleRun row whose heartbeat_at is older than this, with
# no _finish_cycle_run() ever landing, is treated as stuck/crashed rather
# than still in progress.
RUNNING_STALE_THRESHOLD_MINUTES = 10

# The fixed Task Scheduler cadence for the entry-cycle task (see
# scripts/slc_live/run_slc_cycle.bat's own comment): every 5 minutes,
# 9:36 AM through 3:31 PM ET, unconditionally - Windows does not know
# about early closes, so this grid is the same size every weekday.
SESSION_SLOT_START_ET = dt_time(9, 36)
SESSION_SLOT_END_ET = dt_time(15, 31)
SESSION_SLOT_INTERVAL_MINUTES = 5

# 5 seconds after the closeout guardian's own registered 12:55:45 ET
# trigger start - before this, closeout for the day simply hasn't had a
# chance to run yet, which is not a health problem.
CLOSEOUT_PENDING_UNTIL_ET = dt_time(12, 55, 45)


def _et_day_bounds_utc_naive(day: date) -> tuple[datetime, datetime]:
    """[start, end) naive-UTC bounds covering the full ET calendar day,
    matching this codebase's naive-UTC DB convention."""
    start_et = datetime.combine(day, dt_time(0, 0), tzinfo=EASTERN)
    end_et = start_et + timedelta(days=1)
    return (
        start_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
        end_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None),
    )


def _to_et(naive_utc: datetime) -> datetime:
    return naive_utc.replace(tzinfo=ZoneInfo("UTC")).astimezone(EASTERN)


def compute_entry_cutoff_et(day: date):
    """Duplicated-free reuse of run_slc_live.compute_entry_cutoff() - kept
    as a thin wrapper here only to avoid importing run_slc_live (which
    pulls in the Alpaca SDK, broker settings, etc.) into a script that
    must stay read-only and dependency-light."""
    from live_slc.run_slc_live import compute_entry_cutoff

    return compute_entry_cutoff(day)


def _session_slots_et(day: date) -> list[datetime]:
    session = session_for(day)
    if session is None:
        return []
    start = datetime.combine(day, SESSION_SLOT_START_ET, tzinfo=EASTERN)
    end = datetime.combine(day, SESSION_SLOT_END_ET, tzinfo=EASTERN)
    slots = []
    t = start
    step = timedelta(minutes=SESSION_SLOT_INTERVAL_MINUTES)
    while t <= end:
        slots.append(t)
        t += step
    return slots


def _entry_eligible_slots_et(day: date, slots: list[datetime]) -> list[datetime]:
    """slot_time - 1min <= compute_entry_cutoff(day): a cycle firing at a
    given scheduled slot processes the bar that closed one minute
    earlier, so its confirmations carry entry_time = slot_time - 1min."""
    import pandas as pd

    cutoff = compute_entry_cutoff_et(day)
    if cutoff is None:
        return []
    eligible = []
    for slot in slots:
        slot_ts = pd.Timestamp(slot)
        if (slot_ts - timedelta(minutes=1)) <= cutoff:
            eligible.append(slot)
    return eligible


@dataclass(frozen=True)
class CycleRunRow:
    id: str
    cycle_time_utc: datetime
    stage: str
    status: str
    heartbeat_at: datetime


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    uri = db_path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _fetch_cycle_runs(db_path: Path, day: date) -> list[CycleRunRow]:
    start_utc, end_utc = _et_day_bounds_utc_naive(day)
    conn = _read_only_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT id, cycle_time_utc, stage, status, heartbeat_at "
            "FROM slc_cycle_runs WHERE cycle_time_utc >= ? AND cycle_time_utc < ?",
            (start_utc.isoformat(sep=" "), end_utc.isoformat(sep=" ")),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()
    return [
        CycleRunRow(
            id=row[0],
            cycle_time_utc=datetime.fromisoformat(row[1]),
            stage=row[2],
            status=row[3],
            heartbeat_at=datetime.fromisoformat(row[4]),
        )
        for row in rows
    ]


@dataclass(frozen=True)
class PreflightHealth:
    status: str  # PASS | WARNING | NO_TELEMETRY_LEGACY
    detail: str


@dataclass(frozen=True)
class CycleCoverageHealth:
    status: str  # PASS | WARNING
    windows_scheduled_slots: int
    entry_eligible_slots: int
    present_count: int
    missing_slots: list
    failed_rows: int
    stale_running_rows: int
    detail: str


@dataclass(frozen=True)
class CloseoutHealth:
    status: str  # PASS | WARNING | PENDING
    detail: str


@dataclass(frozen=True)
class ScheduleHealthResult:
    check_date: date
    status: str  # PASS | WARNING | NOT_APPLICABLE
    preflight: Optional[PreflightHealth] = None
    cycle: Optional[CycleCoverageHealth] = None
    closeout: Optional[CloseoutHealth] = None
    warnings: list = field(default_factory=list)


def _evaluate_preflight(rows: list[CycleRunRow], day: date, now_et: datetime) -> PreflightHealth:
    preflight_rows = [r for r in rows if r.stage == "preflight"]
    if not preflight_rows:
        if day < PREFLIGHT_TELEMETRY_EFFECTIVE_DATE:
            return PreflightHealth(
                status="NO_TELEMETRY_LEGACY",
                detail="no preflight telemetry - predates instrumentation",
            )
        return PreflightHealth(status="WARNING", detail="no preflight run recorded")

    failed = [r for r in preflight_rows if r.status == "failed"]
    if failed:
        return PreflightHealth(status="WARNING", detail=f"{len(failed)} failed preflight run(s)")

    now_utc_naive = now_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    stale = [
        r for r in preflight_rows
        if r.status == "running"
        and (now_utc_naive - r.heartbeat_at) > timedelta(minutes=RUNNING_STALE_THRESHOLD_MINUTES)
    ]
    if stale:
        return PreflightHealth(status="WARNING", detail=f"{len(stale)} stale (stuck) preflight run(s)")

    completed = [r for r in preflight_rows if r.status == "completed"]
    if not completed:
        return PreflightHealth(status="WARNING", detail="preflight run(s) recorded but none completed")

    earliest = min(completed, key=lambda r: r.cycle_time_utc)
    started_et = _to_et(earliest.cycle_time_utc)
    expected_et = datetime.combine(day, PREFLIGHT_EXPECTED_TIME_ET, tzinfo=EASTERN)
    delay_minutes = (started_et - expected_et).total_seconds() / 60.0
    if delay_minutes > PREFLIGHT_LATE_THRESHOLD_MINUTES:
        return PreflightHealth(
            status="WARNING",
            detail=f"preflight started {delay_minutes:.0f} min late ({started_et.strftime('%H:%M:%S')} ET)",
        )
    return PreflightHealth(status="PASS", detail=f"preflight started {started_et.strftime('%H:%M:%S')} ET")


def _evaluate_cycle_coverage(
    rows: list[CycleRunRow], day: date, now_et: datetime
) -> CycleCoverageHealth:
    all_slots = _session_slots_et(day)
    eligible_slots = _entry_eligible_slots_et(day, all_slots)

    now_utc_naive = now_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    cycle_rows = [r for r in rows if r.stage == "cycle"]
    completed_times_et = [_to_et(r.cycle_time_utc) for r in cycle_rows if r.status == "completed"]
    failed_rows = [r for r in cycle_rows if r.status == "failed"]
    stale_rows = [
        r for r in cycle_rows
        if r.status == "running"
        and (now_utc_naive - r.heartbeat_at) > timedelta(minutes=RUNNING_STALE_THRESHOLD_MINUTES)
    ]

    def _slot_present(slot: datetime) -> bool:
        bucket_end = slot + timedelta(minutes=SESSION_SLOT_INTERVAL_MINUTES)
        return any(slot <= t < bucket_end for t in completed_times_et)

    # A slot is only "expected by now" once its own bucket has fully
    # elapsed - never the instant it starts - so an in-flight cycle is
    # never flagged as missing before it's had its full window to finish.
    expected_so_far = [
        slot for slot in eligible_slots
        if slot + timedelta(minutes=SESSION_SLOT_INTERVAL_MINUTES) <= now_et
    ]
    missing_slots = [slot for slot in expected_so_far if not _slot_present(slot)]
    present_count = len(expected_so_far) - len(missing_slots)

    problems = []
    if missing_slots:
        problems.append(f"{len(missing_slots)} missing scheduled cycle(s)")
    if failed_rows:
        problems.append(f"{len(failed_rows)} failed cycle run(s)")
    if stale_rows:
        problems.append(f"{len(stale_rows)} stale (stuck) cycle run(s)")

    status = "WARNING" if problems else "PASS"
    detail = "; ".join(problems) if problems else f"{present_count}/{len(expected_so_far)} expected cycles present"

    return CycleCoverageHealth(
        status=status,
        windows_scheduled_slots=len(all_slots),
        entry_eligible_slots=len(eligible_slots),
        present_count=present_count,
        missing_slots=[s.strftime("%H:%M") for s in missing_slots],
        failed_rows=len(failed_rows),
        stale_running_rows=len(stale_rows),
        detail=detail,
    )


def _evaluate_closeout(rows: list[CycleRunRow], day: date, now_et: datetime) -> CloseoutHealth:
    pending_until_et = datetime.combine(day, CLOSEOUT_PENDING_UNTIL_ET, tzinfo=EASTERN)
    if now_et < pending_until_et:
        return CloseoutHealth(status="PENDING", detail="closeout window has not opened yet today")

    closeout_rows = [
        r for r in rows
        if r.stage == "closeout" and _to_et(r.cycle_time_utc) >= pending_until_et
    ]
    failed = [r for r in closeout_rows if r.status == "failed"]
    if failed:
        return CloseoutHealth(status="WARNING", detail=f"{len(failed)} failed closeout run(s)")

    now_utc_naive = now_et.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
    stale = [
        r for r in closeout_rows
        if r.status == "running"
        and (now_utc_naive - r.heartbeat_at) > timedelta(minutes=RUNNING_STALE_THRESHOLD_MINUTES)
    ]
    if stale:
        return CloseoutHealth(status="WARNING", detail=f"{len(stale)} stale (stuck) closeout run(s)")

    completed = [r for r in closeout_rows if r.status == "completed"]
    if not completed:
        return CloseoutHealth(status="WARNING", detail="no closeout run recorded after closeout window opened")

    return CloseoutHealth(status="PASS", detail=f"{len(completed)} closeout run(s) completed")


def evaluate_schedule_health(
    check_date: date,
    db_path: Path,
    now_et: Optional[datetime] = None,
) -> ScheduleHealthResult:
    if now_et is None:
        now_et = datetime.now(EASTERN)

    if not is_trading_day(check_date):
        return ScheduleHealthResult(check_date=check_date, status="NOT_APPLICABLE")

    rows = _fetch_cycle_runs(db_path, check_date)

    preflight = _evaluate_preflight(rows, check_date, now_et)
    cycle = _evaluate_cycle_coverage(rows, check_date, now_et)
    closeout = _evaluate_closeout(rows, check_date, now_et)

    warnings = []
    if preflight.status == "WARNING":
        warnings.append(f"preflight: {preflight.detail}")
    if cycle.status == "WARNING":
        warnings.append(f"cycle: {cycle.detail}")
    if closeout.status == "WARNING":
        warnings.append(f"closeout: {closeout.detail}")

    overall = "WARNING" if warnings else "PASS"
    return ScheduleHealthResult(
        check_date=check_date,
        status=overall,
        preflight=preflight,
        cycle=cycle,
        closeout=closeout,
        warnings=warnings,
    )


def main() -> None:
    from live_slc.models import LIVE_SLC_DB_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="YYYY-MM-DD (ET calendar date). Defaults to today (ET) - "
        "this script runs unattended and must determine 'today' itself.",
    )
    args = parser.parse_args()
    check_date = (
        datetime.strptime(args.date, "%Y-%m-%d").date()
        if args.date
        else datetime.now(EASTERN).date()
    )

    result = evaluate_schedule_health(check_date, LIVE_SLC_DB_PATH)

    print(f"live_slc schedule health check - {result.check_date}")
    if result.status == "NOT_APPLICABLE":
        print("NOT_APPLICABLE - not a trading day")
        sys.exit(0)

    print(f"  preflight: {result.preflight.status} - {result.preflight.detail}")
    print(
        f"  cycle:     {result.cycle.status} - {result.cycle.detail} "
        f"(windows_scheduled_slots={result.cycle.windows_scheduled_slots}, "
        f"entry_eligible_slots={result.cycle.entry_eligible_slots})"
    )
    if result.cycle.missing_slots:
        print(f"             missing: {', '.join(result.cycle.missing_slots)}")
    print(f"  closeout:  {result.closeout.status} - {result.closeout.detail}")

    if result.status == "WARNING":
        print(f"WARNING ({len(result.warnings)} issue(s)):")
        for w in result.warnings:
            print(f"  - {w}")
        sys.exit(1)

    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
