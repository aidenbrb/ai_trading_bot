"""Static protections for the Windows scheduler wrapper scripts."""
import subprocess
from pathlib import Path
import re

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts" / "slc_live"
ALL_WRAPPERS = [
    SCRIPT_ROOT / "run_slc_preflight.bat",
    SCRIPT_ROOT / "run_slc_cycle.bat",
    SCRIPT_ROOT / "run_slc_closeout.bat",
    SCRIPT_ROOT / "run_slc_schedule_health.bat",
]


def test_scheduler_stages_use_distinct_log_files():
    """Simultaneous cmd.exe append redirection to scheduler.log caused
    one of the cycle/guardian tasks to return Windows result 1. Each stage
    now owns a separate append target even if Task Scheduler starts both."""
    log_names = []
    for script in ALL_WRAPPERS:
        text = script.read_text(encoding="utf-8").lower()
        matches = re.findall(r"logs\\([a-z0-9_]+\.log)", text)
        assert matches
        assert "scheduler.log" not in matches
        assert len(set(matches)) == 1
        log_names.append(matches[0])
    assert len(set(log_names)) == len(ALL_WRAPPERS)


def test_closeout_wrapper_documents_safe_second_offset():
    text = (SCRIPT_ROOT / "run_slc_closeout.bat").read_text(encoding="utf-8")
    assert "12:55:45" in text
    assert "shared process lock" in text


# -- Exit-code preservation (sleep-fix Part 5) -------------------------------
#
# All four wrappers used to `echo` a completion line after the Python call
# with no exit-code capture, so run_hidden.vbs's exit-code forwarding to
# Task Scheduler always reported success (0) regardless of the real
# result - a scheduled task could be silently failing every day and Task
# Scheduler would never know. A PATH-shadowed fake python.exe can't catch
# this: every wrapper invokes an absolute .venv\Scripts\python.exe path,
# which PATH manipulation cannot intercept.

def test_every_wrapper_captures_and_forwards_the_real_exit_code():
    """Phase 6 Step 2: run_slc_preflight.bat/run_slc_cycle.bat now invoke
    python.exe TWICE - once for the independent Tier-1 verifier, once for
    the actual live_slc launch - so "the first python.exe line" is no
    longer necessarily "the" invocation whose exit code matters for the
    wrapper's own final exit code. Every python.exe invocation must still
    capture its own exit code on the immediately-following line (into
    SOME variable - verify_rc for the verifier, rc for the main launch);
    specifically the FINAL python.exe invocation (the one whose result the
    wrapper ultimately reports) must capture into "rc", matching the
    trailing `exit /b %rc%`."""
    for script in ALL_WRAPPERS:
        text = script.read_text(encoding="utf-8")
        assert re.search(r'set\s+"rc=%ERRORLEVEL%"', text), f"{script.name} missing rc capture"
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        assert lines[-1] == "exit /b %rc%", f"{script.name} must end with exit /b %rc%"
        python_line_indices = [i for i, ln in enumerate(lines) if "python.exe" in ln.lower()]
        assert python_line_indices, f"{script.name} has no python.exe invocation"
        # Every python.exe invocation captures ITS OWN exit code on the
        # very next line - never left unchecked, never captured several
        # lines later after some unrelated command.
        for idx in python_line_indices:
            assert re.match(r'set\s+"\w+=%ERRORLEVEL%"', lines[idx + 1]), (
                f"{script.name}: python.exe invocation at line {idx} has no immediate exit-code capture"
            )
        # The LAST python.exe invocation is the one whose result the
        # wrapper ultimately reports - it must capture into "rc"
        # specifically, matching the trailing exit /b %rc%.
        assert 'set "rc=%ERRORLEVEL%"' in lines[python_line_indices[-1] + 1]


def test_run_hidden_vbs_forwards_exit_code_in_isolation(tmp_path):
    """Isolates run_hidden.vbs's own exit-code forwarding from Python/
    live_slc entirely: a dummy batch file that just exits 37."""
    dummy = tmp_path / "dummy.bat"
    dummy.write_text("@echo off\r\nexit /b 37\r\n", encoding="utf-8")
    vbs = SCRIPT_ROOT / "run_hidden.vbs"

    result = subprocess.run(
        ["cscript.exe", "//nologo", str(vbs), str(dummy)],
        capture_output=True, timeout=30,
    )
    assert result.returncode == 37
