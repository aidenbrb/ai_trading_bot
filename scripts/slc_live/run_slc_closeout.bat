@echo off
:: SLC live paper-forward closeout guardian. Fixed Windows trigger,
:: 12:55pm-4:05pm ET every minute (Windows cannot dynamically schedule
:: relative to the exchange close) - the Python stage itself no-ops until
:: the calendar-derived official-close-minus-5-minutes is reached, then
:: flattens, retries every minute, reconciles after close.
:: Register this trigger at 45 seconds past the minute (12:55:45 start),
:: so a five-minute entry cycle always acquires the shared process lock
:: first. Separate logs prevent simultaneous cmd.exe redirection failures.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\slc_live\)
::
:: Phase 6 Step 2: deliberately does NOT run verify_tier1_independent.py
:: before launching, unlike run_slc_preflight.bat/run_slc_cycle.bat - a
:: hard pre-launch Tier-1 gate would block closeout from ever running
:: whenever no signed baseline exists yet (e.g. right now), which
:: contradicts assert_closeout_preconditions()'s own explicit design:
:: closeout must stay reachable independent of Tier-2/signature problems,
:: because its job is safely exiting/protecting already-open positions
:: regardless of guardrail state elsewhere. Tier-1 also doesn't include
:: anything closeout's own code path touches (execution.py/closeout.py
:: are Tier-2, independently hash-checked inside assert_closeout_
:: preconditions() itself, best-effort).
cd /d "%~dp0..\.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting live_slc closeout check >> logs\slc_closeout_scheduler.log 2>&1
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.run_slc_live --stage closeout >> logs\slc_closeout_scheduler.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] live_slc closeout check complete exit_code=%rc% >> logs\slc_closeout_scheduler.log 2>&1
exit /b %rc%
