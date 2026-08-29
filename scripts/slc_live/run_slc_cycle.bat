@echo off
:: SLC live paper-forward entry cycle. Scheduled every 5 minutes, 9:36am ET
:: through 3:31pm ET (amendment_004) - no entry-cycle invocations after that;
:: the closeout guardian owns everything from there. Task Scheduler's own
:: "don't start a new instance if already running" setting plus
:: live_slc's OS-level process lock both guard against overlap.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\slc_live\)
cd /d "%~dp0..\.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting live_slc cycle >> logs\slc_cycle_scheduler.log 2>&1
:: Phase 6 Step 2: independent Tier-1 verifier - see run_slc_preflight.bat's
:: comment for the full rationale. Fails closed, never launches run_slc_
:: live.py past a nonzero exit here.
"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0verify_tier1_independent.py" >> logs\slc_cycle_scheduler.log 2>&1
set "verify_rc=%ERRORLEVEL%"
if not "%verify_rc%"=="0" (
    echo [%DATE% %TIME%] Tier-1 independent verification FAILED exit_code=%verify_rc% - live_slc NOT launched >> logs\slc_cycle_scheduler.log 2>&1
    exit /b %verify_rc%
)
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.run_slc_live --stage cycle >> logs\slc_cycle_scheduler.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] live_slc cycle complete exit_code=%rc% >> logs\slc_cycle_scheduler.log 2>&1
exit /b %rc%
