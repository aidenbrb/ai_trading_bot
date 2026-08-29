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
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.run_slc_live --stage cycle >> logs\slc_cycle_scheduler.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] live_slc cycle complete exit_code=%rc% >> logs\slc_cycle_scheduler.log 2>&1
exit /b %rc%
