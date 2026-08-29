@echo off
:: SLC live paper-forward preflight. Run ~8:35am ET. Validates guardrails,
:: rule-freeze, and account ID; bootstraps/backfills reducer state for every
:: symbol; detects stock splits and orphaned overnight positions.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\slc_live\)
cd /d "%~dp0..\.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting live_slc preflight >> logs\slc_preflight_scheduler.log 2>&1
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.run_slc_live --stage preflight >> logs\slc_preflight_scheduler.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] live_slc preflight complete exit_code=%rc% >> logs\slc_preflight_scheduler.log 2>&1
exit /b %rc%
