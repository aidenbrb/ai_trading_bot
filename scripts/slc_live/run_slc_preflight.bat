@echo off
:: SLC live paper-forward preflight. Run ~8:35am ET. Validates guardrails,
:: rule-freeze, and account ID; bootstraps/backfills reducer state for every
:: symbol; detects stock splits and orphaned overnight positions.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\slc_live\)
cd /d "%~dp0..\.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting live_slc preflight >> logs\slc_preflight_scheduler.log 2>&1
:: Phase 6 Step 2: independent Tier-1 verifier (scripts/slc_live/verify_
:: tier1_independent.py, zero live_slc imports) runs BEFORE the live
:: process - fails closed, never launches run_slc_live.py past a nonzero
:: exit here.
"%~dp0..\..\.venv\Scripts\python.exe" "%~dp0verify_tier1_independent.py" >> logs\slc_preflight_scheduler.log 2>&1
set "verify_rc=%ERRORLEVEL%"
if not "%verify_rc%"=="0" (
    echo [%DATE% %TIME%] Tier-1 independent verification FAILED exit_code=%verify_rc% - live_slc NOT launched >> logs\slc_preflight_scheduler.log 2>&1
    exit /b %verify_rc%
)
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.run_slc_live --stage preflight >> logs\slc_preflight_scheduler.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] live_slc preflight complete exit_code=%rc% >> logs\slc_preflight_scheduler.log 2>&1
exit /b %rc%
