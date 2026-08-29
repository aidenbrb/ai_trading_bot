@echo off
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\)
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting ai_trading_bot daily run >> logs\scheduler.log 2>&1
"%~dp0..\.venv\Scripts\python.exe" run_pipeline.py >> logs\scheduler.log 2>&1
echo [%DATE% %TIME%] Run complete >> logs\scheduler.log 2>&1
