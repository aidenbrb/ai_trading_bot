@echo off
:: Day-trading (5-min ORB) mode, morning setup. Run ~8:30-8:45am ET, before
:: the 9:30 open, so the 9:36 open job only has to fetch today's opening bar.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\)
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting ai_trading_bot day-mode preflight >> logs\scheduler.log 2>&1
"%~dp0..\.venv\Scripts\python.exe" -m nodes.intraday_preflight_node >> logs\scheduler.log 2>&1
echo [%DATE% %TIME%] Day-mode preflight complete >> logs\scheduler.log 2>&1
