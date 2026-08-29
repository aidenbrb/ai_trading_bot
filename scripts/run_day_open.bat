@echo off
:: Day-trading (5-min ORB) mode, opening signal generation. Run ~9:36am ET,
:: right after the first 5-min candle closes. Signal-only - --stock-strategy
:: day + --market stocks together are required by run_pipeline.py's day-mode
:: safety gate, which hard-blocks risk/execution/options from ever running
:: alongside this flag.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\)
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting ai_trading_bot day-mode open signal run >> logs\scheduler.log 2>&1
"%~dp0..\.venv\Scripts\python.exe" run_pipeline.py --stock-strategy day --nodes strategy --market stocks >> logs\scheduler.log 2>&1
echo [%DATE% %TIME%] Day-mode open signal run complete >> logs\scheduler.log 2>&1
