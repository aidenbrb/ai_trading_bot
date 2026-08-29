@echo off
:: Day-trading (5-min ORB) mode, post-close outcome reconstruction. Run
:: ~4:10pm ET, AFTER the actual close (not before) so it never reads an
:: incomplete candle - see nodes/intraday_shadow_node.py for why.
:: Paths resolve relative to this script (%~dp0 = ...\ai_trading_bot\scripts\)
cd /d "%~dp0.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting ai_trading_bot day-mode shadow reconstruction >> logs\scheduler.log 2>&1
"%~dp0..\.venv\Scripts\python.exe" -m nodes.intraday_shadow_node >> logs\scheduler.log 2>&1
echo [%DATE% %TIME%] Day-mode shadow reconstruction complete >> logs\scheduler.log 2>&1
