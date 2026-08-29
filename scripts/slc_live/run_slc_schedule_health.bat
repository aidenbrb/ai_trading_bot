@echo off
cd /d "%~dp0..\.."
if not exist "logs" mkdir "logs"
echo [%DATE% %TIME%] Starting live_slc schedule health check >> logs\slc_schedule_health.log 2>&1
"%~dp0..\..\.venv\Scripts\python.exe" -m live_slc.check_schedule_health >> logs\slc_schedule_health.log 2>&1
set "rc=%ERRORLEVEL%"
echo [%DATE% %TIME%] schedule health complete exit_code=%rc% >> logs\slc_schedule_health.log 2>&1
exit /b %rc%
