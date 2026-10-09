@echo off
REM Double-click to run every experiment. Options: run_all.bat --estimate  /  run_all.bat --workers 3
cd /d %~dp0
set PYTHONUTF8=1
python run_all.py %*
pause
