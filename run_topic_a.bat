@echo off
REM Double-click to run topic A (continue federated training after unlearning). Options: run_topic_a.bat --estimate
cd /d %~dp0
set PYTHONUTF8=1
python run_all.py --topic-a %*
pause
