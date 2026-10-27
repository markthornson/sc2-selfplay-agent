@echo off
rem Plays one game against the built-in AI. Extra arguments are passed on, e.g. .\smoke.bat --realtime
cd /d "%~dp0"
.venv\Scripts\python scripts\smoke_test.py %*
