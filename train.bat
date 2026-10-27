@echo off
rem Runs scripts\train.py in the project environment. Arguments are passed on.
cd /d "%~dp0"
.venv\Scripts\python scripts\train.py %*
