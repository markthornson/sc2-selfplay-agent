@echo off
rem Runs scripts\evaluate.py in the project environment. Arguments are passed on.
cd /d "%~dp0"
.venv\Scripts\python scripts\evaluate.py %*
