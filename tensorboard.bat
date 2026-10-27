@echo off
rem Starts TensorBoard for the training runs; open http://localhost:6006 in a browser.
cd /d "%~dp0"
.venv\Scripts\tensorboard --logdir runs
