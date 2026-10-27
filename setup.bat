@echo off
rem Creates .venv and installs everything, including PyTorch with CUDA. Run from PowerShell: .\setup.bat
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
    py -3.13 -m venv .venv || py -3 -m venv .venv || exit /b 1
)
.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu124 || exit /b 1
.venv\Scripts\python -m pip install -e ".[dev]" || exit /b 1
.venv\Scripts\python -c "import torch; print('CUDA:', torch.cuda.is_available())"
.venv\Scripts\python -m pytest -q
