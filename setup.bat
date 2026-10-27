@echo off
rem Creates .venv and installs everything, including PyTorch with CUDA. Run from PowerShell: .\setup.bat
rem Set TORCH_INDEX to use another CUDA build, e.g. $env:TORCH_INDEX="https://download.pytorch.org/whl/cu126"
cd /d "%~dp0"
if "%TORCH_INDEX%"=="" set TORCH_INDEX=https://download.pytorch.org/whl/cu128
if not exist .venv\Scripts\python.exe (
    py -3.13 -m venv .venv || py -3 -m venv .venv || exit /b 1
)
set PY=.venv\Scripts\python
%PY% -m pip install --upgrade pip || exit /b 1

rem A CPU-only torch from PyPI must go before the CUDA build can be installed.
%PY% -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul || %PY% -m pip uninstall -y torch
%PY% -m pip install torch --index-url %TORCH_INDEX% || exit /b 1

rem Pin the CUDA torch so the remaining packages cannot swap it for the CPU build on PyPI.
%PY% -c "import torch; print('torch==' + torch.__version__)" > .venv\torch-constraint.txt
%PY% -m pip install -e ".[dev]" -c .venv\torch-constraint.txt || exit /b 1

%PY% -c "import torch; print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())"
%PY% -m pytest -q
