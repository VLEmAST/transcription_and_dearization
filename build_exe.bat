@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "APP_NAME=speech_recognition"
set "BUILD_VENV=.venv-build-py312"
set "PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128"

echo [1/7] Checking Python 3.12...
py -3.12 -c "import struct; raise SystemExit(0 if struct.calcsize('P') * 8 == 64 else 1)"
if errorlevel 1 (
    echo ERROR: Python 3.12 x64 is required.
    goto :failed
)

echo [2/7] Creating the build environment...
if not exist "%BUILD_VENV%\Scripts\python.exe" (
    py -3.12 -m venv "%BUILD_VENV%"
    if errorlevel 1 goto :failed
)

set "BUILD_PYTHON=%CD%\%BUILD_VENV%\Scripts\python.exe"

echo [3/7] Updating build tools...
"%BUILD_PYTHON%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :failed

echo [4/7] Installing CUDA-enabled PyTorch...
"%BUILD_PYTHON%" -m pip install --upgrade torch torchaudio --index-url "%PYTORCH_INDEX_URL%"
if errorlevel 1 (
    echo ERROR: CUDA-enabled PyTorch could not be installed.
    echo You may change PYTORCH_INDEX_URL at the beginning of build_exe.bat.
    goto :failed
)

echo [5/7] Installing application dependencies...
"%BUILD_PYTHON%" -m pip install --upgrade -r requirements.txt
if errorlevel 1 goto :failed

echo [6/7] Building the EXE...
"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean speech_recognition.spec
if errorlevel 1 goto :failed

echo [7/7] Checking the assembled folder...
"%BUILD_PYTHON%" check_build.py "dist\%APP_NAME%"
if errorlevel 1 goto :failed

echo.
echo BUILD COMPLETED.
echo Send the entire folder: dist\%APP_NAME%
echo Add your models manually to: dist\%APP_NAME%\models
echo Do not send only %APP_NAME%.exe.
pause
exit /b 0

:failed
echo.
echo BUILD FAILED. Read the error above and do not use a partial dist folder.
pause
exit /b 1
