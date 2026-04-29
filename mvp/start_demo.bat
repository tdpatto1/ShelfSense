@echo off
setlocal
cd /d "%~dp0"

echo Starting ShelfSense MVP demo...
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python was not found on this computer.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During install, check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo Dependency installation failed. Read the error above.
    echo.
    pause
    exit /b 1
)

python demo.py
if %errorlevel% neq 0 (
    echo.
    echo The demo failed to start. Read the error above.
    echo.
    pause
    exit /b 1
)

pause
