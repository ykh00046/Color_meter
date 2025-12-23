@echo off
chcp 65001 >nul
setlocal
title Color Meter Web UI Launcher

REM UTF-8 output for Python
set PYTHONIOENCODING=utf-8

set VENV_DIR=venv

if exist %VENV_DIR%\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call %VENV_DIR%\Scripts\activate.bat
) else (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        goto :end
    )
    call %VENV_DIR%\Scripts\activate.bat
)

echo.
echo ========================================================
echo   Color Meter System - Web UI Launcher
echo ========================================================
echo.

REM Dependency check and install
python -c "import fastapi, uvicorn, jinja2, multipart" 2>nul
if errorlevel 1 (
    echo [INFO] Installing missing dependencies...
    if exist requirements.txt (
        python -m pip install -r requirements.txt
    ) else (
        python -m pip install fastapi "uvicorn[standard]" python-multipart jinja2
    )
)

REM Open browser after a short delay
start /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

REM Start server
echo.
echo [INFO] Starting Web Server...
echo [INFO] Dashboard will open automatically in 3 seconds.
echo [INFO] URL: http://localhost:8000
echo.
echo [STOP] Press Ctrl+C to stop the server.
echo.

python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload --log-level info

:end
pause
