@echo off
REM Color Meter Web UI 실행 스크립트
REM ====================================

echo.
echo ========================================
echo   Color Meter Web UI
echo   Contact Lens Inspection System
echo ========================================
echo.

REM 가상환경 활성화 (있다면)
if exist venv\Scripts\activate.bat (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Python 버전 확인
echo [INFO] Python version:
python --version
echo.

REM 필수 패키지 확인
echo [INFO] Checking dependencies...
python -c "import fastapi, uvicorn" 2>nul
if errorlevel 1 (
    echo [ERROR] FastAPI or Uvicorn not found!
    echo [INFO] Installing dependencies...
    pip install fastapi uvicorn python-multipart
)

REM 웹 서버 실행
echo.
echo [INFO] Starting web server...
echo [INFO] Access at: http://localhost:8000
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --reload

pause
