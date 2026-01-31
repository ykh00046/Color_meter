@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

REM ============================================
REM Contact Lens Color Inspection System
REM Dependency Installation Script (Windows)
REM ============================================

echo.
echo ========================================
echo  Installing Dependencies
echo ========================================
echo.

REM Check Python version
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not found. Please install Python 3.10 or higher.
    exit /b 1
)

echo.
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/4] Installing core dependencies...
pip install -r requirements.txt

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies.
    exit /b 1
)

echo.
echo [3/4] Verifying installation...
python tools/check_imports.py

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some imports failed. Please check the output above.
    exit /b 1
)

echo.
echo [4/4] Running quick tests...
pytest tests/test_ink_estimator.py::TestInkEstimatorSampling::test_sample_ink_pixels_basic -v

if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Quick test failed. Please investigate.
    exit /b 1
)

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run full test suite: pytest tests/ -v
echo   2. Start Web UI: python -m src.web.app
echo   3. Run pipeline: python -m src.services.inspection_service data/raw_images/SKU001_OK_001.jpg
echo.

exit /b 0
