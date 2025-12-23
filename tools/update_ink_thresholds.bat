@echo off
setlocal
set SUMMARY=results\compare_json\summary.json
set INPUT_DIR=results\compare_json

if "%1"=="" (
  echo Usage: update_ink_thresholds.bat ^<SKU_JSON_PATH^>
  echo Example: update_ink_thresholds.bat config\sku_db\SKU001.json
  exit /b 1
)

python tools\compute_ink_thresholds.py --input-dir "%INPUT_DIR%" --output "%SUMMARY%"
if errorlevel 1 exit /b 1
python tools\generate_ink_threshold_report.py --summary "%SUMMARY%" --output "results\compare_json\report.md"
if errorlevel 1 exit /b 1
python tools\update_sku_ink_thresholds.py --summary "%SUMMARY%" --sku "%~1"
if errorlevel 1 exit /b 1

echo Updated ink thresholds for %~1
