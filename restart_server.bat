@echo off
echo Killing old server...
taskkill /F /IM python.exe 2>nul

echo Cleaning cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul

echo Starting server...
python -m src.web.app
