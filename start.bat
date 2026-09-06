@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================
echo  Attribute Classifier MCP Server
echo ============================================
echo.

echo Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo.
echo Checking dependencies...
python -c "import mcp; import torch; import ultralytics; print('All dependencies OK')" 2>nul
if errorlevel 1 (
    echo [WARN] Some dependencies missing. Installing...
    pip install -r requirements.txt
)

echo.
echo Starting MCP Server via stdio...
echo Press Ctrl+C to stop.
echo.
python src/server.py %*
pause
