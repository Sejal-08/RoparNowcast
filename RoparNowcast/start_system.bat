@echo off
title RoparNowcast Launcher
echo ===================================================
echo 🚀 STARTING ROPAR NOWCAST LOCAL SYSTEM
echo ===================================================

cd /d "%~dp0"
call venv\Scripts\activate.bat

echo.
echo [1/3] Starting Auto-Predictor (Generates Forecasts)...
start "Nowcast Predictor" cmd /k "python auto_run.py"

echo.
echo [2/3] Starting API Server...
start "Nowcast API" cmd /k "uvicorn api:app --reload"

echo.
echo [3/3] Launching Dashboard...
echo    (This will open in your default browser)
streamlit run dashboard.py

pause
