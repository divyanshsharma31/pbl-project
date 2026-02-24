@echo off
REM Streamlit App Launcher for AQI Prediction System

echo.
echo ========================================
echo   AQI Prediction System - Streamlit App
echo ========================================
echo.

REM Check if streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit -q
)

REM Run the app
echo Starting Streamlit App...
echo.
echo The app will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the app.
echo.

streamlit run app.py

pause
