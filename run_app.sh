#!/bin/bash

# Streamlit App Launcher for AQI Prediction System

echo ""
echo "========================================"
echo "  AQI Prediction System - Streamlit App"
echo "========================================"
echo ""

# Check if streamlit is installed
python3 -c "import streamlit" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Installing Streamlit..."
    pip install streamlit -q
fi

# Run the app
echo "Starting Streamlit App..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the app."
echo ""

streamlit run app.py
