@echo off
echo ==========================================
echo      Initializing SonicPulse AI Setup
echo ==========================================

echo.
echo [1/2] Installing Dependencies...
pip install -r requirements.txt

echo.
echo [2/2] Launching Application...
echo.
streamlit run app.py

pause
