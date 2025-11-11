@echo off
echo ================================================================================
echo SynchroChain AI Dashboard
echo ================================================================================
echo.

REM Check if virtual environment exists and activate it
if exist venv_ppo\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv_ppo\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo Warning: No virtual environment found. Using system Python.
)

echo.
echo Installing/checking dashboard dependencies...
python -m pip install -q streamlit plotly pandas numpy

echo.
echo Starting dashboard...
echo Open your browser to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo ================================================================================
echo.

python -m streamlit run src/core/app.py










