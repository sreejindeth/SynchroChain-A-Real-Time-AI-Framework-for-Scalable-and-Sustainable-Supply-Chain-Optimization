@echo off
echo SynchroChain Frontend Launcher
echo ==============================
echo.

REM Try different Python commands
set PYTHON_CMD=
for %%i in (python py python3 python.exe) do (
    %%i --version >nul 2>&1
    if not errorlevel 1 (
        set PYTHON_CMD=%%i
        goto :found_python
    )
)

echo ERROR: Python not found! Please install Python 3.7+
echo Trying alternative method...
echo.

REM Try Microsoft Store Python path
if exist "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" (
    set PYTHON_CMD=%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe
    goto :found_python
)

echo Please run this command manually:
echo cd frontend
echo python api.py
pause
exit /b 1

:found_python
echo SUCCESS: Python found (%PYTHON_CMD%)
echo.

REM Install Flask dependencies if needed
echo Checking Flask dependencies...
%PYTHON_CMD% -c "import flask, flask_cors" >nul 2>&1
if errorlevel 1 (
    echo Installing Flask dependencies...
    %PYTHON_CMD% -m pip install flask flask-cors
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo SUCCESS: Dependencies ready
echo.

REM Change to frontend directory and start server
echo Starting SynchroChain Frontend...
echo    Backend API: http://localhost:5000/api
echo    Frontend UI: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

cd frontend
%PYTHON_CMD% api.py

pause
