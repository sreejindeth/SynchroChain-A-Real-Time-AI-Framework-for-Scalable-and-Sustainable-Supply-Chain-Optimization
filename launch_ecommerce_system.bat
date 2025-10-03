@echo off
echo Starting SynchroChain E-Commerce System...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python and try again.
    pause
    exit /b 1
)

echo Python found. Starting the system...
echo.

REM Change to frontend directory
cd /d "%~dp0frontend"

REM Check if required files exist
if not exist "ecommerce_api.py" (
    echo ERROR: ecommerce_api.py not found!
    pause
    exit /b 1
)

if not exist "ecommerce_user.html" (
    echo ERROR: ecommerce_user.html not found!
    pause
    exit /b 1
)

if not exist "admin_dashboard.html" (
    echo ERROR: admin_dashboard.html not found!
    pause
    exit /b 1
)

echo Starting Enhanced E-Commerce API...
echo.
echo ========================================
echo SynchroChain E-Commerce System
echo ========================================
echo.
echo User Interface: http://localhost:5000
echo Admin Dashboard: http://localhost:5000/admin
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the Flask API
python ecommerce_api.py

pause

