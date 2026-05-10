@echo off
REM ============================================================================
REM START_SERVERS.cmd - Start all three servers for the project
REM ============================================================================
REM
REM This script starts:
REM   1. Frontend (React + Vite) on port 5173
REM   2. Backend (Express + MySQL) on port 5000
REM   3. ML Server (FastAPI + TensorFlow) on port 8000
REM
REM PREREQUISITES:
REM   - Node.js installed
REM   - Python installed with venv_new virtual environment set up
REM   - MySQL running in XAMPP
REM   - All dependencies installed (npm install, pip install -r requirements.txt)
REM
REM ============================================================================

echo.
echo ============================================================================
echo   Starting All Servers for ITE03 + EVENTDP Final Project
echo ============================================================================
echo.

REM Check if MySQL is running
echo [1/4] Checking MySQL status...
netstat -ano | findstr :3306 >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] MySQL is not running!
    echo Please start MySQL in XAMPP Control Panel first.
    echo.
    pause
    exit /b 1
)
echo [OK] MySQL is running on port 3306
echo.

REM Check if venv_new exists
echo [2/4] Checking Python virtual environment...
if not exist "venv_new\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv_new' not found!
    echo Please create it first: python -m venv venv_new
    echo Then install dependencies: venv_new\Scripts\activate ^&^& pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo [OK] Virtual environment found
echo.

REM Check if node_modules exists
echo [3/4] Checking Node.js dependencies...
if not exist "node_modules" (
    echo [WARNING] node_modules not found. Installing dependencies...
    call npm install
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install frontend dependencies
        pause
        exit /b 1
    )
)
if not exist "database\node_modules" (
    echo [WARNING] database/node_modules not found. Installing dependencies...
    cd database
    call npm install
    cd ..
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install backend dependencies
        pause
        exit /b 1
    )
)
echo [OK] Node.js dependencies installed
echo.

echo [4/4] Starting servers...
echo.
echo ============================================================================
echo   Opening 3 terminal windows...
echo ============================================================================
echo.
echo   Terminal 1: Frontend (React + Vite) - http://localhost:5173
echo   Terminal 2: Backend (Express + MySQL) - http://localhost:5000
echo   Terminal 3: ML Server (FastAPI + TensorFlow) - http://localhost:8000
echo.
echo   Press Ctrl+C in each terminal to stop the servers
echo ============================================================================
echo.

REM Start Frontend (React + Vite)
start "Frontend - React + Vite (Port 5173)" cmd /k "npm run dev"

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM Start Backend (Express + MySQL)
start "Backend - Express + MySQL (Port 5000)" cmd /k "cd database && node server.js"

REM Wait 2 seconds
timeout /t 2 /nobreak >nul

REM Start ML Server (FastAPI + TensorFlow)
start "ML Server - FastAPI + TensorFlow (Port 8000)" cmd /k "venv_new\Scripts\activate && python start_ml_server.py"

echo.
echo ============================================================================
echo   All servers are starting!
echo ============================================================================
echo.
echo   Frontend:  http://localhost:5173
echo   Backend:   http://localhost:5000
echo   ML Server: http://localhost:8000
echo.
echo   Note: ML server takes 10-30 seconds to load the model on first startup.
echo.
echo   To stop all servers: Close the 3 terminal windows or press Ctrl+C in each
echo ============================================================================
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul

REM Open browser
start http://localhost:5173

echo.
echo Done! Check the 3 terminal windows for server status.
echo.
pause
