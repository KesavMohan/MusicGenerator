@echo off

REM AI Music Generator Launcher Script for Windows
REM This script activates the virtual environment and runs the application

echo 🎵 AI Music Generator Launcher
echo ================================

REM Check if virtual environment exists
if not exist "venv" (
    echo ❌ Virtual environment not found!
    echo 🔧 Please run the setup first:
    echo    python setup.py
    pause
    exit /b 1
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate

REM Check if requirements are installed
python -c "import flask" 2>nul
if errorlevel 1 (
    echo ❌ Dependencies not installed!
    echo 🔧 Please run the setup first:
    echo    python setup.py
    pause
    exit /b 1
)

REM Start the application
echo 🚀 Starting AI Music Generator...
echo 🌐 Opening browser to http://localhost:5002
echo ⚠️  Press Ctrl+C to stop the server
echo.

REM Try to open browser (optional)
timeout /t 2 /nobreak >nul
start http://localhost:5002 2>nul

REM Run the Flask app
python app.py

pause 