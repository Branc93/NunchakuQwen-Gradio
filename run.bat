@echo off
echo Starting Nunchaku Qwen-Image Interface...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo Error: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade runtime requirements (preserving PyTorch CUDA version)
echo Installing/upgrading runtime requirements...
pip install -r requirements-runtime.txt --upgrade

REM Launch the application
echo.
echo Launching Nunchaku Qwen-Image Interface...
echo The interface will open in your browser at http://localhost:7860
echo.
echo Press Ctrl+C to stop the application
echo.
python app.py

pause
