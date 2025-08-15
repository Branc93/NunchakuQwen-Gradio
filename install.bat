@echo off
echo Installing Nunchaku Qwen-Image Interface...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
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

REM Upgrade pip first
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch first (with CUDA support if available)
echo Installing PyTorch...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

REM Install other requirements
echo Installing other requirements...
pip install -r requirements.txt

echo.
echo Installation complete! You can now run the interface with:
echo python app.py
echo.
pause
