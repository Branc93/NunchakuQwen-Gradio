# PowerShell script to run the Nunchaku Qwen-Image Interface
# Run this script in PowerShell with: .\run.ps1

Write-Host "🎨 Starting Nunchaku Qwen-Image Interface..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    try {
        python -m venv venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        Write-Host "✅ Virtual environment created successfully" -ForegroundColor Green
    } catch {
        Write-Host "❌ Error: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Activate virtual environment
Write-Host "🚀 Activating virtual environment..." -ForegroundColor Yellow
try {
    & "venv\Scripts\Activate.ps1"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to activate virtual environment"
    }
    Write-Host "✅ Virtual environment activated" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to activate virtual environment" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install/upgrade runtime requirements (preserving PyTorch CUDA version)
Write-Host "📥 Installing/upgrading runtime requirements..." -ForegroundColor Yellow
try {
    pip install -r requirements-runtime.txt --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install runtime requirements"
    }
    Write-Host "✅ Runtime requirements installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to install runtime requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Launch the application
Write-Host ""
Write-Host "🎨 Launching Nunchaku Qwen-Image Interface..." -ForegroundColor Cyan
Write-Host "The interface will open in your browser at http://localhost:7860" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

try {
    python app.py
} catch {
    Write-Host "❌ Error running the application: $_" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"
