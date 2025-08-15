# PowerShell script to install the Nunchaku Qwen-Image Interface
# Run this script in PowerShell with: .\install.ps1

Write-Host "🎨 Installing Nunchaku Qwen-Image Interface..." -ForegroundColor Cyan
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

# Upgrade pip first
Write-Host "📥 Upgrading pip..." -ForegroundColor Yellow
try {
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to upgrade pip"
    }
    Write-Host "✅ Pip upgraded successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to upgrade pip" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install PyTorch first (with CUDA support if available)
Write-Host "🔥 Installing PyTorch with CUDA support..." -ForegroundColor Yellow
try {
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install PyTorch"
    }
    Write-Host "✅ PyTorch installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to install PyTorch" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Install other requirements
Write-Host "📦 Installing other requirements..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install requirements"
    }
    Write-Host "✅ Requirements installed successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Failed to install requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "🎉 Installation complete!" -ForegroundColor Green
Write-Host "You can now run the interface with:" -ForegroundColor Yellow
Write-Host "  python app.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Or use the run script:" -ForegroundColor Yellow
Write-Host "  .\run.ps1" -ForegroundColor Cyan
Write-Host ""

Read-Host "Press Enter to exit"
