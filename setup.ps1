# AIML-Practicals Automated Setup Script for Windows PowerShell
# ============================================================

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  AIML Practicals - Environment Setup (PowerShell)    " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Check if Python is installed and available
Write-Host "`n[1/3] Checking for Python installation..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCmd) {
    Write-Host "ERROR: Python is not found in your PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://www.python.org/ or the Microsoft Store." -ForegroundColor Red
    exit 1
}

$pyVersion = python --version
Write-Host "Found Python: $pyVersion" -ForegroundColor Green

# 2. Create virtual environment (.venv) if it does not exist
Write-Host "`n[2/3] Setting up virtual environment (.venv)..." -ForegroundColor Yellow
$venvDir = Join-Path $PSScriptRoot ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating new virtual environment at: $venvDir" -ForegroundColor Gray
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to create virtual environment." -ForegroundColor Red
        exit 1
    }
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 3. Install required packages
Write-Host "`n[3/3] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$reqFile = Join-Path $PSScriptRoot "requirements.txt"

if (Test-Path $reqFile) {
    & $venvPython -m pip install --upgrade pip --quiet
    & $venvPython -m pip install -r $reqFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Dependency installation failed." -ForegroundColor Red
        exit 1
    }
    Write-Host "All dependencies installed successfully!" -ForegroundColor Green
} else {
    Write-Host "WARNING: requirements.txt not found at $reqFile" -ForegroundColor Yellow
}

Write-Host "`n======================================================" -ForegroundColor Cyan
Write-Host "  Setup Complete! You are ready for your practicals.  " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "`nTo activate the virtual environment in PowerShell:" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Yellow
Write-Host "`nTo execute any experiment:" -ForegroundColor White
Write-Host "  python run.py --list" -ForegroundColor Yellow
Write-Host "  python run.py 1" -ForegroundColor Yellow
