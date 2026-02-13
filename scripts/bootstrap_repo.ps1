# scripts/bootstrap_repo.ps1
# One-time bootstrap for local development and LMS automation tooling.

$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
  Write-Host "Creating virtual environment..."
  py -3 -m venv .venv
}

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing project package..."
python -m pip install -e .

Write-Host "Installing LMS/runtime dependencies..."
python -m pip install -r requirements-lms.txt

Write-Host "Installing Playwright browsers (chromium)..."
python -m playwright install chromium

Write-Host "Running tests..."
python -m pytest -q

Write-Host "Bootstrap completed."
