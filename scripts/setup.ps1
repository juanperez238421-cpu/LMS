# scripts/setup.ps1
# Creates a Python virtual environment, installs dependencies, and runs tests.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment..."
py -3.13 -m venv .venv

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install -U pip

Write-Host "Installing project dependencies from pyproject.toml..."
python -m pip install -e .

Write-Host "Running tests..."
# Install pytest if not already installed (should be via -e .)
try {
    pytest --version | Out-Null
} catch {
    Write-Host "pytest not found, installing..."
    python -m pip install pytest
}
pytest

Write-Host "Setup complete."
