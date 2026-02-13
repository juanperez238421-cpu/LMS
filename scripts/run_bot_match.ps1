# scripts/run_bot_match.ps1
# Runs a local bot game match using the console script entry point.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Running bot game match..."
python -m botgame.server.run_match --num_scripted_bots 2 --episode_duration 120 --record_replay $args
