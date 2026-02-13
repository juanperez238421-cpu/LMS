# scripts/train_imitation.ps1
# Generates data by running a match with scripted bots and then trains an imitation policy.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Generating data with scripted bots (1 minute match)..."
# Run a match with scripted bots to generate replay data
botgame-run-match --num_scripted_bots 2 --episode_duration 60 --record_replay

Write-Host "Training imitation policy..."
python -m botgame.training.imitation

Write-Host "Imitation training complete."
