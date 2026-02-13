# scripts/run_server.ps1
# Runs a local bot game server with configurable parameters.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Running botgame server..."
python -m botgame.server --num_scripted_bots 2 --episode_duration 120 --record_replay --learned_model_path artifacts/imitation_policy.pt --learned_model_type imitation $args
