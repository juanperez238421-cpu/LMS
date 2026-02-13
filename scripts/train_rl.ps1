# scripts/train_rl.ps1
# Trains an RL policy using PPO.

# Exit immediately if a command exits with a non-zero status.
$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Training RL policy (PPO)..."
python -m botgame.training.rl_train

Write-Host "RL training complete."
