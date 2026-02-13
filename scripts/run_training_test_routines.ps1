# scripts/run_training_test_routines.ps1
# Ejecuta rutinas repetidas de tests para estabilidad de entrenamiento.
# No modifica codigo ni archivos de configuracion.

param(
    [int]$CollectorRuns = 20,
    [int]$MathReplayRuns = 10
)

$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "=== Bloque A: Sanidad minima ==="
python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
python -m pytest -q tests/test_lms_live_collector_bot.py

Write-Host "=== Bloque B: Collector mock stability ($CollectorRuns runs) ==="
for ($i = 1; $i -le $CollectorRuns; $i++) {
    Write-Host "[Collector $i/$CollectorRuns] parser"
    python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
    Write-Host "[Collector $i/$CollectorRuns] full file"
    python -m pytest -q tests/test_lms_live_collector_bot.py
}

Write-Host "=== Bloque C: AlphaStar math/replay stability ($MathReplayRuns runs) ==="
for ($i = 1; $i -le $MathReplayRuns; $i++) {
    Write-Host "[MathReplay $i/$MathReplayRuns] alphastar math"
    python -m pytest -q tests/test_alphastar_math.py
    Write-Host "[MathReplay $i/$MathReplayRuns] replay buffer"
    python -m pytest -q tests/test_replay_buffer.py
}

Write-Host "Training test routines completed."
