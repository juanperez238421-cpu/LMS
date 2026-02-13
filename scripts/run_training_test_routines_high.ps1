# scripts/run_training_test_routines_high.ps1
# Rutina intensiva de tests para estabilidad de entrenamiento.
# No modifica codigo ni configuracion del proyecto.

param(
    [int]$CollectorRuns = 60,
    [int]$MathReplayRuns = 30
)

$ErrorActionPreference = "Stop"

if ($CollectorRuns -lt 1) { $CollectorRuns = 1 }
if ($MathReplayRuns -lt 1) { $MathReplayRuns = 1 }

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "=== Bloque A: Sanidad minima ==="
python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
if ($LASTEXITCODE -ne 0) { throw "Fallo en sanidad minima (parser)." }
python -m pytest -q tests/test_lms_live_collector_bot.py
if ($LASTEXITCODE -ne 0) { throw "Fallo en sanidad minima (collector mocks)." }

Write-Host "=== Bloque B: Collector mock stress ($CollectorRuns runs) ==="
for ($i = 1; $i -le $CollectorRuns; $i++) {
    Write-Host "[Collector $i/$CollectorRuns] parser"
    python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
    if ($LASTEXITCODE -ne 0) { throw "Fallo en Collector parser, run $i/$CollectorRuns." }

    Write-Host "[Collector $i/$CollectorRuns] full file"
    python -m pytest -q tests/test_lms_live_collector_bot.py
    if ($LASTEXITCODE -ne 0) { throw "Fallo en Collector full file, run $i/$CollectorRuns." }
}

Write-Host "=== Bloque C: AlphaStar math/replay stress ($MathReplayRuns runs) ==="
for ($i = 1; $i -le $MathReplayRuns; $i++) {
    Write-Host "[MathReplay $i/$MathReplayRuns] alphastar math"
    python -m pytest -q tests/test_alphastar_math.py
    if ($LASTEXITCODE -ne 0) { throw "Fallo en alphastar math, run $i/$MathReplayRuns." }

    Write-Host "[MathReplay $i/$MathReplayRuns] replay buffer"
    python -m pytest -q tests/test_replay_buffer.py
    if ($LASTEXITCODE -ne 0) { throw "Fallo en replay buffer, run $i/$MathReplayRuns." }
}

Write-Host "High-volume training test routines completed."
