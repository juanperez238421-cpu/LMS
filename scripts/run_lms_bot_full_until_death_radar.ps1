# scripts/run_lms_bot_full_until_death_radar.ps1
# Full live run: no BOT HUD, telemetry to radar web port, stop on death.

param(
  [string]$TelemetryPath = "reports\\runtime\\telemetry_radar_full.jsonl",
  [double]$TelemetryRateHz = 10.0,
  [int]$RadarPort = 8008,
  [double]$RadarWsHz = 10.0,
  [int]$RunMaxSec = 900
)

$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

$extraArgs = @(
  "--no-bot-debug-hud",
  "--bot-visual-ocr",
  "--bot-run-until-end",
  "--bot-run-stop-on-death-only",
  "--bot-run-max-sec", "$RunMaxSec",
  "--report-every-sec", "0"
)

Write-Host "Running full LMS test until death with radar telemetry..."
& .\tools\run_live_with_telemetry.ps1 `
  -WebGui `
  -NoTui `
  -WebGuiPort $RadarPort `
  -WebGuiWsHz $RadarWsHz `
  -TelemetryPath $TelemetryPath `
  -TelemetryRateHz $TelemetryRateHz `
  -ExtraArgs $extraArgs
