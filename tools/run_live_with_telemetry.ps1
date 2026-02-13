param(
    [string]$TelemetryPath = "reports\\runtime\\telemetry_live.jsonl",
    [double]$TelemetryRateHz = 10.0,
    [switch]$UseAutoScript,
    [switch]$NoTui,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

Write-Host "[LIVE-TELEMETRY] Activating venv..."
. .\.venv\Scripts\Activate.ps1

$telemetryOut = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $TelemetryPath))
$telemetryDir = Split-Path $telemetryOut -Parent
New-Item -ItemType Directory -Force -Path $telemetryDir | Out-Null
if (Test-Path $telemetryOut) {
    Remove-Item $telemetryOut -Force -ErrorAction SilentlyContinue
}

$tuiProc = $null
if (-not $NoTui) {
    $tuiCommand = "Set-Location '$repoRoot'; . .\.venv\Scripts\Activate.ps1; python tools\telemetry_tui.py --path `"$telemetryOut`" --fps 10"
    $tuiProc = Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command", $tuiCommand
    ) -PassThru
    Start-Sleep -Milliseconds 900
    Write-Host "[LIVE-TELEMETRY] TUI launched in a secondary PowerShell window."
}

$runnerScript = if ($UseAutoScript) { ".\\scripts\\run_lms_bot_auto.ps1" } else { ".\\scripts\\run_lms_bot_dual_alphastar.ps1" }
Write-Host "[LIVE-TELEMETRY] Running live bot with telemetry..."

& powershell -ExecutionPolicy Bypass -File $runnerScript `
    --telemetry-jsonl $telemetryOut `
    --telemetry-rate-hz $TelemetryRateHz `
    @ExtraArgs

$runnerExitCode = $LASTEXITCODE

if ($tuiProc -and (-not $tuiProc.HasExited)) {
    try {
        Stop-Process -Id $tuiProc.Id -Force -ErrorAction SilentlyContinue
    } catch {}
}

if (Test-Path $telemetryOut) {
    $lines = (Get-Content $telemetryOut -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    Write-Host "[LIVE-TELEMETRY] telemetry_jsonl=$telemetryOut lines=$lines"
} else {
    Write-Host "[LIVE-TELEMETRY][WARN] telemetry file not found: $telemetryOut"
}

exit $runnerExitCode
