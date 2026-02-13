# scripts/run_lms_bot_batch.ps1
# Runs N consecutive dual-mode matches and stores per-match logs + manifest.

param(
  [int]$Count = 5,
  [int]$RunMaxSec = 900,
  [string]$BatchDir = ""
)

$ErrorActionPreference = "Stop"

if ($Count -lt 1) { $Count = 1 }
if ($RunMaxSec -lt 60) { $RunMaxSec = 60 }

if ([string]::IsNullOrWhiteSpace($BatchDir)) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $BatchDir = "reports/live/batch${Count}_$ts"
}

New-Item -ItemType Directory -Force -Path $BatchDir | Out-Null
$manifestPath = Join-Path $BatchDir "manifest.jsonl"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Batch run started. count=$Count run_max_sec=$RunMaxSec"
Write-Host "Batch directory: $BatchDir"

for ($i = 1; $i -le $Count; $i++) {
  $logPath = Join-Path $BatchDir ("match_{0}.log" -f $i)
  Write-Host ("[BATCH] match={0}/{1} start log={2}" -f $i, $Count, $logPath)

  & "scripts/run_lms_bot_dual.ps1" --bot-run-max-sec $RunMaxSec *>&1 | Tee-Object -FilePath $logPath | Out-Null

  $runtimeLine = Select-String -Path $logPath -Pattern "Runtime feedback en:" | Select-Object -Last 1
  $runJsonl = ""
  if ($runtimeLine) {
    $line = $runtimeLine.Line
    $idx = $line.IndexOf("Runtime feedback en:")
    if ($idx -ge 0) {
      $runJsonl = $line.Substring($idx + ("Runtime feedback en:").Length).Trim()
    }
  }

  $finalLineObj = Select-String -Path $logPath -Pattern "\[BOT\]\[RUN\] Ejecucion completa finalizada\." | Select-Object -Last 1
  $stopLineObj = Select-String -Path $logPath -Pattern "\[BOT\]\[RUN\] Fin detectado por" | Select-Object -Last 1
  $finalLine = if ($finalLineObj) { $finalLineObj.Line } else { "" }
  $stopLine = if ($stopLineObj) { $stopLineObj.Line } else { "" }

  $record = [ordered]@{
    match = $i
    log = $logPath
    runtime_feedback_jsonl = $runJsonl
    run_finish_line = $finalLine
    run_stop_line = $stopLine
    captured_at = (Get-Date).ToString("o")
  }
  ($record | ConvertTo-Json -Compress) | Add-Content -Path $manifestPath -Encoding UTF8

  Write-Host ("[BATCH] match={0}/{1} end stop='{2}'" -f $i, $Count, $stopLine)
}

Write-Host ("[BATCH] done manifest={0}" -f $manifestPath)
