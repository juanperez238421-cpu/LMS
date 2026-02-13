param(
  [string]$RepoRoot = "D:\Procastrinar\LMS",
  [string]$TelemetryPath = "reports\runtime\telemetry_live.jsonl",
  [int]$Port = 8008,
  [double]$WsHz = 10.0,
  [int]$History = 1200,
  [int]$Queue = 64,
  [switch]$SkipSmokeCheck
)

Set-Location $RepoRoot

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  . .\.venv\Scripts\Activate.ps1
}

if (-not $SkipSmokeCheck) {
  python -c "import fastapi, uvicorn, orjson"
  if ($LASTEXITCODE -ne 0) {
    throw "Dependency smoke check failed: python -c `"import fastapi, uvicorn, orjson`""
  }
  python -c "import importlib.util,sys;sys.exit(0 if any(importlib.util.find_spec(m) for m in ('websockets','wsproto')) else 2)"
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[WEB-GUI] Installing WebSocket backend dependencies (websockets/wsproto)..."
    python -m pip install --disable-pip-version-check websockets wsproto
    if ($LASTEXITCODE -ne 0) {
      throw "Failed to install websocket dependencies. Run: python -m pip install websockets wsproto"
    }
    python -c "import importlib.util,sys;sys.exit(0 if any(importlib.util.find_spec(m) for m in ('websockets','wsproto')) else 2)"
    if ($LASTEXITCODE -ne 0) {
      throw "WebSocket backend still unavailable after install."
    }
  }
  Write-Host "[WEB-GUI] Dependency smoke check passed."
}

$env:LMS_TELEMETRY_PATH = $TelemetryPath
$env:LMS_TELEMETRY_WS_HZ = "$WsHz"
$env:LMS_TELEMETRY_HISTORY = "$History"
$env:LMS_TELEMETRY_QUEUE = "$Queue"

Write-Host "[WEB-GUI] telemetry=$TelemetryPath ws_hz=$WsHz history=$History port=$Port"
python -m uvicorn tools.gui_web.backend.app:app --host 127.0.0.1 --port $Port --ws auto
