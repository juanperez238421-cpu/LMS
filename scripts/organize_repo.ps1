# scripts/organize_repo.ps1
# Normalizes the repository layout for data, reports, and docs artifacts.

$ErrorActionPreference = "Stop"

$dirs = @(
  "docs",
  "docs/workflows",
  "docs/architecture",
  "config",
  "tools",
  "data/raw",
  "data/raw/har",
  "data/raw/ws",
  "data/processed",
  "data/processed/sqlite",
  "data/processed/ocr",
  "reports",
  "reports/audit",
  "reports/live",
  "reports/smoke",
  "reports/ops"
)

foreach ($d in $dirs) {
  if (!(Test-Path $d)) {
    New-Item -ItemType Directory -Path $d | Out-Null
  }
}

# Move audit files from root to reports/audit
Get-ChildItem -File -Path . -Filter "audit_step0*" -ErrorAction SilentlyContinue |
  ForEach-Object { Move-Item -Force $_.FullName "reports/audit/" }

# Move HAR captures to canonical raw folder
Get-ChildItem -File -Path . -Filter "*.har" -ErrorAction SilentlyContinue |
  ForEach-Object { Move-Item -Force $_.FullName "data/raw/har/" }

# Move SQLite DB files produced by LMS tools
Get-ChildItem -File -Path . -Filter "lms_*.db" -ErrorAction SilentlyContinue |
  ForEach-Object { Move-Item -Force $_.FullName "data/processed/sqlite/" }

# Move OCR capture folder contents into data/processed/ocr
if (Test-Path "ocr_captures") {
  Get-ChildItem -Force "ocr_captures" | ForEach-Object {
    Move-Item -Force $_.FullName "data/processed/ocr/"
  }
  Remove-Item -Recurse -Force "ocr_captures"
}

# Move websocket sample folders
if (Test-Path "ws_samples") {
  Move-Item -Force "ws_samples" "data/raw/ws/ws_samples"
}
if (Test-Path "ws_samples_test") {
  Move-Item -Force "ws_samples_test" "data/raw/ws/ws_samples_test"
}

# Split reports by concern
Get-ChildItem -File -Path "reports" -ErrorAction SilentlyContinue | ForEach-Object {
  $name = $_.Name
  if ($name -like "bot_*") {
    if ($name -like "*smoke*") {
      Move-Item -Force $_.FullName "reports/smoke/"
    } else {
      Move-Item -Force $_.FullName "reports/live/"
    }
  } elseif ($name -like "step1*" -or $name -like "*text_mining*") {
    Move-Item -Force $_.FullName "reports/audit/"
  }
}

# Ensure OCR config exists under config/
if (Test-Path "lms_ocr_rois.example.json") {
  Copy-Item -Force "lms_ocr_rois.example.json" "config/lms_ocr_rois.example.json"
}

Write-Host "Repository organization completed."
