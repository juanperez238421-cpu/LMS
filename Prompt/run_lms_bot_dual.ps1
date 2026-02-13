# scripts/run_lms_bot_dual.ps1
# Runs game bot + parallel smoke monitor at the same time.

$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

$googleLoginArgs = @()
$googleEmail = [string]$env:LMS_GOOGLE_EMAIL
$googlePassword = [string]$env:LMS_GOOGLE_PASSWORD
if ((-not [string]::IsNullOrWhiteSpace($googleEmail)) -and (-not [string]::IsNullOrWhiteSpace($googlePassword))) {
  Write-Host "Google login pre-game enabled (credentials from environment)."
  $googleLoginArgs += @(
    "--bot-google-login",
    "--stealth-login"
  )
} else {
  Write-Host "Google login pre-game disabled (set LMS_GOOGLE_EMAIL and LMS_GOOGLE_PASSWORD to enable)."
}

Write-Host "Running LMS bot in dual mode (play-game + parallel smoke)..."
python -u lms_live_collector.py `
  --play-game `
  --channel chrome `
  --no-persistent `
  --bot-parallel-smoke `
  --bot-debug-hud `
  --bot-cursor-transition-ms 20 `
  --bot-cursor-log-interval-sec 1 `
  --bot-cursor-idle-amplitude 14 `
  --bot-smoke-move-hold-ms 260 `
  --bot-smoke-move-pattern "KeyW,KeyW,KeyA,KeyD,KeyS" `
  --bot-move-base-hold-ms 340 `
  --bot-move-click-every-steps 3 `
  --bot-move-motion-sample-every 1 `
  --bot-opening-move-sec 14 `
  --bot-opening-hold-multiplier 2.1 `
  --bot-collision-streak-threshold 2 `
  --bot-collision-escape-extra-steps 2 `
  --bot-stuck-repeat-action-streak 2 `
  --bot-stuck-confirm-streak 1 `
  --bot-stuck-recovery-steps 9 `
  --bot-stuck-motion-factor 0.90 `
  --bot-enemy-vision `
  --bot-enemy-vision-interval-ms 260 `
  --bot-enemy-red-ratio-threshold 0.007 `
  --bot-enemy-min-area 45 `
  --bot-open-map-every-sec 12 `
  --bot-ability-every-sec 2.8 `
  --bot-run-until-end `
  --bot-run-stop-on-death-only `
  --bot-run-max-sec 1200 `
  --bot-move-motion-threshold 1.7 `
  --bot-move-stuck-streak 2 `
  --bot-move-escape-steps 5 `
  --bot-history-stuck-max-rows 16000 `
  --bot-history-stuck-min-samples 3 `
  --bot-history-stuck-weight 1.15 `
  --bot-feedback-dir reports/feedback_training/live `
  --bot-feedback-screenshot-every-sec 0.6 `
  --bot-feedback-max-screenshots 450 `
  --bot-visual-ocr `
  --bot-visual-ocr-every-sec 1.2 `
  --bot-visual-ocr-max-frames 360 `
  --bot-feedback-render-video `
  --bot-feedback-video-fps 10 `
  --bot-ui-poll-ms 95 `
  --bot-action-timeout-ms 700 `
  --report-every-sec 15 `
  $googleLoginArgs `
  $args
