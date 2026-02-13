# scripts/run_lms_bot_dual_alphastar.ps1
# Runs visible Chrome live match with AlphaStar decision backend.

$ErrorActionPreference = "Stop"

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

Write-Host "Running LMS bot in dual mode with AlphaStar backend..."
python -u lms_live_collector.py `
  --play-game `
  --channel chrome `
  --no-persistent `
  --no-bot-debug-hud `
  --no-bot-visual-cursor `
  --bot-decision-backend alphastar `
  --bot-alphastar-checkpoint artifacts/alphastar/pi_rl.pt `
  --bot-run-until-end `
  --bot-run-stop-on-death-only `
  --bot-run-max-sec 1200 `
  --bot-ui-poll-ms 60 `
  --bot-feedback-dir reports/feedback_training/live `
  --bot-feedback-max-screenshots 2000 `
  --bot-feedback-screenshot-every-sec 0.6 `
  --bot-feedback-screenshot-every-ms 120 `
  --bot-feedback-max-burst-per-loop 1 `
  --bot-move-max-blocking-hold-ms 45 `
  --bot-hold-capture-slice-ms 10 `
  --bot-enemy-vision `
  --bot-enemy-vision-interval-ms 140 `
  --bot-enemy-red-ratio-threshold 0.003 `
  --bot-enemy-min-area 18 `
  --bot-move-motion-sample-every 999 `
  --no-bot-visual-ocr `
  --bot-feedback-render-video `
  --bot-feedback-video-fps 10 `
  --report-every-sec 0 `
  $args
