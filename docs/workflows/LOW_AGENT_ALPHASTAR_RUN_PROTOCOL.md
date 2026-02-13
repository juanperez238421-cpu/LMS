# Low-Agent AlphaStar Run Protocol (Do Not Edit Code)

This protocol is written for a lower-capability agent. Follow it exactly.

## Hard Rules

1. Do not edit any file in the repository.
2. Do not change script code, Python code, configs, or tests.
3. Only run commands listed in this document.
4. If any command fails, stop and report:
   - command
   - error output
   - last successful step
5. Never delete artifacts in `reports/`, `artifacts/`, or `data/processed/`.

## Goal

Produce training data and run AlphaStar-backed Chrome matches with reproducible evidence.

## Environment

- OS: Windows PowerShell
- Repo: `D:\Procastrinar\LMS`

## Phase 0: Setup (run once)

```powershell
cd D:\Procastrinar\LMS
.\scripts\bootstrap_repo.ps1
.\.venv\Scripts\Activate.ps1
```

## Phase 1: Parser + AlphaStar smoke checks (exactly 1 time)

```powershell
python -m pytest -q tests/test_lms_live_collector_bot.py::test_build_parser_play_game_argument
python -m botgame.training.alphastar_rl `
  --supervised-checkpoint artifacts/alphastar/pi_sup.pt `
  --output artifacts/alphastar/pi_rl_smoke_test.pt `
  --iterations 1 `
  --actor-rollouts 1 `
  --learner-updates 1 `
  --unroll-length 16 `
  --batch-size 2 `
  --replay-size 32 `
  --min-replay-sequences 1 `
  --save-every 1 `
  --cpu
```

Expected:
- test passes
- `artifacts/alphastar/pi_rl_smoke_test.pt` exists

## Phase 2: Data collection for learning (exactly 60 runs)

Run exactly 60 local matches:

```powershell
for ($i = 1; $i -le 60; $i++) {
  python -m botgame.server.run_match `
    --num_scripted_bots 2 `
    --episode_duration 120 `
    --record_replay `
    --seed (5000 + $i)
}
```

## Phase 3: Train checkpoints (exactly once each)

1) Supervised:
```powershell
python -m botgame.training.alphastar_imitation `
  --data-dir data/processed `
  --feedback-dir reports/feedback_training `
  --reports-live-dir reports/live `
  --output artifacts/alphastar/pi_sup.pt `
  --epochs 10 `
  --batch-size 128 `
  --lr 3e-4
```

2) RL:
```powershell
python -m botgame.training.alphastar_rl `
  --supervised-checkpoint artifacts/alphastar/pi_sup.pt `
  --output artifacts/alphastar/pi_rl.pt `
  --iterations 80 `
  --actor-rollouts 6 `
  --learner-updates 3 `
  --unroll-length 64 `
  --batch-size 12 `
  --replay-size 1024 `
  --min-replay-sequences 16 `
  --save-every 10
```

## Phase 4: Live Chrome AlphaStar validation

Run 2 deterministic + 8 stochastic = 10 runs total.

### 4.1 Deterministic runs (exactly 2 runs)

```powershell
for ($i = 1; $i -le 2; $i++) {
  python -u lms_live_collector.py `
    --play-game `
    --channel chrome `
    --no-persistent `
    --bot-parallel-smoke `
    --bot-debug-hud `
    --bot-decision-backend alphastar `
    --bot-alphastar-checkpoint artifacts/alphastar/pi_rl.pt `
    --bot-run-until-end `
    --bot-run-stop-on-death-only `
    --bot-run-max-sec 600 `
    --bot-feedback-dir reports/feedback_training/live `
    --bot-feedback-screenshot-every-sec 0.6 `
    --bot-visual-ocr `
    --bot-visual-ocr-every-sec 1.2 `
    --bot-feedback-render-video `
    --bot-feedback-video-fps 10 `
    --report-every-sec 15
}
```

### 4.2 Stochastic runs (exactly 8 runs)

Use temperature 1.15 for exploration:

```powershell
for ($i = 1; $i -le 8; $i++) {
  python -u lms_live_collector.py `
    --play-game `
    --channel chrome `
    --no-persistent `
    --bot-parallel-smoke `
    --bot-debug-hud `
    --bot-decision-backend alphastar `
    --bot-alphastar-checkpoint artifacts/alphastar/pi_rl.pt `
    --bot-alphastar-stochastic `
    --bot-alphastar-temperature 1.15 `
    --bot-run-until-end `
    --bot-run-stop-on-death-only `
    --bot-run-max-sec 600 `
    --bot-feedback-dir reports/feedback_training/live `
    --bot-feedback-screenshot-every-sec 0.6 `
    --bot-visual-ocr `
    --bot-visual-ocr-every-sec 1.2 `
    --bot-feedback-render-video `
    --bot-feedback-video-fps 10 `
    --report-every-sec 15
}
```

## Phase 5: Save/report artifacts (no modification)

Collect only these paths:

1. Checkpoints:
- `artifacts/alphastar/pi_sup.pt`
- `artifacts/alphastar/pi_rl.pt`
- `artifacts/alphastar/pi_rl_smoke_test.pt`

2. Live runs:
- `reports/feedback_training/live/play_runtime_*/feedback_stream.jsonl`
- `reports/feedback_training/live/play_runtime_*/screens/*.png`
- `reports/feedback_training/live/play_runtime_*/timeline.mp4`

3. Runtime probe:
- `reports/runtime_probe/*.json`

## Operator output format (required)

At the end, print:

1. Number of runs completed per phase.
2. Exact checkpoint file paths generated.
3. Exact live run directories generated.
4. Any failures with command + error.

Do not include opinions. Only factual run status and paths.
