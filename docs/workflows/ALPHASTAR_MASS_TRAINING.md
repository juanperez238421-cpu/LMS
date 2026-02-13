# AlphaStar Mass Training Runbook (Windows PowerShell)

This runbook is for preparing and running many games for the AlphaStar-style stack in `src/botgame/training/alphastar/*`.

## Important Scope Note

`lms_live_collector.py` does not currently expose an AlphaStar inference backend.

- Supported live collector backends today: `legacy`, `lms_re`.
- AlphaStar training is currently run through:
  - `python -m botgame.training.alphastar_imitation`
  - `python -m botgame.training.alphastar_rl`
  - `python -m botgame.training.alphastar_league`

So the AlphaStar "similar test" should target those entrypoints, not `--bot-decision-backend` in `lms_live_collector.py`.

## 0) One-time setup

```powershell
cd D:\Procastrinar\LMS
.\scripts\bootstrap_repo.ps1
```

## 1) AlphaStar quick smoke test (validated command)

This verifies AlphaStar RL path end-to-end using an existing supervised checkpoint.

```powershell
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

Expected artifact:
- `artifacts/alphastar/pi_rl_smoke_test.pt`

## 2) Generate many game trajectories (for learning data)

Use the local server match runner repeatedly with replay recording:

```powershell
for ($i = 1; $i -le 200; $i++) {
  python -m botgame.server.run_match `
    --num_scripted_bots 2 `
    --episode_duration 120 `
    --record_replay `
    --seed (1000 + $i)
}
```

Replay data is written under `data/processed` and consumed by AlphaStar data loaders.

## 3) Train supervised initialization

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

## 4) RL fine-tune at scale

```powershell
python -m botgame.training.alphastar_rl `
  --supervised-checkpoint artifacts/alphastar/pi_sup.pt `
  --output artifacts/alphastar/pi_rl.pt `
  --iterations 200 `
  --actor-rollouts 8 `
  --learner-updates 4 `
  --unroll-length 64 `
  --batch-size 16 `
  --replay-size 2048 `
  --min-replay-sequences 32 `
  --save-every 10
```

## 5) Optional league-style continual training

```powershell
python -m botgame.training.alphastar_league `
  --supervised-checkpoint artifacts/alphastar/pi_sup.pt `
  --output-dir reports/league `
  --rounds 20 `
  --rl-iterations-per-round 20 `
  --actor-rollouts 8 `
  --learner-updates 4
```

Summarize league results:

```powershell
python -m botgame.training.report_league `
  --league-state reports/league/league_state.json `
  --matches reports/league/matches.jsonl `
  --output reports/league/summary.json
```

## 6) Artifact checklist

- Supervised checkpoint: `artifacts/alphastar/pi_sup.pt`
- RL checkpoint: `artifacts/alphastar/pi_rl.pt`
- League state: `reports/league/league_state.json`
- League matches: `reports/league/matches.jsonl`
- League summary: `reports/league/summary.json`

## 7) Practical scaling guidance

- Start with a smoke run, then increase one dimension at a time:
  1. `iterations`
  2. `actor-rollouts`
  3. `learner-updates`
  4. `replay-size`
- Keep periodic checkpoints (`--save-every`) so you can resume quickly.
- Pin seeds for reproducibility when comparing runs.
