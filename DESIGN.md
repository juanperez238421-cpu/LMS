# AlphaStar-Inspired Pipeline Design

## Scope
This design implements an "apply then refine" pipeline in parallel with existing LMS/botgame trainers:
1. Supervised imitation (`pi_sup`) from offline logs.
2. Off-policy RL fine-tuning initialized from `pi_sup`, with KL regularization toward `pi_sup`, V-trace policy correction, TD(lambda) value targets, and optional UPGO.
3. League training with PFSP, frozen snapshots, and exploiter/main agent roles.

No anti-cheat, bypass, or ToS-violating behavior is added.

## Contracts (PHASE 0)
Observation type:
- Primary env observation comes from `WorldState.build_observation(...)` (`src/botgame/server/world.py`) and is converted to a fixed 19-dim vector in `src/botgame/training/alphastar/features.py`.
- Live collector feedback events (`reports/feedback_training/**/feedback_stream.jsonl`) are also projected into the same 19-dim vector space for optional offline learning.

Action type:
- Runtime action API is `botgame.common.types.Action` (continuous move/aim + binary fire/interact + optional ability).
- For scalable policy gradients, actions are factorized into categorical heads:
  `move_x`, `move_y`, `aim_x`, `aim_y`, `fire`, `ability`, `interact`.
- Factorization codec: `src/botgame/training/alphastar/action_codec.py`.

Termination + reward:
- Env termination is read from `BotGameEnv.step(...)` (`terminated/truncated`).
- RL terminal outcome reward defaults to paper-like sparse objective (+1 win / -1 loss; draw configurable as 0 by truncation logic), with pseudo-reward kept optional via `--pseudo-reward-scale`.
- Offline live data gets terminal signal from `reports/live/**/manifest.jsonl` when linked by `runtime_feedback`.

## Module Mapping (Paper -> Repo)
Data flow:
- Unified loader: `src/botgame/training/alphastar/data.py`
  - Loads `data/processed/*.jsonl.gz` recorder trajectories.
  - Loads `reports/feedback_training/**/feedback_stream.jsonl`.
  - Optionally injects outcomes from `reports/live/**/manifest.jsonl`.
- Unified sample schema: `TrajectorySequence` (`src/botgame/training/alphastar/types.py`).

Model interfaces:
- Factorized actor-critic net: `src/botgame/training/alphastar/model.py`
  - Shared torso
  - Per-component policy heads
  - Scalar value head

RL math:
- V-trace: `vtrace(...)` in `src/botgame/training/alphastar/losses.py`
- TD(lambda): `td_lambda_targets(...)` in `src/botgame/training/alphastar/losses.py`
- UPGO: `upgo_returns(...)` in `src/botgame/training/alphastar/losses.py`

Off-policy actor-learner:
- Replay of sequences: `src/botgame/training/alphastar/replay.py`
- Actor collection and learner update: `src/botgame/training/alphastar/actor_learner.py`
  - Policy update uses V-trace-corrected advantages.
  - Value update uses TD(lambda) targets.
  - UPGO is additive and weighted (`--upgo-coef`).
  - KL loss toward supervised policy (`--kl-coef`).

League manager + PFSP:
- PFSP weighting functions and league bookkeeping:
  `src/botgame/training/alphastar/league.py`
  - Includes `f_hard`, `f_var`, and normalized PFSP weights.
  - Supports `main`, `main_exploiter`, `league_exploiter`.
  - Snapshot and reset rules are configurable.

Reporting/eval:
- `src/botgame/training/report_league.py`
  - Win rates by pair
  - Simple Elo estimation from pairwise results
  - Exploitability proxy: min win-rate vs frozen past players

## Entrypoints
- Dataset sanity (PHASE 1):
  - `python -m botgame.training.alphastar.data --data-dir data/processed --feedback-dir reports/feedback_training`
- Imitation (PHASE 2):
  - `python -m botgame.training.alphastar_imitation --epochs 2 --batch-size 64 --output artifacts/alphastar/pi_sup.pt`
- RL fine-tuning (PHASES 3-4):
  - `python -m botgame.training.alphastar_rl --supervised-checkpoint artifacts/alphastar/pi_sup.pt --iterations 10 --actor-rollouts 2 --learner-updates 1 --output artifacts/alphastar/pi_rl.pt`
- League training (PHASE 5):
  - `python -m botgame.training.alphastar_league --supervised-checkpoint artifacts/alphastar/pi_sup.pt --rounds 2 --rl-iterations-per-round 3 --output-dir reports/league`
- League report (PHASE 6):
  - `python -m botgame.training.report_league --league-state reports/league/league_state.json --matches reports/league/matches.jsonl --output reports/league/summary.json`

## Notes on LMS Integration
- Live collector outputs are treated as optional offline signals:
  - Action/feedback sequences from `reports/feedback_training`.
  - Match-level outcomes from `reports/live` manifests.
- OCR-derived metrics remain external to core policy math; they can be merged via `extras` in `TrajectorySequence` without changing trainer internals.

