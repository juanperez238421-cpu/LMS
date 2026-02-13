# scripts/run_alphastar_batch.ps1
# Collect many local matches and run AlphaStar imitation + RL (+ optional league).

param(
    [int]$NumMatches = 100,
    [int]$EpisodeDurationSec = 120,
    [int]$NumScriptedBots = 2,
    [int]$SeedBase = 1000,
    [switch]$SkipBootstrap,
    [switch]$SkipCollect,
    [switch]$SkipImitation,
    [switch]$SkipRL,
    [switch]$RunLeague
)

$ErrorActionPreference = "Stop"

if (-not $SkipBootstrap) {
    Write-Host "Bootstrapping repo..."
    .\scripts\bootstrap_repo.ps1
}

Write-Host "Activating virtual environment..."
.\.venv\Scripts\Activate.ps1

if (-not $SkipCollect) {
    Write-Host "Collecting $NumMatches replay matches..."
    for ($i = 1; $i -le $NumMatches; $i++) {
        $seed = $SeedBase + $i
        Write-Host "[$i/$NumMatches] seed=$seed"
        python -m botgame.server.run_match `
            --num_scripted_bots $NumScriptedBots `
            --episode_duration $EpisodeDurationSec `
            --record_replay `
            --seed $seed
    }
}

if (-not $SkipImitation) {
    Write-Host "Training AlphaStar imitation checkpoint..."
    python -m botgame.training.alphastar_imitation `
        --data-dir data/processed `
        --feedback-dir reports/feedback_training `
        --reports-live-dir reports/live `
        --output artifacts/alphastar/pi_sup.pt `
        --epochs 10 `
        --batch-size 128 `
        --lr 3e-4
}

if (-not $SkipRL) {
    Write-Host "Training AlphaStar RL checkpoint..."
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
}

if ($RunLeague) {
    Write-Host "Running AlphaStar league rounds..."
    python -m botgame.training.alphastar_league `
        --supervised-checkpoint artifacts/alphastar/pi_sup.pt `
        --output-dir reports/league `
        --rounds 20 `
        --rl-iterations-per-round 20 `
        --actor-rollouts 8 `
        --learner-updates 4

    python -m botgame.training.report_league `
        --league-state reports/league/league_state.json `
        --matches reports/league/matches.jsonl `
        --output reports/league/summary.json
}

Write-Host "AlphaStar batch run completed."
