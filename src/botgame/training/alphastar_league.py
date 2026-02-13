from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from botgame.training.alphastar.league import LeagueManager, LeaguePlayer, PFSPConfig
from botgame.training.alphastar_rl import run_training


def _win_prob_from_checkpoints(a: Path, b: Path) -> float:
    """Deterministic lightweight proxy win-prob from checkpoint parameter norms."""
    a_ckpt = torch.load(a, map_location="cpu")
    b_ckpt = torch.load(b, map_location="cpu")
    a_norm = sum(float(v.float().norm().item()) for v in a_ckpt["model_state_dict"].values())
    b_norm = sum(float(v.float().norm().item()) for v in b_ckpt["model_state_dict"].values())
    return 1.0 / (1.0 + pow(10.0, (b_norm - a_norm) / 4000.0))


def _train_once(base_args: argparse.Namespace, checkpoint_in: Path, checkpoint_out: Path) -> None:
    train_args = argparse.Namespace(**vars(base_args))
    train_args.supervised_checkpoint = checkpoint_in
    train_args.output = checkpoint_out
    train_args.iterations = base_args.rl_iterations_per_round
    run_training(train_args)


def run_league(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    matches_path = out_dir / "matches.jsonl"

    manager = LeagueManager(
        config=PFSPConfig(
            main_timeout_steps=args.main_timeout_steps,
            main_exploiter_timeout_steps=args.main_exploiter_timeout_steps,
            league_exploiter_timeout_steps=args.league_exploiter_timeout_steps,
            exploiter_snapshot_threshold=args.snapshot_threshold,
            league_exploiter_reset_prob=args.league_reset_prob,
        ),
        rng_seed=args.seed,
    )

    main_id = "main_0"
    main_ckpt = checkpoints_dir / f"{main_id}.pt"
    torch.save(torch.load(args.supervised_checkpoint, map_location="cpu"), main_ckpt)
    manager.add_player(
        LeaguePlayer(
            player_id=main_id,
            agent_type="main",
            checkpoint_path=str(main_ckpt),
            trainable=True,
        )
    )

    main_exp_id = "main_exploiter_0"
    main_exp_ckpt = checkpoints_dir / f"{main_exp_id}.pt"
    torch.save(torch.load(args.supervised_checkpoint, map_location="cpu"), main_exp_ckpt)
    manager.add_player(
        LeaguePlayer(
            player_id=main_exp_id,
            agent_type="main_exploiter",
            checkpoint_path=str(main_exp_ckpt),
            trainable=True,
        )
    )

    league_exp_id = "league_exploiter_0"
    league_exp_ckpt = checkpoints_dir / f"{league_exp_id}.pt"
    torch.save(torch.load(args.supervised_checkpoint, map_location="cpu"), league_exp_ckpt)
    manager.add_player(
        LeaguePlayer(
            player_id=league_exp_id,
            agent_type="league_exploiter",
            checkpoint_path=str(league_exp_ckpt),
            trainable=True,
        )
    )

    base_rl_args = argparse.Namespace(**vars(args))

    for round_idx in range(args.rounds):
        for trainable in manager.list_players(trainable=True):
            player_id = trainable.player_id
            opp_id = manager.sample_opponent(player_id)
            opp = manager.players[opp_id]
            ckpt_in = Path(trainable.checkpoint_path)
            ckpt_out = checkpoints_dir / f"{player_id}_round{round_idx+1}.pt"

            _train_once(base_rl_args, checkpoint_in=ckpt_in, checkpoint_out=ckpt_out)
            trainable.checkpoint_path = str(ckpt_out)
            manager.tick_train_step(player_id, steps=args.rl_iterations_per_round * args.unroll_length)

            win_prob = _win_prob_from_checkpoints(ckpt_out, Path(opp.checkpoint_path))
            noisy_result = 1.0 if random.random() < win_prob else -1.0
            manager.update_result(
                step=(round_idx + 1) * args.rl_iterations_per_round,
                agent_a=player_id,
                agent_b=opp_id,
                outcome=noisy_result,
            )
            with matches_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "round": round_idx + 1,
                            "agent_a": player_id,
                            "agent_b": opp_id,
                            "win_prob_proxy": win_prob,
                            "outcome": noisy_result,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

            if manager.should_snapshot(player_id):
                frozen = manager.snapshot(
                    trainable_player_id=player_id,
                    step=(round_idx + 1) * args.rl_iterations_per_round,
                    checkpoint_path=str(ckpt_out),
                )
                if manager.should_reset_after_snapshot(player_id):
                    reset_path = checkpoints_dir / f"{player_id}_reset_round{round_idx+1}.pt"
                    torch.save(torch.load(args.supervised_checkpoint, map_location="cpu"), reset_path)
                    trainable.checkpoint_path = str(reset_path)
                print(
                    json.dumps(
                        {
                            "round": round_idx + 1,
                            "snapshot": frozen.player_id,
                            "parent": player_id,
                            "reset": manager.should_reset_after_snapshot(player_id),
                        },
                        ensure_ascii=True,
                    )
                )

        manager.save(out_dir / "league_state.json")
        print(json.dumps({"round": round_idx + 1, "players": len(manager.players)}, ensure_ascii=True))

    manager.save(out_dir / "league_state.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaStar-inspired league training.")
    parser.add_argument("--supervised-checkpoint", type=Path, default=Path("artifacts/alphastar/pi_sup.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/league"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--rl-iterations-per-round", type=int, default=6)
    parser.add_argument("--snapshot-threshold", type=float, default=0.70)
    parser.add_argument("--league-reset-prob", type=float, default=0.25)
    parser.add_argument("--main-timeout-steps", type=int, default=int(2e6))
    parser.add_argument("--main-exploiter-timeout-steps", type=int, default=int(4e6))
    parser.add_argument("--league-exploiter-timeout-steps", type=int, default=int(2e6))
    parser.add_argument("--seed", type=int, default=11)

    # Reuse RL args for inner fine-tuning.
    parser.add_argument("--unroll-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--replay-size", type=int, default=512)
    parser.add_argument("--min-replay-sequences", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.8)
    parser.add_argument("--clip-rho", type=float, default=1.0)
    parser.add_argument("--clip-pg-rho", type=float, default=1.0)
    parser.add_argument("--clip-c", type=float, default=1.0)
    parser.add_argument("--entropy-coef", type=float, default=2e-3)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--upgo-coef", type=float, default=0.2)
    parser.add_argument("--pseudo-reward-scale", type=float, default=0.0)
    parser.add_argument("--terminal-win-reward", type=float, default=1.0)
    parser.add_argument("--terminal-loss-reward", type=float, default=-1.0)
    parser.add_argument("--save-every", type=int, default=3)
    parser.add_argument("--actor-rollouts", type=int, default=4)
    parser.add_argument("--learner-updates", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_league(args)


if __name__ == "__main__":
    main()

