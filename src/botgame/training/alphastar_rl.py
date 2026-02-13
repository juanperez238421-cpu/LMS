from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.actor_learner import RLConfig, collect_actor_sequence, learner_step
from botgame.training.alphastar.model import FactorizedPolicyValueNet
from botgame.training.alphastar.replay import SequenceReplayBuffer
from botgame.training.rl_env import BotGameEnv


def _load_supervised_checkpoint(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device)
    head_sizes = dict(ckpt["head_sizes"])
    model = FactorizedPolicyValueNet(obs_dim=int(ckpt["obs_dim"]), head_sizes=head_sizes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    codec = FactorizedActionCodec(
        bins=int(ckpt.get("bins", 11)),
        ability_size=int(ckpt.get("ability_size", head_sizes.get("ability", 2))),
    )
    return model, codec


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    supervised_model, codec = _load_supervised_checkpoint(args.supervised_checkpoint, device=device)
    model = FactorizedPolicyValueNet(
        obs_dim=supervised_model.obs_dim,
        head_sizes=supervised_model.head_sizes,
        hidden_dim=args.hidden_dim,
    ).to(device)
    model.load_state_dict(supervised_model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    replay = SequenceReplayBuffer(max_sequences=args.replay_size, seed=args.seed)
    env = BotGameEnv(bot_id="alphastar_rl")

    config = RLConfig(
        unroll_length=args.unroll_length,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        clip_rho=args.clip_rho,
        clip_pg_rho=args.clip_pg_rho,
        clip_c=args.clip_c,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        kl_coef=args.kl_coef,
        upgo_coef=args.upgo_coef,
        pseudo_reward_scale=args.pseudo_reward_scale,
        terminal_win_reward=args.terminal_win_reward,
        terminal_loss_reward=args.terminal_loss_reward,
    )

    step = 0
    for iteration in range(args.iterations):
        model.eval()
        for actor_idx in range(args.actor_rollouts):
            seq = collect_actor_sequence(
                env=env,
                model=model,
                codec=codec,
                config=config,
                device=device,
                seed=args.seed + iteration * 1000 + actor_idx,
            )
            replay.add(seq)
            step += seq.length

        if len(replay) < args.min_replay_sequences:
            continue

        model.train()
        for update_idx in range(args.learner_updates):
            batch = replay.sample_batch(batch_size=args.batch_size, unroll_length=args.unroll_length - 1)
            batch_obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
            batch_next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
            batch_rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
            batch_dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=device)
            batch_behavior = torch.as_tensor(batch.behavior_logp, dtype=torch.float32, device=device)
            batch_actions = {
                k: torch.as_tensor(v, dtype=torch.long, device=device) for k, v in batch.actions.items()
            }
            metrics = learner_step(
                model=model,
                supervised_model=supervised_model,
                codec=codec,
                optimizer=optimizer,
                batch_obs=batch_obs,
                batch_next_obs=batch_next_obs,
                batch_rewards=batch_rewards,
                batch_dones=batch_dones,
                batch_behavior_logp=batch_behavior,
                batch_actions=batch_actions,
                config=config,
            )
            metrics.update(
                {
                    "iteration": iteration + 1,
                    "update": update_idx + 1,
                    "env_steps": step,
                    "replay_sequences": len(replay),
                }
            )
            print(json.dumps(metrics, ensure_ascii=True))

        if (iteration + 1) % args.save_every == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "obs_dim": model.obs_dim,
                    "head_sizes": model.head_sizes,
                    "bins": codec.bins,
                    "ability_size": codec.ability_size,
                    "model_state_dict": model.state_dict(),
                    "env_steps": step,
                },
                args.output,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "obs_dim": model.obs_dim,
            "head_sizes": model.head_sizes,
            "bins": codec.bins,
            "ability_size": codec.ability_size,
            "model_state_dict": model.state_dict(),
            "env_steps": step,
        },
        args.output,
    )
    print(f"saved_checkpoint={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaStar-inspired off-policy RL fine-tuning.")
    parser.add_argument("--supervised-checkpoint", type=Path, default=Path("artifacts/alphastar/pi_sup.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/alphastar/pi_rl.pt"))
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--actor-rollouts", type=int, default=4)
    parser.add_argument("--learner-updates", type=int, default=2)
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
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

