from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.data import dataset_sanity, load_unified_sequences
from botgame.training.alphastar.features import OBS_DIM
from botgame.training.alphastar.model import FactorizedPolicyValueNet


def _flatten_sequences(sequences, codec: FactorizedActionCodec):
    obs = np.concatenate([seq.obs for seq in sequences], axis=0)
    actions: Dict[str, np.ndarray] = {}
    for key in codec.head_sizes:
        actions[key] = np.concatenate([seq.actions[key] for seq in sequences], axis=0)
    return obs, actions


def train_imitation(args: argparse.Namespace) -> None:
    codec = FactorizedActionCodec(bins=args.bins, ability_size=args.ability_size)
    sequences = load_unified_sequences(
        data_dir=args.data_dir,
        feedback_dir=args.feedback_dir,
        reports_live_dir=args.reports_live_dir,
        include_live_feedback=not args.no_live_feedback,
        codec=codec,
    )
    if not sequences:
        raise RuntimeError("No trajectories found for imitation training.")

    stats = dataset_sanity(sequences, codec=codec)
    print(json.dumps({"dataset": stats}, ensure_ascii=True, indent=2))

    obs_np, actions_np = _flatten_sequences(sequences, codec=codec)
    obs_tensor = torch.as_tensor(obs_np, dtype=torch.float32)
    action_tensors = {k: torch.as_tensor(v, dtype=torch.long) for k, v in actions_np.items()}
    dataset = TensorDataset(obs_tensor, *[action_tensors[k] for k in codec.head_sizes])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = FactorizedPolicyValueNet(obs_dim=OBS_DIM, head_sizes=codec.head_sizes, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        model.train()
        running = {"loss": 0.0, "entropy": 0.0}
        batches = 0
        for batch in dataloader:
            batch_obs = batch[0].to(device)
            batch_actions = {k: batch[i + 1].to(device) for i, k in enumerate(codec.head_sizes)}

            output = model(batch_obs)
            loss = torch.tensor(0.0, device=device)
            entropy = torch.tensor(0.0, device=device)
            for key in codec.head_sizes:
                loss = loss + F.cross_entropy(output.logits[key], batch_actions[key])
                probs = torch.softmax(output.logits[key], dim=-1)
                log_probs = torch.log_softmax(output.logits[key], dim=-1)
                entropy = entropy + (-(probs * log_probs).sum(dim=-1).mean())

            total = loss - args.entropy_coef * entropy
            optimizer.zero_grad()
            total.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["entropy"] += float(entropy.item())
            batches += 1

        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "loss_ce": running["loss"] / max(1, batches),
                    "entropy": running["entropy"] / max(1, batches),
                },
                ensure_ascii=True,
            )
        )

    ckpt = {
        "obs_dim": OBS_DIM,
        "head_sizes": codec.head_sizes,
        "bins": codec.bins,
        "ability_size": codec.ability_size,
        "model_state_dict": model.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.output)
    print(f"saved_checkpoint={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AlphaStar-inspired imitation training.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--feedback-dir", type=Path, default=Path("reports/feedback_training"))
    parser.add_argument("--reports-live-dir", type=Path, default=Path("reports/live"))
    parser.add_argument("--no-live-feedback", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/alphastar/pi_sup.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=1e-3)
    parser.add_argument("--bins", type=int, default=11)
    parser.add_argument("--ability-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train_imitation(args)


if __name__ == "__main__":
    main()

