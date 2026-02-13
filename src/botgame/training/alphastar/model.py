from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class PolicyValueOutput:
    logits: Dict[str, torch.Tensor]
    values: torch.Tensor


class FactorizedPolicyValueNet(nn.Module):
    """Shared torso with factorized policy heads and a scalar value head."""

    def __init__(
        self,
        obs_dim: int,
        head_sizes: Dict[str, int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.head_sizes = dict(head_sizes)
        self.torso = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, size) for name, size in self.head_sizes.items()}
        )
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> PolicyValueOutput:
        embedding = self.torso(obs)
        logits = {name: head(embedding) for name, head in self.policy_heads.items()}
        values = self.value_head(embedding).squeeze(-1)
        return PolicyValueOutput(logits=logits, values=values)

