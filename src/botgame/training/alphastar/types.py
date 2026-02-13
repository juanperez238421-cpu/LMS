from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping

import numpy as np


@dataclass
class TrajectorySequence:
    """Contiguous sequence of transitions for off-policy training."""

    obs: np.ndarray
    actions: Dict[str, np.ndarray]
    rewards: np.ndarray
    dones: np.ndarray
    behavior_logp: np.ndarray
    extras: Dict[str, np.ndarray | float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = int(self.obs.shape[0])
        if self.rewards.shape[0] != length:
            raise ValueError("obs and rewards must have the same temporal length.")
        if self.dones.shape[0] != length:
            raise ValueError("obs and dones must have the same temporal length.")
        if self.behavior_logp.shape[0] != length:
            raise ValueError("obs and behavior_logp must have the same temporal length.")
        for key, value in self.actions.items():
            if value.shape[0] != length:
                raise ValueError(f"Action component {key} length mismatch.")

    @property
    def length(self) -> int:
        return int(self.obs.shape[0])

    def slice(self, start: int, end: int) -> "TrajectorySequence":
        """Returns a time-sliced view of this sequence."""
        return TrajectorySequence(
            obs=self.obs[start:end],
            actions={k: v[start:end] for k, v in self.actions.items()},
            rewards=self.rewards[start:end],
            dones=self.dones[start:end],
            behavior_logp=self.behavior_logp[start:end],
            extras=dict(self.extras),
        )


@dataclass
class ReplayBatch:
    """Batch format produced by sequence replay."""

    obs: np.ndarray
    next_obs: np.ndarray
    actions: Dict[str, np.ndarray]
    rewards: np.ndarray
    dones: np.ndarray
    behavior_logp: np.ndarray
    extras: Dict[str, np.ndarray]


def stack_action_dicts(action_batches: Mapping[str, list[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stacks per-component action arrays into `[T, B]` tensors."""
    return {k: np.stack(v, axis=1) for k, v in action_batches.items()}

