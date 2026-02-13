from __future__ import annotations

import random
from collections import deque
from typing import Dict, List

import numpy as np

from botgame.training.alphastar.types import ReplayBatch, TrajectorySequence, stack_action_dicts


class SequenceReplayBuffer:
    """Replay buffer storing variable-length trajectories and sampling fixed unrolls."""

    def __init__(self, max_sequences: int = 2048, seed: int = 0):
        self._sequences: deque[TrajectorySequence] = deque(maxlen=max_sequences)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._sequences)

    def add(self, sequence: TrajectorySequence) -> None:
        self._sequences.append(sequence)

    def sample_batch(self, batch_size: int, unroll_length: int) -> ReplayBatch:
        if not self._sequences:
            raise RuntimeError("Replay buffer is empty.")

        obs_batches: List[np.ndarray] = []
        next_obs_batches: List[np.ndarray] = []
        rewards_batches: List[np.ndarray] = []
        dones_batches: List[np.ndarray] = []
        logp_batches: List[np.ndarray] = []
        action_batches: Dict[str, List[np.ndarray]] = {}
        source_ids: List[np.ndarray] = []

        for _ in range(batch_size):
            seq = self._rng.choice(list(self._sequences))
            if seq.length < (unroll_length + 1):
                padded = self._pad_sequence(seq, unroll_length + 1)
                seq = padded
            start = self._rng.randint(0, seq.length - unroll_length - 1)
            end = start + unroll_length

            obs_batches.append(seq.obs[start:end])
            next_obs_batches.append(seq.obs[start + 1 : end + 1])
            rewards_batches.append(seq.rewards[start:end])
            dones_batches.append(seq.dones[start:end])
            logp_batches.append(seq.behavior_logp[start:end])
            for name, values in seq.actions.items():
                action_batches.setdefault(name, []).append(values[start:end])
            source = str(seq.extras.get("source", "unknown"))
            source_ids.append(np.full((unroll_length,), source, dtype=object))

        return ReplayBatch(
            obs=np.stack(obs_batches, axis=1),
            next_obs=np.stack(next_obs_batches, axis=1),
            actions=stack_action_dicts(action_batches),
            rewards=np.stack(rewards_batches, axis=1).astype(np.float32),
            dones=np.stack(dones_batches, axis=1).astype(np.float32),
            behavior_logp=np.stack(logp_batches, axis=1).astype(np.float32),
            extras={"source": np.stack(source_ids, axis=1)},
        )

    def _pad_sequence(self, sequence: TrajectorySequence, min_length: int) -> TrajectorySequence:
        if sequence.length >= min_length:
            return sequence
        pad = min_length - sequence.length
        obs_pad = np.repeat(sequence.obs[-1:, :], repeats=pad, axis=0)
        rewards_pad = np.zeros((pad,), dtype=np.float32)
        dones_pad = np.ones((pad,), dtype=np.float32)
        logp_pad = np.zeros((pad,), dtype=np.float32)
        actions_pad = {k: np.repeat(v[-1:], repeats=pad, axis=0) for k, v in sequence.actions.items()}
        return TrajectorySequence(
            obs=np.concatenate([sequence.obs, obs_pad], axis=0),
            actions={k: np.concatenate([sequence.actions[k], actions_pad[k]], axis=0) for k in sequence.actions},
            rewards=np.concatenate([sequence.rewards, rewards_pad], axis=0),
            dones=np.concatenate([sequence.dones, dones_pad], axis=0),
            behavior_logp=np.concatenate([sequence.behavior_logp, logp_pad], axis=0),
            extras=dict(sequence.extras),
        )

