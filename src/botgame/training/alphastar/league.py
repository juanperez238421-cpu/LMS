from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def f_hard(win_prob: np.ndarray, p: float = 2.0) -> np.ndarray:
    """PFSP hard-opponent weighting `(1 - x)^p`."""
    return np.power(np.clip(1.0 - win_prob, 0.0, 1.0), p)


def f_var(win_prob: np.ndarray) -> np.ndarray:
    """PFSP variance weighting `x * (1 - x)`."""
    x = np.clip(win_prob, 0.0, 1.0)
    return x * (1.0 - x)


def f_linear(win_prob: np.ndarray) -> np.ndarray:
    """Alternative PFSP weighting favoring not-yet-mastered opponents."""
    return np.clip(1.0 - win_prob, 0.0, 1.0)


def pfsp_weights(win_probs: Iterable[float], mode: str = "var", hard_p: float = 2.0) -> np.ndarray:
    probs = np.asarray(list(win_probs), dtype=np.float64)
    if probs.size == 0:
        return probs
    if mode == "hard":
        raw = f_hard(probs, p=hard_p)
    elif mode == "linear":
        raw = f_linear(probs)
    else:
        raw = f_var(probs)
    if float(raw.sum()) <= 1e-9:
        return np.full_like(raw, fill_value=1.0 / raw.size)
    return raw / raw.sum()


@dataclass
class PFSPConfig:
    main_self_play_ratio: float = 0.35
    main_pfsp_all_ratio: float = 0.50
    main_pfsp_forgotten_ratio: float = 0.15
    exploiter_snapshot_threshold: float = 0.70
    main_exploiter_curriculum_threshold: float = 0.20
    league_exploiter_reset_prob: float = 0.25
    main_timeout_steps: int = int(2e6)
    main_exploiter_timeout_steps: int = int(4e6)
    league_exploiter_timeout_steps: int = int(2e6)
    forgotten_threshold: float = 0.30
    payoff_ema_alpha: float = 0.1


@dataclass
class LeaguePlayer:
    player_id: str
    agent_type: str
    checkpoint_path: str
    trainable: bool
    parent_id: str | None = None
    created_at_step: int = 0
    steps_since_snapshot: int = 0


@dataclass
class LeagueMatch:
    step: int
    agent_a: str
    agent_b: str
    outcome: float


@dataclass
class LeagueManager:
    """Maintains league population, payoff estimates, and PFSP matchmaking."""

    config: PFSPConfig = field(default_factory=PFSPConfig)
    rng_seed: int = 0
    players: Dict[str, LeaguePlayer] = field(default_factory=dict)
    payoff: Dict[Tuple[str, str], float] = field(default_factory=dict)
    matches: List[LeagueMatch] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.rng_seed)

    def add_player(self, player: LeaguePlayer) -> None:
        self.players[player.player_id] = player

    def list_players(self, *, trainable: bool | None = None, agent_type: str | None = None) -> List[LeaguePlayer]:
        items = list(self.players.values())
        if trainable is not None:
            items = [p for p in items if p.trainable == trainable]
        if agent_type is not None:
            items = [p for p in items if p.agent_type == agent_type]
        return items

    def estimate_win_prob(self, agent_a: str, agent_b: str) -> float:
        if (agent_a, agent_b) in self.payoff:
            return float(self.payoff[(agent_a, agent_b)])
        if (agent_b, agent_a) in self.payoff:
            return 1.0 - float(self.payoff[(agent_b, agent_a)])
        return 0.5

    def update_result(self, step: int, agent_a: str, agent_b: str, outcome: float) -> None:
        """Stores result where outcome is 1(win), 0(draw), -1(loss) for `agent_a`."""
        result_prob = 0.5 * (outcome + 1.0)
        current = self.estimate_win_prob(agent_a, agent_b)
        alpha = self.config.payoff_ema_alpha
        updated = (1.0 - alpha) * current + alpha * result_prob
        self.payoff[(agent_a, agent_b)] = float(updated)
        self.payoff[(agent_b, agent_a)] = float(1.0 - updated)
        self.matches.append(LeagueMatch(step=step, agent_a=agent_a, agent_b=agent_b, outcome=outcome))

    def frozen_opponents(self, exclude: str | None = None) -> List[LeaguePlayer]:
        opponents = self.list_players(trainable=False)
        if exclude:
            opponents = [p for p in opponents if p.player_id != exclude]
        return opponents

    def forgotten_main_players(self, main_player_id: str) -> List[LeaguePlayer]:
        candidates = [p for p in self.frozen_opponents(exclude=main_player_id) if p.agent_type == "main"]
        return [
            p
            for p in candidates
            if self.estimate_win_prob(main_player_id, p.player_id) < self.config.forgotten_threshold
        ]

    def snapshot(self, trainable_player_id: str, step: int, checkpoint_path: str) -> LeaguePlayer:
        parent = self.players[trainable_player_id]
        frozen_id = f"{trainable_player_id}_snap_{step}"
        frozen = LeaguePlayer(
            player_id=frozen_id,
            agent_type=parent.agent_type,
            checkpoint_path=checkpoint_path,
            trainable=False,
            parent_id=trainable_player_id,
            created_at_step=step,
        )
        self.add_player(frozen)
        parent.steps_since_snapshot = 0
        return frozen

    def should_snapshot(self, player_id: str) -> bool:
        player = self.players[player_id]
        timeout = {
            "main": self.config.main_timeout_steps,
            "main_exploiter": self.config.main_exploiter_timeout_steps,
            "league_exploiter": self.config.league_exploiter_timeout_steps,
        }[player.agent_type]
        if player.steps_since_snapshot >= timeout:
            return True

        opponents = [p.player_id for p in self.frozen_opponents(exclude=player_id)]
        if not opponents:
            return False
        probs = [self.estimate_win_prob(player_id, opp_id) for opp_id in opponents]
        return min(probs) >= self.config.exploiter_snapshot_threshold

    def should_reset_after_snapshot(self, player_id: str) -> bool:
        player = self.players[player_id]
        if player.agent_type == "main_exploiter":
            return True
        if player.agent_type == "league_exploiter":
            return self._rng.random() < self.config.league_exploiter_reset_prob
        return False

    def _sample_pfsp(self, agent_id: str, candidates: List[LeaguePlayer], mode: str) -> str:
        if not candidates:
            return agent_id
        probs = [self.estimate_win_prob(agent_id, c.player_id) for c in candidates]
        weights = pfsp_weights(probs, mode=mode)
        choice = self._rng.choices([c.player_id for c in candidates], weights=weights.tolist(), k=1)[0]
        return str(choice)

    def sample_opponent(self, player_id: str) -> str:
        player = self.players[player_id]
        if player.agent_type == "main":
            roll = self._rng.random()
            if roll < self.config.main_self_play_ratio:
                peers = [p.player_id for p in self.list_players(trainable=True, agent_type="main") if p.player_id != player_id]
                return self._rng.choice(peers) if peers else player_id
            if roll < self.config.main_self_play_ratio + self.config.main_pfsp_all_ratio:
                return self._sample_pfsp(player_id, self.frozen_opponents(exclude=player_id), mode="hard")
            forgotten = self.forgotten_main_players(player_id)
            if forgotten:
                return self._sample_pfsp(player_id, forgotten, mode="var")
            peers = [p.player_id for p in self.list_players(trainable=True, agent_type="main") if p.player_id != player_id]
            return self._rng.choice(peers) if peers else player_id

        if player.agent_type == "main_exploiter":
            mains = [p for p in self.list_players(trainable=True, agent_type="main")]
            if not mains:
                return player_id
            main_probs = [self.estimate_win_prob(player_id, m.player_id) for m in mains]
            if min(main_probs) < self.config.main_exploiter_curriculum_threshold and self._rng.random() < 0.5:
                frozen_mains = [p for p in self.frozen_opponents() if p.agent_type == "main"]
                return self._sample_pfsp(player_id, frozen_mains, mode="var")
            return self._rng.choice([m.player_id for m in mains])

        # league_exploiter
        return self._sample_pfsp(player_id, self.frozen_opponents(exclude=player_id), mode="hard")

    def tick_train_step(self, player_id: str, steps: int) -> None:
        self.players[player_id].steps_since_snapshot += steps

    def to_json(self) -> Dict[str, object]:
        return {
            "players": {k: asdict(v) for k, v in self.players.items()},
            "payoff": {f"{a}::{b}": p for (a, b), p in self.payoff.items()},
            "matches": [asdict(m) for m in self.matches],
            "config": asdict(self.config),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=True, indent=2), encoding="utf-8")

