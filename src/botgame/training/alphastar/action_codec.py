from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from botgame.common.types import Action


@dataclass(frozen=True)
class ActionHeadSpec:
    name: str
    size: int


class FactorizedActionCodec:
    """Encodes structured `Action` objects into factorized categorical heads."""

    def __init__(self, bins: int = 11, ability_size: int = 2):
        self.bins = bins
        self.ability_size = max(2, ability_size)
        self.heads = [
            ActionHeadSpec("move_x", bins),
            ActionHeadSpec("move_y", bins),
            ActionHeadSpec("aim_x", bins),
            ActionHeadSpec("aim_y", bins),
            ActionHeadSpec("fire", 2),
            ActionHeadSpec("ability", self.ability_size),
            ActionHeadSpec("interact", 2),
        ]

    @property
    def head_sizes(self) -> Dict[str, int]:
        return {head.name: head.size for head in self.heads}

    def _quantize(self, value: float) -> int:
        clipped = float(np.clip(value, -1.0, 1.0))
        scaled = (clipped + 1.0) * 0.5
        index = int(round(scaled * (self.bins - 1)))
        return int(np.clip(index, 0, self.bins - 1))

    def _dequantize(self, index: int) -> float:
        scaled = float(index) / float(max(1, self.bins - 1))
        return float(2.0 * scaled - 1.0)

    def encode_action(self, action: Action | Dict[str, Any]) -> Dict[str, int]:
        if isinstance(action, Action):
            action_dict = action.__dict__
        else:
            action_dict = action
        ability_id = action_dict.get("ability_id")
        ability = 0 if ability_id is None else int(np.clip(int(ability_id), 1, self.ability_size - 1))
        return {
            "move_x": self._quantize(float(action_dict.get("move_x", 0.0))),
            "move_y": self._quantize(float(action_dict.get("move_y", 0.0))),
            "aim_x": self._quantize(float(action_dict.get("aim_x", 0.0))),
            "aim_y": self._quantize(float(action_dict.get("aim_y", 0.0))),
            "fire": 1 if bool(action_dict.get("fire", False)) else 0,
            "ability": ability,
            "interact": 1 if bool(action_dict.get("interact", False)) else 0,
        }

    def decode_action(self, encoded: Dict[str, int]) -> Action:
        ability_raw = int(encoded["ability"])
        ability_id = ability_raw if ability_raw > 0 else None
        return Action(
            move_x=self._dequantize(int(encoded["move_x"])),
            move_y=self._dequantize(int(encoded["move_y"])),
            aim_x=self._dequantize(int(encoded["aim_x"])),
            aim_y=self._dequantize(int(encoded["aim_y"])),
            fire=bool(int(encoded["fire"])),
            ability_id=ability_id,
            interact=bool(int(encoded["interact"])),
        )

    def action_to_env(self, encoded: Dict[str, int]) -> Dict[str, np.ndarray | int]:
        decoded = self.decode_action(encoded)
        ability_val = 0 if decoded.ability_id is None else int(decoded.ability_id)
        return {
            "move_x": np.array([decoded.move_x], dtype=np.float32),
            "move_y": np.array([decoded.move_y], dtype=np.float32),
            "aim_x": np.array([decoded.aim_x], dtype=np.float32),
            "aim_y": np.array([decoded.aim_y], dtype=np.float32),
            "fire": int(decoded.fire),
            "ability": int(np.clip(ability_val, 0, 1)),
            "interact": int(decoded.interact),
        }

    def log_prob(
        self,
        logits: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        per_component: Dict[str, torch.Tensor] = {}
        total = None
        for name, head_logits in logits.items():
            logp = F.log_softmax(head_logits, dim=-1)
            selected = logp.gather(-1, actions[name].long().unsqueeze(-1)).squeeze(-1)
            per_component[name] = selected
            total = selected if total is None else total + selected
        if total is None:
            raise ValueError("No logits were provided for log_prob computation.")
        return total, per_component

    def entropy(self, logits: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for head_logits in logits.values():
            logp = F.log_softmax(head_logits, dim=-1)
            p = torch.exp(logp)
            ent = -(p * logp).sum(dim=-1)
            total = ent if total is None else total + ent
        if total is None:
            raise ValueError("No logits were provided for entropy computation.")
        return total

