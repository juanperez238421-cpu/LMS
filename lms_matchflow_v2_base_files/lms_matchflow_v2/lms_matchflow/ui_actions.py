from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import time


@dataclass
class ClickCooldown:
    """Prevent spamming clicks while still reacting fast."""
    min_interval_sec: float = 0.35
    last_click_ts: float = 0.0

    def ready(self) -> bool:
        return (time.time() - self.last_click_ts) >= self.min_interval_sec

    def mark(self) -> None:
        self.last_click_ts = time.time()


@dataclass
class MatchFlowActions:
    """Abstract actions: Codex wires these to your input driver."""
    click_salon: bool = False
    click_map_slot: Optional[int] = None
    choose_map: Optional[str] = None
    notes: Tuple[str, ...] = ()


def decide_actions_for_phase(
    phase: str,
    *,
    phase_score: float,
    evidence_text: str,
    cooldowns: Dict[str, ClickCooldown],
) -> MatchFlowActions:
    phase = str(phase)

    if phase == "dead":
        cd = cooldowns.setdefault("salon", ClickCooldown())
        if cd.ready():
            cd.mark()
            return MatchFlowActions(click_salon=True, notes=("dead->click_salon",))
        return MatchFlowActions(click_salon=False, notes=("dead_wait_cooldown",))

    if phase == "countdown":
        return MatchFlowActions(notes=("countdown_noop_base",))

    if phase == "lobby":
        return MatchFlowActions(notes=("lobby_noop_base",))

    return MatchFlowActions()
