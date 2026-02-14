from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple
import re
import time
import unicodedata


class GamePhase(str, Enum):
    """High-level phase for throttling expensive perception + taking UI actions."""

    UNKNOWN = "unknown"
    LOBBY = "lobby"
    COUNTDOWN = "countdown"
    IN_MATCH = "in_match"
    DEAD = "dead"
    END_OF_COMBAT = "end_of_combat"
    LOADING = "loading"


@dataclass(frozen=True)
class PhaseEvidence:
    phase: GamePhase
    score: float
    reason: str
    tokens: Tuple[str, ...] = ()


@dataclass
class PhaseState:
    game_phase: GamePhase = GamePhase.UNKNOWN
    phase_confidence: float = 0.0
    phase_reason: str = "boot"
    changed_at: float = 0.0

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "game_phase": self.game_phase.value,
            "phase_confidence": float(self.phase_confidence),
            "phase_reason": str(self.phase_reason),
        }


@dataclass
class PhaseMachine:
    """Simple hysteresis machine to reduce phase flicker."""

    state: PhaseState = field(default_factory=PhaseState)
    min_confidence_switch: float = 0.55
    dead_priority_confidence: float = 0.65

    def update(self, ev: PhaseEvidence) -> PhaseState:
        prev_phase = self.state.game_phase
        now = time.time()

        if ev.phase == prev_phase:
            self.state.phase_confidence = max(float(self.state.phase_confidence), float(ev.score))
            self.state.phase_reason = ev.reason
            return self.state

        can_switch = float(ev.score) >= float(self.min_confidence_switch)
        if ev.phase == GamePhase.DEAD and float(ev.score) >= float(self.dead_priority_confidence):
            can_switch = True

        if can_switch:
            self.state.game_phase = ev.phase
            self.state.phase_confidence = float(ev.score)
            self.state.phase_reason = ev.reason
            self.state.changed_at = now

        return self.state


_TIME_MMSS_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")


def normalize_text_for_match(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "")).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9:# \n\t]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _hit_count(text_norm: str, tokens: Tuple[str, ...]) -> int:
    return sum(1 for t in tokens if t in text_norm)


def infer_phase_from_text(
    merged_ocr_text: str,
    *,
    extra_flags: Optional[Dict[str, Any]] = None,
) -> PhaseEvidence:
    """Fast phase classifier driven by OCR text snippets."""
    t = normalize_text_for_match(merged_ocr_text)
    flags = extra_flags or {}

    end_tokens = (
        "fin del combate",
        "recompensas",
        "mazmorra real",
        "tiempo con vida",
        "eliminaciones",
        "objetos construidos",
    )
    dead_tokens = (
        "estas fuera",
        "observando",
        "spectating",
        "you died",
        "has muerto",
        "has sido eliminado",
        "derrotado",
        "game over",
    )
    lobby_tokens = (
        "lobby",
        "salon",
        "jugar",
        "play",
        "ready",
        "matchmaking",
        "queue",
        "seleccionar",
        "select",
        "personaje",
        "character",
        "preparacion",
    )
    loading_tokens = (
        "cargando",
        "loading",
        "conectando",
        "connecting",
        "reconectando",
        "reconnecting",
    )

    end_hits = _hit_count(t, end_tokens)
    if end_hits >= 2:
        return PhaseEvidence(GamePhase.END_OF_COMBAT, min(1.0, 0.95 + 0.01 * end_hits), "end_tokens", end_tokens)

    dead_hits = _hit_count(t, dead_tokens)
    if dead_hits >= 1 and ("salon" in t or flags.get("salon_button_seen")):
        return PhaseEvidence(GamePhase.DEAD, min(1.0, 0.85 + 0.02 * dead_hits), "dead_tokens+salon", dead_tokens)

    if "comienza en" in t or "empieza en" in t or "inicio en" in t or "comenzando" in t:
        return PhaseEvidence(GamePhase.COUNTDOWN, 0.8, "countdown_keyword", ("comienza en",))
    if _TIME_MMSS_RE.search(t) and flags.get("countdown_roi_used", False):
        return PhaseEvidence(GamePhase.COUNTDOWN, 0.75, "countdown_time_mmss", ())

    lobby_hits = _hit_count(t, lobby_tokens)
    if lobby_hits >= 2 or flags.get("play_button_seen"):
        return PhaseEvidence(GamePhase.LOBBY, min(1.0, 0.6 + 0.05 * lobby_hits), "lobby_tokens", lobby_tokens)

    loading_hits = _hit_count(t, loading_tokens)
    if loading_hits >= 1:
        return PhaseEvidence(GamePhase.LOADING, min(1.0, 0.55 + 0.05 * loading_hits), "loading_tokens", loading_tokens)

    if flags.get("in_match_hud_seen"):
        return PhaseEvidence(GamePhase.IN_MATCH, 0.7, "hud_hint", ())

    return PhaseEvidence(GamePhase.UNKNOWN, 0.1, "no_strong_signal", ())


def is_full_pipeline_phase(phase: GamePhase) -> bool:
    return phase == GamePhase.IN_MATCH


def phase_target_fps(phase: GamePhase) -> int:
    """Outside IN_MATCH keep 2-5 FPS to avoid wasting frames."""
    if phase == GamePhase.IN_MATCH:
        return 10
    if phase in {GamePhase.COUNTDOWN, GamePhase.DEAD, GamePhase.END_OF_COMBAT}:
        return 5
    return 3
