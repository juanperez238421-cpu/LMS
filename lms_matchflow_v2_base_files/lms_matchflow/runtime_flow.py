from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from .end_metrics import EndOfCombatMetrics, extract_end_of_combat_metrics
from .map_catalog import MapDef, match_map_name_from_ocr
from .map_fingerprint import MapFingerprintDB
from .match_state import (
    GamePhase,
    PhaseMachine,
    infer_phase_from_text,
    is_full_pipeline_phase,
    phase_target_fps,
)


@dataclass
class MatchIdentity:
    map_id: Optional[str] = None
    map_name: Optional[str] = None
    map_size_class: Optional[str] = None

    def set_map(self, m: MapDef) -> None:
        self.map_id = m.map_id
        self.map_name = m.display_name
        self.map_size_class = m.size_class

    def to_telemetry(self) -> Dict[str, Optional[str]]:
        return {
            "match.map_id": self.map_id,
            "match.map_name": self.map_name,
            "match.map_size_class": self.map_size_class,
        }


@dataclass
class MatchFlowRuntime:
    """Runtime helper for map identification + phase gating + end metrics."""

    map_db: Optional[MapFingerprintDB] = None
    phase_machine: PhaseMachine = field(default_factory=PhaseMachine)
    match_identity: MatchIdentity = field(default_factory=MatchIdentity)
    match_end: EndOfCombatMetrics = field(default_factory=EndOfCombatMetrics)

    @classmethod
    def from_roi_config(cls, cfg: Dict[str, Any]) -> "MatchFlowRuntime":
        big_default = str(cfg.get("assets_map_dir_big_default", "assets/Mapas/Big") or "assets/Mapas/Big")
        small_default = str(cfg.get("assets_map_dir_small_default", "assets/Mapas/Small") or "assets/Mapas/Small")
        db = MapFingerprintDB(ref_dirs=[Path(big_default), Path(small_default)])
        return cls(map_db=db)

    def detect_map(
        self,
        *,
        ocr_map_name_text: str = "",
        minimap_crop: Optional[Image.Image] = None,
        fingerprint_max_dist: int = 18,
    ) -> Optional[MapDef]:
        """(A) OCR map-name preferred; (B) minimap fingerprint fallback."""
        if self.match_identity.map_id:
            return None

        found = match_map_name_from_ocr(ocr_map_name_text)
        if found is not None:
            self.match_identity.set_map(found)
            return found

        if minimap_crop is None or self.map_db is None or not self.map_db.is_ready():
            return None

        found = self.map_db.match_minimap(minimap_crop, max_dist=fingerprint_max_dist)
        if found is not None:
            self.match_identity.set_map(found)
            return found

        return None

    def update_phase(self, merged_ocr_text: str, *, extra_flags: Optional[Dict[str, Any]] = None) -> None:
        ev = infer_phase_from_text(merged_ocr_text, extra_flags=extra_flags)
        self.phase_machine.update(ev)

    def should_run_full_pipeline(self) -> bool:
        return is_full_pipeline_phase(self.phase_machine.state.game_phase)

    def target_fps(self) -> int:
        return phase_target_fps(self.phase_machine.state.game_phase)

    def maybe_extract_end_metrics(self, ocr_by_field: Dict[str, str]) -> Optional[EndOfCombatMetrics]:
        if self.phase_machine.state.game_phase != GamePhase.END_OF_COMBAT:
            return None
        self.match_end = extract_end_of_combat_metrics(ocr_by_field)
        return self.match_end

    def telemetry_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        out.update(self.phase_machine.state.to_telemetry())
        out.update(self.match_identity.to_telemetry())
        out.update(
            {
                "match_end.time_alive_sec": self.match_end.time_alive_sec,
                "match_end.eliminations": self.match_end.eliminations,
                "match_end.objects_built": self.match_end.objects_built,
            }
        )
        return out
