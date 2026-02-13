from __future__ import annotations

from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field

TelemetrySource = Literal["ws", "ocr", "heur", "fallback", "sim", "unknown"]

class EntityTrack(BaseModel):
    name: str = ""
    kind: Literal["player", "bot", "npc", "unknown"] = "unknown"
    hp_pct: Optional[float] = None          # 0..100
    conf: float = 0.0                       # 0..1
    anchor_xy: Optional[Tuple[float, float]] = None  # normalized (0..1,0..1)

class TelemetryFrame(BaseModel):
    # --- identity/time ---
    ts_ms: int = Field(..., description="epoch ms")
    run_id: str = ""
    match_id: str = ""

    # --- vitals ---
    hp_pct: Optional[float] = None
    hp_conf: float = 0.0
    hp_src: TelemetrySource = "unknown"

    stamina_pct: Optional[float] = None
    stamina_conf: float = 0.0
    stamina_src: TelemetrySource = "unknown"

    # --- damage ---
    dmg_in_total: float = 0.0
    dmg_out_total: float = 0.0
    dmg_in_tick: float = 0.0
    dmg_out_tick: float = 0.0

    # --- perception (enemy) ---
    enemy_visible: bool = False
    enemy_conf: float = 0.0
    enemy_dir_deg: Optional[float] = None
    enemy_xy: Optional[Tuple[float, float]] = None  # normalized (0..1,0..1)

    # --- zone ---
    zone_outside: bool = False
    zone_toxic: bool = False
    zone_countdown_s: Optional[float] = None

    # --- decisions / actions ---
    decision: str = ""
    action_last: str = ""
    action_ok: Optional[bool] = None

    # --- movement / collision proxies ---
    motion_score: Optional[float] = None
    stuck: Optional[bool] = None
    collision_proxy: Optional[bool] = None

    # --- entities ---
    entities: List[EntityTrack] = Field(default_factory=list)
