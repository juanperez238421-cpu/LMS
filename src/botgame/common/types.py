from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class SelfState:
    position: List[float]  # [x, y]
    velocity: List[float]  # [vx, vy]
    hp: float
    cooldowns: Dict[str, float] # Ability name -> time left
    ammo: int

@dataclass
class Entity:
    id: str
    position: List[float]
    type: str # e.g., "enemy", "friend", "projectile"
    hp: Optional[float] = None # Only for living entities

@dataclass
class Item:
    id: str
    position: List[float]
    type: str # e.g., "health_pack", "ammo_crate", "weapon"

@dataclass
class ZoneState:
    position: List[float] # [x, y] of zone center
    radius: float
    is_safe: bool

@dataclass
class Observation:
    self_state: SelfState
    zone_state: ZoneState
    tick_id: int
    visible_entities: List[Entity] = field(default_factory=list) # Enemies/friends within vision
    visible_items: List[Item] = field(default_factory=list) # Loot near you
    # Potentially add more for map data, navmesh queries, etc.

@dataclass
class Action:
    move_x: float  # Normalized, -1 to 1
    move_y: float  # Normalized, -1 to 1
    aim_x: float   # Normalized, -1 to 1 (direction vector)
    aim_y: float   # Normalized, -1 to 1 (direction vector)
    fire: bool     # True to fire
    ability_id: Optional[int] = None # ID of ability to use, or None
    interact: bool = False # True to interact (loot/open/revive)

    def clamp_and_validate(self):
        # Clamp move and aim to -1 to 1
        self.move_x = max(-1.0, min(1.0, self.move_x))
        self.move_y = max(-1.0, min(1.0, self.move_y))
        self.aim_x = max(-1.0, min(1.0, self.aim_x))
        self.aim_y = max(-1.0, min(1.0, self.aim_y))

        # Ensure aim vector is not (0,0) if trying to aim
        if self.aim_x == 0 and self.aim_y == 0 and (self.fire or self.ability_id is not None):
            # Default to aiming forward or some sensible default if trying to do something that requires aim
            self.aim_x = 1.0
