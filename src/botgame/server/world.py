from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import math

from botgame.common.types import Observation, Action, SelfState, Entity, Item, ZoneState
from botgame.common.rng import PRNG

# Constants for the game world
PLAYER_VISION_RADIUS = 10.0 # units
PLAYER_MAX_HP = 100.0
PLAYER_MAX_AMMO = 30
PLAYER_MOVE_SPEED = 5.0 # units per second
PROJECTILE_SPEED = 20.0 # units per second
PROJECTILE_DAMAGE = 10.0
TICK_RATE = 30 # ticks per second

@dataclass
class Player:
    id: str
    is_bot: bool
    position: List[float] # [x, y]
    velocity: List[float] # [vx, vy]
    hp: float
    ammo: int
    cooldowns: Dict[str, float] # Ability name -> time left (e.g., "fire": 0.5)
    last_fire_time: float = 0.0 # To manage fire rate
    fire_rate_cooldown: float = 0.2 # seconds between shots

@dataclass
class Projectile:
    id: str
    owner_id: str
    position: List[float]
    velocity: List[float]
    damage: float
    creation_time: float

@dataclass
class GameItem:
    id: str
    position: List[float]
    type: str # e.g., "health_pack", "ammo_crate"
    value: float # amount of health or ammo

@dataclass
class WorldState:
    players: Dict[str, Player]
    projectiles: Dict[str, Projectile]
    items: Dict[str, GameItem]
    zone_state: ZoneState
    _rng: PRNG = field(repr=False) # Internal RNG for deterministic events
    tick_id: int = 0
    time: float = 0.0 # Simulation time in seconds

    def tick(self, dt: float) -> None:
        """
        Advances the world simulation by a given delta time.
        Args:
            dt: The delta time in seconds for this tick.
        """
        self.time += dt
        self.tick_id += 1

        # 1. Update player cooldowns
        for player in self.players.values():
            for cooldown_name in list(player.cooldowns.keys()):
                player.cooldowns[cooldown_name] = max(0.0, player.cooldowns[cooldown_name] - dt)
            player.last_fire_time = max(0.0, player.last_fire_time - dt)


        # 2. Move players (based on velocity from applied actions)
        for player in self.players.values():
            player.position[0] += player.velocity[0] * dt
            player.position[1] += player.velocity[1] * dt
            # Simple boundary check (e.g., wrap around or clamp)
            # For now, let's assume an unbounded world or handle boundaries later


        # 3. Move projectiles and handle collisions
        projectiles_to_remove = []
        new_projectiles: Dict[str, Projectile] = {}
        for proj_id, projectile in self.projectiles.items():
            projectile.position[0] += projectile.velocity[0] * dt
            projectile.position[1] += projectile.velocity[1] * dt

            # Check collision with players
            for player_id, player in self.players.items():
                if player_id != projectile.owner_id and self._distance(player.position, projectile.position) < 1.0: # Simple collision
                    player.hp -= projectile.damage
                    projectiles_to_remove.append(proj_id)
                    break # Projectile hit one player, so it's "destroyed"

            # Remove if out of bounds or too old (for simplicity, just remove after a time)
            if self.time - projectile.creation_time > 5.0: # Projectiles last 5 seconds
                 projectiles_to_remove.append(proj_id)

            if proj_id not in projectiles_to_remove:
                new_projectiles[proj_id] = projectile
        self.projectiles = new_projectiles

        # 4. Handle zone
        # For now, just update is_safe based on player position relative to zone
        for player in self.players.values():
            if self._distance(player.position, self.zone_state.position) > self.zone_state.radius:
                # Player is outside zone, take damage (placeholder)
                player.hp -= 1.0 * dt # 1 damage per second outside zone
            player.hp = max(0.0, player.hp) # Clamp HP

        # 5. Remove dead players
        self.players = {p_id: p for p_id, p in self.players.items() if p.hp > 0}

        # 6. Spawn items (simple placeholder)
        if self.tick_id % (TICK_RATE * 5) == 0: # Every 5 seconds
            if len(self.items) < 5:
                item_id = f"item_{self.tick_id}_{self._rng.randint(0, 1000)}"
                item_type = self._rng.choice(["health_pack", "ammo_crate"])
                item_value = 25.0 if item_type == "health_pack" else 15.0
                pos = [self._rng.random() * 100 - 50, self._rng.random() * 100 - 50] # Random pos
                self.items[item_id] = GameItem(id=item_id, position=pos, type=item_type, value=item_value)

        # 7. Player-item interaction (if a player "interacted") - handled by apply_action for now
        #    But a bot could "interact" when close to an item.

    def build_observation(self, player_id: str) -> Observation:
        """
        Constructs a partial observation for a given player, representing fog-of-war.
        """
        player = self.players.get(player_id)
        if not player:
            # Return a default/empty observation or raise an error
            raise ValueError(f"Player with ID {player_id} not found.")

        self_state = SelfState(
            position=player.position,
            velocity=player.velocity,
            hp=player.hp,
            cooldowns=player.cooldowns,
            ammo=player.ammo
        )

        visible_entities: List[Entity] = []
        for other_player_id, other_player in self.players.items():
            if other_player_id == player_id:
                continue
            if self._distance(player.position, other_player.position) <= PLAYER_VISION_RADIUS:
                visible_entities.append(Entity(
                    id=other_player.id,
                    position=other_player.position,
                    type="enemy" if other_player.is_bot != player.is_bot else "friend", # Simplified
                    hp=other_player.hp
                ))

        visible_items: List[Item] = []
        for item_id, item in self.items.items():
            if self._distance(player.position, item.position) <= PLAYER_VISION_RADIUS:
                visible_items.append(Item(
                    id=item.id,
                    position=item.position,
                    type=item.type
                ))

        # Zone state is global for now
        zone_state = self.zone_state
        zone_state.is_safe = self._distance(player.position, zone_state.position) <= zone_state.radius

        return Observation(
            self_state=self_state,
            visible_entities=visible_entities,
            visible_items=visible_items,
            zone_state=zone_state,
            tick_id=self.tick_id
        )

    def apply_action(self, player_id: str, action: Action) -> None:
        """
        Applies a player's action to the world state.
        """
        player = self.players.get(player_id)
        if not player:
            return

        action.clamp_and_validate()

        # Movement
        player.velocity[0] = action.move_x * PLAYER_MOVE_SPEED
        player.velocity[1] = action.move_y * PLAYER_MOVE_SPEED

        # Firing
        if action.fire and player.ammo > 0 and player.last_fire_time <= 0.0:
            if action.aim_x != 0 or action.aim_y != 0:
                # Normalize aim vector
                mag = math.sqrt(action.aim_x**2 + action.aim_y**2)
                if mag > 0:
                    aim_dir_x = action.aim_x / mag
                    aim_dir_y = action.aim_y / mag
                else: # Default if aim is (0,0)
                    aim_dir_x, aim_dir_y = 1.0, 0.0

                proj_id = f"proj_{self.tick_id}_{player_id}_{self._rng.randint(0, 1000)}"
                projectile = Projectile(
                    id=proj_id,
                    owner_id=player_id,
                    position=list(player.position), # Projectile starts at player pos
                    velocity=[aim_dir_x * PROJECTILE_SPEED, aim_dir_y * PROJECTILE_SPEED],
                    damage=PROJECTILE_DAMAGE,
                    creation_time=self.time
                )
                self.projectiles[proj_id] = projectile
                player.ammo -= 1
                player.last_fire_time = player.fire_rate_cooldown
                player.cooldowns["fire"] = player.fire_rate_cooldown # Also update generic cooldown

        # Ability usage (placeholder)
        if action.ability_id is not None:
            # Implement ability logic here
            pass

        # Interaction
        if action.interact:
            items_to_remove = []
            for item_id, item in self.items.items():
                if self._distance(player.position, item.position) < 2.0: # Close enough to interact
                    if item.type == "health_pack":
                        player.hp = min(PLAYER_MAX_HP, player.hp + item.value)
                    elif item.type == "ammo_crate":
                        player.ammo = min(PLAYER_MAX_AMMO, player.ammo + int(item.value))
                    items_to_remove.append(item_id)
            for item_id in items_to_remove:
                del self.items[item_id]


    def _distance(self, pos1: List[float], pos2: List[float]) -> float:
        """Calculates Euclidean distance between two 2D points."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
