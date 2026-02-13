import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from botgame.bots.base import BotPolicy
from botgame.common.rng import PRNG
from botgame.common.types import Action, Entity, Item, Observation, SelfState
from botgame.server.world import TICK_RATE

DEFAULT_MODE_RULES_PATH = Path("reports/reverse_engineering/mode_rules_summary.json")
DEFAULT_CATALOG_SUMMARY_PATH = Path("reports/reverse_engineering/catalog_summary.json")


@dataclass
class LMSModeRuleSnapshot:
    source_path: str = ""
    mode_name: str = ""
    resource_collection_amount: Optional[float] = None
    global_mana_regen_multiplier: Optional[float] = None
    respawn_time_increase_per_death: Optional[float] = None
    chest_regen_seconds: Optional[float] = None
    mana_resource_only: Optional[bool] = None
    owned_object_limits: Dict[str, float] = field(default_factory=dict)
    ability_costs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ability_cooldowns: Dict[str, float] = field(default_factory=dict)


@dataclass
class LMSCatalogSnapshot:
    source_path: str = ""
    player_prefabs: List[str] = field(default_factory=list)
    enemy_prefabs: List[str] = field(default_factory=list)
    ability_sprites: List[str] = field(default_factory=list)
    loot_assets: List[str] = field(default_factory=list)
    zone_assets: List[str] = field(default_factory=list)


def load_lms_mode_rule_snapshot(
    mode_name: str = "royale_mode",
    path: Path = DEFAULT_MODE_RULES_PATH,
) -> LMSModeRuleSnapshot:
    if not path.exists():
        return LMSModeRuleSnapshot(mode_name=mode_name)

    payload = json.loads(path.read_text(encoding="utf-8"))
    mode_payload = (payload.get("modes") or {}).get(mode_name, {})
    return LMSModeRuleSnapshot(
        source_path=str(path),
        mode_name=mode_name,
        resource_collection_amount=mode_payload.get("resourceCollectionAmount"),
        global_mana_regen_multiplier=mode_payload.get("globalPlayerManaRegenMultiplier"),
        respawn_time_increase_per_death=mode_payload.get("respawnTimeIncreasePerDeath"),
        chest_regen_seconds=mode_payload.get("regenerateChestSeconds"),
        mana_resource_only=mode_payload.get("manaResourceOnly"),
        owned_object_limits=dict(mode_payload.get("ownedObjectLimits", {}) or {}),
        ability_costs=dict(mode_payload.get("abilityCosts", {}) or {}),
        ability_cooldowns=dict(mode_payload.get("abilityCooldowns", {}) or {}),
    )


def load_lms_catalog_snapshot(path: Path = DEFAULT_CATALOG_SUMMARY_PATH) -> LMSCatalogSnapshot:
    if not path.exists():
        return LMSCatalogSnapshot()

    payload = json.loads(path.read_text(encoding="utf-8"))
    return LMSCatalogSnapshot(
        source_path=str(path),
        player_prefabs=list(payload.get("player_prefabs", []) or []),
        enemy_prefabs=list(payload.get("enemy_prefabs", []) or []),
        ability_sprites=list(payload.get("ability_sprites", []) or []),
        loot_assets=list(payload.get("loot_assets", []) or []),
        zone_assets=list(payload.get("zone_assets", []) or []),
    )


class LMSReverseEngineeredBot(BotPolicy):
    """
    Rule-based policy modeled from extracted LMS symbols and mode configs.

    Priority order:
    1) Zone survival and toxic escape.
    2) Combat decisions with ability queue and cooldown gating.
    3) Loot collection (chests/ability drops/resources).
    4) Zone-centered patrol.
    """

    _ABILITY_ALIASES = {
        1: ("digit1", "ability1", "slot1", "skill1"),
        2: ("digit2", "ability2", "slot2", "skill2", "shift", "dash"),
        3: ("digit3", "ability3", "slot3", "skill3"),
    }

    def __init__(
        self,
        player_id: str,
        mode_name: str = "royale_mode",
        seed: Optional[int] = None,
        retreat_hp_threshold: float = 35.0,
        fire_range: float = 10.0,
        strafe_min_range: float = 4.0,
        strafe_max_range: float = 9.0,
        dash_cooldown_fallback_sec: float = 2.0,
        mode_rules_path: Path = DEFAULT_MODE_RULES_PATH,
        catalog_summary_path: Path = DEFAULT_CATALOG_SUMMARY_PATH,
    ) -> None:
        self.player_id = player_id
        self.retreat_hp_threshold = retreat_hp_threshold
        self.fire_range = fire_range
        self.strafe_min_range = strafe_min_range
        self.strafe_max_range = strafe_max_range
        self.mode_rules = load_lms_mode_rule_snapshot(mode_name=mode_name, path=mode_rules_path)
        self.catalog = load_lms_catalog_snapshot(path=catalog_summary_path)

        self._rng: Optional[PRNG] = PRNG(seed if seed is not None else 0)
        self._last_hp: Optional[float] = None
        self._zone_pressure_streak: int = 0
        self._last_seen_enemy_tick: int = -10_000
        self._fallback_ready_tick: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self._fallback_cooldown_sec: Dict[int, float] = self._build_slot_cooldowns(
            dash_cooldown_fallback_sec=dash_cooldown_fallback_sec
        )

        if seed is not None:
            self.reset(seed)

    def reset(self, seed: int) -> None:
        self._rng = PRNG(seed)
        self._last_hp = None
        self._zone_pressure_streak = 0
        self._last_seen_enemy_tick = -10_000
        self._fallback_ready_tick = {1: 0, 2: 0, 3: 0}

    def act(self, observation: Observation) -> Action:
        if self._rng is None:
            raise RuntimeError("BotPolicy not reset. Call reset() first.")

        self_state = observation.self_state
        enemy = self._closest_enemy(observation)
        item = self._best_item(observation)

        self._update_adaptation_state(observation, enemy_present=enemy is not None)

        if self._must_escape_zone(observation):
            return self._zone_escape_action(self_state, observation, enemy)

        if enemy is not None:
            if self_state.hp <= self.retreat_hp_threshold:
                return self._retreat_action(self_state, observation, enemy)
            return self._engage_action(self_state, observation, enemy)

        if item is not None:
            return self._loot_action(self_state, item)

        return self._patrol_action(self_state, observation)

    def _build_slot_cooldowns(self, dash_cooldown_fallback_sec: float) -> Dict[int, float]:
        cooldown_values = [
            float(v)
            for v in self.mode_rules.ability_cooldowns.values()
            if isinstance(v, (float, int)) and float(v) > 0
        ]
        offense_default = min(cooldown_values) if cooldown_values else 5.0
        defense_default = max(cooldown_values) if cooldown_values else 12.0
        return {
            1: offense_default,
            2: float(dash_cooldown_fallback_sec),
            3: defense_default,
        }

    def _update_adaptation_state(self, observation: Observation, enemy_present: bool) -> None:
        current_hp = float(observation.self_state.hp)
        hp_delta = 0.0
        if self._last_hp is not None:
            hp_delta = max(0.0, self._last_hp - current_hp)
        self._last_hp = current_hp

        if enemy_present:
            self._last_seen_enemy_tick = int(observation.tick_id)

        if not observation.zone_state.is_safe:
            self._zone_pressure_streak = min(10, self._zone_pressure_streak + (2 if hp_delta > 0.1 else 1))
        elif hp_delta > 0.0 and enemy_present:
            self._zone_pressure_streak = min(10, self._zone_pressure_streak + 1)
        else:
            self._zone_pressure_streak = max(0, self._zone_pressure_streak - 1)

    def _must_escape_zone(self, observation: Observation) -> bool:
        if not observation.zone_state.is_safe:
            return True
        if self._zone_pressure_streak >= 4:
            return True
        return False

    def _zone_escape_action(
        self,
        self_state: SelfState,
        observation: Observation,
        enemy: Optional[Entity],
    ) -> Action:
        zone_center = observation.zone_state.position
        move_x, move_y = self._unit_vector(self_state.position, zone_center)
        action = Action(move_x=move_x, move_y=move_y, aim_x=move_x, aim_y=move_y, fire=False, interact=False)

        if enemy is not None:
            action.aim_x, action.aim_y = self._unit_vector(self_state.position, enemy.position)
            if self._can_fire(self_state) and self._distance(self_state.position, enemy.position) <= self.fire_range:
                action.fire = True

        if self._ability_ready(observation, slot=2):
            dist_zone = self._distance(self_state.position, zone_center)
            if self._zone_pressure_streak >= 2 or dist_zone > max(4.0, observation.zone_state.radius * 0.35):
                action.ability_id = 2
                self._register_ability_use(observation.tick_id, slot=2)

        return action

    def _retreat_action(self, self_state: SelfState, observation: Observation, enemy: Entity) -> Action:
        run_x, run_y = self._unit_vector(enemy.position, self_state.position)
        zone_x, zone_y = self._unit_vector(self_state.position, observation.zone_state.position)
        blend_x = (0.7 * run_x) + (0.3 * zone_x)
        blend_y = (0.7 * run_y) + (0.3 * zone_y)
        move_x, move_y = self._normalize(blend_x, blend_y)

        aim_x, aim_y = self._unit_vector(self_state.position, enemy.position)
        action = Action(move_x=move_x, move_y=move_y, aim_x=aim_x, aim_y=aim_y, fire=False, interact=False)

        if self._ability_ready(observation, slot=3):
            action.ability_id = 3
            self._register_ability_use(observation.tick_id, slot=3)
        elif self._ability_ready(observation, slot=2) and self._distance(self_state.position, enemy.position) < 3.5:
            action.ability_id = 2
            self._register_ability_use(observation.tick_id, slot=2)

        if self._can_fire(self_state) and self._distance(self_state.position, enemy.position) <= self.fire_range:
            action.fire = True

        return action

    def _engage_action(self, self_state: SelfState, observation: Observation, enemy: Entity) -> Action:
        dist_enemy = self._distance(self_state.position, enemy.position)
        aim_x, aim_y = self._unit_vector(self_state.position, enemy.position)
        move_x, move_y = 0.0, 0.0

        if dist_enemy > self.strafe_max_range:
            move_x, move_y = aim_x, aim_y
        elif dist_enemy < self.strafe_min_range:
            move_x, move_y = self._unit_vector(enemy.position, self_state.position)
        else:
            strafe_sign = 1.0 if ((observation.tick_id + len(self.player_id)) % 2 == 0) else -1.0
            move_x, move_y = self._normalize(-aim_y * strafe_sign, aim_x * strafe_sign)

        action = Action(move_x=move_x, move_y=move_y, aim_x=aim_x, aim_y=aim_y, fire=False, interact=False)

        if self._ability_ready(observation, slot=1) and dist_enemy <= self.fire_range:
            action.ability_id = 1
            self._register_ability_use(observation.tick_id, slot=1)
        elif self._can_fire(self_state) and dist_enemy <= self.fire_range:
            action.fire = True
        elif self._ability_ready(observation, slot=2) and dist_enemy > self.fire_range:
            action.ability_id = 2
            self._register_ability_use(observation.tick_id, slot=2)

        return action

    def _loot_action(self, self_state: SelfState, item: Item) -> Action:
        move_x, move_y = self._unit_vector(self_state.position, item.position)
        action = Action(move_x=move_x, move_y=move_y, aim_x=move_x, aim_y=move_y, fire=False, interact=False)
        if self._distance(self_state.position, item.position) <= 1.8:
            action.interact = True
        return action

    def _patrol_action(self, self_state: SelfState, observation: Observation) -> Action:
        center = observation.zone_state.position
        orbit_r = max(2.0, observation.zone_state.radius * 0.2)
        phase = (observation.tick_id % (TICK_RATE * 12)) / float(TICK_RATE * 12)
        angle = (phase * 2.0 * math.pi) + (len(self.player_id) * 0.17)
        target = [center[0] + math.cos(angle) * orbit_r, center[1] + math.sin(angle) * orbit_r]
        move_x, move_y = self._unit_vector(self_state.position, target)
        return Action(move_x=move_x, move_y=move_y, aim_x=move_x, aim_y=move_y, fire=False, interact=False)

    def _ability_ready(self, observation: Observation, slot: int) -> bool:
        tick_id = int(observation.tick_id)
        if tick_id < self._fallback_ready_tick.get(slot, 0):
            return False

        cooldowns = observation.self_state.cooldowns or {}
        norm_cooldowns: Dict[str, float] = {}
        for raw_k, raw_v in cooldowns.items():
            norm_key = self._normalize_key(str(raw_k))
            if isinstance(raw_v, (float, int)):
                norm_cooldowns[norm_key] = float(raw_v)

        aliases = self._ABILITY_ALIASES.get(slot, ())
        for alias in aliases:
            norm_alias = self._normalize_key(alias)
            if norm_alias in norm_cooldowns:
                if norm_cooldowns[norm_alias] > 0.0:
                    return False

        return True

    def _register_ability_use(self, tick_id: int, slot: int) -> None:
        cooldown_sec = float(self._fallback_cooldown_sec.get(slot, 2.0))
        cooldown_ticks = max(1, int(round(cooldown_sec * TICK_RATE)))
        self._fallback_ready_tick[slot] = int(tick_id) + cooldown_ticks

    @staticmethod
    def _can_fire(self_state: SelfState) -> bool:
        if int(self_state.ammo) <= 0:
            return False
        fire_cd = float((self_state.cooldowns or {}).get("fire", 0.0) or 0.0)
        return fire_cd <= 0.0

    def _closest_enemy(self, observation: Observation) -> Optional[Entity]:
        enemies = [e for e in observation.visible_entities if e.type == "enemy"]
        if not enemies:
            return None
        enemies.sort(key=lambda e: self._distance(observation.self_state.position, e.position))
        return enemies[0]

    def _best_item(self, observation: Observation) -> Optional[Item]:
        if not observation.visible_items:
            return None

        def score(item: Item) -> Tuple[float, float]:
            t = (item.type or "").lower()
            category_score = 1.0
            if any(tok in t for tok in ("ability", "spell", "rune", "orb", "scroll")):
                category_score = 4.0
            elif "chest" in t:
                category_score = 3.0
            elif any(tok in t for tok in ("gold", "coin", "mana")):
                category_score = 2.5
            elif any(tok in t for tok in ("ammo", "health", "potion")):
                category_score = 2.0
            distance = self._distance(observation.self_state.position, item.position)
            return (category_score, -distance)

        return max(observation.visible_items, key=score)

    @staticmethod
    def _normalize_key(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    @staticmethod
    def _distance(p1: List[float], p2: List[float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _unit_vector(self, src: List[float], dst: List[float]) -> Tuple[float, float]:
        return self._normalize(dst[0] - src[0], dst[1] - src[1])

    @staticmethod
    def _normalize(x: float, y: float) -> Tuple[float, float]:
        mag = math.sqrt((x * x) + (y * y))
        if mag <= 1e-9:
            return 0.0, 0.0
        return x / mag, y / mag
