import math
import random
from typing import Optional, List, Dict

from botgame.bots.base import BotPolicy
from botgame.common.types import Observation, Action, Entity, Item, SelfState
from botgame.common.rng import PRNG
from botgame.server.world import PLAYER_VISION_RADIUS, PLAYER_MAX_HP, PLAYER_MOVE_SPEED, PROJECTILE_SPEED, TICK_RATE

class ScriptedUtilityBot(BotPolicy):
    def __init__(self,
                 player_id: str,
                 seed: Optional[int] = None,
                 reaction_delay_ticks: int = 0, # Delay in ticks
                 aim_noise_degrees: float = 5.0,
                 aggressiveness: float = 0.5, # 0.0 (retreats early) to 1.0 (fights to death)
                 loot_bias: float = 0.5): # 0.0 (ignores loot) to 1.0 (prioritizes loot)
        self.player_id = player_id
        self.reaction_delay_ticks = reaction_delay_ticks
        self.aim_noise_degrees = aim_noise_degrees
        self.aggressiveness = aggressiveness
        self.loot_bias = loot_bias
        self._rng: Optional[PRNG] = PRNG(seed if seed is not None else 0)
        self._last_observation: Optional[Observation] = None
        self._delay_counter: int = 0

        if seed is not None:
            self.reset(seed)

    def reset(self, seed: int) -> None:
        self._rng = PRNG(seed)
        self._last_observation = None
        self._delay_counter = 0

    def act(self, observation: Observation) -> Action:
        if self._rng is None:
            raise RuntimeError("BotPolicy not reset. Call reset() first.")

        # Simulate reaction delay
        if self.reaction_delay_ticks > 0:
            if self._delay_counter <= self.reaction_delay_ticks:
                self._delay_counter += 1
                self._last_observation = observation
                return Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)

            observation = self._last_observation if self._last_observation is not None else observation
            self._delay_counter = 0 # Reset for next action decision
        self._last_observation = observation

        action = Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)

        self_state = observation.self_state
        closest_enemy = self._find_closest_enemy(observation)
        closest_item = self._find_closest_item(observation)

        # Retreat logic
        if self_state.hp < PLAYER_MAX_HP * (1.0 - self.aggressiveness) and closest_enemy:
            return self._retreat_action(self_state, closest_enemy)

        # Combat logic
        if closest_enemy:
            action = self._combat_action(self_state, closest_enemy)
        else:
            # Exploration/Looting logic
            if closest_item and self._rng.random() < self.loot_bias:
                action = self._move_towards_target(self_state, closest_item.position)
                if self._distance(self_state.position, closest_item.position) < 2.0: # Close enough to interact
                    action.interact = True
            else:
                # Move towards zone center or a random point
                action = self._move_towards_target(self_state, observation.zone_state.position)

        return action

    def _find_closest_enemy(self, observation: Observation) -> Optional[Entity]:
        for entity in observation.visible_entities:
            if entity.type == "enemy":
                # Observations are expected to be pre-prioritized by the server pipeline.
                return entity
        return None

    def _find_closest_item(self, observation: Observation) -> Optional[Item]:
        closest_item: Optional[Item] = None
        min_dist = float('inf')
        for item in observation.visible_items:
            dist = self._distance(observation.self_state.position, item.position)
            if dist < min_dist:
                min_dist = dist
                closest_item = item
        return closest_item

    def _combat_action(self, self_state: SelfState, enemy: Entity) -> Action:
        action = Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)

        # Aim at enemy with noise
        aim_vec = [enemy.position[0] - self_state.position[0],
                   enemy.position[1] - self_state.position[1]]
        aim_angle = math.atan2(aim_vec[1], aim_vec[0])
        aim_angle += math.radians(self._rng.random() * self.aim_noise_degrees - self.aim_noise_degrees / 2)

        action.aim_x = math.cos(aim_angle)
        action.aim_y = math.sin(aim_angle)

        # Strafe perpendicular to enemy if within a certain range
        dist_to_enemy = self._distance(self_state.position, enemy.position)
        if 5.0 < dist_to_enemy <= PLAYER_VISION_RADIUS: # Within effective combat range
            strafe_angle = aim_angle + math.pi / 2 # Perpendicular
            if self._rng.random() < 0.5: # Randomly choose strafe direction
                strafe_angle -= math.pi
            action.move_x = math.cos(strafe_angle)
            action.move_y = math.sin(strafe_angle)
        elif dist_to_enemy > PLAYER_VISION_RADIUS: # Too far, move closer
            action.move_x = aim_vec[0]
            action.move_y = aim_vec[1]
            mag = math.sqrt(action.move_x**2 + action.move_y**2)
            if mag > 0:
                action.move_x /= mag
                action.move_y /= mag

        # Fire if aim is good and not on cooldown
        if self_state.cooldowns.get("fire", 0) <= 0 and self_state.ammo > 0:
            # Simple check: if aim vector is roughly pointing at enemy
            # (A more advanced check would be line of sight in world.py)
            fire_threshold_angle = math.radians(self.aim_noise_degrees * 2) # Allow some error
            current_aim_angle = math.atan2(action.aim_y, action.aim_x)
            actual_aim_angle = math.atan2(aim_vec[1], aim_vec[0])

            angle_diff = abs(current_aim_angle - actual_aim_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            if angle_diff <= fire_threshold_angle:
                action.fire = True

        return action

    def _retreat_action(self, self_state: SelfState, enemy: Entity) -> Action:
        action = Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)

        # Move directly away from the enemy
        run_vec = [self_state.position[0] - enemy.position[0],
                   self_state.position[1] - enemy.position[1]]
        mag = math.sqrt(run_vec[0]**2 + run_vec[1]**2)
        if mag > 0:
            action.move_x = run_vec[0] / mag
            action.move_y = run_vec[1] / mag

        # While retreating, aim at enemy to potentially get a shot off
        action.aim_x = -action.move_x # Aim back towards the enemy
        action.aim_y = -action.move_y

        # Optionally fire while retreating if possible
        if self_state.cooldowns.get("fire", 0) <= 0 and self_state.ammo > 0 and self._rng.random() < 0.3:
             action.fire = True

        return action

    def _move_towards_target(self, self_state: SelfState, target_pos: List[float]) -> Action:
        move_vec = [target_pos[0] - self_state.position[0],
                    target_pos[1] - self_state.position[1]]
        mag = math.sqrt(move_vec[0]**2 + move_vec[1]**2)
        action = Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)
        if mag > 0:
            action.move_x = move_vec[0] / mag
            action.move_y = move_vec[1] / mag
        return action

    def _distance(self, pos1: List[float], pos2: List[float]) -> float:
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
