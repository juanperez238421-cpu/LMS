import json
import gzip
from typing import Dict, Any, List
import os
import time

from botgame.common.types import Observation, Action
from botgame.server.world import WorldState, Player

class EpisodeRecorder:
    """
    Records per-tick (obs, action, reward, next_obs, done) for each bot
    and stores them in data/processed as jsonl.gz.
    """
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.episode_data: Dict[str, List[Dict[str, Any]]] = {} # player_id -> list of transitions
        self.initial_player_hp: Dict[str, float] = {}
        self.episode_start_time: float = 0.0

    def start_episode(self, world_state: WorldState) -> None:
        """Initializes recording for a new episode."""
        self.episode_data = {player_id: [] for player_id in world_state.players.keys() if world_state.players[player_id].is_bot}
        self.initial_player_hp = {player_id: player.hp for player_id, player in world_state.players.items()}
        self.episode_start_time = time.time()

    def record_step(self,
                    player_id: str,
                    observation: Observation,
                    action: Action,
                    world_state_before: WorldState, # For reward calculation
                    world_state_after: WorldState,  # For reward calculation
                    is_done: bool) -> None:
        """Records a single step for a given bot."""
        if player_id not in self.episode_data:
            return # Not a bot we are tracking

        reward = self._calculate_reward(player_id, world_state_before, world_state_after, is_done)

        transition = {
            "tick_id": observation.tick_id,
            "observation": observation.__dict__, # Convert dataclass to dict
            "action": action.__dict__,           # Convert dataclass to dict
            "reward": reward,
            "done": is_done,
        }
        # Note: next_observation is implicitly the observation of the next step
        # For simplicity, we'll assume the training pipeline reconstructs next_obs

        self.episode_data[player_id].append(transition)

    def _calculate_reward(self,
                          player_id: str,
                          world_state_before: WorldState,
                          world_state_after: WorldState,
                          is_done: bool) -> float:
        """
        Calculates reward based on survival, damage dealt, damage taken, and objectives.
        """
        reward = 0.0

        player_before = world_state_before.players.get(player_id)
        player_after = world_state_after.players.get(player_id)

        if not player_before or not player_after:
            return -100.0 if is_done else 0.0 # Significant penalty for dying

        # Survival reward (small positive for each tick survived)
        reward += 0.01

        # HP change
        hp_diff = player_after.hp - player_before.hp
        reward += hp_diff * 0.1 # Small reward for gaining HP, penalty for losing HP

        # Ammo change (reward for picking up ammo)
        ammo_diff = player_after.ammo - player_before.ammo
        if ammo_diff > 0:
            reward += ammo_diff * 0.05

        # Zone interaction (reward for being in safe zone, penalty for being outside)
        if world_state_after.zone_state.is_safe:
            reward += 0.02
        else:
            reward -= 0.05

        # Placeholder for damage dealt - this would require tracking damage events from world.py
        # For now, we don't have explicit damage dealt events, so it's omitted.
        # If we had such events, we would add: reward += damage_dealt * X

        if is_done and player_after.hp <= 0:
            reward -= 5.0 # Additional penalty for dying

        return reward

    def end_episode(self, episode_id: str = None) -> None:
        """
        Saves the recorded episode data for all bots to gzipped JSON Lines files.
        """
        if episode_id is None:
            episode_id = f"episode_{int(self.episode_start_time)}_{WorldState.TICK_RATE}" # Should use world's tick rate

        for player_id, transitions in self.episode_data.items():
            filename = os.path.join(self.output_dir, f"{episode_id}_{player_id}.jsonl.gz")
            with gzip.open(filename, 'wt', encoding='utf-8') as f:
                for transition in transitions:
                    # Convert dataclass objects within observation/action back to dicts if they weren't already
                    # This is handled by the __dict__ conversion above, but defensive coding for nested objects
                    transition_copy = json.loads(json.dumps(transition, default=lambda o: o.__dict__))
                    f.write(json.dumps(transition_copy) + "\n")
        self.episode_data = {} # Clear data after saving
        self.initial_player_hp = {}
        self.episode_start_time = 0.0
