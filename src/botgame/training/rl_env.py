import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from botgame.server.world import (
    WorldState,
    Player,
    ZoneState,
    TICK_RATE,
    PLAYER_MAX_HP,
    PLAYER_MAX_AMMO,
    PLAYER_VISION_RADIUS,
)
from botgame.common.types import Observation, Action, SelfState, Entity, Item
from botgame.common.rng import PRNG

class BotGameEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": TICK_RATE}

    def __init__(self, bot_id: str, seed: Optional[int] = None):
        super().__init__()
        self.bot_id = bot_id
        self._seed = seed
        self.world: Optional[WorldState] = None
        self.current_player: Optional[Player] = None

        # Define Observation Space - needs to match the feature vector in BotDataset
        # Order: self_pos(2), self_vel(2), self_hp(1), self_ammo(1),
        #        zone_pos(2), zone_radius(1), zone_is_safe(1),
        #        closest_enemy_pos(2), closest_enemy_hp(1),
        #        closest_item_pos(2)
        # Total: 2+2+1+1 + 2+1+1 + 2+1 + 2 = 15 features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)

        # Define Action Space
        # move_x, move_y, aim_x, aim_y (continuous Box from -1 to 1)
        # fire (Discrete 0 or 1)
        # ability_id (Discrete N+1 where N is number of abilities, 0 for None) - for simplicity, let's make it 0 or 1 for now (no ability/use ability)
        # interact (Discrete 0 or 1)
        self.action_space = spaces.Dict({
            "move_x": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "move_y": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "aim_x": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "aim_y": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "fire": spaces.Discrete(2), # 0: no fire, 1: fire
            "ability": spaces.Discrete(2), # 0: no ability, 1: use ability 0
            "interact": spaces.Discrete(2) # 0: no interact, 1: interact
        })


    def _get_obs(self) -> np.ndarray:
        # This logic should mirror the feature extraction in BotDataset
        observation = self.world.build_observation(self.bot_id)
        self_state = observation.self_state
        zone_state = observation.zone_state

        obs_features = [
            self_state.position[0], self_state.position[1],
            self_state.velocity[0], self_state.velocity[1],
            self_state.hp,
            self_state.ammo,
            zone_state.position[0], zone_state.position[1],
            zone_state.radius,
            1.0 if zone_state.is_safe else 0.0,
        ]

        closest_enemy_pos = [0.0, 0.0]
        closest_enemy_hp = 0.0
        enemies = [e for e in observation.visible_entities if e.type == "enemy"]
        if enemies:
            # For simplicity, take the first one or find the actual closest later
            # (assuming world.build_observation returns entities sorted by distance)
            enemy_entity = enemies[0]
            closest_enemy_pos = enemy_entity.position
            closest_enemy_hp = enemy_entity.hp if enemy_entity.hp is not None else 0.0
        obs_features.extend(closest_enemy_pos)
        obs_features.append(closest_enemy_hp)

        closest_item_pos = [0.0, 0.0]
        if observation.visible_items:
            # For simplicity, take the first one
            item_entity = observation.visible_items[0]
            closest_item_pos = item_entity.position
        obs_features.extend(closest_item_pos)

        return np.array(obs_features, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "player_hp": self.current_player.hp if self.current_player else 0,
            "player_pos": self.current_player.position if self.current_player else [0,0],
            "world_time": self.world.time if self.world else 0
        }

    @staticmethod
    def _as_scalar(value: Any) -> Any:
        return value.item() if hasattr(value, "item") else value

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed

        # Initialize a new WorldState for the episode
        rng = PRNG(self._seed if self._seed is not None else np.random.randint(0, 100000))
        # Create a single bot and a dummy opponent for interaction
        bot_player = Player(id=self.bot_id, is_bot=True, position=[0.0, 0.0], velocity=[0.0, 0.0], hp=PLAYER_MAX_HP, ammo=PLAYER_MAX_AMMO, cooldowns={})
        opponent_player = Player(id="opponent_1", is_bot=False, position=[10.0, 10.0], velocity=[0.0, 0.0], hp=PLAYER_MAX_HP, ammo=PLAYER_MAX_AMMO, cooldowns={})
        
        initial_zone_state = ZoneState(position=[0.0, 0.0], radius=50.0, is_safe=True)
        self.world = WorldState(players={self.bot_id: bot_player, opponent_player.id: opponent_player},
                                projectiles={},
                                items={},
                                zone_state=initial_zone_state,
                                _rng=rng)
        self.current_player = self.world.players[self.bot_id]

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action_dict: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.world or not self.current_player:
            raise RuntimeError("Environment not reset. Call reset() first.")

        if self.current_player.hp <= 1.0:
            self.current_player.hp = 0.0
            self.world.players.pop(self.bot_id, None)
            self.current_player = None
            terminated = True
            truncated = False
            reward = -10.0
            observation = np.zeros_like(self.observation_space.sample())
            info = self._get_info()
            return observation, reward, terminated, truncated, info

        # Convert gymnasium action_dict to botgame.common.types.Action
        fire_flag = bool(self._as_scalar(action_dict["fire"]))
        ability_raw = int(self._as_scalar(action_dict["ability"]))
        interact_flag = bool(self._as_scalar(action_dict["interact"]))
        bot_action = Action(
            move_x=float(self._as_scalar(action_dict["move_x"])),
            move_y=float(self._as_scalar(action_dict["move_y"])),
            aim_x=float(self._as_scalar(action_dict["aim_x"])),
            aim_y=float(self._as_scalar(action_dict["aim_y"])),
            fire=fire_flag,
            ability_id=ability_raw if ability_raw == 1 else None,
            interact=interact_flag
        )

        # Apply the bot's action
        self.world.apply_action(self.bot_id, bot_action)

        # Let the opponent take a dummy action for now (e.g., stand still)
        # In a more complex RL setup, the opponent might also be controlled by a policy
        opponent = self.world.players.get("opponent_1")
        if opponent:
            self.world.apply_action("opponent_1", Action(0,0,0,0,False, None, False)) # Stand still

        # Advance the world state
        dt = 1.0 / TICK_RATE
        world_state_before_tick = self.world # Store for reward calculation if needed

        self.world.tick(dt)

        # Re-fetch the player after tick, as it might have been removed if dead
        self.current_player = self.world.players.get(self.bot_id)

        # Check termination condition
        terminated = False
        if not self.current_player or self.current_player.hp <= 0:
            terminated = True # Bot died
            reward = -10.0 # Penalty for dying
        else:
            # Simple placeholder reward: encourage staying alive and dealing damage
            # More sophisticated reward shaping would be needed here
            reward = 0.1 # Small survival reward
            # Example: distance to opponent as reward (encourage getting closer to fight)
            opponent = self.world.players.get("opponent_1")
            if opponent:
                dist_to_opponent = np.linalg.norm(np.array(self.current_player.position) - np.array(opponent.position))
                reward += (PLAYER_VISION_RADIUS - dist_to_opponent) * 0.01 # Reward for being closer

        truncated = False # For now, no truncation based on episode length

        observation = self._get_obs() if not terminated else np.zeros_like(self.observation_space.sample()) # Return dummy obs if terminated
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        # In a CLI environment, "rendering" could mean printing current world state
        # or logging key events. For a visual game, this would display graphics.
        if self.world and self.current_player:
            print(f"Tick: {self.world.tick_id}, Time: {self.world.time:.2f}s, "
                  f"Bot Pos: {self.current_player.position}, HP: {self.current_player.hp:.1f}, Ammo: {self.current_player.ammo}")
            for entity in self.world.build_observation(self.bot_id).visible_entities:
                print(f"  Visible {entity.type}: {entity.id} at {entity.position}, HP: {entity.hp}")
        else:
            print("Environment not initialized or bot is dead.")

    def close(self) -> None:
        self.world = None
        self.current_player = None

# For testing the environment
if __name__ == "__main__":
    env = BotGameEnv(bot_id="test_bot")
    obs, info = env.reset()
    print("Initial Observation:", obs)
    print("Initial Info:", info)

    for _ in range(100):
        # Sample random actions
        action = {
            "move_x": env.action_space["move_x"].sample(),
            "move_y": env.action_space["move_y"].sample(),
            "aim_x": env.action_space["aim_x"].sample(),
            "aim_y": env.action_space["aim_y"].sample(),
            "fire": env.action_space["fire"].sample(),
            "ability": env.action_space["ability"].sample(),
            "interact": env.action_space["interact"].sample(),
        }
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print("Episode finished.")
            obs, info = env.reset()
    env.close()
