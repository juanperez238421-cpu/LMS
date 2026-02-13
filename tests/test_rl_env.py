import pytest
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

from botgame.training.rl_env import BotGameEnv
from botgame.common.types import Action
from botgame.server.world import PLAYER_MAX_HP, PLAYER_MAX_AMMO

def test_rl_env_creation():
    env = BotGameEnv(bot_id="test_rl_bot")
    assert isinstance(env, gym.Env)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Dict)
    env.close()

def test_rl_env_reset():
    env = BotGameEnv(bot_id="test_rl_bot")
    observation, info = env.reset(seed=42)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert observation[4] == PLAYER_MAX_HP # Initial HP
    assert observation[5] == PLAYER_MAX_AMMO # Initial Ammo

    assert isinstance(info, dict)
    assert "player_hp" in info
    assert "player_pos" in info
    assert "world_time" in info
    env.close()

def test_rl_env_step():
    env = BotGameEnv(bot_id="test_rl_bot")
    env.reset(seed=42)

    # Example action: move forward, aim right, fire, no ability, no interact
    action_dict = {
        "move_x": np.array([1.0], dtype=np.float32),
        "move_y": np.array([0.0], dtype=np.float32),
        "aim_x": np.array([1.0], dtype=np.float32),
        "aim_y": np.array([0.0], dtype=np.float32),
        "fire": 1,
        "ability": 0,
        "interact": 0
    }

    observation, reward, terminated, truncated, info = env.step(action_dict)

    assert isinstance(observation, np.ndarray)
    assert observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    # Check if player moved
    initial_pos = env.current_player.position if env.current_player else [0.0, 0.0]
    assert initial_pos[0] > 0.0 # Should have moved right
    
    # Check if player fired (ammo should decrease if fire was successful)
    # The dummy opponent is too far to hit, but ammo should decrease if fire action is processed
    assert env.current_player.ammo < PLAYER_MAX_AMMO # Ammo should be consumed
    
    env.close()

def test_rl_env_termination_on_death():
    env = BotGameEnv(bot_id="test_rl_bot")
    obs, info = env.reset(seed=42)

    # Drain player HP to simulate death
    env.world.players[env.bot_id].hp = 1.0 # Set HP to a low value
    
    # Perform an action and expect termination
    action_dict = {
        "move_x": np.array([0.0], dtype=np.float32),
        "move_y": np.array([0.0], dtype=np.float32),
        "aim_x": np.array([0.0], dtype=np.float32),
        "aim_y": np.array([0.0], dtype=np.float32),
        "fire": 0,
        "ability": 0,
        "interact": 0
    }
    
    observation, reward, terminated, truncated, info = env.step(action_dict)
    assert terminated
    assert reward < 0 # Expect a negative reward for dying
    env.close()

def test_rl_env_action_space_conversion():
    env = BotGameEnv(bot_id="test_rl_bot")
    env.reset(seed=42)

    action_dict = {
        "move_x": np.array([0.5], dtype=np.float32),
        "move_y": np.array([-0.5], dtype=np.float32),
        "aim_x": np.array([0.8], dtype=np.float32),
        "aim_y": np.array([-0.6], dtype=np.float32),
        "fire": 1,
        "ability": 1,
        "interact": 1
    }

    # Internal check of the conversion logic within step
    # This is more of an integration test for the internal Action conversion
    # We can't directly inspect the converted Action object in `step`,
    # but we can verify its effects indirectly.
    
    # Stub the world.apply_action to capture the Action object
    original_apply_action = env.world.apply_action
    captured_action = None

    def mock_apply_action(player_id, action):
        nonlocal captured_action
        if player_id == env.bot_id:
            captured_action = action
        original_apply_action(player_id, action) # Still call original to progress world state

    env.world.apply_action = mock_apply_action
    
    env.step(action_dict)
    
    assert captured_action is not None
    assert isinstance(captured_action, Action)
    assert captured_action.move_x == pytest.approx(action_dict["move_x"].item())
    assert captured_action.move_y == pytest.approx(action_dict["move_y"].item())
    assert captured_action.aim_x == pytest.approx(action_dict["aim_x"].item())
    assert captured_action.aim_y == pytest.approx(action_dict["aim_y"].item())
    assert captured_action.fire == bool(action_dict["fire"])
    assert captured_action.ability_id == action_dict["ability"]
    assert captured_action.interact == bool(action_dict["interact"])

    env.close()
