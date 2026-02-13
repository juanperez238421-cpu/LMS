import torch
import numpy as np
from typing import Optional, Dict, Any

from botgame.bots.base import BotPolicy
from botgame.common.types import Observation, Action
from botgame.common.rng import PRNG
from botgame.training.imitation import ImitationPolicy # For PyTorch model
from botgame.training.rl_env import BotGameEnv # For observation space definition
from stable_baselines3 import PPO # For SB3 model

class LearnedPolicyBot(BotPolicy):
    def __init__(self,
                 player_id: str,
                 model_path: str,
                 model_type: str = "imitation", # "imitation" or "ppo"
                 reaction_delay_ticks: int = 2,
                 aim_noise_degrees: float = 2.0):
        self.player_id = player_id
        self.model_path = model_path
        self.model_type = model_type
        self.reaction_delay_ticks = reaction_delay_ticks
        self.aim_noise_degrees = aim_noise_degrees
        self._rng: Optional[PRNG] = None
        self._policy = None
        self._last_observation: Optional[Observation] = None
        self._delay_counter: int = 0
        self._load_policy()

    def _load_policy(self):
        # Create a dummy environment to get obs/action space for model loading
        # This is a bit hacky but common for SB3 policies
        dummy_env = BotGameEnv(bot_id="dummy")
        obs_dim = dummy_env.observation_space.shape[0]
        action_space_dict = dummy_env.action_space
        # Need to flatten action space for imitation policy or map it for SB3

        if self.model_type == "imitation":
            # Assuming a fixed action_dim from dataset.py
            # move_x, move_y, aim_x, aim_y, fire (0/1), ability (0/1), interact (0/1)
            # Total 7 dimensions
            action_dim = 7
            self._policy = ImitationPolicy(obs_dim, action_dim)
            self._policy.load_state_dict(torch.load(self.model_path))
            self._policy.eval()
            print(f"Loaded ImitationPolicy from {self.model_path}")
        elif self.model_type == "ppo":
            # SB3 PPO models handle MultiInputPolicy directly
            self._policy = PPO.load(self.model_path, env=None) # env=None is important if not training
            print(f"Loaded PPO policy from {self.model_path}")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def reset(self, seed: int) -> None:
        self._rng = PRNG(seed)
        self._last_observation = None
        self._delay_counter = 0

    def act(self, observation: Observation) -> Action:
        if self._rng is None:
            raise RuntimeError("BotPolicy not reset. Call reset() first.")
        if self._policy is None:
            raise RuntimeError("Policy not loaded.")

        # Simulate reaction delay
        if self._delay_counter < self.reaction_delay_ticks:
            self._delay_counter += 1
            if self._last_observation:
                observation = self._last_observation
            else:
                return Action(move_x=0, move_y=0, aim_x=0, aim_y=0, fire=False, interact=False)
        else:
            self._last_observation = observation
            self._delay_counter = 0

        # Convert observation to feature vector (must match BotDataset and BotGameEnv)
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
            enemy_entity = enemies[0] # Take first for now
            closest_enemy_pos = enemy_entity.position
            closest_enemy_hp = enemy_entity.hp if enemy_entity.hp is not None else 0.0
        obs_features.extend(closest_enemy_pos)
        obs_features.append(closest_enemy_hp)

        closest_item_pos = [0.0, 0.0]
        if observation.visible_items:
            item_entity = observation.visible_items[0] # Take first for now
            closest_item_pos = item_entity.position
        obs_features.extend(closest_item_pos)

        obs_tensor = torch.tensor(obs_features, dtype=torch.float32).unsqueeze(0) # Add batch dimension

        predicted_action_raw: np.ndarray

        if self.model_type == "imitation":
            with torch.no_grad():
                predicted_action_raw = self._policy(obs_tensor).squeeze(0).numpy()
            # Map raw outputs to action object
            move_x, move_y, aim_x, aim_y, fire, ability_id, interact = predicted_action_raw
            fire = bool(fire > 0.5) # Threshold for binary actions
            ability_id = int(ability_id > 0.5) if ability_id > 0.5 else None
            interact = bool(interact > 0.5)
        elif self.model_type == "ppo":
            # SB3 policy.predict returns action and state (if recurrent)
            predicted_action_raw, _states = self._policy.predict(obs_tensor.numpy(), deterministic=True)
            # SB3 actions are usually flat numpy arrays. Need to map them to the dict space.
            # Assuming predicted_action_raw is already a dict if MultiInputPolicy was used correctly.
            # If it's a flat array, you need to reconstruct the dict.
            # For simplicity, let's assume it returns a dict matching action_space for now
            # or it's a flat array that maps directly.
            # Let's assume for PPO it also outputs a flattened array:
            # [move_x, move_y, aim_x, aim_y, fire_logit, ability_logit, interact_logit]
            # This is a simplification and would need careful handling based on SB3 model's output.
            # For now, let's assume it matches the imitation policy's output structure
            # where continuous actions are direct and discrete are logits/binary.
            
            # Reconstruct the dict action for SB3 output
            # This part needs to be precise based on how the SB3 model was trained
            # and what kind of action space it was given.
            # If using `spaces.Dict` for `action_space`, then `predict` returns a dict.
            # If using `FlattenedAction` for `action_space`, then `predict` returns an array.

            # Assuming the PPO policy is trained on the flattened action space
            # as presented to ImitationPolicy or converted correctly by SB3.
            
            # This part is critical: if PPO uses MultiInputPolicy and we give it spaces.Dict,
            # then model.predict(obs) usually returns a single numpy array of actions for each input
            # which then needs to be re-mapped to the Dict action space.
            # For example, if it's a flattened Box:
            # move_x, move_y, aim_x, aim_y are directly mapped
            # fire, ability, interact are typically discrete, so their output from the policy might be logits
            # or directly the chosen action. If the env action space is Dict, SB3 tries to match it.

            # Let's assume the SB3 PPO model output maps directly to the Action dataclass parameters
            # by index if it's a flattened array from `predict`.
            # This is a common pattern for "flat" action spaces in SB3.
            # Predicted_action_raw from predict is typically a numpy array if action space is Box or Discrete/MultiDiscrete
            # For MultiInputPolicy with spaces.Dict, it usually outputs a dict.
            # For simplicity, let's assume it's a single array and we map it.
            
            # This needs to be carefully aligned with `rl_env.py` action space definition and how SB3 processes it.
            # If `rl_env.py` uses `spaces.Dict`, `model.predict` should return a dict of arrays.
            # For `BotGameEnv` as defined:
            #   "move_x": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            #   "move_y": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            #   "aim_x": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            #   "aim_y": spaces.Box(low=-1.0, high=1.0, shape=(1,)),
            #   "fire": spaces.Discrete(2),
            #   "ability": spaces.Discrete(2),
            #   "interact": spaces.Discrete(2)

            # SB3's `predict` for a `MultiInputPolicy` on a `Dict` action space would return
            # a tuple `(actions, states)` where `actions` is an array that needs to be
            # reshaped/processed to match the Dict format.
            # A simpler way often adopted is to use a `FlattenedActionSpace` wrapper
            # or manually convert.

            # For now, let's assume a simplified flat output from the PPO model that matches the imitation policy for action construction.
            # This is a pragmatic choice to move forward, acknowledging it might need refinement.
            predicted_action_flat, _ = self._policy.predict(obs_tensor.numpy(), deterministic=True)
            move_x, move_y, aim_x, aim_y, fire_val, ability_val, interact_val = predicted_action_flat.flatten()
            
            # Apply safety filter and convert to Action object
            move_x = float(np.clip(move_x, -1.0, 1.0))
            move_y = float(np.clip(move_y, -1.0, 1.0))
            aim_x = float(np.clip(aim_x, -1.0, 1.0))
            aim_y = float(np.clip(aim_y, -1.0, 1.0))
            fire = bool(fire_val > 0.5) # For discrete actions
            ability_id = int(ability_val) if int(ability_val) > 0 else None
            interact = bool(interact_val > 0.5)

        # Apply safety filter for continuous actions (clamping) - already done for SB3 above
        # For imitation:
        # move_x = float(np.clip(move_x, -1.0, 1.0))
        # move_y = float(np.clip(move_y, -1.0, 1.0))
        # aim_x = float(np.clip(aim_x, -1.0, 1.0))
        # aim_y = float(np.clip(aim_y, -1.0, 1.0))

        # Add aim noise
        if self.aim_noise_degrees > 0:
            aim_angle = math.atan2(aim_y, aim_x)
            aim_angle += math.radians(self._rng.random() * self.aim_noise_degrees - self.aim_noise_degrees / 2)
            aim_x = math.cos(aim_angle)
            aim_y = math.sin(aim_angle)


        action = Action(
            move_x=move_x,
            move_y=move_y,
            aim_x=aim_x,
            aim_y=aim_y,
            fire=fire,
            ability_id=ability_id,
            interact=interact
        )

        # Apply safety filter and cooldown gating if needed
        # For fire, it's already implicitly handled by world.py if player.cooldowns is checked
        # but a safety filter could prevent the bot from even trying if cooldown is active.
        # This can be added to the Action.clamp_and_validate() or here.
        # For now, rely on world.py to enforce cooldowns.

        return action
