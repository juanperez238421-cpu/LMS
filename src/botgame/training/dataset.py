import json
import gzip
import os
import glob
from typing import List, Dict, Any, Tuple

import torch
import numpy as np

from botgame.common.types import Observation, Action, SelfState, Entity, Item, ZoneState

class BotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, sequence_length: int = 1):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.transitions: List[Dict[str, Any]] = []
        self._load_data()

    def _load_data(self):
        jsonl_files = glob.glob(os.path.join(self.data_dir, "*.jsonl.gz"))
        for file_path in jsonl_files:
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    transition = json.loads(line)
                    # Convert dicts back to dataclasses if needed, or process directly
                    # For imitation learning, we primarily need observation and action
                    self.transitions.append(transition)

        print(f"Loaded {len(self.transitions)} transitions from {len(jsonl_files)} files.")

    def __len__(self):
        return len(self.transitions) - self.sequence_length + 1 if self.sequence_length > 1 else len(self.transitions)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # For simplicity, let's assume sequence_length = 1 for now for basic imitation
        # and focus on converting observation and action to tensors.

        transition = self.transitions[idx]
        observation_dict = transition["observation"]
        action_dict = transition["action"]

        # Convert observation_dict to a flat tensor (example, needs proper feature extraction)
        # This is a very simplified example. A real implementation would need careful flattening
        # of visible entities/items, padding, etc.
        # For now, let's extract some key numerical features.

        self_state = SelfState(**observation_dict['self_state'])
        zone_state = ZoneState(**observation_dict['zone_state'])

        obs_features = [
            self_state.position[0], self_state.position[1],
            self_state.velocity[0], self_state.velocity[1],
            self_state.hp,
            self_state.ammo,
            zone_state.position[0], zone_state.position[1],
            zone_state.radius,
            1.0 if zone_state.is_safe else 0.0,
        ]

        # Add features for closest enemy and item if available
        closest_enemy_pos = [0.0, 0.0]
        closest_enemy_hp = 0.0
        if observation_dict['visible_entities']:
            # Find closest enemy (simplified: just take the first one for now)
            enemy_entity = Entity(**observation_dict['visible_entities'][0])
            closest_enemy_pos = enemy_entity.position
            closest_enemy_hp = enemy_entity.hp if enemy_entity.hp is not None else 0.0
        obs_features.extend([closest_enemy_pos[0], closest_enemy_pos[1], closest_enemy_hp])

        closest_item_pos = [0.0, 0.0]
        if observation_dict['visible_items']:
            # Find closest item
            item_entity = Item(**observation_dict['visible_items'][0])
            closest_item_pos = item_entity.position
        obs_features.extend([closest_item_pos[0], closest_item_pos[1]])

        # Convert action_dict to a tensor
        action_target = [
            action_dict['move_x'], action_dict['move_y'],
            action_dict['aim_x'], action_dict['aim_y'],
            1.0 if action_dict['fire'] else 0.0,
            float(action_dict['ability_id'] if action_dict['ability_id'] is not None else -1), # -1 for no ability
            1.0 if action_dict['interact'] else 0.0,
        ]

        return torch.tensor(obs_features, dtype=torch.float32), torch.tensor(action_target, dtype=torch.float32)

if __name__ == "__main__":
    # Example usage:
    # This will only work if you have some jsonl.gz files in data/processed
    # To generate data, you need to run a match with bots and the recorder enabled.
    dataset = BotDataset("data/processed")
    if len(dataset) > 0:
        obs, act = dataset[0]
        print(f"Sample Observation Features: {obs.shape}")
        print(f"Sample Action Target: {act.shape}")
    else:
        print("No data found in data/processed. Please generate some data first.")

