from typing import Dict, Optional

from botgame.common.types import Action, Observation
from botgame.server.world import WorldState, Player
from botgame.bots.base import BotPolicy

class BotManager:
    """Manages bot instances and integrates them into the world's tick loop."""

    def __init__(self, world: WorldState):
        self.world = world
        self.bot_policies: Dict[str, BotPolicy] = {} # player_id -> BotPolicy instance

    def add_bot(self, player_id: str, bot_policy: BotPolicy, seed: int) -> None:
        """Adds a bot policy for a given player ID."""
        if player_id not in self.world.players or not self.world.players[player_id].is_bot:
            raise ValueError(f"Player {player_id} is not a bot or does not exist in the world.")
        self.bot_policies[player_id] = bot_policy
        bot_policy.reset(seed)

    def tick_bots(self) -> None:
        """
        Builds observations for all managed bots, asks for actions, and applies them.
        """
        for player_id, bot_policy in self.bot_policies.items():
            if player_id in self.world.players: # Ensure bot is still alive
                observation = self.world.build_observation(player_id)
                action = bot_policy.act(observation)
                self.world.apply_action(player_id, action)

