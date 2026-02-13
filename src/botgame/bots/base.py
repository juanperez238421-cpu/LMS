from abc import ABC, abstractmethod
from typing import Any

from botgame.common.types import Observation, Action

class BotPolicy(ABC):
    @abstractmethod
    def reset(self, seed: int) -> None:
        """
        Resets the bot's internal state for a new episode.
        Args:
            seed: A seed for any internal randomness.
        """
        pass

    @abstractmethod
    def act(self, observation: Observation) -> Action:
        """
        Takes an observation and returns an action.
        Args:
            observation: The current observation of the game world.
        Returns:
            An Action object.
        """
        pass
