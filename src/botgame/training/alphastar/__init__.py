"""AlphaStar-inspired training components for botgame."""

from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.league import LeagueManager, PFSPConfig
from botgame.training.alphastar.losses import td_lambda_targets, upgo_returns, vtrace
from botgame.training.alphastar.model import FactorizedPolicyValueNet
from botgame.training.alphastar.replay import SequenceReplayBuffer
from botgame.training.alphastar.types import TrajectorySequence

__all__ = [
    "FactorizedActionCodec",
    "FactorizedPolicyValueNet",
    "LeagueManager",
    "PFSPConfig",
    "SequenceReplayBuffer",
    "TrajectorySequence",
    "td_lambda_targets",
    "upgo_returns",
    "vtrace",
]

