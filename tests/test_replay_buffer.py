import numpy as np

from botgame.training.alphastar.replay import SequenceReplayBuffer
from botgame.training.alphastar.types import TrajectorySequence


def _make_sequence(length: int, obs_dim: int = 4, source: str = "test") -> TrajectorySequence:
    obs = np.arange(length * obs_dim, dtype=np.float32).reshape(length, obs_dim)
    actions = {
        "move": np.zeros((length, 2), dtype=np.float32),
        "cast": np.zeros((length, 1), dtype=np.float32),
    }
    rewards = np.linspace(0.0, 1.0, num=length, dtype=np.float32)
    dones = np.zeros((length,), dtype=np.float32)
    dones[-1] = 1.0
    behavior_logp = np.zeros((length,), dtype=np.float32)
    return TrajectorySequence(
        obs=obs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        behavior_logp=behavior_logp,
        extras={"source": source},
    )


def test_replay_buffer_push_and_sample_shapes():
    buffer = SequenceReplayBuffer(max_sequences=10, seed=42)
    buffer.add(_make_sequence(length=8, obs_dim=4, source="ep_a"))
    buffer.add(_make_sequence(length=10, obs_dim=4, source="ep_b"))

    batch = buffer.sample_batch(batch_size=2, unroll_length=5)

    assert batch.obs.shape == (5, 2, 4)
    assert batch.next_obs.shape == (5, 2, 4)
    assert batch.actions["move"].shape == (5, 2, 2)
    assert batch.actions["cast"].shape == (5, 2, 1)
    assert batch.rewards.shape == (5, 2)
    assert batch.dones.shape == (5, 2)
    assert batch.behavior_logp.shape == (5, 2)
    assert batch.extras["source"].shape == (5, 2)


def test_replay_buffer_padding_marks_terminal_steps():
    buffer = SequenceReplayBuffer(max_sequences=10, seed=7)
    buffer.add(_make_sequence(length=3, obs_dim=2, source="short"))

    batch = buffer.sample_batch(batch_size=1, unroll_length=5)

    assert batch.obs.shape == (5, 1, 2)
    assert batch.next_obs.shape == (5, 1, 2)
    # Last two steps come from replay padding and must be terminal.
    assert batch.dones[3, 0] == 1.0
    assert batch.dones[4, 0] == 1.0
    assert np.all(batch.behavior_logp[3:, 0] == 0.0)
