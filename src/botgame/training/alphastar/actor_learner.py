from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from botgame.training.alphastar.action_codec import FactorizedActionCodec
from botgame.training.alphastar.losses import td_lambda_targets, upgo_returns, vtrace
from botgame.training.alphastar.model import FactorizedPolicyValueNet
from botgame.training.alphastar.types import TrajectorySequence
from botgame.training.rl_env import BotGameEnv


@dataclass
class RLConfig:
    unroll_length: int = 64
    gamma: float = 0.99
    td_lambda: float = 0.8
    clip_rho: float = 1.0
    clip_pg_rho: float = 1.0
    clip_c: float = 1.0
    entropy_coef: float = 0.002
    value_coef: float = 0.5
    kl_coef: float = 0.05
    upgo_coef: float = 0.2
    pseudo_reward_scale: float = 0.0
    terminal_win_reward: float = 1.0
    terminal_loss_reward: float = -1.0


def _adapt_obs_dim(obs: np.ndarray, expected_dim: int) -> np.ndarray:
    if obs.shape[0] == expected_dim:
        return obs.astype(np.float32, copy=False)
    if obs.shape[0] > expected_dim:
        return obs[:expected_dim].astype(np.float32, copy=False)
    pad = np.zeros((expected_dim - obs.shape[0],), dtype=np.float32)
    return np.concatenate([obs.astype(np.float32, copy=False), pad], axis=0)


def _sample_actions_from_logits(
    logits: Dict[str, torch.Tensor],
    codec: FactorizedActionCodec,
) -> tuple[Dict[str, int], float]:
    action_dict: Dict[str, int] = {}
    logp_sum = 0.0
    for name in codec.head_sizes:
        dist = torch.distributions.Categorical(logits=logits[name].squeeze(0))
        sampled = int(dist.sample().item())
        action_dict[name] = sampled
        logp_sum += float(dist.log_prob(torch.tensor(sampled, device=dist.logits.device)).item())
    return action_dict, logp_sum


def collect_actor_sequence(
    env: BotGameEnv,
    model: FactorizedPolicyValueNet,
    codec: FactorizedActionCodec,
    config: RLConfig,
    device: torch.device,
    seed: int | None = None,
) -> TrajectorySequence:
    """Collects one off-policy trajectory using the current actor parameters."""
    obs, _info = env.reset(seed=seed)
    obs = _adapt_obs_dim(np.asarray(obs, dtype=np.float32), model.obs_dim)
    obs_list = []
    rewards = []
    dones = []
    behavior_logp = []
    actions: Dict[str, list[int]] = {k: [] for k in codec.head_sizes}
    alive = True

    for _ in range(config.unroll_length):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_value = model(obs_tensor)
        encoded_action, logp = _sample_actions_from_logits(policy_value.logits, codec)
        env_action = codec.action_to_env(encoded_action)
        next_obs, env_reward, terminated, truncated, _ = env.step(env_action)
        next_obs = _adapt_obs_dim(np.asarray(next_obs, dtype=np.float32), model.obs_dim)
        done = bool(terminated or truncated)

        reward = float(config.pseudo_reward_scale * env_reward)
        if done:
            alive = not bool(terminated)
            reward += config.terminal_win_reward if alive else config.terminal_loss_reward

        obs_list.append(np.asarray(obs, dtype=np.float32))
        rewards.append(reward)
        dones.append(float(done))
        behavior_logp.append(float(logp))
        for key in actions:
            actions[key].append(int(encoded_action[key]))

        obs = next_obs
        if done:
            break

    if len(obs_list) < 2:
        obs_list.append(np.asarray(obs, dtype=np.float32))
        rewards.append(0.0)
        dones.append(1.0)
        behavior_logp.append(0.0)
        for key in actions:
            actions[key].append(actions[key][-1] if actions[key] else 0)

    return TrajectorySequence(
        obs=np.asarray(obs_list, dtype=np.float32),
        actions={k: np.asarray(v, dtype=np.int64) for k, v in actions.items()},
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.float32),
        behavior_logp=np.asarray(behavior_logp, dtype=np.float32),
        extras={"source": "actor_rollout", "alive": float(alive)},
    )


def learner_step(
    model: FactorizedPolicyValueNet,
    supervised_model: FactorizedPolicyValueNet,
    codec: FactorizedActionCodec,
    optimizer: torch.optim.Optimizer,
    batch_obs: torch.Tensor,
    batch_next_obs: torch.Tensor,
    batch_rewards: torch.Tensor,
    batch_dones: torch.Tensor,
    batch_behavior_logp: torch.Tensor,
    batch_actions: Dict[str, torch.Tensor],
    config: RLConfig,
) -> Dict[str, float]:
    """Runs one learner update from sampled off-policy sequences."""
    t_steps, batch_size, obs_dim = batch_obs.shape
    flat_obs = batch_obs.reshape(t_steps * batch_size, obs_dim)
    flat_next_obs = batch_next_obs.reshape(t_steps * batch_size, obs_dim)

    policy_value = model(flat_obs)
    with torch.no_grad():
        next_policy_value = model(flat_next_obs)
        sup_policy = supervised_model(flat_obs)

    logits_t = {k: v.reshape(t_steps, batch_size, -1) for k, v in policy_value.logits.items()}
    sup_logits_t = {k: v.reshape(t_steps, batch_size, -1) for k, v in sup_policy.logits.items()}
    values_t = policy_value.values.reshape(t_steps, batch_size)
    next_values_t = next_policy_value.values.reshape(t_steps, batch_size)
    bootstrap_value = next_values_t[-1]

    target_logp, _ = codec.log_prob(logits=logits_t, actions=batch_actions)
    vtrace_returns = vtrace(
        rewards=batch_rewards,
        values=values_t,
        bootstrap_value=bootstrap_value,
        behavior_logp=batch_behavior_logp,
        target_logp=target_logp,
        dones=batch_dones,
        gamma=config.gamma,
        lambda_=config.td_lambda,
        clip_rho=config.clip_rho,
        clip_pg_rho=config.clip_pg_rho,
        clip_c=config.clip_c,
    )

    td_targets = td_lambda_targets(
        rewards=batch_rewards,
        values=values_t,
        bootstrap_value=bootstrap_value,
        dones=batch_dones,
        gamma=config.gamma,
        lambda_=config.td_lambda,
    )

    upgo = upgo_returns(
        rewards=batch_rewards,
        values=values_t,
        bootstrap_value=bootstrap_value,
        dones=batch_dones,
        behavior_logp=batch_behavior_logp,
        target_logp=target_logp,
        gamma=config.gamma,
        clip_rho=config.clip_rho,
    )

    mask = 1.0 - batch_dones
    policy_loss = -((target_logp * vtrace_returns.pg_advantages.detach()) * mask).sum() / mask.sum().clamp(min=1.0)
    upgo_loss = -((target_logp * upgo.advantages.detach()) * mask).sum() / mask.sum().clamp(min=1.0)
    value_loss = F.mse_loss(values_t, td_targets.detach())
    entropy = codec.entropy(logits_t).mean()

    kl_terms = []
    for key in logits_t:
        p = F.log_softmax(logits_t[key], dim=-1)
        q = F.softmax(sup_logits_t[key], dim=-1)
        kl_terms.append(F.kl_div(p, q, reduction="batchmean"))
    kl_loss = torch.stack(kl_terms).mean()

    total_loss = (
        policy_loss
        + config.value_coef * value_loss
        + config.kl_coef * kl_loss
        + config.upgo_coef * upgo_loss
        - config.entropy_coef * entropy
    )

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    return {
        "loss_total": float(total_loss.item()),
        "loss_policy": float(policy_loss.item()),
        "loss_upgo": float(upgo_loss.item()),
        "loss_value": float(value_loss.item()),
        "loss_kl_sup": float(kl_loss.item()),
        "entropy": float(entropy.item()),
        "avg_reward": float(batch_rewards.mean().item()),
    }
