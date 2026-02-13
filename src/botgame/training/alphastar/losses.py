from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VTraceReturns:
    vs: torch.Tensor
    pg_advantages: torch.Tensor
    clipped_rho: torch.Tensor


def vtrace(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    behavior_logp: torch.Tensor,
    target_logp: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 1.0,
    clip_rho: float = 1.0,
    clip_pg_rho: float = 1.0,
    clip_c: float = 1.0,
) -> VTraceReturns:
    """Computes V-trace corrected targets for off-policy actor-critic."""
    if values.shape != rewards.shape:
        raise ValueError("values and rewards must have shape [T, B].")
    if behavior_logp.shape != rewards.shape or target_logp.shape != rewards.shape:
        raise ValueError("Log-prob tensors must have shape [T, B].")

    t_steps, batch_size = rewards.shape
    device = rewards.device
    vs = torch.zeros_like(rewards)
    not_done = 1.0 - dones.float()

    rho = torch.exp(target_logp - behavior_logp)
    clipped_rho = torch.clamp(rho, max=clip_rho)
    clipped_pg_rho = torch.clamp(rho, max=clip_pg_rho)
    c_t = torch.clamp(rho, max=clip_c)

    next_vs = bootstrap_value
    for t in reversed(range(t_steps)):
        next_value = bootstrap_value if t == (t_steps - 1) else values[t + 1]
        delta = clipped_rho[t] * (rewards[t] + gamma * not_done[t] * next_value - values[t])
        next_minus_value = next_vs - next_value
        vs[t] = values[t] + delta + gamma * lambda_ * c_t[t] * not_done[t] * next_minus_value
        next_vs = vs[t]

    next_vs_for_pg = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
    pg_adv = clipped_pg_rho * (rewards + gamma * not_done * next_vs_for_pg - values)
    return VTraceReturns(vs=vs, pg_advantages=pg_adv, clipped_rho=clipped_rho.to(device))


def td_lambda_targets(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lambda_: float = 0.8,
) -> torch.Tensor:
    """Computes TD(lambda) targets with bootstrapping on `values`."""
    if values.shape != rewards.shape:
        raise ValueError("values and rewards must have shape [T, B].")

    t_steps = rewards.shape[0]
    targets = torch.zeros_like(rewards)
    not_done = 1.0 - dones.float()
    next_target = bootstrap_value

    for t in reversed(range(t_steps)):
        next_value = bootstrap_value if t == (t_steps - 1) else values[t + 1]
        mixed_bootstrap = (1.0 - lambda_) * next_value + lambda_ * next_target
        targets[t] = rewards[t] + gamma * not_done[t] * mixed_bootstrap
        next_target = targets[t]
    return targets


@dataclass
class UPGOReturns:
    upgo_targets: torch.Tensor
    advantages: torch.Tensor
    clipped_rho: torch.Tensor


def upgo_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_value: torch.Tensor,
    dones: torch.Tensor,
    behavior_logp: torch.Tensor,
    target_logp: torch.Tensor,
    gamma: float = 0.99,
    clip_rho: float = 1.0,
) -> UPGOReturns:
    """Computes UPGO targets and clipped-importance advantages."""
    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have shape [T, B].")

    t_steps = rewards.shape[0]
    upgo = torch.zeros_like(rewards)
    not_done = 1.0 - dones.float()
    next_return = bootstrap_value

    q = rewards + gamma * not_done * torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
    for t in reversed(range(t_steps)):
        next_value = bootstrap_value if t == (t_steps - 1) else values[t + 1]
        if t == (t_steps - 1):
            continue_upgoing = torch.zeros_like(next_value, dtype=torch.bool)
        else:
            continue_upgoing = q[t + 1] >= next_value
        bootstrap = torch.where(continue_upgoing, next_return, next_value)
        upgo[t] = rewards[t] + gamma * not_done[t] * bootstrap
        next_return = upgo[t]

    rho = torch.exp(target_logp - behavior_logp)
    clipped_rho = torch.clamp(rho, max=clip_rho)
    advantages = clipped_rho * (upgo - values)
    return UPGOReturns(upgo_targets=upgo, advantages=advantages, clipped_rho=clipped_rho)

