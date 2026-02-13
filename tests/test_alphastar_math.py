import torch
import numpy as np
import pytest

from botgame.training.alphastar.league import pfsp_weights
from botgame.training.alphastar.losses import td_lambda_targets, upgo_returns, vtrace


# New test for V-trace math
@pytest.fixture
def vtrace_toy_data():
    T = 4
    rewards = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=torch.float32)
    values = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    bootstrap_value = torch.tensor([0.5], dtype=torch.float32) # V(s_T+1)
    behavior_logp = torch.zeros((T,1), dtype=torch.float32) # on-policy
    target_logp = torch.zeros_like(rewards) # on-policy
    dones = torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32)
    gamma = 0.99
    lambda_ = 0.95
    return T, rewards, values, bootstrap_value, behavior_logp, target_logp, dones, gamma, lambda_

def test_vtrace_properties(vtrace_toy_data):
    T, rewards, values, bootstrap_value, behavior_logp, target_logp, dones, gamma, lambda_ = vtrace_toy_data

    # Test with on-policy data (rho=1, c=1)
    # vtrace should approximate TD(lambda) for on-policy
    out_on_policy = vtrace(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        behavior_logp=behavior_logp,
        target_logp=target_logp,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
        clip_rho=1e9, # effectively no clipping
        clip_pg_rho=1e9,
        clip_c=1e9,
    )
    assert out_on_policy.vs.shape == (T, 1)
    assert not torch.isnan(out_on_policy.vs).any()
    assert not torch.isinf(out_on_policy.vs).any()

    # Test with off-policy data and clipping
    off_policy_behavior_logp = torch.tensor([[-1.0], [-0.5], [0.5], [1.0]], dtype=torch.float32) # Some actions less/more probable
    off_policy_target_logp = torch.zeros((T,1), dtype=torch.float32)
    
    # Large clipping values - should behave closer to unclipped for ratios within bounds
    out_large_clip = vtrace(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        behavior_logp=off_policy_behavior_logp,
        target_logp=off_policy_target_logp,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
        clip_rho=10.0,
        clip_pg_rho=1.0, # PG rho often has tighter clip
        clip_c=10.0,
    )
    assert out_large_clip.vs.shape == (T, 1)
    assert not torch.isnan(out_large_clip.vs).any()

    # Smaller clipping values - check effect of clipping
    out_small_clip = vtrace(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        behavior_logp=off_policy_behavior_logp,
        target_logp=off_policy_target_logp,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
        clip_rho=0.5,
        clip_pg_rho=0.5,
        clip_c=0.5,
    )
    # The vs values should be "more stable" (less variance) with smaller clips, though exact values are hard to assert
    # We can check that they are different from large clip and not NaN
    assert not torch.allclose(out_large_clip.vs, out_small_clip.vs)
    assert not torch.isnan(out_small_clip.vs).any()

    # Test vs equals baseline when ratios=1 and rho/c are large
    # This implies target_logp = behavior_logp, so behavior_logp = target_logp (rho=1, c=1)
    out_baseline_check = vtrace(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        behavior_logp=target_logp, # Make behavior_logp == target_logp
        target_logp=target_logp,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
        clip_rho=1e9,
        clip_pg_rho=1e9,
        clip_c=1e9,
    )
    # If behavior_logp == target_logp, then rho and c are 1, and V-trace should fall back to TD(lambda)
    # The TD(lambda) target function can be used to compute the expected vs
    td_lambda_vs_expected = td_lambda_targets(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        dones=dones,
        gamma=gamma,
        lambda_=lambda_,
    )
    assert torch.allclose(out_baseline_check.vs, td_lambda_vs_expected, atol=1e-5)


def test_vtrace_matches_mc_when_on_policy_and_zero_values():
    # This is essentially covered by test_vtrace_properties with large clips and on-policy behavior
    # but keeping it for historical context or specific small check
    # Re-using vtrace_toy_data setup for consistency
    rewards = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    values = torch.zeros_like(rewards)
    bootstrap = torch.tensor([0.0], dtype=torch.float32)
    behavior = torch.zeros_like(rewards)
    target = torch.zeros_like(rewards)
    dones = torch.zeros_like(rewards)

    out = vtrace(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        behavior_logp=behavior,
        target_logp=target,
        dones=dones,
        gamma=1.0,
        lambda_=1.0,
        clip_rho=1.0,
        clip_pg_rho=1.0,
        clip_c=1.0,
    )
    assert torch.allclose(out.vs.squeeze(-1), torch.tensor([2.0, 1.0]), atol=1e-5)


def test_td_lambda_mc_limit():
    rewards = torch.tensor([[2.0], [3.0]])
    values = torch.zeros_like(rewards)
    bootstrap = torch.tensor([0.0])
    dones = torch.zeros_like(rewards)
    targets = td_lambda_targets(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        dones=dones,
        gamma=1.0,
        lambda_=1.0,
    )
    assert torch.allclose(targets.squeeze(-1), torch.tensor([5.0, 3.0]), atol=1e-5)


# New test for TD(lambda) targets
@pytest.fixture
def td_lambda_toy_data():
    rewards = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float32) # T=4
    values = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    bootstrap_value = torch.tensor([0.5], dtype=torch.float32) # V(s_T+1)
    dones = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32) # Last step is terminal
    gamma = 0.9
    return rewards, values, bootstrap_value, dones, gamma

def test_td_lambda_zero(td_lambda_toy_data):
    rewards, values, bootstrap_value, dones, gamma = td_lambda_toy_data
    # lambda = 0 should equal 1-step TD target: r_t + gamma * V(s_t+1) * (1-d_t)
    targets_lambda_0 = td_lambda_targets(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        dones=dones,
        gamma=gamma,
        lambda_=0.0,
    )
    expected_1_step_td = rewards[:-1] + gamma * values[1:] * (1 - dones[:-1])
    # For the last step before bootstrap, it's r_T + gamma * bootstrap_value * (1-d_T)
    expected_1_step_td = torch.cat((expected_1_step_td, rewards[-1:] + gamma * bootstrap_value * (1 - dones[-1:])))

    assert torch.allclose(targets_lambda_0, expected_1_step_td, atol=1e-5)
    assert targets_lambda_0.shape == rewards.shape
    assert not torch.isnan(targets_lambda_0).any()

def test_td_lambda_one_mc_limit(td_lambda_toy_data):
    rewards, values, bootstrap_value, dones, gamma = td_lambda_toy_data
    # lambda = 1 should equal Monte Carlo return (with bootstrap at truncation)
    targets_lambda_1 = td_lambda_targets(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap_value,
        dones=dones,
        gamma=gamma,
        lambda_=1.0,
    )

    # Manual MC calculation with bootstrap
    T = rewards.shape[0]
    expected_mc = torch.zeros_like(rewards)
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            expected_mc[t] = rewards[t] + gamma * bootstrap_value * (1 - dones[t])
        else:
            expected_mc[t] = rewards[t] + gamma * expected_mc[t+1] * (1 - dones[t]) # (1-dones) ensures no reward after terminal
    
    assert torch.allclose(targets_lambda_1, expected_mc, atol=1e-5)
    assert targets_lambda_1.shape == rewards.shape
    assert not torch.isnan(targets_lambda_1).any()


def test_pfsp_var_prefers_midrange():
    probs = [0.1, 0.5, 0.9]
    weights = pfsp_weights(probs, mode="var")
    assert abs(weights.sum() - 1.0) < 1e-6
    assert weights[1] > weights[0]
    assert weights[1] > weights[2]

# New test for PFSP weights
def test_pfsp_properties():
    # Test symmetry around 0.5
    assert pfsp_weights([0.1], mode="var")[0] == pfsp_weights([0.9], mode="var")[0]
    assert pfsp_weights([0.2], mode="var")[0] == pfsp_weights([0.8], mode="var")[0]

    # Extremes should be weighted less than mid-range values.
    extreme_mix = pfsp_weights([0.0, 0.5, 1.0], mode="var")
    assert extreme_mix[1] > extreme_mix[0]
    assert extreme_mix[1] > extreme_mix[2]

    # Test normalization
    probs_norm = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    weights_norm = pfsp_weights(probs_norm, mode="var")
    assert abs(weights_norm.sum() - 1.0) < 1e-6

    # Test stable handling of p outside [0,1] via clipping (assuming internal clipping)
    # The function should handle this gracefully without error or NaN
    probs_clipped = [-0.1, 0.5, 1.1]
    weights_clipped = pfsp_weights(probs_clipped, mode="var")
    assert not np.isnan(weights_clipped).any()
    assert abs(weights_clipped.sum() - 1.0) < 1e-6 # Should still normalize after internal clipping



def test_upgo_pushes_more_when_next_action_is_better_than_value():
    behavior = torch.zeros((2, 1), dtype=torch.float32)
    target = torch.zeros((2, 1), dtype=torch.float32)
    dones = torch.zeros((2, 1), dtype=torch.float32)
    bootstrap = torch.tensor([0.0])

    # Better-than-value next step should propagate return backward.
    rewards_a = torch.tensor([[0.0], [1.0]])
    values_a = torch.tensor([[0.0], [0.5]])
    upgo_a = upgo_returns(rewards_a, values_a, bootstrap, dones, behavior, target, gamma=1.0)

    # Worse-than-value next step should bootstrap to V and reduce pressure.
    rewards_b = torch.tensor([[0.0], [0.0]])
    values_b = torch.tensor([[0.0], [0.5]])
    upgo_b = upgo_returns(rewards_b, values_b, bootstrap, dones, behavior, target, gamma=1.0)

    assert upgo_a.advantages[0, 0] > upgo_b.advantages[0, 0]

# New test for UPGO targets
@pytest.fixture
def upgo_toy_data():
    rewards = torch.tensor([[0.0], [0.0], [1.0], [0.0]], dtype=torch.float32) # T=4
    values = torch.tensor([[0.1], [0.2], [0.3], [0.4]], dtype=torch.float32)
    bootstrap_value = torch.tensor([0.5], dtype=torch.float32) # V(s_T+1)
    # Simulate an off-policy scenario
    behavior_logp = torch.tensor([[-1.0], [-0.5], [0.0], [0.5]], dtype=torch.float32)
    target_logp = torch.zeros_like(rewards) # on-policy
    
    # dones = torch.tensor([[0.0], [0.0], [0.0], [1.0]], dtype=torch.float32) # Last step is terminal
    dones = torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32) # For now, no terminal in main sequence
    gamma = 0.99
    return rewards, values, bootstrap_value, behavior_logp, target_logp, dones, gamma

def test_upgo_rules(upgo_toy_data):
    rewards, values, bootstrap_value, behavior_logp, target_logp, dones, gamma = upgo_toy_data
    T = rewards.shape[0]

    # Test the basic UPGO calculation
    out = upgo_returns(rewards, values, bootstrap_value, dones, behavior_logp, target_logp, gamma)
    assert out.advantages.shape == (T, 1)
    assert not torch.isnan(out.advantages).any()
    assert not torch.isinf(out.advantages).any()

    # Verify the "upgoing" rule: only segments where Q(s,a) > V(s) contribute
    # We expect advantages to be non-negative where Q > V
    # This is a conceptual check, hard to verify exact numbers without reference implementation
    # Let's create a specific scenario
    rewards_qv = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    values_qv = torch.tensor([[0.5], [0.8]], dtype=torch.float32)
    bootstrap_qv = torch.tensor([1.0], dtype=torch.float32)
    behavior_logp_qv = torch.zeros_like(rewards_qv)
    target_logp_qv = torch.zeros_like(rewards_qv)
    dones_qv = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    gamma_qv = 1.0

    # Frame 0: r=0, V=0.5, bootstrap_V=0.8 (from frame 1). Q(s0,a0) = r0 + 1*0.8 = 0.8. Q>V. Should propagate.
    # Frame 1: r=0, V=0.8, bootstrap_V=1.0 (from s_T+1). Q(s1,a1) = r1 + 1*1.0 = 1.0. Q>V. Should propagate.
    upgo_qv_all_up = upgo_returns(rewards_qv, values_qv, bootstrap_qv, dones_qv, 
                                  behavior_logp_qv, target_logp_qv, gamma_qv)
    # Both advantages should be positive, indicating they contribute to returns
    assert upgo_qv_all_up.advantages[0,0] > 0
    assert upgo_qv_all_up.advantages[1,0] > 0

    # Scenario where some Q < V
    rewards_qv_mixed = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    values_qv_mixed = torch.tensor([[0.8], [0.1]], dtype=torch.float32) # V0=0.8, V1=0.1
    bootstrap_qv_mixed = torch.tensor([0.0], dtype=torch.float32) # V(s_T+1)=0.0
    
    # Q(s1,a1) = r1 + 1*0.0 = 0.0. Q < V (0.0 < 0.1). Should stop propagation here or bootstrap to V.
    # Q(s0,a0) = r0 + 1*0.1 = 0.1. Q < V (0.1 < 0.8). Should stop propagation here or bootstrap to V.
    upgo_qv_mixed = upgo_returns(rewards_qv_mixed, values_qv_mixed, bootstrap_qv_mixed, dones_qv,
                                 behavior_logp_qv, target_logp_qv, gamma_qv)
    # The advantages should be significantly smaller than in the all_up case, potentially negative or zero
    assert upgo_qv_mixed.advantages[0,0] < upgo_qv_all_up.advantages[0,0]
    assert upgo_qv_mixed.advantages[1,0] < upgo_qv_all_up.advantages[1,0]


    # Test behavior with terminal transitions (dones)
    rewards_term = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    values_term = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    bootstrap_term = torch.tensor([0.0], dtype=torch.float32)
    behavior_logp_term = torch.zeros_like(rewards_term)
    target_logp_term = torch.zeros_like(rewards_term)
    dones_term = torch.tensor([[0.0], [1.0]], dtype=torch.float32) # Frame 1 is terminal

    out_term = upgo_returns(rewards_term, values_term, bootstrap_term, dones_term,
                            behavior_logp_term, target_logp_term, gamma_qv)
    
    assert not torch.isnan(out_term.advantages).any()
    assert out_term.advantages[0,0] > 0
    assert out_term.advantages[1,0] > 0

