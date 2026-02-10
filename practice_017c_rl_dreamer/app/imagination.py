"""Imagination rollouts and lambda-return computation.

This is the "dreaming" module. Given initial latent states from real experience,
unroll the world model forward using only the transition prior (no observations)
and the actor's policy. Compute lambda-returns on these imagined trajectories
for training the actor and critic.

Why lambda-returns instead of simple Monte Carlo returns?
Lambda-returns (TD(lambda)) blend 1-step TD targets with multi-step returns.
Lambda=0 is pure bootstrapping (low variance, high bias).
Lambda=1 is pure Monte Carlo (high variance, low bias).
Lambda=0.95 is a good balance for imagination where trajectories are short.

Reference: https://arxiv.org/abs/1912.01603 Section 4 (Behavior Learning).
"""

import torch

from app.config import DreamerConfig
from app.rssm import RSSM
from app.actor_critic import Actor, Critic
from app.decoder import RewardDecoder


@torch.no_grad()
def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_gae: float,
) -> torch.Tensor:
    """Compute lambda-returns for an imagined trajectory.

    V_lambda_t = r_t + gamma * ((1 - lambda) * V_{t+1} + lambda * V_lambda_{t+1})

    This recursion blends bootstrapped value estimates with actual (imagined)
    rewards. Computed backwards from the last timestep.

    Args:
        rewards: (H, batch) -- imagined rewards at each step.
        values: (H+1, batch) -- critic value estimates (H steps + bootstrap).
        gamma: Discount factor.
        lambda_gae: Lambda for blending TD and MC.

    Returns:
        (H, batch) -- lambda-return for each timestep.
    """
    # TODO(human): Implement the lambda-return computation.
    #
    # This is computed BACKWARDS from the last timestep.
    #
    # Algorithm:
    #   1. Initialize: last_lambda_return = values[-1]  (bootstrap from critic)
    #   2. Create a list or tensor to store returns
    #   3. For t = H-1, H-2, ..., 0 (reverse order):
    #       td_target = rewards[t] + gamma * values[t + 1]
    #       last_lambda_return = (1 - lambda_gae) * td_target + lambda_gae * (rewards[t] + gamma * last_lambda_return)
    #       store last_lambda_return
    #   4. Reverse the list and stack into tensor
    #
    # Simplified equivalent formula per step:
    #   last_lambda_return = rewards[t] + gamma * ((1 - lambda) * values[t+1] + lambda * last_lambda_return)
    #
    # Think of it like a DP recurrence (similar to competitive programming):
    # dp[t] = r[t] + gamma * ((1-lam) * V[t+1] + lam * dp[t+1])
    # Base case: dp[H] = V[H]
    raise NotImplementedError("TODO(human): lambda-return computation")


def imagine_rollout(
    initial_h: torch.Tensor,
    initial_z: torch.Tensor,
    rssm: RSSM,
    actor: Actor,
    reward_decoder: RewardDecoder,
    critic: Critic,
    config: DreamerConfig,
) -> dict[str, torch.Tensor]:
    """Perform imagination rollout starting from real latent states.

    Starting from (h, z) obtained from real experience, roll forward H steps
    using only the transition model (prior) and actor -- no real observations.

    Args:
        initial_h: (batch, deterministic_dim) -- starting deterministic states.
        initial_z: (batch, stochastic_dim) -- starting stochastic states.
        rssm: The world model's RSSM.
        actor: Policy network.
        reward_decoder: Predicts rewards from latent states.
        critic: Value network (for computing lambda-returns).
        config: Hyperparameters.

    Returns:
        Dictionary with:
            "latent_states": (H, batch, latent_dim) -- imagined latent states
            "actions": (H, batch) -- actions chosen by actor
            "rewards": (H, batch) -- predicted rewards
            "lambda_returns": (H, batch) -- computed lambda-returns
            "values": (H, batch) -- critic value estimates at each step
    """
    # TODO(human): Implement the imagination rollout loop.
    #
    # This is the "dreaming" loop. The agent generates its own training data
    # by rolling the world model forward using the actor's policy.
    #
    # Steps:
    #   1. Initialize lists: latent_states, actions_list, rewards_list
    #   2. Set h, z = initial_h, initial_z
    #   3. For t in range(config.imagination_horizon):
    #       a. Concatenate h and z to form latent = torch.cat([h, z], dim=-1)
    #       b. Store latent in latent_states list
    #       c. Get action distribution: dist = actor(latent)
    #       d. Sample action: action = dist.sample()
    #       e. Store action in actions_list
    #       f. Create one-hot action: action_onehot = F.one_hot(action, config.action_dim).float()
    #       g. Predict reward: reward = reward_decoder(latent).squeeze(-1)
    #       h. Store reward in rewards_list
    #       i. Advance world model: h, z = rssm.imagine_step(h, z, action_onehot)
    #
    #   4. Get bootstrap value for the final state:
    #       final_latent = torch.cat([h, z], dim=-1)
    #       final_value = critic(final_latent)
    #
    #   5. Stack lists into tensors:
    #       latent_states = torch.stack(latent_states)   # (H, batch, latent_dim)
    #       actions = torch.stack(actions_list)           # (H, batch)
    #       rewards = torch.stack(rewards_list)           # (H, batch)
    #
    #   6. Compute values for all imagined states:
    #       values = critic(latent_states)                # (H, batch)
    #       all_values = torch.cat([values, final_value.unsqueeze(0)], dim=0)  # (H+1, batch)
    #
    #   7. Compute lambda-returns:
    #       lambda_rets = compute_lambda_returns(rewards, all_values, config.gamma, config.lambda_gae)
    #
    #   8. Return the dict (see docstring)
    #
    # IMPORTANT: This entire function should run WITHOUT gradients through the
    # RSSM (the world model is fixed during imagination). However, the actor
    # and critic DO need gradients. The caller handles this by wrapping
    # appropriately. For now, just implement the logic.
    #
    # Hint on F.one_hot: import torch.nn.functional as F at the top.
    raise NotImplementedError("TODO(human): imagination rollout")
