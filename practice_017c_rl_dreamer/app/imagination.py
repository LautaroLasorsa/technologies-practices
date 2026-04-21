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
import torch.nn.functional as F

from app.config import DreamerConfig
from app.rssm import RSSM
from app.actor_critic import Actor, Critic
from app.decoder import RewardDecoder


# -- TODO 1 ---------------------------------------------------------------


@torch.no_grad()
def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambda_gae: float,
) -> torch.Tensor:
    """Backward DP computing lambda-returns for an imagined trajectory.

    Per-step recurrence (computed from the last timestep backwards):

        R_t = r_t + gamma * ((1 - lambda) * V_{t+1} + lambda * R_{t+1})
        R_H = V_H                                       (bootstrap)

    Think of it like a classic DP pass:
        dp[t] = r[t] + gamma * ((1 - lam) * V[t+1] + lam * dp[t+1])
        dp[H] = V[H]

    Args:
        rewards:    (H, batch)   -- imagined rewards at each step.
        values:     (H+1, batch) -- critic value estimates, the last is the bootstrap.
        gamma:      discount factor.
        lambda_gae: lambda for blending TD and MC.

    Returns:
        (H, batch) -- lambda-return for each timestep, in forward order.
    """
    # TODO(human): iterate t = H-1 .. 0 accumulating last_lambda_return using
    # the recurrence above (base case: last_lambda_return = values[-1]), then
    # stack the results in forward order and return a (H, batch) tensor.
    raise NotImplementedError("Implement compute_lambda_returns()")


# -- TODO 2 ---------------------------------------------------------------


def imagine_rollout(
    initial_h: torch.Tensor,
    initial_z: torch.Tensor,
    rssm: RSSM,
    actor: Actor,
    reward_decoder: RewardDecoder,
    critic: Critic,
    config: DreamerConfig,
) -> dict[str, torch.Tensor]:
    """Roll the world model forward `imagination_horizon` steps in latent space.

    Starting from real latent states (`initial_h`, `initial_z`), at each step:
      1. latent = torch.cat([h, z], dim=-1)                      # store
      2. dist   = actor(latent); action = dist.sample()          # store
      3. action_onehot = F.one_hot(action, config.action_dim).float()
      4. reward = reward_decoder(latent).squeeze(-1)             # store
      5. h, z   = rssm.imagine_step(h, z, action_onehot)

    After the loop:
      - bootstrap with `final_value = critic(torch.cat([h, z], dim=-1))`
      - `values     = critic(stacked_latent_states)`             # (H, batch)
      - `all_values = torch.cat([values, final_value[None]], 0)` # (H+1, batch)
      - `lambda_rets = compute_lambda_returns(rewards, all_values, gamma, lambda)`

    IMPORTANT: the RSSM is fixed during imagination — the caller wraps this
    function to manage gradient flow (actor/critic need gradients, world
    model does not). Just implement the rollout logic here.

    Returns a dict with:
        "latent_states":  (H, batch, latent_dim)
        "actions":        (H, batch)
        "rewards":        (H, batch)
        "values":         (H, batch)
        "lambda_returns": (H, batch)
    """
    # TODO(human): implement the H-step rollout + lambda-return computation
    # following the steps above and return the dict described in the docstring.
    raise NotImplementedError("Implement imagine_rollout()")
