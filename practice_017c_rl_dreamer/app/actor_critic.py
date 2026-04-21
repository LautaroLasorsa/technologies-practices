"""Actor-Critic networks that operate entirely in latent space.

Key insight of Dreamer: the actor and critic NEVER see raw observations.
They receive the latent state (h, z) and learn from imagined trajectories.

- Actor: outputs a categorical distribution over actions given latent state.
- Critic: estimates the value (expected discounted return) of a latent state.

Training:
- Critic is trained with MSE loss against lambda-returns computed on
  imagined trajectories.
- Actor is trained with REINFORCE-style policy gradient using advantages
  (lambda-return minus critic baseline), plus an entropy bonus for exploration.

The MLPs are pre-built in `__init__`; the exercises focus on:
  - Wrapping the actor's logits into a `Categorical` distribution.
  - The REINFORCE + entropy actor loss.
  - The MSE critic loss against lambda-return targets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from app.config import DreamerConfig


class Actor(nn.Module):
    """Policy network: latent state -> action distribution.

    Architecture: 2-layer MLP -> logits -> Categorical distribution.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config
        # 2-layer MLP producing raw logits (Categorical applies the softmax).
        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, config.actor_hidden), nn.ELU(),
            nn.Linear(config.actor_hidden, config.actor_hidden), nn.ELU(),
            nn.Linear(config.actor_hidden, config.action_dim),
        )

    # -- TODO 1 ---------------------------------------------------------------

    def forward(self, latent: torch.Tensor) -> Categorical:
        """Turn a latent state into a `Categorical` distribution over actions.

        Pass `latent` through `self.network` to get raw logits, then return
        `Categorical(logits=logits)`.

        Why return a distribution object instead of an action? Different
        callers need different things:
          - Environment interaction: `.sample()` for exploration.
          - Loss computation:         `.log_prob(action)` and `.entropy()`.
        """
        # TODO(human): compute logits and wrap them in a Categorical.
        raise NotImplementedError("Implement Actor.forward()")

    # -- TODO 2 ---------------------------------------------------------------

    def loss(
        self,
        latent_states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """REINFORCE + entropy bonus actor loss.

            L_actor = -mean(log_prob(a) * advantage) - entropy_weight * mean(entropy)

        The first term pushes the policy toward actions with positive
        advantage; the second encourages exploration by rewarding entropy.

        Steps:
          1. `dist = self(latent_states)`
          2. `log_probs = dist.log_prob(actions)`
          3. `entropy   = dist.entropy()`
          4. `policy_loss  = -(log_probs * advantages.detach()).mean()`
          5. `entropy_loss = -self.config.entropy_weight * entropy.mean()`
          6. return `policy_loss + entropy_loss`

        CRITICAL: `advantages.detach()` — the actor should treat advantages
        as fixed targets; do not backprop through the critic / return
        computation into the actor.
        """
        # TODO(human): implement the REINFORCE + entropy loss as above.
        raise NotImplementedError("Implement Actor.loss()")


class Critic(nn.Module):
    """Value network: latent state -> scalar value estimate.

    Architecture: 2-layer MLP -> single scalar output.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.latent_dim, config.critic_hidden), nn.ELU(),
            nn.Linear(config.critic_hidden, config.critic_hidden), nn.ELU(),
            nn.Linear(config.critic_hidden, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate value of a latent state (batch / (T, batch), latent_dim) -> same leading dims."""
        return self.network(latent).squeeze(-1)

    # -- TODO 3 ---------------------------------------------------------------

    def loss(
        self,
        latent_states: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        """MSE between predicted values and lambda-return targets.

        Steps:
          1. `values = self(latent_states)`            # (T, batch)
          2. return `F.mse_loss(values, lambda_returns.detach())`

        `.detach()` on targets: the critic must not backprop through the
        lambda-return computation — treat returns as fixed regression targets.
        """
        # TODO(human): implement the MSE critic loss against detached lambda returns.
        raise NotImplementedError("Implement Critic.loss()")
