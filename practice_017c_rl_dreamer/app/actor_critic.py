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

        # TODO(human): Build a 2-layer MLP that maps latent_dim -> action_dim logits.
        #
        # Architecture:
        #   self.network = nn.Sequential(
        #       nn.Linear(config.latent_dim, config.actor_hidden), nn.ELU(),
        #       nn.Linear(config.actor_hidden, config.actor_hidden), nn.ELU(),
        #       nn.Linear(config.actor_hidden, config.action_dim),
        #   )
        #
        # The output are raw logits (unnormalized log-probabilities).
        # Categorical() will handle the softmax internally.
        raise NotImplementedError("TODO(human): build actor network")

    def forward(self, latent: torch.Tensor) -> Categorical:
        """Compute action distribution from latent state.

        Args:
            latent: (batch, latent_dim) -- concat of h and z.

        Returns:
            Categorical distribution over actions.
        """
        # TODO(human): Pass latent through self.network to get logits,
        # then return Categorical(logits=logits).
        #
        # Why return a distribution object instead of an action?
        # Because different callers need different things:
        # - Environment interaction: .sample() for exploration
        # - Loss computation: .log_prob(action) and .entropy()
        raise NotImplementedError("TODO(human): forward pass")

    def loss(
        self,
        latent_states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute actor loss: REINFORCE with entropy bonus.

        L_actor = -mean(log_prob(a) * advantage) - entropy_weight * mean(entropy)

        The first term pushes the policy toward actions with positive advantage.
        The entropy term encourages exploration by penalizing overly deterministic
        policies.

        Args:
            latent_states: (T, batch, latent_dim) -- imagined trajectory states.
            actions: (T, batch) -- actions taken (long indices).
            advantages: (T, batch) -- lambda-return minus critic value.

        Returns:
            Scalar loss.
        """
        # TODO(human): Implement the REINFORCE + entropy loss.
        #
        # Steps:
        #   1. dist = self.forward(latent_states)     # get distribution for all states
        #   2. log_probs = dist.log_prob(actions)      # log pi(a|s) for chosen actions
        #   3. entropy = dist.entropy()                # H[pi(.|s)]
        #   4. policy_loss = -(log_probs * advantages.detach()).mean()
        #   5. entropy_loss = -self.config.entropy_weight * entropy.mean()
        #   6. return policy_loss + entropy_loss
        #
        # CRITICAL: advantages.detach() -- don't backprop through the return
        # computation into the actor. The actor should treat advantages as fixed
        # targets, not try to change the world model to increase them.
        raise NotImplementedError("TODO(human): actor loss")


class Critic(nn.Module):
    """Value network: latent state -> scalar value estimate.

    Architecture: 2-layer MLP -> single scalar output.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config

        # TODO(human): Build a 2-layer MLP that maps latent_dim -> 1.
        #
        # Same structure as Actor but output dim is 1 (scalar value).
        #   self.network = nn.Sequential(
        #       nn.Linear(config.latent_dim, config.critic_hidden), nn.ELU(),
        #       nn.Linear(config.critic_hidden, config.critic_hidden), nn.ELU(),
        #       nn.Linear(config.critic_hidden, 1),
        #   )
        raise NotImplementedError("TODO(human): build critic network")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate value of a latent state.

        Args:
            latent: (batch, latent_dim) or (T, batch, latent_dim).

        Returns:
            Value estimate, same leading dims as input but last dim squeezed.
        """
        # TODO(human): Pass latent through self.network and squeeze the last dim.
        #
        # return self.network(latent).squeeze(-1)
        raise NotImplementedError("TODO(human): forward pass")

    def loss(
        self,
        latent_states: torch.Tensor,
        lambda_returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute critic loss: MSE between predicted values and lambda-returns.

        Args:
            latent_states: (T, batch, latent_dim)
            lambda_returns: (T, batch) -- target values from imagination.

        Returns:
            Scalar MSE loss.
        """
        # TODO(human): Implement critic loss.
        #
        # Steps:
        #   1. values = self.forward(latent_states)           # (T, batch)
        #   2. return F.mse_loss(values, lambda_returns.detach())
        #
        # .detach() on targets: the critic should not backprop through the
        # lambda-return computation. Targets are treated as fixed.
        raise NotImplementedError("TODO(human): critic loss")
