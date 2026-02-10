"""Recurrent State-Space Model (RSSM) -- the world model core.

The RSSM maintains a latent state with two parts:
  - h (deterministic): GRU hidden state, captures long-term temporal structure.
  - z (stochastic): Gaussian latent variable, captures environmental uncertainty.

Three sub-models compose the RSSM:

  1. Recurrent model:  h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
     Takes the previous stochastic state and action, advances the deterministic state.

  2. Representation model (posterior):  z_t ~ q(z_t | h_t, o_t)
     Given the new deterministic state AND the actual observation embedding,
     infers what stochastic state we're *really* in. Used during training.

  3. Transition model (prior):  z_t ~ p(z_t | h_t)
     Given ONLY the deterministic state (no observation), predicts the stochastic
     state. Used during imagination -- the agent must "guess" without seeing.

The KL divergence between posterior and prior is a key loss term: it trains the
prior to predict well AND regularizes the posterior to not rely too heavily on
observations.

Reference: https://arxiv.org/abs/2010.02193 (DreamerV2), Section 2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from app.config import DreamerConfig


class RSSM(nn.Module):
    """Simplified RSSM with Gaussian stochastic latent.

    Full DreamerV2 uses 32 categorical distributions x 32 classes.
    We simplify to a diagonal Gaussian for clarity.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config
        det = config.deterministic_dim
        sto = config.stochastic_dim
        hidden = config.rssm_hidden
        emb = config.embedding_dim
        act = config.action_dim

        # --- Recurrent model ---
        # Input to GRU: concat of previous stochastic state z_{t-1} and action a_{t-1}
        # The GRU cell updates the deterministic state h.
        self.recurrent_input_fc = nn.Linear(sto + act, hidden)
        self.gru_cell = nn.GRUCell(input_size=hidden, hidden_size=det)

        # --- Representation model (posterior) ---
        # Input: concat of deterministic state h_t and observation embedding e_t
        # Output: mean and (log) std of Gaussian over z_t
        #
        # TODO(human): Build the posterior network.
        #
        # Architecture:
        #   self.posterior_fc = nn.Sequential(
        #       nn.Linear(det + emb, hidden), nn.ELU(),
        #   )
        #   self.posterior_mean = nn.Linear(hidden, sto)
        #   self.posterior_log_std = nn.Linear(hidden, sto)
        #
        # The fc layer processes the concatenated input, then two heads
        # produce the mean and log-std of the Gaussian distribution over z.
        raise NotImplementedError("TODO(human): build posterior network")

        # --- Transition model (prior) ---
        # Input: deterministic state h_t only (no observation!)
        # Output: mean and (log) std of Gaussian over z_t
        #
        # TODO(human): Build the prior network. Same structure as posterior
        # but input is only h_t (size det), not h_t + embedding.
        #
        # Architecture:
        #   self.prior_fc = nn.Sequential(
        #       nn.Linear(det, hidden), nn.ELU(),
        #   )
        #   self.prior_mean = nn.Linear(hidden, sto)
        #   self.prior_log_std = nn.Linear(hidden, sto)
        raise NotImplementedError("TODO(human): build prior network")

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, z) for a batch.

        Returns:
            h: (batch, deterministic_dim) -- zeros
            z: (batch, stochastic_dim) -- zeros
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.config.deterministic_dim, device=device)
        z = torch.zeros(batch_size, self.config.stochastic_dim, device=device)
        return h, z

    def recurrent_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the deterministic state using the GRU.

        h_t = GRU(h_{t-1}, fc([z_{t-1}, a_{t-1}]))

        Args:
            prev_h: (batch, deterministic_dim)
            prev_z: (batch, stochastic_dim)
            prev_action: (batch, action_dim) -- one-hot encoded

        Returns:
            h_t: (batch, deterministic_dim) -- new deterministic state
        """
        # TODO(human): Implement the recurrent step.
        #
        # Steps:
        #   1. Concatenate prev_z and prev_action along dim=-1
        #   2. Pass through self.recurrent_input_fc + ELU activation
        #   3. Feed into self.gru_cell(result, prev_h) to get new h
        #   4. Return h
        #
        # Hint: x = torch.cat([prev_z, prev_action], dim=-1)
        #       x = F.elu(self.recurrent_input_fc(x))
        #       h = self.gru_cell(x, prev_h)
        raise NotImplementedError("TODO(human): recurrent step")

    def posterior(
        self,
        h: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, Normal]:
        """Representation model: infer z from h + observation embedding.

        z_t ~ q(z_t | h_t, embedding_t)

        Args:
            h: (batch, deterministic_dim)
            embedding: (batch, embedding_dim) -- from ObservationEncoder

        Returns:
            z: (batch, stochastic_dim) -- sampled using reparameterization trick
            dist: Normal distribution object (for KL computation)
        """
        # TODO(human): Implement the posterior.
        #
        # Steps:
        #   1. Concatenate h and embedding: x = torch.cat([h, embedding], dim=-1)
        #   2. Pass through self.posterior_fc: features = self.posterior_fc(x)
        #   3. Compute mean = self.posterior_mean(features)
        #   4. Compute log_std = self.posterior_log_std(features)
        #   5. Clamp log_std to [-5, 2] for numerical stability
        #   6. Create dist = Normal(mean, log_std.exp())
        #   7. Sample z using rsample() (reparameterized -- allows gradients to flow!)
        #   8. Return (z, dist)
        #
        # Why rsample() not sample()? rsample uses the reparameterization trick:
        # z = mean + std * epsilon, where epsilon ~ N(0,1). This makes z
        # differentiable w.r.t. mean and std, which is essential for backprop
        # through the world model.
        raise NotImplementedError("TODO(human): posterior inference")

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        """Transition model: predict z from h only (no observation).

        z_t ~ p(z_t | h_t)

        Args:
            h: (batch, deterministic_dim)

        Returns:
            z: (batch, stochastic_dim) -- sampled
            dist: Normal distribution object (for KL computation)
        """
        # TODO(human): Implement the prior. Same structure as posterior,
        # but uses self.prior_fc/mean/log_std and input is just h (not h + emb).
        #
        # This is nearly identical to posterior() but with different networks
        # and different input. The prior must learn to predict z without
        # seeing the observation -- this is what makes imagination possible.
        raise NotImplementedError("TODO(human): prior prediction")

    def observe_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Normal, Normal]:
        """One step of RSSM during training (with observations).

        Combines recurrent step + posterior + prior.

        Args:
            prev_h: (batch, deterministic_dim)
            prev_z: (batch, stochastic_dim)
            prev_action: (batch, action_dim) -- one-hot
            embedding: (batch, embedding_dim) -- current observation encoded

        Returns:
            h: new deterministic state
            z: sampled from posterior (used for reconstruction)
            posterior_dist: for KL loss
            prior_dist: for KL loss
        """
        # TODO(human): Compose the three sub-models.
        #
        # Steps:
        #   1. h = self.recurrent_step(prev_h, prev_z, prev_action)
        #   2. z_post, post_dist = self.posterior(h, embedding)
        #   3. _, prior_dist = self.prior(h)
        #   4. Return (h, z_post, post_dist, prior_dist)
        #
        # Note: we use z from posterior (not prior) during training because
        # the posterior has access to the real observation and is more accurate.
        # The prior is only used for the KL loss term here.
        raise NotImplementedError("TODO(human): observe step")

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One step of RSSM during imagination (NO observations).

        Uses the prior (transition model) instead of posterior.

        Args:
            prev_h: (batch, deterministic_dim)
            prev_z: (batch, stochastic_dim)
            action: (batch, action_dim) -- one-hot

        Returns:
            h: new deterministic state
            z: sampled from prior
        """
        # TODO(human): Implement imagination step.
        #
        # Steps:
        #   1. h = self.recurrent_step(prev_h, prev_z, action)
        #   2. z, _ = self.prior(h)
        #   3. Return (h, z)
        #
        # This is the core of "dreaming": advance the world model forward
        # using only the learned dynamics, with no real observations.
        raise NotImplementedError("TODO(human): imagine step")
