"""WorldModel: bundles encoder, RSSM, and decoders into one trainable unit.

The world model is trained end-to-end with three loss terms:
  1. Observation reconstruction loss (MSE) -- trains encoder + RSSM + obs decoder
  2. Reward prediction loss (MSE) -- trains RSSM + reward decoder
  3. KL divergence loss -- trains prior to match posterior, regularizes posterior

This module handles the forward pass over a sequence of observations and
computes the combined loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence

from app.config import DreamerConfig
from app.encoder import ObservationEncoder
from app.rssm import RSSM
from app.decoder import ObservationDecoder, RewardDecoder


class WorldModel(nn.Module):
    """Bundles all world-model components for joint training."""

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ObservationEncoder(config)
        self.rssm = RSSM(config)
        self.obs_decoder = ObservationDecoder(config)
        self.reward_decoder = RewardDecoder(config)

    def observe_sequence(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Process a sequence of observations through the world model.

        For each timestep t:
          1. Encode observation -> embedding
          2. RSSM observe_step -> (h, z, posterior, prior)
          3. Decode (h, z) -> predicted obs, predicted reward

        Args:
            observations: (batch, T, obs_dim) -- sequence of observations.
            actions: (batch, T, action_dim) -- one-hot actions taken.

        Returns:
            Dictionary with:
                "h_states": (batch, T, deterministic_dim)
                "z_states": (batch, T, stochastic_dim)
                "obs_preds": (batch, T, obs_dim)
                "reward_preds": (batch, T, 1)
                "posterior_dists": list of T Normal distributions
                "prior_dists": list of T Normal distributions
        """
        batch_size, seq_len, _ = observations.shape
        h, z = self.rssm.initial_state(batch_size)

        h_states = []
        z_states = []
        obs_preds = []
        reward_preds = []
        posterior_dists = []
        prior_dists = []

        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t] if t > 0 else torch.zeros_like(actions[:, 0])

            embedding = self.encoder(obs_t)
            h, z, post_dist, prior_dist = self.rssm.observe_step(h, z, action_t, embedding)

            latent = torch.cat([h, z], dim=-1)
            obs_pred = self.obs_decoder(latent)
            reward_pred = self.reward_decoder(latent)

            h_states.append(h)
            z_states.append(z)
            obs_preds.append(obs_pred)
            reward_preds.append(reward_pred)
            posterior_dists.append(post_dist)
            prior_dists.append(prior_dist)

        return {
            "h_states": torch.stack(h_states, dim=1),
            "z_states": torch.stack(z_states, dim=1),
            "obs_preds": torch.stack(obs_preds, dim=1),
            "reward_preds": torch.stack(reward_preds, dim=1),
            "posterior_dists": posterior_dists,
            "prior_dists": prior_dists,
        }

    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute the full world-model loss.

        Args:
            observations: (batch, T, obs_dim)
            actions: (batch, T, action_dim) -- one-hot
            rewards: (batch, T)

        Returns:
            Dictionary with individual and total losses.
        """
        outputs = self.observe_sequence(observations, actions)

        # --- Observation reconstruction loss ---
        obs_loss = F.mse_loss(outputs["obs_preds"], observations)

        # --- Reward prediction loss ---
        reward_targets = rewards.unsqueeze(-1)
        reward_loss = F.mse_loss(outputs["reward_preds"], reward_targets)

        # --- KL divergence loss ---
        # Average KL over the sequence. KL(posterior || prior) trains the prior
        # to predict well and regularizes the posterior.
        kl_losses = []
        for post_dist, prior_dist in zip(
            outputs["posterior_dists"], outputs["prior_dists"]
        ):
            kl = kl_divergence(post_dist, prior_dist)
            kl = kl.sum(dim=-1)  # sum over stochastic dimensions
            kl = torch.clamp(kl, min=self.config.kl_free_nats)  # free nats
            kl_losses.append(kl.mean())
        kl_loss = torch.stack(kl_losses).mean()

        total_loss = obs_loss + reward_loss + self.config.kl_weight * kl_loss

        return {
            "total": total_loss,
            "obs_loss": obs_loss.detach(),
            "reward_loss": reward_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }
