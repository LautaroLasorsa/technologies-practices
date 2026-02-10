"""Decoders: reconstruct observations and predict rewards from latent states.

The decoders take the full latent state (h concatenated with z) as input and
produce predictions. Their reconstruction losses provide the main learning
signal for the world model -- without them, the RSSM would have no reason
to learn meaningful representations.

In full DreamerV2/V3, there's also a "continue" (discount) predictor that
learns to predict episode termination. We simplify by using a fixed gamma.
"""

import torch
import torch.nn as nn

from app.config import DreamerConfig


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
) -> nn.Sequential:
    """Helper: 2-layer MLP with ELU activations (no activation on output)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ObservationDecoder(nn.Module):
    """Reconstruct observations from latent state.

    latent (h cat z) -> MLP -> predicted observation

    The reconstruction loss (MSE) trains the encoder + RSSM to preserve
    observation information in the latent state.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        # TODO(human): Create self.network using _build_mlp.
        #
        # Input dimension: config.latent_dim  (= deterministic_dim + stochastic_dim)
        # Hidden dimension: config.decoder_hidden
        # Output dimension: config.obs_dim
        #
        # One line: self.network = _build_mlp(...)
        raise NotImplementedError("TODO(human): build observation decoder")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to observation prediction.

        Args:
            latent: (batch, latent_dim) -- concat of h and z.

        Returns:
            (batch, obs_dim) -- predicted observation.
        """
        # TODO(human): One-liner -- pass latent through self.network.
        raise NotImplementedError("TODO(human): forward pass")


class RewardDecoder(nn.Module):
    """Predict scalar reward from latent state.

    latent (h cat z) -> MLP -> predicted reward (scalar)

    The reward prediction loss trains the world model to understand which
    latent states lead to high/low rewards -- critical for imagination.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        # This one is provided as reference for the observation decoder above.
        self.network = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden,
            output_dim=1,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state to reward prediction.

        Args:
            latent: (batch, latent_dim)

        Returns:
            (batch, 1) -- predicted reward.
        """
        return self.network(latent)
