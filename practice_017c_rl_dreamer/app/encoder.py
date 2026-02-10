"""Observation encoder: maps raw observations to an embedding vector.

In full DreamerV2/V3 with pixel observations, this would be a CNN.
For CartPole's 4-dimensional observation, a small MLP suffices.

The encoder output is fed into the RSSM's representation model (posterior)
so the world model can infer what stochastic latent state corresponds to
the current observation.
"""

import torch
import torch.nn as nn

from app.config import DreamerConfig


class ObservationEncoder(nn.Module):
    """MLP encoder: obs (obs_dim,) -> embedding (embedding_dim,).

    Architecture: two hidden layers with ELU activation.
    ELU is the standard activation in Dreamer (avoids dead neurons).
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config

        # TODO(human): Build a 2-layer MLP that maps obs_dim -> embedding_dim.
        #
        # Layers:
        #   1. Linear(obs_dim -> encoder_hidden) + ELU
        #   2. Linear(encoder_hidden -> embedding_dim) + ELU
        #
        # Use nn.Sequential and store it as self.network.
        #
        # Hint: nn.Sequential(
        #     nn.Linear(...), nn.ELU(),
        #     nn.Linear(...), nn.ELU(),
        # )
        raise NotImplementedError("TODO(human): build self.network")

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations into embeddings.

        Args:
            obs: Tensor of shape (batch, obs_dim).

        Returns:
            Tensor of shape (batch, embedding_dim).
        """
        # TODO(human): Pass obs through self.network and return the result.
        # This is a one-liner.
        raise NotImplementedError("TODO(human): forward pass")
