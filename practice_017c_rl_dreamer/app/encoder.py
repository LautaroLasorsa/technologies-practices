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
        self.network = self._build_network()

    # -- TODO -----------------------------------------------------------------

    def _build_network(self) -> nn.Sequential:
        """Return the 2-layer MLP that maps obs_dim -> embedding_dim.

        Architecture:
          Linear(obs_dim -> encoder_hidden) + ELU
          Linear(encoder_hidden -> embedding_dim) + ELU

        ELU (not ReLU) is the Dreamer-standard activation: it keeps a small
        negative slope and avoids "dead neurons" that occur when ReLU gets
        stuck at zero — important because every gradient flows through this
        encoder into the RSSM.
        """
        # TODO(human): build and return an nn.Sequential with the two
        # Linear + ELU blocks described above, using self.config.obs_dim,
        # self.config.encoder_hidden and self.config.embedding_dim.
        raise NotImplementedError("Implement ObservationEncoder._build_network()")

    # -- Scaffolded forward ---------------------------------------------------

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations (batch, obs_dim) -> (batch, embedding_dim)."""
        return self.network(obs)
