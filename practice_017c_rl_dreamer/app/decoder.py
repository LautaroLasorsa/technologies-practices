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
        self.network = self._build_network(config)

    # -- TODO -----------------------------------------------------------------

    @staticmethod
    def _build_network(config: DreamerConfig) -> nn.Sequential:
        """Return an MLP that maps the full latent state to an obs prediction.

        Use the provided `_build_mlp` helper:
          input  = config.latent_dim   (= deterministic_dim + stochastic_dim)
          hidden = config.decoder_hidden
          output = config.obs_dim

        This decoder is what forces the RSSM to learn an informative latent:
        if (h, z) cannot be decoded back to the observation, the
        reconstruction loss pushes the whole stack to fix it.
        """
        # TODO(human): return _build_mlp(input_dim=..., hidden_dim=..., output_dim=...).
        raise NotImplementedError("Implement ObservationDecoder._build_network()")

    # -- Scaffolded forward ---------------------------------------------------

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state (batch, latent_dim) -> (batch, obs_dim)."""
        return self.network(latent)


class RewardDecoder(nn.Module):
    """Predict scalar reward from latent state.

    latent (h cat z) -> MLP -> predicted reward (scalar)

    The reward prediction loss trains the world model to understand which
    latent states lead to high/low rewards -- critical for imagination.

    Provided as a reference for the observation decoder above.
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.network = _build_mlp(
            input_dim=config.latent_dim,
            hidden_dim=config.decoder_hidden,
            output_dim=1,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent state (batch, latent_dim) -> (batch, 1)."""
        return self.network(latent)
