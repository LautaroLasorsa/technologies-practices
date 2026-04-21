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
from torch.distributions import Normal

from app.config import DreamerConfig


class RSSM(nn.Module):
    """Simplified RSSM with Gaussian stochastic latent.

    Full DreamerV2 uses 32 categorical distributions x 32 classes.
    We simplify to a diagonal Gaussian for clarity.

    Construction of the sub-networks is pre-scaffolded so the exercises can
    focus on the three forward passes (recurrent / posterior / prior) and how
    they compose into `observe_step` (training) and `imagine_step` (dreaming).
    """

    def __init__(self, config: DreamerConfig) -> None:
        super().__init__()
        self.config = config
        det = config.deterministic_dim
        sto = config.stochastic_dim
        hidden = config.rssm_hidden
        emb = config.embedding_dim
        act = config.action_dim

        # Recurrent model: fc(prev_z, prev_action) -> GRUCell -> h_t.
        self.recurrent_input_fc = nn.Linear(sto + act, hidden)
        self.gru_cell = nn.GRUCell(input_size=hidden, hidden_size=det)

        # Representation model (posterior): q(z_t | h_t, embedding_t).
        # Shared trunk + separate heads for mean / log_std of a diagonal Gaussian.
        self.posterior_fc = nn.Sequential(nn.Linear(det + emb, hidden), nn.ELU())
        self.posterior_mean = nn.Linear(hidden, sto)
        self.posterior_log_std = nn.Linear(hidden, sto)

        # Transition model (prior): p(z_t | h_t) -- no observation input.
        # Same structure as the posterior; this is what imagination uses.
        self.prior_fc = nn.Sequential(nn.Linear(det, hidden), nn.ELU())
        self.prior_mean = nn.Linear(hidden, sto)
        self.prior_log_std = nn.Linear(hidden, sto)

    def initial_state(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialized (h, z) for a batch."""
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, self.config.deterministic_dim, device=device)
        z = torch.zeros(batch_size, self.config.stochastic_dim, device=device)
        return h, z

    # -- TODO 1 ---------------------------------------------------------------

    def recurrent_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the deterministic state: h_t = GRU(h_{t-1}, fc([z_{t-1}, a_{t-1}])).

        This is the deterministic backbone of the RSSM — the GRU that carries
        sequential information across timesteps. Concatenate `prev_z` and
        `prev_action` along the last dim, push through `self.recurrent_input_fc`
        with an ELU activation, then feed the result (together with `prev_h`)
        into `self.gru_cell` and return the new hidden state.
        """
        # TODO(human): implement the 3-line recurrent step.
        raise NotImplementedError("Implement RSSM.recurrent_step()")

    # -- TODO 2 ---------------------------------------------------------------

    def posterior(
        self,
        h: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, Normal]:
        """Representation model: q(z_t | h_t, embedding_t).

        Concatenate `h` with the observation `embedding`, run it through
        `posterior_fc`, produce `mean` and `log_std` via the two heads, clamp
        `log_std` to `[-5, 2]` for numerical stability, build a `Normal`
        distribution, and sample with `rsample()`.

        Why `rsample()` and not `sample()`? The reparameterization trick
        z = mean + std * epsilon with epsilon ~ N(0, 1) makes z differentiable
        w.r.t. mean and std — essential for backprop through the world model
        (reconstruction loss has to flow back into the posterior parameters).

        Returns: (z_sampled, Normal distribution for KL loss).
        """
        # TODO(human): implement the posterior forward pass as described.
        raise NotImplementedError("Implement RSSM.posterior()")

    # -- TODO 3 ---------------------------------------------------------------

    def prior(self, h: torch.Tensor) -> tuple[torch.Tensor, Normal]:
        """Transition model: p(z_t | h_t) — no observation.

        Structurally identical to `posterior`, but input is just `h` and the
        layers are `prior_fc / prior_mean / prior_log_std`. This is what makes
        imagination possible: the prior alone must predict plausible next
        stochastic states, without ever seeing the real observation.

        Returns: (z_sampled, Normal distribution for KL loss).
        """
        # TODO(human): implement the prior forward pass (mirror of posterior).
        raise NotImplementedError("Implement RSSM.prior()")

    # -- TODO 4 ---------------------------------------------------------------

    def observe_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        prev_action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Normal, Normal]:
        """One RSSM step during *training* (observations available).

        Compose the three sub-models:
          1. h = recurrent_step(prev_h, prev_z, prev_action)
          2. z_post, post_dist = posterior(h, embedding)
          3. _,      prior_dist = prior(h)

        Return `(h, z_post, post_dist, prior_dist)`. We keep `z` from the
        posterior (it has the real observation) for reconstruction; the prior
        distribution is only used for the KL loss term.
        """
        # TODO(human): compose recurrent_step + posterior + prior.
        raise NotImplementedError("Implement RSSM.observe_step()")

    # -- TODO 5 ---------------------------------------------------------------

    def imagine_step(
        self,
        prev_h: torch.Tensor,
        prev_z: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One RSSM step during *imagination* (no observations).

        Same recurrent step, but the stochastic state comes from the prior,
        not the posterior:
          1. h = recurrent_step(prev_h, prev_z, action)
          2. z, _ = prior(h)

        This is the core of "dreaming": advance the world model forward using
        only the learned dynamics.
        """
        # TODO(human): compose recurrent_step + prior.
        raise NotImplementedError("Implement RSSM.imagine_step()")
