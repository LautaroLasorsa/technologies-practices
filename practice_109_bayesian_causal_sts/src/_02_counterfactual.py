"""Phase 2 — posterior sampling and the counterfactual forecast.

The model only ever sees pre-intervention data (Phase 1). Once we have a
posterior over its latent level, its innovation scale, its regression
coefficients and its observation noise, "what would the target have done
without the intervention" becomes a genuine forecasting question: for each
posterior draw, roll the local-level random walk forward from its last
pre-period value for `n_post` more steps, add the (drawn) regression
contribution from the *post-period* controls, and add observation noise.
Repeating this once per posterior draw turns "the counterfactual" from a
single number into a full posterior over counterfactual *paths* — the
object CausalImpact-style reports are built from.

Run on its own to sample the Phase 1 model and print the counterfactual's
posterior mean for the first few post-period days:
    uv run python -m src._02_counterfactual
"""
from __future__ import annotations

import arviz as az
import numpy as np
import pymc as pm

from ._01_structural_model import build_local_level_model
from .datasets import load_dataset


def run_mcmc(model: pm.Model, draws: int = 500, tune: int = 1000, chains: int = 4, seed: int = 0) -> az.InferenceData:
    """Sample the posterior. Kept small (500 draws, 4 short chains) so the
    whole practice finishes in under a minute on a laptop CPU — this is
    teaching-scale inference, not a production BSTS fit. Uses `nutpie`
    (a prebuilt, Rust-backed NUTS implementation) rather than PyMC's default
    sampler so sampling never depends on a working system C/C++ compiler —
    see `src/__init__.py` for why that matters on this stack."""
    with model:
        return pm.sample(
            draws, tune=tune, chains=chains, cores=1, target_accept=0.95,
            progressbar=False, random_seed=seed, nuts_sampler="nutpie",
        )


# TODO(human) — forecast the counterfactual forward from the intervention point
# ---------------------------------------------------------------------------
# Goal: given the posterior over the pre-period local-level model, simulate
# `n_post` steps of the local-level random walk forward for every posterior
# draw, add each draw's regression contribution from the post-period
# controls, and add observation noise — producing a `(n_draws, n_post)`
# array of counterfactual *paths*, not a single forecast line.
#
# Why this matters: this is the actual counterfactual-generation step BSTS
# is for. A point forecast alone would silently pretend the model is certain
# about the future; forecasting draw-by-draw is what makes the growing
# uncertainty (the random walk's cumulative variance) show up honestly in
# the credible band Phase 4's plots shade.
#
# Steps:
#   1. Stack chains and draws into one sample axis:
#      `post = idata.posterior.stack(sample=("chain", "draw"))`.
#   2. Pull out, as plain NumPy arrays: `level_last = post["level"].values[-1, :]`
#      (the last pre-period level per draw, shape `(n_draws,)`),
#      `sigma_level = post["sigma_level"].values`, `sigma_obs = post["sigma_obs"].values`
#      (each `(n_draws,)`), and `beta = post["beta"].values` (shape `(k, n_draws)`).
#   3. For each draw, simulate the level's `n_post` forward innovations with
#      `rng.normal(0, sigma_level[:, None], size=(n_draws, n_post))` and
#      `np.cumsum` along the time axis, then add `level_last[:, None]` — this
#      is the random walk continuing exactly where the pre-period posterior
#      left off, per draw.
#   4. Add the regression contribution `X_post @ beta` (shape `(n_post, n_draws)`,
#      transpose to line up with the level array) and observation noise
#      `rng.normal(0, sigma_obs[:, None], size=(n_draws, n_post))` (note the
#      `[:, None]` — `sigma_obs` is `(n_draws,)` and must be reshaped to
#      broadcast against the `(n_draws, n_post)` output, the same trick used
#      for `sigma_level` in step 3).
#   5. Return the `(n_draws, n_post)` sum of all three terms.
# ---------------------------------------------------------------------------
def counterfactual_forecast(idata: az.InferenceData, X_post: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulate `n_draws` counterfactual paths over the post-period.

    `X_post` is `(n_post, k)`. Returns a `(n_draws, n_post)` array — one
    simulated counterfactual path per posterior draw, ready to be reduced
    to a mean/credible-band (Phase 4) or compared against the observed
    post-period series (Phase 3).
    """
    raise NotImplementedError("TODO(human): forecast the local-level random walk forward")


def main() -> None:
    data = load_dataset(n_pre=90, n_post=30, seed=0)
    (y_pre, X_pre), (y_post, X_post) = data.split()
    try:
        model = build_local_level_model(y_pre, X_pre)
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")
        return
    idata = run_mcmc(model)
    try:
        rng = np.random.default_rng(1)
        cf = counterfactual_forecast(idata, X_post, rng)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    print(f"counterfactual draws shape: {cf.shape}")
    print(f"posterior mean, first 5 post-period days: {cf.mean(axis=0)[:5]}")
    print(f"observed,      first 5 post-period days: {y_post[:5]}")


if __name__ == "__main__":
    main()
