"""Phase 1 — the Bayesian structural time series (BSTS) model specification.

A structural time series decomposes a series into unobserved *state*
components — typically a local level, a local trend, and a seasonal
component — plus a regression on control series, all evolving through a
state-space model rather than being fit as one static regression. The
canonical special case (and CausalImpact's own default) drops trend and
seasonality and keeps just:

    y_t = level_t + X_t @ beta + eps_obs,   level_t = level_{t-1} + eps_level

i.e. a random-walk "local level" plus a static regression coefficient on the
controls. That is the model this phase implements: it is the simplest
member of the BSTS family that still gives every later phase a genuine
posterior over a latent state to forecast forward from, and it keeps
sampling fast enough to run on a laptop CPU in the time-boxed session (full
trend/seasonal components are a natural, more expensive extension — see this
practice's CLAUDE.md Theoretical Context).

Run on its own to fit the model to the synthetic pre-period and print a
posterior summary:
    uv run python -m src._01_structural_model
"""
from __future__ import annotations

import numpy as np
import pymc as pm

from .datasets import load_dataset


# TODO(human) — local-level structural model with regression on controls
# ---------------------------------------------------------------------------
# Goal: build (but do not sample) a PyMC model implementing the local-level
# BSTS specification above, trained ONLY on the pre-intervention data.
#
# Why this matters: everything downstream — the counterfactual forecast
# (Phase 2), the effect posterior (Phase 3) — is only as good as this
# specification. Getting the state-space wiring right (the level is a
# *latent random walk*, not a fixed intercept) is what lets the model
# generate a genuine forward forecast with growing uncertainty, instead of
# a static regression line that is equally confident at every horizon.
#
# Steps:
#   1. `sigma_level = pm.HalfNormal("sigma_level", sigma=1.0)` — the
#      innovation scale of the latent level's random walk (must be
#      positive).
#   2. `sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)` — the
#      observation noise scale.
#   3. `level = pm.GaussianRandomWalk("level", sigma=sigma_level,
#      init_dist=pm.Normal.dist(0.0, 5.0), shape=n)` — the latent local
#      level over the `n` pre-period days. This is the state whose *last*
#      value and innovation scale Phase 2 forecasts forward from.
#   4. `beta = pm.Normal("beta", mu=0.0, sigma=2.0, shape=k)` — the
#      regression coefficients on the `k` control series (static: the
#      controls' relationship to the target does not itself drift).
#   5. `mu = level + X_pre @ beta` — the model's expected value.
#   6. `pm.Normal("y_obs", mu=mu, sigma=sigma_obs, observed=y_pre)` — the
#      likelihood.
#
# All of the above must be declared inside a `with pm.Model() as model:`
# block; return that `model` object un-sampled (Phase 2 calls `pm.sample`
# on it).
# ---------------------------------------------------------------------------
def build_local_level_model(y_pre: np.ndarray, X_pre: np.ndarray) -> pm.Model:
    """Build the un-sampled local-level BSTS model for the pre-period data.

    `y_pre` is `(n,)`, `X_pre` is `(n, k)`. Returns a `pymc.Model` with
    named variables `sigma_level`, `sigma_obs`, `level`, `beta`, `y_obs` —
    later phases read the posterior of `level`, `sigma_level`, `beta` and
    `sigma_obs` by name.
    """
    raise NotImplementedError("TODO(human): implement the local-level BSTS model")


def main() -> None:
    data = load_dataset(n_pre=90, n_post=30, seed=0)
    (y_pre, X_pre), _ = data.split()
    try:
        model = build_local_level_model(y_pre, X_pre)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    print(model)
    with model:
        idata = pm.sample(
            500, tune=1000, chains=4, cores=1, target_accept=0.95,
            progressbar=False, random_seed=0, nuts_sampler="nutpie",
        )
    import arviz as az

    print(az.summary(idata, var_names=["sigma_level", "sigma_obs", "beta"]))


if __name__ == "__main__":
    main()
