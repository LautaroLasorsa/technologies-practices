"""Phase 3 — the pointwise and cumulative effect posteriors.

Once we have a posterior over the counterfactual (Phase 2), the causal
effect at each post-period day is just `observed - counterfactual` — but
computed *per posterior draw*, so the effect itself is a distribution, not
a point estimate with a bolted-on standard error. Summing that pointwise
effect over time gives the cumulative effect: the CausalImpact-style
"total impact of the intervention so far," complete with a full posterior
(and therefore a real credible interval, not a normal-approximation one)
at every day since the intervention.

Run on its own to fit through Phase 2 and print the cumulative effect's
posterior mean on the last post-period day:
    uv run python -m src._03_effect
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._01_structural_model import build_local_level_model
from ._02_counterfactual import counterfactual_forecast, run_mcmc
from .datasets import load_dataset


@dataclass
class EffectPosterior:
    """Posterior over the causal effect, one row per posterior draw."""

    pointwise: np.ndarray  # (n_draws, n_post) — observed - counterfactual, per day
    cumulative: np.ndarray  # (n_draws, n_post) — running sum of `pointwise` over days


# TODO(human) — pointwise and cumulative effect posteriors
# ---------------------------------------------------------------------------
# Goal: turn `(observed post-period series, counterfactual draws)` into the
# effect posterior — the object every later summary (a credible interval on
# the total impact, the probability the effect is positive) is read off of.
#
# Why this matters: because `counterfactual_draws` already carries one
# simulated path per posterior sample, subtracting it from the (fixed,
# observed) `y_post` propagates the *full* posterior uncertainty into the
# effect — no delta-method or normal-approximation step is needed. This is
# the concrete difference from practice 105's synthetic control: there, one
# set of donor weights gives one counterfactual path and the effect is a
# single number (or a placebo-test p-value bolted on afterward); here, the
# effect *is* a distribution from the start.
#
# Steps:
#   1. Broadcast-subtract: `pointwise = y_post[None, :] - counterfactual_draws`
#      — `y_post` is `(n_post,)`, `counterfactual_draws` is `(n_draws, n_post)`,
#      broadcasting gives `(n_draws, n_post)`.
#   2. `cumulative = np.cumsum(pointwise, axis=1)` — the running total impact
#      as of each post-period day, per draw.
#   3. Return both, wrapped in an `EffectPosterior`.
# ---------------------------------------------------------------------------
def cumulative_effect(y_post: np.ndarray, counterfactual_draws: np.ndarray) -> EffectPosterior:
    """Compute the pointwise and cumulative effect posteriors.

    `y_post` is `(n_post,)`, `counterfactual_draws` is `(n_draws, n_post)`
    (Phase 2's output). Returns an `EffectPosterior` with both arrays at
    `(n_draws, n_post)`.
    """
    raise NotImplementedError("TODO(human): compute the pointwise and cumulative effect posteriors")


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
        print(f"(skipped — Phase 2 not implemented yet — {e})")
        return
    try:
        effect = cumulative_effect(y_post, cf)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    total = effect.cumulative[:, -1]
    print(f"true cumulative effect:      {data.true_cumulative_effect:.2f}")
    print(f"posterior mean, last day:    {total.mean():.2f}")
    print(f"P(total effect > 0):         {(total > 0).mean():.3f}")


if __name__ == "__main__":
    main()
