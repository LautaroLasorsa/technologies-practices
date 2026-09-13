"""Synthetic CausalImpact-style intervention data with a known ground truth.

The data-generating process is the standard "shared latent trend" setup used
to teach Bayesian structural time series / CausalImpact: a target series and
two control series all move together through a common, unobserved local-level
trend (so the controls are genuinely informative about what the target would
have done absent the intervention), and only the target receives an
additional, known effect starting at a fixed intervention day. Because the
controls never receive the effect, they stay valid predictors of the
counterfactual throughout the post-period — this is the identifying
assumption every synthetic-control/BSTS-style method needs, made explicit and
checkable here since we control the DGP.

A fixed seed makes results reproducible across runs and learners. No TODO
here: data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# True per-day treatment effect injected into the target series from
# `intervention_day` onward. Every phase compares its recovered effect
# against this same ground truth.
TRUE_EFFECT_PER_DAY = 4.0


@dataclass
class STSData:
    """One simulated pre/post intervention panel."""

    t: np.ndarray  # (n,) day index, 0..n-1
    y: np.ndarray  # (n,) observed target series (includes the injected effect post-intervention)
    X: np.ndarray  # (n, k) control series, never touched by the intervention
    intervention_day: int  # first day index that carries the effect
    true_effect_per_day: float
    true_cumulative_effect: float  # true_effect_per_day * n_post, the post-period ground truth

    @property
    def n_pre(self) -> int:
        return self.intervention_day

    @property
    def n_post(self) -> int:
        return len(self.t) - self.intervention_day

    def split(self):
        """Pre/post slices of (y, X) — the model only ever trains on the pre split."""
        b = self.intervention_day
        return (self.y[:b], self.X[:b, :]), (self.y[b:], self.X[b:, :])

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        cols = {f"x{i + 1}": self.X[:, i] for i in range(self.X.shape[1])}
        return pd.DataFrame({"t": self.t, "y": self.y, **cols})


def load_dataset(
    n_pre: int = 90,
    n_post: int = 30,
    effect_per_day: float = TRUE_EFFECT_PER_DAY,
    seed: int = 0,
) -> STSData:
    """Generate a synthetic target + 2 control series with a known injected effect.

    A shared latent local-level walk drives the target and both controls
    (different loadings each), plus idiosyncratic AR(1)-free Gaussian noise.
    From `n_pre` onward the target additionally receives a constant
    `effect_per_day` bump — the quantity every later phase tries to recover.
    """
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    t = np.arange(n)

    # Shared latent local-level trend: a random walk every series partially loads on.
    latent = np.cumsum(rng.normal(scale=0.6, size=n))

    x1 = 0.9 * latent + rng.normal(scale=1.0, size=n) + 10.0
    x2 = 0.5 * latent + rng.normal(scale=1.2, size=n) + 5.0
    X = np.column_stack([x1, x2])

    beta_true = np.array([0.8, 0.6])
    y_counterfactual = 3.0 + 0.7 * latent + X @ beta_true + rng.normal(scale=1.0, size=n)

    effect = np.where(t >= n_pre, effect_per_day, 0.0)
    y_observed = y_counterfactual + effect

    return STSData(
        t=t,
        y=y_observed,
        X=X,
        intervention_day=n_pre,
        true_effect_per_day=effect_per_day,
        true_cumulative_effect=effect_per_day * n_post,
    )
