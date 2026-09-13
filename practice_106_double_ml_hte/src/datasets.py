"""Synthetic double-ML dataset with known heterogeneous treatment effects.

The data-generating process (DGP) has a *known* CATE function tau(x) and a
*known* average treatment effect, so every phase can compare an estimator's
output against ground truth instead of against nothing. Treatment is
confounded with the outcome through a shared covariate (x0 enters both the
propensity and the CATE/baseline), which is exactly the setting where a
naive plug-in estimator gets it wrong and orthogonalization matters. A
fixed seed makes results reproducible across runs and across learners. No
TODO here: data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_COVARIATES = 4

# True DGP parameters for the heterogeneous treatment effect
# tau(x) = TAU0 + TAU1 * x0 -- every scenario shares this ground truth so
# estimates are always compared against the same target.
TAU0 = 2.0
TAU1 = 1.5


def _expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class DMLData:
    """One simulated double-ML dataset."""

    X: np.ndarray  # (n, k) covariates
    D: np.ndarray  # (n,) binary treatment indicator
    Y: np.ndarray  # (n,) outcome
    tau_true: np.ndarray  # (n,) true CATE, tau(X_i)
    propensity_true: np.ndarray  # (n,) true e(X_i) = P(D_i=1 | X_i)

    @property
    def ate_true(self) -> float:
        """True average treatment effect E[tau(X)] -- the target of Phases 1-3."""
        return float(np.mean(self.tau_true))

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        cols = {f"x{i}": self.X[:, i] for i in range(self.X.shape[1])}
        return pd.DataFrame({"D": self.D, "Y": self.Y, **cols, "tau_true": self.tau_true})


def load_dataset(n: int = 1000, seed: int = 0) -> DMLData:
    """Generate a synthetic partially-linear dataset with heterogeneous effects.

    DGP:
      Y = tau(X) * D + g0(X) + U
      D ~ Bernoulli(e(X))
      tau(X) = TAU0 + TAU1 * x0            (known heterogeneous CATE)
      e(X)   = clip(expit(0.75*x0 - 0.5*x1), 0.05, 0.95)   (confounded propensity)
      g0(X)  = sin(x0) + 0.5*x1^2 - 0.5*x2  (nonlinear baseline)
      U ~ Normal(0, 1)

    x0 drives both the propensity and the CATE/baseline, so a naive
    estimator that ignores D's dependence on X (Phase 1) is confounded by
    construction -- this is what makes the bias comparison in Phase 3 show
    up clearly rather than being a rounding-level effect.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, N_COVARIATES))

    propensity = np.clip(_expit(0.75 * X[:, 0] - 0.5 * X[:, 1]), 0.05, 0.95)
    D = rng.binomial(1, propensity)

    tau = TAU0 + TAU1 * X[:, 0]
    g0 = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 - 0.5 * X[:, 2]
    U = rng.normal(scale=1.0, size=n)
    Y = tau * D + g0 + U

    return DMLData(X=X, D=D, Y=Y, tau_true=tau, propensity_true=propensity)
