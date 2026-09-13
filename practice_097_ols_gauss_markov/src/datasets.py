"""Synthetic OLS datasets with known ground truth.

Every dataset here comes from a data-generating process (DGP) with a
*known* true beta, so every phase can compare an estimator's output against
the ground truth instead of against nothing. A fixed seed makes results
reproducible across runs and across learners — this is the standard
"simulate data with a known true effect, show the estimator recovers it"
pattern used throughout applied econometrics teaching. No TODO here: data
generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# True DGP coefficients: [intercept, x1, x2]. Every scenario shares this
# ground truth so estimates are always compared against the same target.
TRUE_BETA = np.array([2.0, -1.5, 0.75])


@dataclass
class OLSData:
    """One simulated OLS dataset."""

    X: np.ndarray  # (n, k) design matrix; column 0 is the intercept (all ones)
    y: np.ndarray  # (n,) response
    beta_true: np.ndarray  # (k,) DGP coefficients
    cluster_id: np.ndarray | None = None  # (n,) cluster labels, only when scenario == "clustered"

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        cols = {f"x{i}": self.X[:, i] for i in range(1, self.X.shape[1])}
        df = pd.DataFrame({"y": self.y, **cols})
        if self.cluster_id is not None:
            df["cluster_id"] = self.cluster_id
        return df


def load_dataset(scenario: str = "homoskedastic", n: int = 200, seed: int = 0) -> OLSData:
    """Generate a synthetic OLS dataset for the given Gauss-Markov scenario.

    scenario:
      - "homoskedastic":   iid Normal(0, sigma^2) errors — every Gauss-Markov
        assumption (A1-A5) holds, so OLS is BLUE and the classical variance
        formula is exactly right.
      - "heteroskedastic": error scale grows multiplicatively with x1
        (sigma_i = exp(0.6 * x1)) — violates A5. The dependence is
        *monotone* in x1 on purpose: Breusch-Pagan regresses squared
        residuals on X itself, so a symmetric form like |x1| would be
        invisible to it however strong the heteroskedasticity is.
      - "clustered":       errors share a per-cluster shock on top of an
        idiosyncratic term — violates the independence part of A4.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])

    cluster_id = None
    if scenario == "homoskedastic":
        errors = rng.normal(scale=1.0, size=n)
    elif scenario == "heteroskedastic":
        sigma_i = np.exp(0.6 * x1)
        errors = rng.normal(scale=1.0, size=n) * sigma_i
    elif scenario == "clustered":
        n_clusters = max(4, n // 20)
        cluster_id = rng.integers(0, n_clusters, size=n)
        cluster_shock = rng.normal(scale=1.5, size=n_clusters)[cluster_id]
        errors = cluster_shock + rng.normal(scale=0.5, size=n)
    else:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    y = X @ TRUE_BETA + errors
    return OLSData(X=X, y=y, beta_true=TRUE_BETA, cluster_id=cluster_id)
