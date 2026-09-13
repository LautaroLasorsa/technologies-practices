"""Synthetic RCT datasets with a full science table.

Rubin's potential-outcomes framework defines the causal effect for unit `i`
as `tau_i = Y_i(1) - Y_i(0)` — the difference between what would have
happened under treatment and under control, *for the same unit*. In any
real experiment only one of the two is ever observed (the "fundamental
problem of causal inference"). Because this is synthetic data, we can
cheat and keep both `Y_i(0)` and `Y_i(1)` around as the "science table" —
letting every phase compare its estimate against the true, fully-known
average treatment effect (ATE) instead of against nothing. A fixed seed
makes results reproducible across runs and across learners. No TODO here:
data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RCTData:
    """One simulated randomized-experiment dataset, science table included."""

    y0: np.ndarray  # (n,) potential outcome under control, Y_i(0)
    y1: np.ndarray  # (n,) potential outcome under treatment, Y_i(1)
    x: np.ndarray  # (n,) pre-period covariate, correlated with y0/y1 (for CUPED)
    treatment: np.ndarray  # (n,) 0/1 actual random assignment
    tau_true: float  # true ATE = mean(y1 - y0)

    @property
    def y_observed(self) -> np.ndarray:
        """The single potential outcome actually revealed for each unit —
        the only column a real experiment would ever hand you."""
        return np.where(self.treatment == 1, self.y1, self.y0)

    def science_table(self) -> pd.DataFrame:
        """Full table with both potential outcomes — only visible because
        the data is synthetic. Real experiments never have this table."""
        return pd.DataFrame(
            {
                "x": self.x,
                "y0": self.y0,
                "y1": self.y1,
                "tau_i": self.y1 - self.y0,
                "treatment": self.treatment,
                "y_observed": self.y_observed,
            }
        )


def load_dataset(n: int = 500, tau: float = 2.0, seed: int = 0) -> RCTData:
    """Generate a synthetic RCT with a known, constant true ATE `tau`.

    `x` is a pre-period covariate (e.g. last week's value of the same
    metric) correlated with the outcome, used by CUPED in Phase 5. `y0`
    is generated from `x` plus idiosyncratic noise; `y1 = y0 + tau + noise`
    adds a small amount of individual-level heterogeneity around the
    constant effect so `tau_i` isn't perfectly identical for every unit.
    Treatment is assigned by **complete randomization**: exactly `n // 2`
    units are treated, chosen uniformly at random — the classical Neyman
    design every formula in this practice assumes.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(loc=50.0, scale=10.0, size=n)
    y0 = 0.6 * x + rng.normal(scale=8.0, size=n)
    y1 = y0 + tau + rng.normal(scale=1.0, size=n)

    treated = np.zeros(n, dtype=int)
    treated_idx = rng.permutation(n)[: n // 2]
    treated[treated_idx] = 1

    return RCTData(y0=y0, y1=y1, x=x, treatment=treated, tau_true=tau)
