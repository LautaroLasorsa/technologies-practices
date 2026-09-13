"""Synthetic IV datasets with known ground truth.

Two data-generating processes, matched to the two halves of this practice:

- `simulate_iv_linear`: a just-identified linear model with one endogenous
  regressor and one excluded instrument, whose *relevance* (the first-stage
  coefficient `pi`) can be dialed from strong to weak. Used by Phases 1-3
  (2SLS mechanics, validation, and the weak-instrument problem). The
  treatment effect is homogeneous here on purpose -- Phases 1-3 are about
  2SLS's mechanics and finite-sample behavior, not effect heterogeneity.
- `simulate_iv_binary`: a binary encouragement design with three complier
  types (never-taker, complier, always-taker; monotonicity assumed, so no
  defiers) and a *heterogeneous* treatment effect. Used by Phase 4 (LATE
  vs. ATE) -- this is what makes "IV identifies the complier-average
  effect, not the population-average effect" a fact about the data instead
  of an abstract claim.

A fixed seed makes every run reproducible; every DGP includes an
unobserved confounder correlated with treatment, so plain OLS is always
biased -- that bias is the reason an instrument is needed at all. No TODO
here: data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# True structural effect of `endog` on `y` in the linear DGP. Homogeneous
# by construction, so it plays the role of "the one number 2SLS is after"
# in Phases 1-3, before Phase 4 shows effects need not be homogeneous.
TRUE_BETA = 2.0


@dataclass
class IVLinearData:
    """One simulated just-identified IV dataset (single endogenous regressor,
    single excluded instrument)."""

    exog: np.ndarray         # (n, 1) included exogenous regressors -- intercept only
    endog: np.ndarray        # (n,) endogenous regressor D
    instruments: np.ndarray  # (n, 1) excluded instrument Z
    y: np.ndarray            # (n,) outcome
    beta_true: float         # true causal effect of D on y

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        return pd.DataFrame({"z": self.instruments[:, 0], "d": self.endog, "y": self.y})


def simulate_iv_linear(n: int = 500, pi: float = 1.0, seed: int = 0) -> IVLinearData:
    """Generate a just-identified linear IV dataset with a known causal effect.

    `pi` is the first-stage coefficient on the excluded instrument -- the
    instrument's *relevance*. `pi` far from 0 is a strong instrument; `pi`
    near 0 is weak (swept across Phase 3's weak-instrument simulation). An
    unobserved confounder `v` enters both the endogenous regressor and the
    outcome, so plain OLS of `y` on `d` is biased -- `v` is the whole
    reason an instrument is needed.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    v = rng.normal(size=n)  # unobserved confounder -- correlated with endog AND y
    d = pi * z + 1.0 * v + rng.normal(scale=1.0, size=n)
    y = 1.0 + TRUE_BETA * d + 1.5 * v + rng.normal(scale=1.0, size=n)

    exog = np.ones((n, 1))
    instruments = z.reshape(-1, 1)
    return IVLinearData(exog=exog, endog=d, instruments=instruments, y=y, beta_true=TRUE_BETA)


@dataclass
class IVBinaryData:
    """One simulated binary-instrument / binary-treatment IV dataset with
    known complier types -- what makes LATE vs. ATE visible instead of
    abstract."""

    z: np.ndarray             # (n,) instrument, 0/1
    d: np.ndarray             # (n,) treatment actually received, 0/1
    y: np.ndarray             # (n,) outcome
    complier_type: np.ndarray  # (n,) one of "never", "complier", "always"
    ate_true: float           # true population average treatment effect
    late_true: float          # true average effect among compliers only

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        return pd.DataFrame({"z": self.z, "d": self.d, "y": self.y, "type": self.complier_type})


def simulate_iv_binary(n: int = 4000, complier_share: float = 0.4, seed: int = 0) -> IVBinaryData:
    """Generate a binary encouragement design with three complier types
    (monotonicity assumed -> no defiers).

    "never"-takers never get treated regardless of `z`; "always"-takers
    always do; "complier"s take treatment exactly when `z` == 1. Each type
    has its own individual treatment effect on `y` (never: 0.0, complier:
    ~3.0, always: 8.0 -- always-takers benefit far more, but that effect is
    never revealed by any instrument-driven variation). An unobserved
    "ability" shifts `y` directly and skews higher for always-takers and
    lower for never-takers -- the source of naive OLS's selection bias.
    `ate_true` averages every type's effect; `late_true` averages only the
    compliers' -- the one IV can actually identify.
    """
    rng = np.random.default_rng(seed)
    remainder = (1.0 - complier_share) / 2.0
    complier_type = rng.choice(
        ["never", "complier", "always"],
        size=n,
        p=[remainder, complier_share, remainder],
    )
    z = rng.integers(0, 2, size=n)

    d = np.where(
        complier_type == "complier",
        z,
        np.where(complier_type == "always", 1, 0),
    )

    ability = rng.normal(size=n)
    ability += np.where(complier_type == "always", 0.8, np.where(complier_type == "never", -0.8, 0.0))

    tau_i = np.select(
        [complier_type == "never", complier_type == "complier", complier_type == "always"],
        [0.0, 3.0 + rng.normal(scale=0.3, size=n), 8.0],
    )
    y = 1.0 + tau_i * d + 1.2 * ability + rng.normal(scale=1.0, size=n)

    ate_true = float(np.mean(tau_i))
    late_true = float(np.mean(tau_i[complier_type == "complier"]))
    return IVBinaryData(z=z, d=d, y=y, complier_type=complier_type, ate_true=ate_true, late_true=late_true)
