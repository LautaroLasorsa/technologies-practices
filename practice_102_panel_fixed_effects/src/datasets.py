"""Synthetic panel datasets with known unit/time effects and a known true beta.

The data-generating process (DGP) is built so that every estimator this
practice implements has something to be checked against:

  - `alpha_i` (the unit effect) is correlated with `x1`'s unit-level mean,
    so **pooled OLS is biased** for beta1 (the classic omitted-variable-bias
    motivation for fixed effects).
  - `gamma_t` (the time effect) trends upward together with `x2`'s rollout,
    so **one-way (entity-only) FE is still biased** for beta2 unless time
    effects are also absorbed (the motivation for two-way FE).
  - `x2` ("treatment") is assigned at the **cluster** level (constant across
    units within a cluster-period, e.g. a regional policy) and the error
    term carries a **cluster-period shock**, so inference on beta2 is only
    valid when standard errors are clustered at the cluster level — the
    level at which the regressor of interest actually varies.

A fixed seed makes every run reproducible. No TODO here: data generation is
infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# True DGP coefficients on [x1, x2]. No intercept: with unit fixed effects
# in the picture an intercept is not separately identified (see CLAUDE.md,
# "Why the intercept disappears").
TRUE_BETA = np.array([1.5, -2.0])


@dataclass
class PanelData:
    """One simulated balanced panel, long format, plus DGP ground truth."""

    df: pd.DataFrame  # columns: unit_id, time_id, cluster_id, x1, x2, y
    beta_true: np.ndarray  # (2,) true [beta_x1, beta_x2]
    n_units: int
    n_periods: int
    n_clusters: int


def generate_panel(
    n_clusters: int = 8,
    units_per_cluster: int = 5,
    n_periods: int = 10,
    seed: int = 0,
) -> PanelData:
    """Generate a balanced synthetic panel with known unit/time effects.

    Layout: `n_clusters * units_per_cluster` units, each observed for
    `n_periods` consecutive periods (a balanced panel — every unit has
    every period, which is what lets Phase 2's two-way demeaning use the
    closed-form double-demeaning formula instead of iterative alternating
    projections).
    """
    rng = np.random.default_rng(seed)
    n_units = n_clusters * units_per_cluster
    obs_per_unit = n_periods

    unit_id = np.repeat(np.arange(n_units), obs_per_unit)
    time_id = np.tile(np.arange(n_periods), n_units)
    cluster_id_by_unit = np.arange(n_units) // units_per_cluster
    cluster_id = cluster_id_by_unit[unit_id]

    # x1: individual-level regressor with a unit-specific baseline (this is
    # what makes it correlated with alpha_i below).
    unit_baseline = rng.normal(size=n_units)
    x1 = unit_baseline[unit_id] + rng.normal(scale=0.6, size=n_units * obs_per_unit)

    # x2 ("treatment"): assigned at the cluster-period level — every unit in
    # a cluster shares the same x2 in a given period — with a rollout trend
    # that grows over time (correlated with gamma_t below).
    cluster_offset = rng.normal(scale=0.8, size=n_clusters)
    x2_cluster_period = (
        0.20 * np.arange(n_periods)[None, :]
        + cluster_offset[:, None]
        + rng.normal(scale=0.3, size=(n_clusters, n_periods))
    )
    x2 = x2_cluster_period[cluster_id, time_id]

    # Unit fixed effect: correlated with the unit's average x1 -> omitted
    # variable bias for pooled OLS, exactly what the within transformation
    # is built to remove.
    unit_mean_x1 = np.array(
        [x1[unit_id == u].mean() for u in range(n_units)]
    )
    alpha_i = 0.9 * unit_mean_x1 + rng.normal(scale=1.0, size=n_units)

    # Time fixed effect: a trend, correlated with x2's rollout -> one-way
    # (entity-only) FE is not enough; two-way FE is needed.
    gamma_t = 0.45 * np.arange(n_periods) + rng.normal(scale=0.3, size=n_periods)

    # Error: a cluster-period shock (shared by every unit in a cluster in a
    # given period — the source of within-cluster correlation) plus
    # idiosyncratic noise.
    cluster_period_shock = rng.normal(scale=0.9, size=(n_clusters, n_periods))
    idiosyncratic = rng.normal(scale=0.5, size=n_units * obs_per_unit)

    y = (
        TRUE_BETA[0] * x1
        + TRUE_BETA[1] * x2
        + alpha_i[unit_id]
        + gamma_t[time_id]
        + cluster_period_shock[cluster_id, time_id]
        + idiosyncratic
    )

    df = pd.DataFrame(
        {
            "unit_id": unit_id,
            "time_id": time_id,
            "cluster_id": cluster_id,
            "x1": x1,
            "x2": x2,
            "y": y,
        }
    )
    return PanelData(
        df=df,
        beta_true=TRUE_BETA,
        n_units=n_units,
        n_periods=n_periods,
        n_clusters=n_clusters,
    )


# Parameter presets used across the notebook and the benchmark/simulation
# harnesses, so every phase generates panels the same way.
SMALL_PANEL = dict(n_clusters=8, units_per_cluster=5, n_periods=10)
LARGE_PANEL = dict(n_clusters=200, units_per_cluster=25, n_periods=20)  # ~100k rows, for the speed benchmark
