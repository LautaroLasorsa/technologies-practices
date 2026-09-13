"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) — it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the estimators and inference procedures that feed these plots
are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def randomization_distribution_plot(
    null_stats: np.ndarray,
    observed_stat: float,
    title: str = "Randomization distribution under the sharp null",
):
    """Histogram of the diff-in-means statistic across many random
    reassignments of the treatment label, with the actually-observed
    statistic marked — the picture Fisher's exact test is built on."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(null_stats, bins=40, alpha=0.8)
    ax.axvline(observed_stat, linestyle="--", label="observed statistic")
    ax.set_xlabel("difference in means")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    return fig


def power_curve_plot(
    n_grid: np.ndarray,
    power_grid: np.ndarray,
    target_power: float = 0.8,
    title: str = "Power vs. sample size",
):
    """Statistical power as a function of per-arm sample size for a fixed
    effect size, with a reference line at the conventional 80% target."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(n_grid, power_grid, marker="o")
    ax.axhline(target_power, linestyle="--", label=f"{target_power:.0%} power")
    ax.set_xlabel("sample size per arm")
    ax.set_ylabel("power")
    ax.set_title(title)
    ax.legend()
    return fig


def cuped_variance_plot(
    var_raw: float,
    var_cuped: float,
    title: str = "CUPED variance reduction",
):
    """Bar chart comparing the outcome's variance before and after the
    CUPED covariate adjustment — the shorter bar is the whole point of
    CUPED: same estimator, smaller variance, for free."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["raw", "CUPED-adjusted"], [var_raw, var_cuped])
    ax.set_ylabel("Var(Y)")
    ax.set_title(title)
    return fig
