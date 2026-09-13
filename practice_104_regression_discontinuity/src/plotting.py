"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) — it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the estimators that feed these plots are. Every function here
takes already-computed arrays (bin means, fitted lines, bandwidth grids) —
none of them fit anything themselves.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def rdd_scatter_plot(
    running: np.ndarray,
    outcome: np.ndarray,
    cutoff: float,
    bin_centers: np.ndarray | None = None,
    bin_means: np.ndarray | None = None,
    fit_left: tuple[np.ndarray, np.ndarray] | None = None,
    fit_right: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Regression discontinuity",
):
    """The canonical RDD figure: raw scatter, binned means, and separate
    local-linear fits either side of the cutoff, with a vertical cutoff line.

    `fit_left`/`fit_right` are each `(x_grid, y_grid)` pairs — the local
    linear fit evaluated on a grid restricted to that side of the cutoff,
    so the two fits are drawn as visibly distinct lines that need not
    agree at the cutoff (that gap *is* the RDD estimate).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(running, outcome, alpha=0.15, label="raw data")
    if bin_centers is not None and bin_means is not None:
        ax.scatter(bin_centers, bin_means, alpha=0.9, label="binned means")
    if fit_left is not None:
        ax.plot(fit_left[0], fit_left[1], linewidth=2.0, label="local linear fit (left)")
    if fit_right is not None:
        ax.plot(fit_right[0], fit_right[1], linewidth=2.0, label="local linear fit (right)")
    ax.axvline(cutoff, linestyle="--")
    ax.set_xlabel("Running variable")
    ax.set_ylabel("Outcome")
    ax.set_title(title)
    ax.legend()
    return fig


def bandwidth_sensitivity_plot(
    bandwidths: np.ndarray,
    estimates: np.ndarray,
    selected_bandwidth: float | None = None,
    true_value: float | None = None,
    title: str = "RDD estimate across bandwidths",
):
    """Sensitivity plot: the sharp/fuzzy RDD estimate as a function of the
    bandwidth. A stable plateau around the selected bandwidth is reassuring;
    an estimate that swings wildly means the result is bandwidth-fragile."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bandwidths, estimates, marker="o")
    if selected_bandwidth is not None:
        ax.axvline(selected_bandwidth, linestyle="--", label="selected bandwidth")
    if true_value is not None:
        ax.axhline(true_value, linestyle=":", label="true value")
    ax.set_xlabel("Bandwidth")
    ax.set_ylabel("Estimated jump")
    ax.set_title(title)
    ax.legend()
    return fig


def mccrary_density_plot(
    bin_centers: np.ndarray,
    bin_density: np.ndarray,
    cutoff: float,
    fit_left: tuple[np.ndarray, np.ndarray] | None = None,
    fit_right: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "McCrary density test",
):
    """Binned density of the running variable either side of the cutoff,
    with local linear density fits — a visible jump here is what the
    McCrary test formalizes into a p-value."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(bin_centers, bin_density, alpha=0.8, label="binned density")
    if fit_left is not None:
        ax.plot(fit_left[0], fit_left[1], linewidth=2.0, label="fit (left)")
    if fit_right is not None:
        ax.plot(fit_right[0], fit_right[1], linewidth=2.0, label="fit (right)")
    ax.axvline(cutoff, linestyle="--")
    ax.set_xlabel("Running variable")
    ax.set_ylabel("Estimated density")
    ax.set_title(title)
    ax.legend()
    return fig
