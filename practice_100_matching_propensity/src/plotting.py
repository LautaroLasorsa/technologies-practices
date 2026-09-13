"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) with no interop with real
matplotlib `Axes`/`Figure` objects, so every plot here is built with
`xy.pyplot` calls only. No TODO here: plotting is not the taught technique,
the estimators that feed these plots are.

Two `xy` limitations are worked around here (see CLAUDE.md Notes):
  - `xy` has no native KDE/density mark, so `overlap_density_plot` computes
    the density curves itself with `scipy.stats.gaussian_kde` and hands xy
    plain numeric (x, y) line series.
  - `xy`'s categorical-axis handling has an open upstream bug
    (reflex-dev/xy#471), so `love_plot` and `estimate_comparison_plot` never
    pass category labels as plotted data — they plot against a plain
    `np.arange(...)` position and attach the labels with `ax.set_yticklabels`
    instead, exactly like `practice_097`'s `coefficient_plot`.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt
from scipy.stats import gaussian_kde


def love_plot(
    names: list[str],
    smd_before: np.ndarray,
    smd_after: np.ndarray,
    after_label: str = "after",
    title: str = "Covariate balance (Love plot)",
):
    """Standardized mean differences before vs. after adjustment, one row
    per covariate. Points inside +-0.1 are conventionally "balanced"."""
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(names) + 1.5))
    ax.scatter(smd_before, y, label="before", alpha=0.7)
    ax.scatter(smd_after, y, label=after_label, alpha=0.9)
    ax.axvline(0.0, linestyle="-")
    ax.axvline(0.1, linestyle="--")
    ax.axvline(-0.1, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(names)
    ax.set_xlabel("Standardized mean difference")
    ax.set_title(title)
    ax.legend()
    return fig


def overlap_density_plot(
    ps_treated: np.ndarray, ps_control: np.ndarray, title: str = "Propensity score overlap"
):
    """Overlapping KDE curves of the propensity score by treatment arm —
    the standard visual check for the common-support/overlap assumption."""
    grid = np.linspace(0.0, 1.0, 200)
    dens_t = gaussian_kde(ps_treated)(grid)
    dens_c = gaussian_kde(ps_control)(grid)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, dens_t, label="treated")
    ax.plot(grid, dens_c, label="control")
    ax.set_xlabel("Propensity score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return fig


def estimate_comparison_plot(
    labels: list[str],
    estimates: np.ndarray,
    ci_lo: np.ndarray,
    ci_hi: np.ndarray,
    title: str = "ATE estimates: experimental benchmark vs. observational methods",
):
    """Horizontal comparison of point estimates with asymmetric CIs, one row
    per method — the plot the LaLonde replication is building toward."""
    y = np.arange(len(labels))
    estimates = np.asarray(estimates, dtype=float)
    err_lo = estimates - np.asarray(ci_lo, dtype=float)
    err_hi = np.asarray(ci_hi, dtype=float) - estimates
    fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(labels) + 1.5))
    ax.errorbar(estimates, y, xerr=[err_lo, err_hi], fmt="o", capsize=3)
    ax.axvline(0.0, linestyle="--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Estimated ATE ($, re78)")
    ax.set_title(title)
    return fig
