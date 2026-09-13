"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) — it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the estimators that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def coefficient_comparison_plot(
    names: list[str],
    estimate_by_method: dict[str, np.ndarray],
    se_by_method: dict[str, np.ndarray],
    truth: np.ndarray,
    title: str = "Pooled OLS vs. within vs. two-way FE",
):
    """Coefficient point estimates (95% CI) for several estimators, one
    offset row of points per method, with the true beta marked.

    `estimate_by_method` and `se_by_method` share the same keys (method
    name -> (k,) array); `truth` is the (k,) true beta these estimators are
    all trying to recover.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_methods = len(estimate_by_method)
    offsets = np.linspace(-0.2, 0.2, n_methods) if n_methods > 1 else [0.0]
    x = np.arange(len(names))
    for offset, method in zip(offsets, estimate_by_method):
        est = estimate_by_method[method]
        se = se_by_method[method]
        ax.errorbar(x + offset, est, yerr=1.96 * se, fmt="o", label=method, capsize=3)
    for xi, t in zip(x, truth):
        ax.plot([xi - 0.3, xi + 0.3], [t, t], linestyle="--", color="black", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Coefficient estimate (95% CI)")
    ax.set_title(title)
    ax.legend()
    return fig


def coverage_plot(
    coverage_by_method: dict[str, float],
    nominal: float = 0.95,
    title: str = "Empirical coverage of nominal 95% CIs",
):
    """Bar chart of empirical CI coverage rate per method, with the nominal
    coverage level marked — the picture that makes "ignoring clustering
    understates uncertainty" concrete."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    methods = list(coverage_by_method.keys())
    values = [coverage_by_method[m] for m in methods]
    ax.bar(methods, values)
    ax.axhline(nominal, linestyle="--", color="black", label=f"nominal ({nominal:.0%})")
    ax.set_ylabel("Empirical coverage rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    return fig
