"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) -- it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the estimators that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def first_stage_fit_plot(z: np.ndarray, d: np.ndarray, d_hat: np.ndarray, title: str = "First stage: D on Z"):
    """Scatter of the endogenous regressor D against the instrument Z, with
    the first-stage fitted line D_hat overlaid -- the steeper this line,
    the more of D's variation is explained by Z (relevance)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(z, d, alpha=0.4, label="observed D")
    order = np.argsort(z)
    ax.plot(z[order], d_hat[order], linestyle="-", label="fitted D_hat")
    ax.set_xlabel("Instrument Z")
    ax.set_ylabel("Endogenous regressor D")
    ax.set_title(title)
    ax.legend()
    return fig


def weak_instrument_plot(
    avg_f: np.ndarray,
    bias: dict[str, np.ndarray],
    coverage: np.ndarray,
    title: str = "2SLS under a weakening instrument",
):
    """Two-panel figure: bias of each method against the truth, and 95% CI
    coverage, both plotted against the average first-stage F statistic
    (log x-axis) as the instrument weakens -- the classic weak-instrument
    picture, including 2SLS's bias converging toward OLS's as F shrinks.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    for method, values in bias.items():
        ax1.plot(avg_f, values, marker="o", label=method)
    ax1.axhline(0.0, linestyle=":")
    ax1.axvline(10.0, linestyle=":", label="F = 10 rule of thumb")
    ax1.set_xscale("log")
    ax1.set_xlabel("First-stage F (avg. across replications)")
    ax1.set_ylabel("Bias (estimate - true beta)")
    ax1.set_title("Bias vs. instrument strength")
    ax1.legend()

    ax2.plot(avg_f, coverage, marker="o")
    ax2.axhline(0.95, linestyle=":", label="nominal 95%")
    ax2.axvline(10.0, linestyle=":", label="F = 10 rule of thumb")
    ax2.set_xscale("log")
    ax2.set_xlabel("First-stage F (avg. across replications)")
    ax2.set_ylabel("95% CI coverage of true beta")
    ax2.set_title("Coverage vs. instrument strength")
    ax2.legend()

    fig.suptitle(title)
    return fig


def estimate_comparison_plot(
    names: list[str],
    estimates: list[float],
    ses: list[float],
    reference_lines: dict[str, float],
    title: str = "OLS vs. 2SLS vs. truth",
):
    """Point estimates with 95% CI error bars for each method (e.g. OLS,
    2SLS/Wald), plus horizontal reference lines for known ground-truth
    values (e.g. true ATE, true LATE) -- the picture that makes "IV
    identifies a different parameter than OLS is even estimating" visible.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(names))
    ax.errorbar(x, estimates, yerr=1.96 * np.asarray(ses), fmt="o", capsize=4, label="estimate (95% CI)")
    for label, value in reference_lines.items():
        ax.axhline(value, linestyle="--", label=label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Effect estimate")
    ax.set_title(title)
    ax.legend()
    return fig
