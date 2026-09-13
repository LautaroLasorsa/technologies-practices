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


def residual_diagnostic_plot(
    fitted: np.ndarray, resid: np.ndarray, title: str = "Residuals vs fitted"
):
    """Residuals-vs-fitted scatter — the standard visual heteroskedasticity/
    nonlinearity check: a random horizontal band means A5 is plausible, a
    funnel or curve means it isn't."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(fitted, resid, alpha=0.6)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title(title)
    return fig


def coefficient_plot(
    names: list[str],
    estimate: np.ndarray,
    se_by_method: dict[str, np.ndarray],
    title: str = "Coefficient estimates by standard-error method",
):
    """Coefficient point estimates with 95% CI error bars, one offset row of
    points per standard-error method so the CIs can be compared side by side.

    `se_by_method` maps a method name (e.g. "classical", "HC3", "cluster")
    to a (k,) array of standard errors for the *same* point estimate.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_methods = len(se_by_method)
    offsets = np.linspace(-0.2, 0.2, n_methods) if n_methods > 1 else [0.0]
    x = np.arange(len(names))
    for offset, (method, se) in zip(offsets, se_by_method.items()):
        ax.errorbar(x + offset, estimate, yerr=1.96 * se, fmt="o", label=method, capsize=3)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Coefficient estimate (95% CI)")
    ax.set_title(title)
    ax.legend()
    return fig


def sampling_distribution_plot(beta_hats: np.ndarray, true_value: float, coef_name: str = "beta_1"):
    """Histogram of one coefficient's empirical sampling distribution across
    Monte Carlo replications, with the true value and the empirical mean
    marked — the picture the Gauss-Markov theorem is actually making a claim
    about."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(beta_hats, bins=30, alpha=0.8)
    ax.axvline(true_value, linestyle="--", label="true value")
    ax.axvline(float(np.mean(beta_hats)), linestyle=":", label="mean(beta_hat)")
    ax.set_xlabel(coef_name)
    ax.set_ylabel("count")
    ax.set_title(f"Sampling distribution of {coef_name} across replications")
    ax.legend()
    return fig
