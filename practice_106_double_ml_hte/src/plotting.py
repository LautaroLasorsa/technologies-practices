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


def bias_comparison_plot(
    biases: dict[str, np.ndarray],
    title: str = "Bias across estimators (Monte Carlo)",
):
    """Box plot of theta_hat - true_ATE across simulation replications, one
    box per estimator. A box centered away from zero is a biased estimator;
    the box's width reflects sampling variance, not bias."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(biases.keys())
    data = [biases[k] for k in labels]
    ax.boxplot(data, labels=labels)
    ax.axhline(0.0, linestyle="--")
    ax.set_ylabel("theta_hat - true ATE")
    ax.set_title(title)
    return fig


def cate_calibration_scatter(
    tau_true: np.ndarray,
    tau_hat: np.ndarray,
    title: str = "CATE calibration: estimated vs. true",
):
    """Scatter of estimated CATE against the known true CATE, with the y=x
    line marking perfect calibration -- points scattered tightly around the
    diagonal indicate a well-calibrated heterogeneity estimator."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(tau_true, tau_hat, alpha=0.4)
    lo = float(min(tau_true.min(), tau_hat.min()))
    hi = float(max(tau_true.max(), tau_hat.max()))
    ax.plot([lo, hi], [lo, hi], linestyle="--")
    ax.set_xlabel("true CATE tau(x)")
    ax.set_ylabel("estimated CATE tau_hat(x)")
    ax.set_title(title)
    return fig


def policy_value_curve_plot(
    budgets: np.ndarray,
    values: dict[str, np.ndarray],
    title: str = "Policy value vs. treatment budget",
):
    """Policy value as the treatment budget varies, one line per policy
    (e.g. "CATE-targeted" vs. a random-assignment baseline) -- the value a
    budget-constrained decision-maker actually cares about."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, v in values.items():
        ax.plot(budgets, v, marker="o", label=label)
    ax.set_xlabel("treatment budget (fraction of units treated)")
    ax.set_ylabel("policy value  V(pi) = E[Y | pi]")
    ax.set_title(title)
    ax.legend()
    return fig
