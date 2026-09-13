"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) and is used for every plot *we*
compose: the observed-vs-counterfactual series, the cumulative-effect band,
and the posterior density of the total effect. `xy` has no native
statistical-density primitive (no KDE, no HDI), so both are computed here
with plain NumPy before handing xy a `line`/`area`/`fill_between` call.

None of the axes below are categorical (day index and effect magnitude are
both continuous), so xy's open low-cardinality-categorical-axis bug
(reflex-dev/xy#471, which bites horizontal dot/love plots) does not apply
here — noted explicitly since the scaffold spec flags it as a risk for this
practice's density plot; it turned out not to be one, precisely because we
never build a categorical axis.

The two arviz diagnostic plots this practice also produces (`plot_trace`,
`plot_posterior` in the notebook) draw onto a *real* `matplotlib.axes.Axes`
that arviz creates itself — xy cannot be substituted there, so those two
calls use `import matplotlib.pyplot as mplt` directly in the notebook, kept
entirely separate from the `xy.pyplot` calls in this module. No TODO here:
plotting is not the taught technique, the model and inference that feed
these plots are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def observed_vs_counterfactual_plot(
    t: np.ndarray,
    y: np.ndarray,
    intervention_day: int,
    counterfactual_mean: np.ndarray,
    counterfactual_lo: np.ndarray,
    counterfactual_hi: np.ndarray,
    title: str = "Observed vs. counterfactual",
):
    """Observed series (full range) against the post-period counterfactual
    forecast, with a shaded credible band and a vertical intervention marker
    — the CausalImpact-style headline plot."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, y, label="observed", color="black")
    t_post = t[intervention_day:]
    ax.plot(t_post, counterfactual_mean, label="counterfactual (posterior mean)", linestyle="--")
    ax.fill_between(t_post, counterfactual_lo, counterfactual_hi, alpha=0.25, label="94% credible band")
    ax.axvline(t[intervention_day], linestyle=":", color="red", label="intervention")
    ax.set_xlabel("day")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    return fig


def cumulative_effect_plot(
    t_post: np.ndarray,
    cumulative_mean: np.ndarray,
    cumulative_lo: np.ndarray,
    cumulative_hi: np.ndarray,
    true_cumulative_effect: float,
    title: str = "Posterior cumulative effect",
):
    """Cumulative effect posterior over the post-period: mean path, a
    per-day credible band, a zero reference line, and the (known, synthetic)
    ground truth for comparison."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_post, cumulative_mean, label="posterior mean")
    ax.fill_between(t_post, cumulative_lo, cumulative_hi, alpha=0.25, label="94% credible band")
    ax.axhline(0.0, linestyle="--", color="gray", label="no effect")
    ax.axhline(true_cumulative_effect, linestyle=":", color="red", label="true cumulative effect")
    ax.set_xlabel("day")
    ax.set_ylabel("cumulative effect")
    ax.set_title(title)
    ax.legend()
    return fig


def effect_posterior_density_plot(
    total_effect_draws: np.ndarray,
    hdi_lo: float,
    hdi_hi: float,
    true_cumulative_effect: float,
    n_bins: int = 40,
    title: str = "Posterior density of the total effect",
):
    """Density (via a plain NumPy histogram, since xy has no native KDE) of
    the total post-period effect, with the 94% HDI shaded and the true
    value marked. A histogram is a coarser density estimate than a KDE, but
    needs no extra dependency and is honest about being an estimate."""
    counts, edges = np.histogram(total_effect_draws, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(centers, counts, color="black", label="posterior density")
    in_hdi = (centers >= hdi_lo) & (centers <= hdi_hi)
    ax.fill_between(centers, 0, counts, where=in_hdi, alpha=0.35, label="94% HDI")
    ax.axvline(0.0, linestyle="--", color="gray", label="no effect")
    ax.axvline(true_cumulative_effect, linestyle=":", color="red", label="true cumulative effect")
    ax.set_xlabel("total post-period effect")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    return fig
