"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) -- it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the estimators that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xy.pyplot as plt


def event_study_plot(table: pd.DataFrame, title: str = "Event study") -> plt.Figure:
    """Point estimates with 95% CIs over event time -- a vertical line at
    t=0 (treatment starts) and a horizontal line at 0 (no effect).

    `table` needs columns `rel_time` (or `event_time`), `estimate` (or
    `att`), `ci_lo`, `ci_hi` (or `se`, from which a symmetric 95% CI is
    derived).
    """
    x_col = "rel_time" if "rel_time" in table.columns else "event_time"
    y_col = "estimate" if "estimate" in table.columns else "att"
    if "ci_lo" in table.columns:
        lo, hi = table["ci_lo"], table["ci_hi"]
    else:
        lo = table[y_col] - 1.96 * table["se"]
        hi = table[y_col] + 1.96 * table["se"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    yerr = np.vstack([table[y_col] - lo, hi - table[y_col]])
    ax.errorbar(table[x_col], table[y_col], yerr=yerr, fmt="o", capsize=3)
    ax.axvline(-0.5, linestyle="--")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Event time (periods since treatment)")
    ax.set_ylabel("Estimated effect")
    ax.set_title(title)
    return fig


def bacon_decomposition_plot(bacon: pd.DataFrame, title: str = "Goodman-Bacon decomposition") -> plt.Figure:
    """Scatter of each pairwise 2x2 DiD estimate against its weight, one
    point per comparison, colored by whether it is a clean or a "forbidden"
    (already-treated-as-control) comparison.

    `bacon` needs columns `weight`, `estimate`, `type` (values `"clean"` /
    `"forbidden"`), and `comparison` (a label per row).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind, marker in (("clean", "o"), ("forbidden", "x")):
        sub = bacon[bacon["type"] == kind]
        ax.scatter(sub["weight"], sub["estimate"], marker=marker, label=kind, s=60)
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Comparison weight")
    ax.set_ylabel("2x2 DiD estimate")
    ax.set_title(title)
    ax.legend()
    return fig


def method_comparison_plot(
    labels: list[str],
    estimates: list[float],
    ses: list[float | None],
    true_att: float,
    title: str = "TWFE vs. Callaway-Sant'Anna vs. truth",
) -> plt.Figure:
    """One point per estimation method (e.g. "TWFE", "Callaway-Sant'Anna"),
    with a 95% CI where a standard error is available, against a horizontal
    line marking the true overall ATT.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(len(labels))
    for xi, est, se in zip(x, estimates, ses):
        if se is not None:
            ax.errorbar([xi], [est], yerr=[1.96 * se], fmt="o", capsize=4, color="tab:blue")
        else:
            ax.scatter([xi], [est], color="tab:blue")
    ax.axhline(true_att, linestyle="--", label="true overall ATT")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Estimated overall effect")
    ax.set_title(title)
    ax.legend()
    return fig
