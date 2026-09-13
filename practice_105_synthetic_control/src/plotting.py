"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) — it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot here is built with
`xy.pyplot` calls only. No TODO here: plotting is not the taught technique,
the estimators that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def treated_vs_synthetic_plot(
    years: np.ndarray,
    y_treated: np.ndarray,
    y_synthetic: np.ndarray,
    treatment_year: int,
    title: str = "Treated vs. synthetic control",
):
    """California's actual outcome path against its synthetic counterpart,
    with a vertical marker at the treatment date — the picture the whole
    method exists to produce."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(years, y_treated, label="California (treated)", linewidth=2)
    ax.plot(years, y_synthetic, label="Synthetic California", linestyle="--", linewidth=2)
    ax.axvline(treatment_year, linestyle=":", color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Cigarette sales (packs per capita)")
    ax.set_title(title)
    ax.legend()
    return fig


def gap_plot(
    years: np.ndarray,
    gap: np.ndarray,
    treatment_year: int,
    title: str = "Gap: treated minus synthetic",
):
    """The estimated treatment effect over time — treated minus synthetic,
    which should hover near zero pre-treatment (that's the fit check) and
    move away from zero post-treatment (that's the effect)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, gap, linewidth=2)
    ax.axhline(0.0, linestyle="--", color="grey")
    ax.axvline(treatment_year, linestyle=":", color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap (packs per capita)")
    ax.set_title(title)
    return fig


def placebo_spaghetti_plot(
    years: np.ndarray,
    placebo_gaps: np.ndarray,
    true_gap: np.ndarray,
    treatment_year: int,
    title: str = "Placebo test: gap for every donor treated as if it were California",
):
    """Every donor's placebo gap path in thin grey, the true treated unit's
    gap path highlighted on top — the visual form of the permutation test:
    California's post-treatment gap should stand out from the null-
    distribution "spaghetti" of donors that were never actually treated.

    `placebo_gaps` is (n_donors, T); each row is one donor's gap path when
    that donor is (falsely) treated as the treated unit.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for row in placebo_gaps:
        ax.plot(years, row, color="grey", alpha=0.3, linewidth=1)
    ax.plot(years, true_gap, color="crimson", linewidth=2.5, label="California (true)")
    ax.axhline(0.0, linestyle="--", color="grey")
    ax.axvline(treatment_year, linestyle=":", color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("Gap (packs per capita)")
    ax.set_title(title)
    ax.legend()
    return fig
