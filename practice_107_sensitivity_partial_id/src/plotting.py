"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) -- it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice is built
with `xy.pyplot` calls only. No TODO here: plotting is not the taught
technique, the bounds/statistics that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt


def sensitivity_contour_plot(
    gamma_grid: np.ndarray,
    delta_grid: np.ndarray,
    pvalue_grid: np.ndarray,
    alpha: float = 0.05,
    title: str = "Sensitivity contour: p-value bound over confounding strength",
):
    """Contour of a sensitivity statistic (e.g. Rosenbaum's upper-bound
    p-value) over a 2D grid of confounding strength in two directions, with
    the "explained away" frontier (where the bound crosses `alpha`) marked.

    `gamma_grid`, `delta_grid` are 1D axis values; `pvalue_grid` is a 2D
    array of shape `(len(delta_grid), len(gamma_grid))`.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cs = ax.contourf(gamma_grid, delta_grid, pvalue_grid, levels=20, cmap="viridis")
    ax.contour(gamma_grid, delta_grid, pvalue_grid, levels=[alpha], colors="red", linewidths=2)
    fig.colorbar(cs, ax=ax, label="upper-bound p-value")
    ax.set_xlabel("confounder-treatment association (Gamma)")
    ax.set_ylabel("confounder-outcome association")
    ax.set_title(title)
    return fig


def manski_bound_narrowing_plot(
    bound_labels: list[str],
    lower_bounds: list[float],
    upper_bounds: list[float],
    naive_estimate: float,
    naive_ci: tuple[float, float],
    title: str = "Bounds narrow as assumptions are added",
):
    """Horizontal bound-width chart: one row per assumption regime (e.g.
    "no assumption" -> "monotonicity"), each drawn as a bar from its lower
    to upper bound, plus the naive point estimate + sampling CI marked as a
    single narrow reference row -- so the contrast between "sampling error
    only" and "identification error" is visually obvious.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(bound_labels))
    widths = [u - l for l, u in zip(lower_bounds, upper_bounds)]
    ax.barh(y, widths, left=lower_bounds, height=0.5, alpha=0.8)
    ax.errorbar(
        [naive_estimate],
        [len(bound_labels)],
        xerr=[[naive_estimate - naive_ci[0]], [naive_ci[1] - naive_estimate]],
        fmt="o",
        color="red",
        capsize=4,
        label="naive point estimate + 95% CI (sampling error only)",
    )
    ax.set_yticks(list(y) + [len(bound_labels)])
    ax.set_yticklabels(bound_labels + ["naive CI"])
    ax.axvline(0.0, linestyle="--", color="gray")
    ax.set_xlabel("treatment effect")
    ax.set_title(title)
    ax.legend()
    return fig


def e_value_plot(
    point_rr: float,
    ci_limit_rr: float,
    e_value_point: float,
    e_value_ci: float,
    title: str = "E-values: strength of confounding needed to explain away the effect",
):
    """Ding & VanderWeele-style display: the point estimate and CI-limit
    risk ratios plotted against the confounder-association strength
    (E-value) that would be needed, on both the confounder-treatment and
    confounder-outcome axes, to fully explain away each one."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter([1.0], [point_rr], s=80, label=f"point estimate RR={point_rr:.2f}")
    ax.scatter([1.0], [ci_limit_rr], s=80, marker="^", label=f"CI limit RR={ci_limit_rr:.2f}")
    ax.axhline(point_rr, linestyle=":", alpha=0.5)
    ax.axhline(ci_limit_rr, linestyle=":", alpha=0.5)
    ax.annotate(f"E-value = {e_value_point:.2f}", xy=(1.0, point_rr), xytext=(1.05, point_rr))
    ax.annotate(f"E-value = {e_value_ci:.2f}", xy=(1.0, ci_limit_rr), xytext=(1.05, ci_limit_rr))
    ax.set_xlim(0.9, 1.5)
    ax.set_xticks([])
    ax.set_ylabel("risk ratio")
    ax.set_title(title)
    ax.legend()
    return fig
