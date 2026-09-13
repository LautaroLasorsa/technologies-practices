"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "DAG drawing" note) — it has no interop with real
matplotlib `Axes`/`Figure` objects, so every plot in this practice,
including the DAGs themselves, is built with `xy.pyplot` calls only: nodes
as `ax.scatter` points + `ax.text` labels, edges as `ax.annotate("", ...)`
arrows — all part of `xy.pyplot`'s matplotlib-compatible 2-D Axes surface,
so no separate graph-layout library is needed for these small (3-5 node),
hand-laid-out DAGs. No TODO here: plotting is not the taught technique, the
identification logic that feeds these plots is.
"""
from __future__ import annotations

import numpy as np
import xy.pyplot as plt

from .datasets import Graph


def draw_dag(graph: Graph, positions: dict[str, tuple[float, float]], title: str = "DAG"):
    """Draw a small DAG: one `ax.scatter` point per node (placed at its fixed
    `positions` coordinate), an `ax.text` label above each, and one
    `ax.annotate("", ...)` arrow per directed edge."""
    fig, ax = plt.subplots(figsize=(5, 4))
    xs = [positions[n][0] for n in positions]
    ys = [positions[n][1] for n in positions]
    ax.scatter(xs, ys, s=800, zorder=2)
    for node, (x, y) in positions.items():
        ax.text(x, y + 0.12, node, ha="center", va="bottom", fontsize=12)
    for parent, children in graph.items():
        for child in children:
            x0, y0 = positions[parent]
            x1, y1 = positions[child]
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", lw=1.5),
            )
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-1.0, 1.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    return fig


def estimate_vs_truth_plot(
    labels: list[str],
    estimates: list[float],
    true_value: float,
    title: str = "Effect estimate vs. truth, by adjustment set",
):
    """One point per adjustment-set label, with a horizontal reference line
    at the true effect — the headline "controlling for more is not safer"
    figure: some adjustment sets land on the line, others don't."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.scatter(x, estimates, s=60, zorder=2)
    ax.axhline(true_value, linestyle="--", label="true effect")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("estimated effect")
    ax.set_title(title)
    ax.legend()
    return fig


def bias_magnitude_plot(
    labels: list[str],
    bias_values: list[float],
    title: str = "Bias introduced by conditioning on a collider",
):
    """Bar chart of induced bias magnitude, one bar per scenario/condition —
    used to compare how badly collider-stratification bias bites across the
    collider and M-bias scenarios."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax.bar(x, bias_values)
    ax.axhline(0.0, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("bias (adjusted - unadjusted estimate)")
    ax.set_title(title)
    return fig
