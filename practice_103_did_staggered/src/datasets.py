"""Synthetic staggered-adoption panel with known, heterogeneous, dynamic
treatment effects.

Three cohorts ("early", "mid", "late") adopt treatment at different times,
plus a "never" cohort that stays untreated for the whole panel. Every
cohort's treatment effect starts at the same base level the moment it
switches on (`TAU_BASE`), but only the "early" cohort's effect keeps
growing the longer it stays treated (`TAU_SLOPE`) — "mid" and "late" have
flat, constant effects. All cohorts share the same unit-fixed-effect
distribution and the same common linear time trend, so the untreated
potential-outcome path is *exactly* parallel across cohorts by
construction: parallel trends holds by design, isolating "staggered timing
+ dynamic effects" as the only thing that can break an estimator here.

This calibration (see `_02_answer-xy.md`-adjacent research and Goodman-Bacon
2021 / de Chaisemartin & D'Haultfoeuille 2020) is picked specifically so
that naive two-way-fixed-effects (Phase 3) returns an estimate *outside the
range of every true effect in the data — with the wrong sign* — even
though every single true treatment effect is strictly positive. No TODO
here: the DGP is infrastructure the later phases' estimators are judged
against, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Cohort -> first treated period (g). "never" never switches on; it is
# encoded as +inf so "is this unit treated at time t" is always `t >= g`
# (False for every finite t) without a special case anywhere downstream.
COHORT_TIMING: dict[str, float] = {"early": 2, "mid": 5, "late": 8, "never": np.inf}
N_PERIODS = 11  # t = 0 .. 10
N_UNITS_PER_COHORT = 80

# True dynamic-effect path: tau(cohort, e) = TAU_BASE[cohort] + TAU_SLOPE[cohort] * e,
# for event time e = t - g >= 0 (0 for e < 0, i.e. before treatment).
# "early" grows steeply post-treatment; "mid"/"late" are flat at the same
# starting level.
TAU_BASE: dict[str, float] = {"early": 1.0, "mid": 1.0, "late": 1.0}
TAU_SLOPE: dict[str, float] = {"early": 4.0, "mid": 0.0, "late": 0.0}

# Sentinel used only when handing the panel to the `differences` package,
# which requires the cohort column to be finite numeric (never-treated
# entities must have a cohort value strictly after the last observed
# period, not +inf).
NEVER_TREATED_SENTINEL = 10_000


def true_effect(cohort: str, event_time: float) -> float:
    """The known, ground-truth treatment effect for `cohort` at relative
    event time `event_time` (= t - g). Zero before treatment starts."""
    if event_time < 0:
        return 0.0
    return TAU_BASE[cohort] + TAU_SLOPE[cohort] * event_time


@dataclass
class StaggeredPanel:
    """A long-format staggered-adoption panel plus its ground truth."""

    df: pd.DataFrame  # columns: unit, time, cohort, g, treated, event_time, y
    true_att_by_cell: pd.DataFrame  # columns: g, t, true_att — ground truth for every (g, t) with t >= g


def load_staggered_panel(seed: int = 0) -> StaggeredPanel:
    """Simulate the staggered-adoption panel described in this module's
    docstring: one row per (unit, time), with `y` driven by a unit fixed
    effect, a common linear time trend, the unit's true treatment effect
    (0 if not yet treated), and idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    unit_id = 0
    for cohort, g in COHORT_TIMING.items():
        unit_fe = rng.normal(scale=1.0, size=N_UNITS_PER_COHORT)
        for u in range(N_UNITS_PER_COHORT):
            uid = unit_id
            unit_id += 1
            for t in range(N_PERIODS):
                treated = np.isfinite(g) and t >= g
                event_time = (t - g) if treated else -1
                tau = true_effect(cohort, event_time) if treated else 0.0
                y = 5.0 + unit_fe[u] + 0.2 * t + tau + rng.normal(scale=0.7)
                rows.append(
                    {
                        "unit": uid,
                        "time": t,
                        "cohort": cohort,
                        "g": g,
                        "treated": int(treated),
                        "event_time": event_time,
                        "y": y,
                    }
                )
    df = pd.DataFrame(rows)

    true_cells = pd.DataFrame(
        [
            {"g": g, "t": t, "true_att": true_effect(cohort, t - g)}
            for cohort, g in COHORT_TIMING.items()
            if np.isfinite(g)
            for t in range(int(g), N_PERIODS)
        ]
    )
    return StaggeredPanel(df=df, true_att_by_cell=true_cells)


def load_2x2_subset(panel: StaggeredPanel, cohort: str = "mid", control: str = "never") -> pd.DataFrame:
    """Extract a clean canonical 2x2 slice: one treated cohort vs. one
    control cohort, restricted to `cohort`'s treatment period `g` and the
    single period right before it. Used by Phase 1 (the textbook 2x2 case,
    where "the period right before treatment" and "the period of
    treatment" are the whole story).
    """
    g = COHORT_TIMING[cohort]
    pre, post = int(g) - 1, int(g)
    sub = panel.df[(panel.df["cohort"].isin([cohort, control])) & (panel.df["time"].isin([pre, post]))].copy()
    sub["treat"] = (sub["cohort"] == cohort).astype(int)
    sub["post"] = (sub["time"] == post).astype(int)
    return sub.reset_index(drop=True)


def load_event_study_subset(panel: StaggeredPanel, cohort: str = "mid", control: str = "never") -> pd.DataFrame:
    """Extract a multi-period slice for one treated cohort vs. one control
    cohort, covering every period in the panel (not just pre/post) so
    Phase 2 can test pre-trends as well as the post-treatment path.
    """
    g = COHORT_TIMING[cohort]
    sub = panel.df[panel.df["cohort"].isin([cohort, control])].copy()
    sub["treat"] = (sub["cohort"] == cohort).astype(int)
    sub["rel_time"] = np.where(sub["treat"] == 1, sub["time"] - g, np.nan)
    return sub.reset_index(drop=True)


def to_differences_cohort_column(g: pd.Series) -> pd.Series:
    """Map our `+inf`-for-never-treated convention to the finite sentinel
    the `differences` package expects (a cohort value strictly after the
    panel's last period)."""
    return g.replace(np.inf, NEVER_TREATED_SENTINEL).astype(int)
