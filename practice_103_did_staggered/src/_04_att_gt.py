"""Phase 4 -- the repair: Callaway-Sant'Anna group-time ATT(g,t).

Phase 3 showed the failure mode: one aggregate TWFE coefficient hides an
average over comparisons that should never have been made. Callaway &
Sant'Anna's (2021) fix is conceptually simple even though the estimator's
name sounds intimidating: instead of one coefficient, estimate *one clean
2x2 DiD per (cohort, period) cell* -- `ATT(g, t)`, the average effect on
cohort `g` at calendar time `t` -- always comparing against a group that is
genuinely untreated at both endpoints. No "forbidden" comparisons are ever
formed, because each cell is its own self-contained 2x2. Python's staggered
-DiD tooling is immature relative to R's `did` package (see this
practice's CLAUDE.md), so this phase hand-rolls the estimator directly on
top of `did_2x2` from Phase 1 -- which is the actual point of this
exercise, not a workaround for a missing library.

Run on its own to compute every ATT(g,t) cell and compare against ground
truth:
    uv run python -m src._04_att_gt
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._01_two_by_two_did import did_2x2  # used inside the TODO(human) below
from .datasets import load_staggered_panel


@dataclass
class AttGtResult:
    """One group-time ATT(g, t) cell."""

    g: int
    t: int
    att: float
    se: float
    n_treated: int
    n_control: int


# TODO(human) -- one group-time ATT(g, t) cell
# ---------------------------------------------------------------------------
# Goal: compute a single Callaway-Sant'Anna group-time average treatment
# effect on the treated, `ATT(g, t)`, for cohort `g` at calendar time `t`.
#
# Why this matters: `ATT(g, t)` is *exactly* Phase 1's 2x2 DiD -- comparing
# cohort `g`'s change in `y` from its last pre-treatment period (`g - 1`)
# to period `t`, against the control group's change over that same
# window -- applied to one specific (cohort, period) cell instead of a
# whole regression. Doing this once per cell, always against a genuinely
# untreated control, is what makes the Callaway-Sant'Anna estimator immune
# to Phase 3's "already-treated units contaminating the control group"
# problem: there is no pooled regression here for a forbidden comparison
# to sneak into.
#
# Steps:
#   1. base = g - 1 (the last pre-treatment period for cohort g).
#   2. Build the treated sub-frame: rows where `df["g"] == g` and
#      `df["time"]` is `base` or `t`.
#   3. Build the control sub-frame: rows where `df["cohort"] == control_group`
#      (the "never-treated" cohort, by default) and `df["time"]` is `base`
#      or `t`.
#   4. Concatenate the two, add `treat` (1 for the cohort-g rows) and
#      `post` (1 where `time == t`) columns -- exactly the shape `did_2x2`
#      (Phase 1) expects -- and call `did_2x2` on it.
#   5. Return an `AttGtResult` with `g`, `t`, the estimate and se from
#      `did_2x2`, and the treated/control group sizes (`len(...)` of each
#      sub-frame, counting both periods).
# ---------------------------------------------------------------------------
def att_gt(df: pd.DataFrame, g: int, t: int, control_group: str = "never") -> AttGtResult:
    """One group-time ATT(g, t): cohort `g`'s DiD from period `g - 1` to
    `t`, against `control_group` (default: the never-treated cohort).

    Returns an `AttGtResult`. Requires Phase 1's `did_2x2` to be implemented.
    """
    raise NotImplementedError("TODO(human): implement one ATT(g, t) cell")


def compute_all_att_gt(df: pd.DataFrame, control_group: str = "never") -> pd.DataFrame:
    """Loop `att_gt` over every valid (g, t) cell (t >= g, for every
    finite-g cohort) and collect the results into a tidy table. Fully
    scaffolded: the looping/bookkeeping is not the teaching point, one
    cell's estimator (`att_gt` above) is.
    """
    cohorts = sorted(g for g in df["g"].unique() if np.isfinite(g))
    max_t = int(df["time"].max())
    rows = []
    for g in cohorts:
        for t in range(int(g), max_t + 1):
            result = att_gt(df, int(g), t, control_group=control_group)
            rows.append(vars(result))
    return pd.DataFrame(rows)


def main() -> None:
    panel = load_staggered_panel(seed=0)
    try:
        table = compute_all_att_gt(panel.df)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")
        return
    merged = table.merge(panel.true_att_by_cell, on=["g", "t"])
    merged["abs_error"] = (merged["att"] - merged["true_att"]).abs()
    print(merged.to_string(index=False))
    print(f"\nMax |att_hat - att_true|: {merged['abs_error'].max():.3f}")


if __name__ == "__main__":
    main()
