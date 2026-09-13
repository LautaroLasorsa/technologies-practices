"""Phase 5 -- aggregating ATT(g, t) into numbers you can actually report.

Phase 4 produces one ATT estimate per (cohort, period) cell -- useful for
inspection, useless for a headline result or an event-study plot on its
own. Callaway & Sant'Anna's aggregation step turns the (g, t) grid into
two things applied papers actually report: an **event-study** path (one
number per relative time, averaged across every cohort that reaches that
relative time) and a **single overall ATT** (one number, averaged across
every post-treatment cell). Both aggregations are the same idea applied
along different axes of the same table: a cohort-size-weighted average.

Run on its own to compute both aggregations and compare against the
TWFE estimate from Phase 3:
    uv run python -m src._05_aggregate_att
"""
from __future__ import annotations

import numpy as np  # used inside the TODO(human) functions below
import pandas as pd

from ._03_twfe_and_bacon import fit_naive_twfe
from ._04_att_gt import compute_all_att_gt
from .datasets import load_staggered_panel


# TODO(human) -- aggregate ATT(g, t) into an event-study path and one overall ATT
# ---------------------------------------------------------------------------
# Goal: implement two cohort-size-weighted aggregations of the ATT(g, t)
# table from Phase 4 -- the two numbers Callaway-Sant'Anna papers actually
# report.
#
# Why this matters: a raw (g, t) grid answers "what happened to cohort g at
# time t", which is exactly right for diagnosing Phase 3's failure but not
# what anyone quotes as "the effect of the policy". Both aggregations below
# collapse the grid along a different axis using the same principle --
# weight each cell by how much treated data it represents (`n_treated`) so
# a cohort with more units counts for more, then average:
#   - `aggregate_event_study`: group cells by event time `e = t - g`
#     (how long a cohort has been treated), average `att` within each `e`
#     across every cohort that reaches it, weighted by `n_treated`.
#   - `aggregate_overall_att`: average `att` across *every* (g, t) cell,
#     weighted by `n_treated` -- the single-number summary.
# For both, propagate uncertainty with the standard weighted-independent-
# estimates formula `se = sqrt(sum(w_i^2 * se_i^2))` (treating each cell's
# ATT as independent is a simplification -- a full CS implementation uses a
# multiplier bootstrap across cells to account for their shared control
# group; out of scope here, see Callaway & Sant'Anna 2021 sec. 3.3).
#
# Steps (`aggregate_event_study`):
#   1. Add `event_time = t - g` to a copy of `att_gt_df`.
#   2. Group by `event_time`; within each group, compute the `n_treated`-
#      weighted average of `att` (`np.average(att, weights=n_treated)`) and
#      the weighted-independent-estimates `se` above (weights normalized
#      to sum to 1 *within that group*).
#   3. Return a tidy DataFrame with columns `event_time`, `att`, `se`,
#      sorted by `event_time`.
#
# Steps (`aggregate_overall_att`):
#   1. Normalize `n_treated` across the *whole* table to sum to 1 -> `w`.
#   2. `att = sum(w * att_gt_df["att"])`.
#   3. `se = sqrt(sum(w**2 * att_gt_df["se"]**2))`.
#   4. Return `(att, se)`.
# ---------------------------------------------------------------------------
def aggregate_event_study(att_gt_df: pd.DataFrame) -> pd.DataFrame:
    """Cohort-size-weighted event-study aggregation of ATT(g, t).

    Returns columns `event_time`, `att`, `se`, one row per relative time.
    """
    raise NotImplementedError("TODO(human): implement the event-study aggregation")


def aggregate_overall_att(att_gt_df: pd.DataFrame) -> tuple[float, float]:
    """Cohort-size-weighted single overall ATT across every (g, t) cell.

    Returns `(att, se)`.
    """
    raise NotImplementedError("TODO(human): implement the overall-ATT aggregation")


def main() -> None:
    panel = load_staggered_panel(seed=0)
    df = panel.df
    twfe = fit_naive_twfe(df)
    print(f"Naive TWFE estimate: {twfe.coef()['treated']:.3f}")

    try:
        att_gt_df = compute_all_att_gt(df)
    except NotImplementedError as e:
        print(f"(skipped -- ATT(g,t) not implemented yet -- {e})")
        return

    try:
        es = aggregate_event_study(att_gt_df)
        print("\nEvent-study aggregation:")
        print(es.to_string(index=False))

        overall_att, overall_se = aggregate_overall_att(att_gt_df)
        print(f"\nOverall (CS) ATT: {overall_att:.3f} (se={overall_se:.3f})")
        print(f"Naive TWFE:       {twfe.coef()['treated']:.3f}")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
