"""Phase 3 -- staggered adoption breaks two-way fixed effects, and the
Goodman-Bacon decomposition shows exactly why.

Phase 1's canonical 2x2 generalises badly the moment adoption is staggered
*and* treatment effects are dynamic (they grow the longer a unit has been
treated). The single "treated" dummy in a two-way-fixed-effects (TWFE)
regression `y ~ treated | unit + time` is no longer just comparing treated
units to never-treated ones -- Goodman-Bacon (2021) shows it is a weighted
average of *every possible pair* of cohorts' 2x2 DiDs, including pairs
where an **already-treated** cohort plays "control" for a later-treated
one. That comparison is contaminated: the "control" group's own change
over the window includes its own (growing) treatment effect, which gets
subtracted off as if it were a time trend. When effects are dynamic, that
contamination can be large enough to pull the aggregate TWFE coefficient
outside the range of every true effect in the data -- even flipping its
sign, despite every true effect being strictly positive.

Run on its own to see the naive TWFE estimate, the true effect range, and
the pairwise decomposition that explains the gap:
    uv run python -m src._03_twfe_and_bacon
"""
from __future__ import annotations

import pandas as pd
import pyfixest as pf

from ._01_two_by_two_did import did_2x2  # used inside the TODO(human) below
from .datasets import load_staggered_panel


def fit_naive_twfe(df: pd.DataFrame) -> pf.Feols:
    """Fit the naive static TWFE regression `y ~ treated | unit + time` --
    the regression every applied paper ran on staggered panels before 2020.
    Fully scaffolded: the point of this phase is what this single
    coefficient is secretly averaging together, not how to call `feols`.
    """
    return pf.feols("y ~ treated | unit + time", data=df, vcov={"CRV1": "unit"})


# TODO(human) -- the Goodman-Bacon pairwise decomposition
# ---------------------------------------------------------------------------
# Goal: decompose the single TWFE coefficient into the many pairwise 2x2
# DiDs it is secretly averaging, and label each one "clean" or "forbidden"
# so the source of the bias is visible instead of just its symptom (a
# coefficient outside the true-effect range).
#
# Why this matters: Goodman-Bacon (2021) proves the TWFE coefficient
# equals a weighted average of the 2x2 DiD between *every pair* of cohorts
# (including "vs. never-treated"). For a pair of cohorts that are BOTH
# eventually treated, there are two such comparisons, and they are not
# symmetric:
#   - "clean": cohort a's own switch (pre = g_a - 1, post = g_a), using
#     cohort b as control, while b is *not yet treated* (g_b > g_a's post
#     period) -- this is a textbook-valid 2x2, no different from Phase 1.
#   - "forbidden": same setup, but b is *already treated* by the time this
#     window starts (g_b < g_a) -- b's own change over the window bundles
#     in whatever b's treatment effect did between those two periods. If
#     b's effect is growing (as "early" is, in this DGP), that growth gets
#     subtracted from a's estimated effect, and the resulting 2x2 can be
#     wildly wrong -- including strongly negative, even though every true
#     effect in the data is positive.
#
# Steps:
#   1. Get the distinct (cohort, g) pairs from `df` (one row per cohort).
#   2. For every cohort `a` with finite `g_a`, and every *other* cohort `b`:
#      - pre, post = g_a - 1, g_a
#      - build a 2-row-per-unit sub-frame restricted to cohorts {a, b} and
#        times {pre, post}, with `treat` = 1 for cohort a, `post` = 1 at
#        the post period -- exactly the shape `did_2x2` (Phase 1) expects.
#      - call `did_2x2` on it to get the pairwise estimate.
#      - label the comparison `"forbidden"` if `g_b < g_a` (b already
#        treated going in), else `"clean"`.
#   3. Weight each comparison by its sample size (`n_a + n_b` observations
#      in that sub-frame), normalized so all weights sum to 1 -- a
#      simplified stand-in for Goodman-Bacon's variance-based weights
#      (exact formula: Goodman-Bacon 2021, eq. 9-10), good enough to see
#      *which* comparisons dominate and how the sign gets pulled around.
#   4. Return a DataFrame with columns `comparison` (e.g. `"late vs early"`),
#      `type` (`"clean"`/`"forbidden"`), `estimate`, `weight`.
# ---------------------------------------------------------------------------
def goodman_bacon_decompose(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Goodman-Bacon-style decomposition of the TWFE coefficient.

    Returns one row per ordered (treatment cohort, control cohort) pair,
    with columns `comparison`, `type`, `estimate`, `weight` (weights sum
    to 1 across all rows).
    """
    raise NotImplementedError("TODO(human): implement the Goodman-Bacon pairwise decomposition")


def main() -> None:
    panel = load_staggered_panel(seed=0)
    df = panel.df
    true_min = panel.true_att_by_cell["true_att"].min()
    true_max = panel.true_att_by_cell["true_att"].max()
    print(f"True effect range across every (g, t) cell: [{true_min:.2f}, {true_max:.2f}]")

    twfe = fit_naive_twfe(df)
    print(f"Naive TWFE estimate:  {twfe.coef()['treated']:.3f} (se={twfe.se()['treated']:.3f})")
    if not (true_min <= twfe.coef()["treated"] <= true_max):
        print("--> outside the range of every true effect in the data.")

    try:
        bacon = goodman_bacon_decompose(df)
        print("\nPairwise decomposition:")
        print(bacon.to_string(index=False))
        print(f"\nWeighted average of pairwise estimates: {(bacon['estimate'] * bacon['weight']).sum():.3f}")
        print("(a simplified weighting scheme -- see the module docstring -- so this won't equal")
        print(" the TWFE coefficient exactly, but the 'forbidden' rows should be visibly the outliers.)")
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
