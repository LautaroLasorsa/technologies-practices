"""Phase 1 -- the canonical 2x2 difference-in-differences.

Every staggered-adoption estimator in this practice (Phases 3-5) is built
out of many copies of this one comparison: one treated group, one control
group, one pre period, one post period. Get the 2x2 case exactly right --
what it estimates, what it assumes, how its standard error is built -- and
the rest of the practice is "do this correctly, many times, and combine
the results without breaking anything."

Run on its own to see the 2x2 estimate on one treated cohort vs. the
never-treated group:
    uv run python -m src._01_two_by_two_did
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # used inside the TODO(human) below
import pandas as pd
import pyfixest as pf

from .datasets import load_2x2_subset, load_staggered_panel


@dataclass
class DiD2x2Result:
    """Result of one canonical 2x2 DiD comparison."""

    estimate: float  # the double-difference point estimate
    se: float  # its standard error (Neyman/design-based, no covariates)
    group_means: dict[str, float]  # the 4 cell means the estimate is built from


# TODO(human) -- the double-difference estimator
# ---------------------------------------------------------------------------
# Goal: implement the canonical 2x2 DiD "by hand" -- as a literal
# difference of differences of group means, not as a regression coefficient
# (Phase 1's `compare_to_regression` below checks the two agree).
#
# Why this matters: every later phase's estimator (Phase 3's TWFE, Phase 4's
# ATT(g,t)) is *this exact computation*, run many times over different
# (group, period) pairs and then combined. If the 2x2 case isn't crisp in
# your head -- what it estimates (average treatment effect on the treated,
# under parallel trends) and what breaks it (a control group whose
# untreated potential outcome doesn't move in parallel with the treated
# group's) -- the staggered-adoption failure in Phase 3 will look like
# arbitrary regression arithmetic instead of "many 2x2s combined badly."
#
# Steps:
#   1. Compute the four cell means of `y`: treated-pre, treated-post,
#      control-pre, control-post (columns `treat`, `post`, `y` in `df`).
#   2. The double difference:
#        (treated_post - treated_pre) - (control_post - control_pre)
#   3. Its standard error, assuming independent observations within each of
#      the 4 cells (the "Neyman" / design-based variance, no covariates):
#        se = sqrt(sum(var(y in cell) / n(cell) for each of the 4 cells))
#      using the *sample* variance (ddof=1) within each cell.
# ---------------------------------------------------------------------------
def did_2x2(df: pd.DataFrame) -> DiD2x2Result:
    """Canonical 2x2 DiD on a frame with `treat`, `post`, `y` columns.

    Returns the double-difference point estimate, its standard error, and
    the four cell means it was built from (keyed `"treat_pre"`,
    `"treat_post"`, `"control_pre"`, `"control_post"`).
    """
    raise NotImplementedError("TODO(human): implement the 2x2 double-difference estimator")


def compare_to_regression(df: pd.DataFrame) -> None:
    """Cross-check: the `treat:post` interaction coefficient from
    `y ~ treat + post + treat:post` should equal the hand-computed double
    difference to numerical precision -- this is the textbook fact that
    the 2x2 DiD *is* that regression coefficient, just computed two ways.
    """
    result = did_2x2(df)
    model = pf.feols("y ~ treat + post + treat*post", data=df, vcov="hetero")
    reg_estimate = model.coef()["treat:post"]
    print(f"Hand-computed DD estimate: {result.estimate:.4f} (se={result.se:.4f})")
    print(f"Regression treat:post:    {reg_estimate:.4f}")
    diff = abs(result.estimate - reg_estimate)
    print(f"|diff|:                   {diff:.2e}")
    assert diff < 1e-6, "the hand-computed DD should match the regression interaction exactly"


def main() -> None:
    panel = load_staggered_panel(seed=0)
    df = load_2x2_subset(panel, cohort="mid", control="never")
    print(f"True effect at treatment (mid, event_time=0): {5.0 - 5.0 + 1.0:.1f}")  # TAU_BASE['mid']
    try:
        compare_to_regression(df)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
