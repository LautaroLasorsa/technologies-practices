"""Phase 4 — Manski worst-case bounds and Lee (2009) trimming bounds.

Phases 1-3 all quantified how much *hidden confounding* could be biasing a
point-identified estimate. This phase drops point identification
altogether: outcomes are missing for some units (`response == 0` in
`src/datasets.py`), and missingness depends on the outcome itself -- so no
amount of covariate adjustment recovers a single point estimate without an
extra, untestable assumption. Manski's (1990) worst-case bounds make no
such assumption at all and report the full range of ATE values consistent
with the observed data. Lee's (2009) trimming bounds add one monotonicity
assumption (treatment shifts *selection into being observed* in only one
direction) and are provably tighter whenever that assumption holds.

Run on its own to see both sets of bounds narrow around the true ATE:
    uv run python -m src._04_manski_bounds
"""
from __future__ import annotations

import numpy as np

from .datasets import load_dataset


# TODO(human) #1 — Manski worst-case bounds on the ATE
# ---------------------------------------------------------------------------
# Goal: implement Manski's (1990) worst-case bounds on E[Y(1)] - E[Y(0)]
# when some outcomes are missing and missingness is *not* assumed ignorable.
#
# Why this matters: with a response rate `p_a` in a group, the true group
# mean is a mixture of the observed-subgroup mean (weight `p_a`) and the
# *unobserved* subgroup's mean (weight `1 - p_a`) -- and with zero
# assumptions about the missing subgroup, its mean could be anything in
# `[y_min, y_max]`. That gives the widest possible (but fully
# assumption-free) bound on each group mean; combining the two groups'
# bounds gives a bound on the ATE.
#
# Procedure (per treatment arm a in {0, 1}):
#   1. p_a = response rate in arm a (mean of `response` among units with t == a).
#   2. ybar_a = mean of `y_obs` among the *observed* units in arm a.
#   3. Lower bound on E[Y(a)]:  p_a * ybar_a + (1 - p_a) * y_min
#      Upper bound on E[Y(a)]:  p_a * ybar_a + (1 - p_a) * y_max
#   4. ATE lower bound = LB(E[Y(1)]) - UB(E[Y(0)])
#      ATE upper bound = UB(E[Y(1)]) - LB(E[Y(0)])
#      (subtracting the *opposing* bound each time is what makes this the
#      worst case for the difference, not just for each mean separately.)
# ---------------------------------------------------------------------------
def manski_worst_case_bounds(
    y_obs: np.ndarray, response: np.ndarray, t: np.ndarray, y_min: float, y_max: float
) -> tuple[float, float]:
    """Manski's (1990) assumption-free worst-case bounds on the ATE.

    `y_obs` has `NaN` for unobserved units; `response` and `t` are the
    same length, all aligned by row. Returns `(lower, upper)`.
    """
    raise NotImplementedError("TODO(human): implement Manski's worst-case ATE bounds")


# TODO(human) #2 — Lee (2009) monotonicity/trimming bounds on the ATE
# ---------------------------------------------------------------------------
# Goal: implement Lee's (2009) trimmed bounds, which narrow Manski's
# worst-case bounds under one added assumption: monotonicity of selection
# -- treatment can only move a unit's probability of being *observed* in
# one direction (here: treatment only ever helps response, never hurts it,
# matching this dataset's `-0.3 * t` term in `logit_r`, which lowers
# response under treatment -- so here it's the *control* arm that has the
# higher response rate; trim from control).
#
# Why this matters: instead of imputing the missing subgroup's mean as the
# extreme `y_min`/`y_max` (Manski), monotonicity lets you reason about
# *which* units in the higher-response arm are the "extra" ones relative
# to the lower-response arm, and bound the effect by trimming them out of
# the *observed* distribution instead of guessing at unobserved values --
# a strictly narrower, still assumption-justified, bound.
#
# Procedure (Lee, 2009, "Training, Wages, and Sample Selection", sec. 3):
#   1. p1, p0 = response rates in the treated / control arms.
#   2. Let the *higher*-response arm be the one to trim; let
#      trim_share = 1 - min(p1, p0) / max(p1, p0) (the fraction of that
#      arm's *observed* units that must be dropped to equalize response
#      rates across arms).
#   3. Sort that arm's observed y values. The lower bound drops the
#      top `trim_share` fraction before averaging (worst case: the "extra"
#      responders were the high performers); the upper bound drops the
#      bottom `trim_share` fraction instead.
#   4. Combine the trimmed mean of the higher-response arm with the
#      untrimmed mean of the lower-response arm (in whichever order gives
#      the ATE's lower vs. upper bound) to get `(lower, upper)`.
# ---------------------------------------------------------------------------
def lee_trimming_bounds(y_obs: np.ndarray, response: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Lee's (2009) monotone-selection trimming bounds on the ATE.

    Assumes treatment shifts the response probability in one direction for
    every unit (monotonicity). Returns `(lower, upper)`, always contained
    within `manski_worst_case_bounds`'s interval when the assumption holds.
    """
    raise NotImplementedError("TODO(human): implement Lee's (2009) trimming bounds")


def main() -> None:
    data = load_dataset(n=1000, confound_strength=1.2, seed=0)
    df = data.df
    y_obs, response, t = df["y_obs"].to_numpy(), df["response"].to_numpy(), df["t"].to_numpy()
    print(f"Response rate, treated: {response[t == 1].mean():.3f}  control: {response[t == 0].mean():.3f}")
    try:
        lo, hi = manski_worst_case_bounds(y_obs, response, t, data.y_min, data.y_max)
        print(f"Manski worst-case bounds: [{lo:.2f}, {hi:.2f}]  (true ATE: {data.ate_true})")
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    try:
        lo_lee, hi_lee = lee_trimming_bounds(y_obs, response, t)
        print(f"Lee trimming bounds:      [{lo_lee:.2f}, {hi_lee:.2f}]")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
