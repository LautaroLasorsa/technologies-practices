"""Phase 4 — M-bias and selection bias: a correct model made worse.

Phase 3 showed conditioning on a *collider* manufactures bias where there
was none. This phase shows the same mechanism biting in two more subtle,
very common situations:

  - **M-bias**: two independent, unobserved causes U1 (of D and M) and U2
    (of M and Y) make M a collider, even though M looks like a plausible
    "pre-treatment control" (it's correlated with both D and Y). Not
    conditioning on it leaves the D -> Y estimate unbiased; a well-meaning
    analyst who adds M "just in case" opens the collider path and biases
    it — the textbook case of a correct model getting *worse* when a
    variable is added.
  - **Selection bias**: the same mechanism, but the "conditioning" happens
    via *sample selection* rather than regression — e.g. a survey that only
    reaches respondents above some collider threshold. Selecting on a
    collider is d-separation-equivalent to conditioning on it.

Run on its own to see both scenarios' bias magnitude:
    uv run python -m src._04_mbias_selection
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ._03_structures import estimate_effect_by_adjustment
from .datasets import load_dataset


@dataclass
class BiasResult:
    """Effect estimates with and without conditioning on a collider, and the
    bias that conditioning introduces."""

    unadjusted: float  # estimate with the collider left alone
    adjusted: float  # estimate after conditioning on the collider
    bias: float  # adjusted - unadjusted


# TODO(human) — collider-stratification bias magnitude
# ---------------------------------------------------------------------------
# Goal: quantify how much conditioning on a collider moves the treatment
# effect estimate, by calling Phase 3's `estimate_effect_by_adjustment`
# twice on the same data — once without the collider, once with it as a
# covariate — and returning both plus their difference.
#
# Why this matters: Phase 1-3 established *whether* a variable is a
# collider (graphically) and *that* conditioning on one biases an estimate
# (via one worked example each). This function turns that into a
# measurement you can compare across scenarios (M-bias vs. plain collider
# vs. varying M-bias strength) and plot — the "bias magnitude" figure this
# practice's notebook builds in its end-to-end run.
#
# Steps:
#   1. Call `estimate_effect_by_adjustment(df, treatment, outcome, [])` for
#      the unadjusted estimate.
#   2. Call it again with `adjust_for=[collider]` for the adjusted estimate.
#   3. Return a `BiasResult` with both values and `bias = adjusted -
#      unadjusted`.
# ---------------------------------------------------------------------------
def collider_stratification_bias(df: pd.DataFrame, treatment: str, outcome: str, collider: str) -> BiasResult:
    """Bias introduced by conditioning on `collider` when estimating `treatment`'s effect on `outcome`.

    Runs the Phase 3 estimator with and without `collider` in the
    adjustment set and returns both estimates plus their difference.
    """
    raise NotImplementedError("TODO(human): compute the collider-stratification bias")


def selection_bias_demo(data, selection_var: str, keep_above_median: bool = True) -> BiasResult:
    """Selection bias as sample-filtering rather than regression-adjustment.

    Filtering the sample on `selection_var` (e.g. "only respondents above
    the median") is d-separation-equivalent to conditioning on it — both
    open the same collider path. Fully scaffolded: the interesting part
    (why filtering = conditioning) is conceptual, not a new algorithm; the
    bias *measurement* is Phase 4's real exercise, reused here unchanged.
    """
    threshold = data.df[selection_var].median()
    mask = data.df[selection_var] > threshold if keep_above_median else data.df[selection_var] <= threshold
    selected = data.df.loc[mask]
    unadjusted_full = estimate_effect_by_adjustment(data.df, data.treatment, data.outcome, [])
    unadjusted_selected = estimate_effect_by_adjustment(selected, data.treatment, data.outcome, [])
    return BiasResult(
        unadjusted=unadjusted_full,
        adjusted=unadjusted_selected,
        bias=unadjusted_selected - unadjusted_full,
    )


def main() -> None:
    mbias_data = load_dataset("mbias", n=1000, seed=0)
    collider_data = load_dataset("collider", n=1000, seed=0)
    print(f"[mbias]    true ATE = {mbias_data.true_effect:.3f}")
    try:
        mbias_result = collider_stratification_bias(mbias_data.df, "D", "Y", "M")
        collider_result = collider_stratification_bias(collider_data.df, "D", "Y", "C")
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    print(f"  unadjusted (correct):        {mbias_result.unadjusted:.3f}")
    print(f"  adjusted for M (worse!):     {mbias_result.adjusted:.3f}  (bias = {mbias_result.bias:+.3f})")
    print(f"[collider] true effect = {collider_data.true_effect:.3f}")
    print(f"  unadjusted (correct):        {collider_result.unadjusted:.3f}")
    print(f"  adjusted for C (worse!):     {collider_result.adjusted:.3f}  (bias = {collider_result.bias:+.3f})")

    selection_result = selection_bias_demo(collider_data, selection_var="C")
    print("\n[collider -> selection] selecting on C instead of regressing on it:")
    print(f"  full sample:                 {selection_result.unadjusted:.3f}")
    print(f"  selected subsample (worse!): {selection_result.adjusted:.3f}  (bias = {selection_result.bias:+.3f})")


if __name__ == "__main__":
    main()
