"""Phase 1 — Pre-treatment fit loss (RMSPE).

Every synthetic-control decision in this practice — which weights are
"good," which placebo is a "close call," which unit's effect stands out —
reduces to one number: how far apart are two trajectories over some window?
Root-mean-squared prediction error (RMSPE) is that number, computed once
here and reused everywhere downstream: as the pre-treatment fit-quality
report (Phase 2), and as the numerator/denominator of the post/pre ratio
that ranks the true effect against the placebo distribution (Phase 4).

Run on its own for a quick sanity check (treated vs. an equal-weighted
average of all donors — a naive baseline, not yet the optimized synthetic
control from Phase 2):
    uv run python -m src._01_fit_loss
"""
from __future__ import annotations

import numpy as np

from .datasets import load_dataset


# TODO(human) — root-mean-squared prediction error
# ---------------------------------------------------------------------------
# Goal: implement RMSPE, the standard fit-quality metric in the synthetic
# control literature (Abadie, Diamond & Hainmueller 2010 use it to define
# both "good pre-treatment fit" and the placebo ranking statistic in
# Phase 4).
#
# Why this matters: mean squared error alone is in squared outcome units
# (packs-per-capita squared here), which is hard to interpret and hard to
# compare across donors with different sales levels. Taking the square root
# puts RMSPE back in the outcome's own units — "California and its
# synthetic differ by about N packs per capita, on average, over this
# window" — which is what lets Phase 4 build a single post/pre *ratio*
# that is comparable across every donor in the placebo distribution
# regardless of that donor's baseline sales level.
#
# Steps:
#   1. Compute the element-wise difference between `actual` and `synthetic`.
#   2. Square it, take the mean, then the square root.
#   3. Return a plain Python float (not a 0-d numpy array) so downstream
#      code can print/compare it without surprises.
# ---------------------------------------------------------------------------
def rmspe(actual: np.ndarray, synthetic: np.ndarray) -> float:
    """Root-mean-squared prediction error between two equal-length arrays.

    Returns a single non-negative float in the same units as `actual`;
    `actual` and `synthetic` are typically the treated and synthetic
    outcome paths restricted to one window (pre- or post-treatment).
    """
    raise NotImplementedError("TODO(human): implement RMSPE")


def main() -> None:
    data = load_dataset()
    # Naive baseline: equal-weighted average of every donor — not the
    # optimized synthetic control (that's Phase 2), just enough of a
    # trajectory to sanity-check `rmspe` in isolation.
    y_naive = data.Y_donors.mean(axis=1)
    try:
        pre = rmspe(data.y_treated[data.pre_mask], y_naive[data.pre_mask])
        post = rmspe(data.y_treated[data.post_mask], y_naive[data.post_mask])
        print(f"Naive (equal-weight) donor average vs. California:")
        print(f"  pre-treatment RMSPE:  {pre:.3f}")
        print(f"  post-treatment RMSPE: {post:.3f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
