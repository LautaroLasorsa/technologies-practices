"""Phase 3 — The gap: treated minus synthetic, and the effect it implies.

Once Phase 2 produces a synthetic California, the *estimate* is simply the
difference between the two trajectories at every point in time. Before the
treatment date that difference is a fit diagnostic (it should be small); after
the treatment date it is the causal estimate (how much lower/higher would
sales have been without Prop 99).

Run on its own to fit a synthetic control and report its gap:
    uv run python -m src._03_gap_effect
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._02_synthetic_weights import fit_synthetic_control
from .datasets import SyntheticControlData, load_dataset


@dataclass
class GapResult:
    """The full gap series plus a scalar summary of the post-treatment effect."""

    gap: np.ndarray            # (T,) y_treated - y_synthetic, all years
    post_treatment_effect: float  # mean of gap over the post-treatment window


# TODO(human) — the gap and its post-treatment summary
# ---------------------------------------------------------------------------
# Goal: compute the treated-minus-synthetic gap series, and summarize the
# post-treatment portion of it as a single average effect.
#
# Why this matters: the gap series *is* the estimated dynamic treatment
# effect — there is no separate "effect estimation" step beyond this
# subtraction, which is what makes synthetic control easy to explain to a
# non-technical audience compared to a regression coefficient. The average
# post-treatment gap is the standard single-number summary reported
# alongside the plot (e.g. "Prop 99 reduced per-capita cigarette sales by
# about N packs/year on average over 1989-2000"), and it is exactly the
# quantity Phase 4 needs at the numerator of its post/pre ratio.
#
# Steps:
#   1. Compute `gap = y_treated - y_synthetic` element-wise, over ALL years
#      (both `data.pre_mask` and `data.post_mask` positions).
#   2. Compute `post_treatment_effect` as the mean of `gap` restricted to
#      `data.post_mask`.
#   3. Return a `GapResult` with both.
# ---------------------------------------------------------------------------
def compute_gap(data: SyntheticControlData, y_synthetic: np.ndarray) -> GapResult:
    """Compute the treated-minus-synthetic gap and its post-treatment mean.

    `y_synthetic` is `(T,)`, aligned with `data.years`/`data.y_treated` (as
    returned by `SyntheticControlFit.y_synthetic`). Returns a `GapResult`.
    """
    raise NotImplementedError("TODO(human): implement the gap and post-treatment effect")


def main() -> None:
    data = load_dataset()
    try:
        fit = fit_synthetic_control(data)
        result = compute_gap(data, fit.y_synthetic)
    except NotImplementedError as e:
        print(f"(skipped — {e})")
        return
    print(f"Pre-treatment RMSPE:        {fit.pre_rmspe:.3f}")
    print(f"Post-treatment avg. effect: {result.post_treatment_effect:.3f} packs/capita")


if __name__ == "__main__":
    main()
