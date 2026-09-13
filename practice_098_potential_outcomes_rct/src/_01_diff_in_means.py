"""Phase 1 — The science table and the difference-in-means ATE estimator.

Rubin's potential-outcomes framework says every unit `i` has *two*
potential outcomes, `Y_i(0)` and `Y_i(1)`, and the individual treatment
effect is `tau_i = Y_i(1) - Y_i(0)`. In any real experiment you only ever
observe one of the two per unit — that's the fundamental problem of
causal inference. Because this practice's data is synthetic, `src/datasets.py`
hands you the full science table (both potential outcomes, for every
unit) so you can see directly what a real experiment can never show you:
the *individual* effects `tau_i`, not just their average.

Randomization is what makes the simple difference-in-means comparison a
valid estimator of the population ATE `tau = E[Y_i(1) - Y_i(0)]`: because
treatment assignment is independent of the potential outcomes (`T ⊥
(Y(0), Y(1))` by construction, since we flipped the coin ourselves),
`E[Y_obs | T=1] = E[Y(1)]` and `E[Y_obs | T=0] = E[Y(0)]`, so their
difference is unbiased for `tau` — no assumptions about the *shape* of the
outcome distribution are needed, unlike observational-data estimators.

Run on its own to compare the estimator against the true ATE:
    uv run python -m src._01_diff_in_means
"""
from __future__ import annotations

import numpy as np

from .datasets import load_dataset


# TODO(human) — difference-in-means ATE estimator
# ---------------------------------------------------------------------------
# Goal: implement the simplest possible causal estimator — the sample
# analogue of `E[Y_obs | T=1] - E[Y_obs | T=0]` — on the *observed* outcome
# only (never peek at the science table's y0/y1 columns here; a real
# experiment could never do that).
#
# Why this matters: this is the estimator every other phase in this
# practice builds on (Neyman's variance formula in Phase 2 is this
# estimator's variance; Fisher's randomization test in Phase 3 re-uses it
# as the test statistic; CUPED in Phase 5 is this same estimator applied
# to an adjusted outcome). Randomization is what licenses treating this
# simple mean difference as a valid estimate of the *causal* effect,
# rather than a merely descriptive correlation.
#
# Steps:
#   1. Split `y` into the treated group (`treatment == 1`) and the control
#      group (`treatment == 0`).
#   2. Return `mean(y[treated]) - mean(y[control])`.
# ---------------------------------------------------------------------------
def difference_in_means(y: np.ndarray, treatment: np.ndarray) -> float:
    """Sample difference-in-means estimate of the average treatment effect.

    `y` is the observed outcome (never the science table's `y0`/`y1`);
    `treatment` is the 0/1 assignment vector. Returns a single float.
    """
    raise NotImplementedError("TODO(human): implement the difference-in-means ATE estimator")


def main() -> None:
    data = load_dataset(n=500, tau=2.0, seed=0)
    print(f"True ATE:              {data.tau_true:.4f}")
    try:
        ate_hat = difference_in_means(data.y_observed, data.treatment)
        print(f"Difference-in-means:   {ate_hat:.4f}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")

    # The science table is only inspectable because the data is synthetic —
    # a real experiment never gets this view.
    print("\nScience table (first 5 rows, synthetic-only view):")
    print(data.science_table().head())


if __name__ == "__main__":
    main()
