"""Phase 5 — CUPED variance reduction using a pre-period covariate.

CUPED ("Controlled-experiment Using Pre-Experiment Data", Deng et al.
2013) is the standard industry trick for shrinking an experiment's
variance for free: if a pre-period covariate `X` (e.g. last week's value
of the same metric) is correlated with the outcome `Y` but was measured
*before* treatment assignment — so it cannot itself be affected by
treatment — then `Y' = Y - theta * (X - mean(X))` has the same
expectation as `Y` (so the diff-in-means ATE estimate is unchanged) but
strictly lower variance whenever `theta = Cov(Y, X) / Var(X)` is chosen
optimally. The intuition: `theta * (X - mean(X))` predicts and removes the
part of each unit's outcome that pre-experiment noise already explains,
leaving a "cleaner" residual to compare across arms.

Run on its own to see the ATE estimate and variance before/after CUPED:
    uv run python -m src._05_cuped
"""
from __future__ import annotations

import numpy as np

from ._01_diff_in_means import difference_in_means
from ._02_neyman_variance import neyman_variance
from .datasets import load_dataset


# TODO(human) — CUPED covariate adjustment
# ---------------------------------------------------------------------------
# Goal: implement the CUPED adjustment `Y' = Y - theta * (X - mean(X))`
# with the variance-minimizing `theta = Cov(Y, X) / Var(X)`.
#
# Why this matters: this is a genuinely different way to reduce variance
# than Phases 2-3's inference machinery — instead of computing a *better*
# standard error for the same estimator, CUPED changes the *outcome
# itself* (in a way that provably doesn't touch its expectation) so the
# same difference-in-means estimator from Phase 1 gets a tighter variance
# "for free," using data that was already available before the experiment
# even started. `theta` must be estimated from the *whole* sample (not
# per-arm) — using the same `theta` for both arms is what keeps `E[Y']`
# unchanged.
#
# Steps:
#   1. Compute `theta = Cov(y, x) / Var(x)` using `np.cov(y, x, ddof=1)`
#      (a 2x2 matrix; the covariance is the off-diagonal entry) and
#      `np.var(x, ddof=1)`.
#   2. Return `y - theta * (x - mean(x))` as the adjusted outcome, and
#      `theta` itself (useful for inspection/debugging).
# ---------------------------------------------------------------------------
def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED-adjust an outcome using a pre-period covariate.

    Returns `(y_adjusted, theta)`; `y_adjusted` has (approximately) the
    same mean as `y` but lower variance whenever `x` is correlated with
    `y`.
    """
    theta = np.cov(y,x, ddof=1)[0,1] / np.var(x, ddof=1)
    return y - theta * (x - np.mean(x)), theta


def main() -> None:
    data = load_dataset(n=500, tau=2.0, seed=0)
    ate_raw = difference_in_means(data.y_observed, data.treatment)
    var_raw = neyman_variance(data.y_observed, data.treatment)

    try:
        y_adj, theta = cuped_adjust(data.y_observed, data.x)
        ate_cuped = difference_in_means(y_adj, data.treatment)
        var_cuped = neyman_variance(y_adj, data.treatment)
        print(f"theta:                {theta:.4f}")
        print(f"ATE (raw):            {ate_raw:.4f}   Neyman var: {var_raw:.4f}")
        print(f"ATE (CUPED-adjusted): {ate_cuped:.4f}   Neyman var: {var_cuped:.4f}")
        print(f"Variance reduction:   {(1 - var_cuped / var_raw):.1%}")
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
