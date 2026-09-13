"""Phase 4 -- LATE vs. ATE: the Wald estimator for a binary instrument.

Every earlier phase assumed a homogeneous treatment effect -- 2SLS
estimated one number, THE effect of D on y. Real treatment effects vary
across individuals, and with a binary instrument, IV does not recover the
population Average Treatment Effect (ATE) -- it recovers the Local Average
Treatment Effect (LATE): the average effect *among compliers*, the
subpopulation whose treatment status is actually moved by the instrument
(Imbens & Angrist, 1994; Angrist, Imbens & Rubin, 1996). Under
monotonicity (no defiers -- nobody does the opposite of what the
instrument nudges them toward), the ratio of the instrument's effect on
the outcome to its effect on treatment -- the Wald estimator -- identifies
exactly that complier-average effect, nothing more.

Run standalone:
    uv run python -m src._04_late_wald
"""
from __future__ import annotations

import numpy as np

from .datasets import simulate_iv_binary


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# TODO(human) -- the Wald estimator for a binary instrument
# ---------------------------------------------------------------------------
# Goal: implement the Wald (ratio) estimator, the special case of 2SLS
# that applies when both the instrument Z and the treatment D are binary.
#
# Why this matters: with a binary instrument, 2SLS's "purify D, then
# regress" machinery from Phases 1-2 collapses to a single ratio -- the
# jump in average outcome when Z flips from 0 to 1, divided by the jump in
# average treatment take-up over the same flip:
#     LATE_hat = (E[y | z=1] - E[y | z=0]) / (E[d | z=1] - E[d | z=0])
# The numerator is the instrument's "reduced-form" effect on y; the
# denominator is its first-stage effect on D. Under monotonicity (assumed
# by this practice's DGP -- see datasets.py, there are no "defiers"), this
# ratio identifies exactly the average effect among compliers: the people
# whose d moves with z. It is numerically identical to 2SLS of y on d
# using z as the sole instrument -- this practice's notebook verifies that
# equivalence directly against Phase 2's `tsls_projection`.
#
# Steps:
#   1. Split y and d by z == 0 vs. z == 1 and take each group's mean.
#   2. numerator = mean(y[z==1]) - mean(y[z==0]).
#   3. denominator = mean(d[z==1]) - mean(d[z==0]).
#   4. Return numerator / denominator.
# ---------------------------------------------------------------------------
def wald_estimator(y: np.ndarray, d: np.ndarray, z: np.ndarray) -> float:
    """Wald (ratio) estimator of the LATE for a binary instrument and binary treatment.

    Returns a single float: the estimated Local Average Treatment Effect.
    """
    raise NotImplementedError("TODO(human): implement the Wald estimator")


def compare_ols_wald_truth(data) -> None:
    """Print naive OLS, the Wald/2SLS estimate, and both ground-truth effects side by side."""
    X_ols = np.column_stack([np.ones_like(data.d, dtype=float), data.d])
    beta_ols = _ols(X_ols, data.y)[-1]
    late_hat = wald_estimator(data.y, data.d, data.z)

    print(f"OLS (biased):        {beta_ols:.3f}")
    print(f"Wald / 2SLS (LATE):  {late_hat:.3f}")
    print(f"True ATE:            {data.ate_true:.3f}")
    print(f"True LATE:           {data.late_true:.3f}")


def bootstrap_se(data, n_boot: int = 300, seed: int = 1) -> tuple[float, float]:
    """Nonparametric bootstrap standard errors for OLS and the Wald
    estimator on this dataset -- infrastructure for the end-to-end
    comparison plot, not the taught estimator itself."""
    rng = np.random.default_rng(seed)
    n = len(data.y)
    ols_draws = np.empty(n_boot)
    wald_draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b, d_b, z_b = data.y[idx], data.d[idx], data.z[idx]
        X_ols = np.column_stack([np.ones(n), d_b])
        ols_draws[b] = _ols(X_ols, y_b)[-1]
        wald_draws[b] = wald_estimator(y_b, d_b, z_b)
    return float(np.std(ols_draws, ddof=1)), float(np.std(wald_draws, ddof=1))


def main() -> None:
    data = simulate_iv_binary(n=4000, complier_share=0.4, seed=0)
    try:
        compare_ols_wald_truth(data)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
