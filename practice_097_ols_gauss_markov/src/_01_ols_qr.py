"""Phase 1 — OLS estimation from scratch via QR decomposition.

The textbook "normal equations" formula beta_hat = (X'X)^-1 X'y is what
every intro-econometrics course writes on the board, but it's not how
real software computes OLS: explicitly forming and inverting X'X squares
the condition number of X, so any near-collinearity in the regressors gets
amplified into numerical garbage. statsmodels, R's `lm()`, and every other
serious implementation instead factor X = QR and solve a triangular system.
This phase implements that route and checks it against statsmodels to
numerical precision.

Run on its own to see the from-scratch estimator matched against
statsmodels:
    uv run python -m src._01_ols_qr
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm

from .datasets import load_dataset


@dataclass
class OLSFit:
    """Result of fitting OLS to one (X, y) dataset."""

    beta: np.ndarray  # (k,) estimated coefficients
    fitted: np.ndarray  # (n,) X @ beta
    resid: np.ndarray  # (n,) y - fitted
    XtX_inv: np.ndarray  # (k, k) (X'X)^-1, reused by later phases' vcov formulas


# TODO(human) — OLS via QR decomposition
# ---------------------------------------------------------------------------
# Goal: implement OLS estimation *without* calling statsmodels or
# np.linalg.lstsq — via the QR decomposition, the numerically stable way
# real software computes OLS (this is what statsmodels/R's lm() do under
# the hood instead of literally inverting X'X).
#
# Why this matters: the normal-equations formula beta_hat = (X'X)^-1 X'y is
# the textbook formula, but explicitly inverting X'X is numerically
# unstable when regressor columns are near-collinear (X'X becomes
# ill-conditioned — its condition number is the *square* of X's). QR
# avoids ever forming X'X: with X = QR (Q orthonormal columns, R square
# upper-triangular), the least-squares problem reduces to solving the
# triangular system R @ beta = Q.T @ y by back-substitution — O(nk^2) and
# numerically stable.
#
# Steps:
#   1. Compute the *reduced* QR decomposition: Q, R = np.linalg.qr(X).
#   2. Solve R @ beta = Q.T @ y for beta with np.linalg.solve (R is square
#      upper-triangular, so this is exact back-substitution, not an
#      approximation).
#   3. Compute fitted values (X @ beta) and residuals (y - fitted).
#   4. Compute (X'X)^-1 from R alone — since X'X = R'R, its inverse is
#      solve(R, solve(R.T, I)) — this reuses the same decomposition instead
#      of ever forming or inverting X'X directly, and later phases (2-5)
#      reuse the returned XtX_inv for their variance formulas.
#
# Do NOT use np.linalg.lstsq or np.linalg.inv(X.T @ X) — the point of this
# exercise is the QR route.
# ---------------------------------------------------------------------------
def ols_via_qr(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Estimate OLS coefficients via QR decomposition.

    Returns an `OLSFit` with the coefficient vector, fitted values,
    residuals, and `(X'X)^-1` — the last computed from `R`, never by
    inverting `X'X` directly.
    """
    raise NotImplementedError("TODO(human): implement OLS via QR decomposition")


def compare_to_statsmodels(X: np.ndarray, y: np.ndarray) -> None:
    """Fit both the from-scratch estimator and statsmodels; assert they match."""
    fit = ols_via_qr(X, y)
    sm_fit = sm.OLS(y, X).fit()
    diff = np.max(np.abs(fit.beta - sm_fit.params))
    print(f"beta (ours):        {fit.beta}")
    print(f"beta (statsmodels): {sm_fit.params}")
    print(f"max |diff|:         {diff:.2e}")
    assert diff < 1e-8, "OLS via QR should match statsmodels to numerical precision"


def main() -> None:
    data = load_dataset("homoskedastic", n=200, seed=0)
    print(f"True beta: {data.beta_true}\n")
    try:
        compare_to_statsmodels(data.X, data.y)
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
