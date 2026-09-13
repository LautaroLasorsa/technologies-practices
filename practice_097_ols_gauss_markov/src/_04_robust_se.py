"""Phase 4 — Heteroskedasticity-robust standard errors (HC0-HC3).

Once Phase 3's Breusch-Pagan test rejects homoskedasticity, the fix isn't
to re-estimate beta_hat (OLS is still unbiased) — it's to re-estimate its
*covariance matrix* without assuming a constant error variance. White's
(1980) sandwich estimator and its HC1-HC3 finite-sample variants are the
standard tool, and are what `statsmodels`' `cov_type="HC3"` computes under
the hood.

Run on its own to see all four HC variants on the heteroskedastic scenario:
    uv run python -m src._04_robust_se
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from ._01_ols_qr import ols_via_qr
from .datasets import load_dataset

HCKind = Literal["HC0", "HC1", "HC2", "HC3"]


# TODO(human) — heteroskedasticity-consistent (sandwich) covariance estimator
# ---------------------------------------------------------------------------
# Goal: implement White's (1980) sandwich estimator and its HC1-HC3
# finite-sample corrections.
#
# Why this matters: instead of assuming a *constant* sigma^2 (Phase 2),
# the sandwich estimator lets each observation contribute its own squared
# residual to the covariance — Var(beta_hat) stays consistent under *any*
# form of heteroskedasticity, at the cost of being noisier in small
# samples. HC1-3 are increasingly aggressive small-sample corrections;
# HC3 approximates the jackknife and is the safest default under a few
# hundred rows.
#
# The "meat" of the sandwich is always M = X' diag(w) X for some
# per-observation weight w_i built from the squared residual u_i^2 and
# (for HC2/HC3) the leverage h_i = X_i (X'X)^-1 X_i' (i.e. the i-th
# diagonal element of the hat matrix X (X'X)^-1 X'). The "bread" is
# (X'X)^-1 on both sides:
#   Var(beta_hat) = (X'X)^-1 @ M @ (X'X)^-1
#
# Per-observation weight w_i by kind (n = observations, k = regressors):
#   HC0: u_i^2
#   HC1: u_i^2 * n / (n - k)                 (simple dof correction)
#   HC2: u_i^2 / (1 - h_i)                   (leverage-adjusted)
#   HC3: u_i^2 / (1 - h_i)^2                 (jackknife-like)
# ---------------------------------------------------------------------------
def hc_vcov(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray, kind: HCKind = "HC3") -> np.ndarray:
    """Heteroskedasticity-consistent covariance matrix (White sandwich, HC0-HC3).

    `XtX_inv` is `(X'X)^-1`, as returned by `ols_via_qr`. Returns the
    (k, k) robust covariance matrix.
    """

    w = resid**2

    match kind:
        case "HC1":
            w = w * X.shape[0] / (X.shape[0] - X.shape[1])
        case "HC2":
            h = (X @ XtX_inv * X).sum(axis=1)
            w = w / (1-h)
        case "HC3":
            h = (X @ XtX_inv * X).sum(axis=1)
            w = w / (1-h)**2
    M = X.T @ (X * w[:, None])
    return XtX_inv @ M @ XtX_inv                    


def main() -> None:
    data = load_dataset("heteroskedastic", n=300, seed=0)
    try:
        fit = ols_via_qr(data.X, data.y)
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")
        return
    for kind in ("HC0", "HC1", "HC2", "HC3"):
        try:
            vcov = hc_vcov(data.X, fit.resid, fit.XtX_inv, kind=kind)
            print(f"{kind}: SE = {np.sqrt(np.diag(vcov))}")
        except NotImplementedError as e:
            print(f"(skipped — {e})")
            break


if __name__ == "__main__":
    main()
