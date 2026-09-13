"""Shared regression plumbing: OLS and lag-matrix construction.

Practice 097 makes OLS itself the exercise; here it is infrastructure, so
it is fully scaffolded. Every estimator in this practice — the ADF
regression, conditional least squares for an AR(p), the Granger F-test,
the cointegrating regression and the ECM — is "OLS on a cleverly built
design matrix". Sharing one `ols()` and one `build_lag_matrix()` keeps
each `TODO(human)` about the *econometrics* (which columns, which
statistic, which null) rather than about array bookkeeping.

No TODO here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinFit:
    """Result of one OLS fit, with everything the later statistics need."""

    beta: np.ndarray  # (k,) coefficients
    se: np.ndarray  # (k,) classical standard errors
    tstat: np.ndarray  # (k,) beta / se
    resid: np.ndarray  # (n,) residuals
    fitted: np.ndarray  # (n,) X @ beta
    rss: float  # residual sum of squares
    r2: float  # uncentered-adjusted R^2 (centered when the design has a constant)
    n: int  # observations used
    k: int  # regressors (including any constant)
    xtx_inv: np.ndarray  # (k, k) (X'X)^-1


def add_const(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones to a (n, k) design matrix."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if X.shape[0] == 1 and X.shape[1] != 1:
        X = X.T
    return np.column_stack([np.ones(X.shape[0]), X])


def ols(X: np.ndarray, y: np.ndarray) -> LinFit:
    """Fit `y = X @ beta + u` by least squares (no constant added for you).

    Returns a `LinFit` with coefficients, classical standard errors,
    t-statistics, residuals, RSS and R^2. Solved through `lstsq`, which is
    QR-based internally — never by inverting `X'X`.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    rss = float(resid @ resid)
    dof = max(n - k, 1)
    sigma2 = rss / dof
    xtx_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.maximum(np.diag(sigma2 * xtx_inv), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    has_const = bool(np.any(np.all(np.isclose(X, X[0, :]), axis=0)))
    tss = float(((y - y.mean()) ** 2).sum()) if has_const else float(y @ y)
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return LinFit(
        beta=beta, se=se, tstat=tstat, resid=resid, fitted=fitted,
        rss=rss, r2=r2, n=n, k=k, xtx_inv=xtx_inv,
    )


def build_lag_matrix(v: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    """Align a series with its own first `p` lags.

    Returns `(v[p:], L)` where `L` has `p` columns and `L[:, j - 1]` is `v`
    lagged `j` periods, row-aligned with `v[p:]`. With `p == 0` the lag
    matrix is an empty `(len(v), 0)` array, so `np.column_stack` still works.
    """
    v = np.asarray(v, dtype=float).ravel()
    if p == 0:
        return v, np.empty((v.size, 0))
    n = v.size - p
    lags = np.column_stack([v[p - j: p - j + n] for j in range(1, p + 1)])
    return v[p:], lags
