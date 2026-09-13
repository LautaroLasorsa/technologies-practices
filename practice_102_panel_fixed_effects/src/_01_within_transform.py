"""Phase 1 — The within (demeaning) transformation, one-way fixed effects.

Pooled OLS on panel data assumes there's no unit-specific unobserved
confounder left in the error term. In this practice's DGP, there is one:
`alpha_i` (the unit effect) is built to correlate with `x1`'s unit-level
mean, so pooled OLS is *biased* for beta1 — the standard motivation for
fixed effects. The "within" (or "fixed effects") estimator sidesteps the
problem without ever estimating `alpha_i`: subtract each unit's own mean
from `y` and every regressor, then run OLS on the demeaned data. Any
time-invariant quantity — `alpha_i` included — is identical to its own
unit mean, so demeaning subtracts it out exactly, whatever its value.

The same demeaning kills time-invariant *regressors*, not just
confounders: a variable that never varies within a unit has zero
within-unit variation, so its coefficient is not identified after
demeaning (this is also why the intercept disappears — it's the most
extreme time-invariant "regressor" of all, a column of ones).

Run on its own to see the within estimator matched against pyfixest and
linearmodels.PanelOLS:
    uv run python -m src._01_within_transform
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datasets import SMALL_PANEL, generate_panel


@dataclass
class OLSResult:
    """Result of an OLS fit on (possibly transformed) data."""

    beta: np.ndarray  # (k,) estimated coefficients
    fitted: np.ndarray  # (n,) X @ beta
    resid: np.ndarray  # (n,) y - fitted
    XtX_inv: np.ndarray  # (k, k) (X'X)^-1, reused by Phase 4's cluster vcov


def ols_lstsq(X: np.ndarray, y: np.ndarray) -> OLSResult:
    """Plain OLS via `np.linalg.lstsq` — not the taught technique here (Phase
    1's point is the *demeaning*, not re-deriving OLS itself; see practice
    097 for the QR route), just the fitting step applied after demeaning."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    XtX_inv = np.linalg.inv(X.T @ X)
    return OLSResult(beta=beta, fitted=fitted, resid=resid, XtX_inv=XtX_inv)


# TODO(human) — the within (demeaning) transformation
# ---------------------------------------------------------------------------
# Goal: implement the one-way within transformation — subtract each unit's
# own mean from y and from every column of X.
#
# Why this matters: this is the fixed-effects estimator. It is numerically
# and algebraically identical to including a full set of unit dummies in
# the regression (LSDV — "least squares dummy variables") and reading off
# the coefficients on x1/x2, but it never estimates the N unit-effect
# coefficients — for a panel with thousands of units, that's the
# difference between inverting a (k, k) matrix and a (N+k, N+k) one.
# Phase 3 proves the equivalence to LSDV directly, via the
# Frisch-Waugh-Lovell theorem; this phase just implements the shortcut.
#
# Steps:
#   1. For each unique unit_id, compute that unit's mean of y and of every
#      column of X (`pandas.DataFrame.groupby(...).transform("mean")` is
#      the natural tool — X and y can be handled as pandas objects here
#      even though the rest of this module works in numpy, because you
#      need a groupby-mean, not a matrix operation).
#   2. Subtract: y_tilde = y - mean_within_unit(y), same for each column of
#      X.
#   3. Return (y_tilde, X_tilde) as numpy arrays, same shape as the inputs.
#
# Do not add an intercept column to X_tilde — after demeaning it would be
# a column of exact zeros (see this module's docstring).
# ---------------------------------------------------------------------------
def within_demean(
    y: np.ndarray, X: np.ndarray, unit_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each unit's own mean from `y` and every column of `X`.

    `unit_id` is a (n,) array of unit labels, one per row. Returns
    `(y_tilde, X_tilde)`, the demeaned arrays, same shapes as the inputs.
    Any time-invariant column of `X` becomes all zeros.
    """
    raise NotImplementedError("TODO(human): implement the within (demeaning) transformation")


def fit_within(df: pd.DataFrame, y_col: str, x_cols: list[str], unit_col: str) -> OLSResult:
    """Demean by `unit_col`, then fit OLS on the demeaned data."""
    y = df[y_col].to_numpy()
    X = df[x_cols].to_numpy()
    unit_id = df[unit_col].to_numpy()
    y_tilde, X_tilde = within_demean(y, X, unit_id)
    return ols_lstsq(X_tilde, y_tilde)


def validate_against_libraries(df: pd.DataFrame) -> None:
    """Fit the within estimator, pyfixest, and linearmodels.PanelOLS on the
    same panel; assert all three agree on beta to numerical precision."""
    import pyfixest as pf
    from linearmodels.panel import PanelOLS

    ours = fit_within(df, "y", ["x1", "x2"], "unit_id")

    fixest_fit = pf.feols("y ~ x1 + x2 | unit_id", data=df)
    fixest_beta = fixest_fit.coef().to_numpy()

    panel_df = df.set_index(["unit_id", "time_id"])
    lm_fit = PanelOLS(
        panel_df["y"], panel_df[["x1", "x2"]], entity_effects=True
    ).fit()
    lm_beta = lm_fit.params.to_numpy()

    print(f"beta (ours):        {ours.beta}")
    print(f"beta (pyfixest):    {fixest_beta}")
    print(f"beta (linearmodels): {lm_beta}")
    assert np.allclose(ours.beta, fixest_beta, atol=1e-6), "within vs pyfixest mismatch"
    assert np.allclose(ours.beta, lm_beta, atol=1e-6), "within vs linearmodels mismatch"


def main() -> None:
    panel = generate_panel(**SMALL_PANEL, seed=0)
    print(f"True beta: {panel.beta_true}\n")
    try:
        validate_against_libraries(panel.df)
    except NotImplementedError as e:
        print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
