"""Phase 1 -- Two-stage least squares, as two literal regressions.

Plain OLS of `y` on an endogenous regressor `D` is biased whenever `D` is
correlated with the structural error -- the endogeneity problem this whole
practice is about. An instrument `Z` fixes this if it satisfies two
conditions (this practice's CLAUDE.md covers both in depth): *relevance*
(Z is correlated with D, checked by Phase 3's F-statistic) and the
*exclusion restriction* (Z affects y only through D, never directly --
untestable, simply assumed by the DGP in `datasets.py`). Two-stage least
squares is the textbook fix, and it is introduced, in every
intro-econometrics course, as exactly two literal OLS regressions -- the
form implemented here, before Phase 2 shows the algebraically equivalent
single-matrix formula.

Run standalone:
    uv run python -m src._01_tsls_two_stage
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import simulate_iv_linear


@dataclass
class TSLSFit:
    """Result of fitting 2SLS to one just-identified IV dataset."""

    beta_exog: np.ndarray  # coefficients on the included exogenous regressors (here: intercept)
    beta_endog: float      # the 2SLS estimate of the endogenous regressor's coefficient
    d_hat: np.ndarray       # (n,) first-stage fitted values for the endogenous regressor
    fitted: np.ndarray      # (n,) second-stage fitted values for y
    resid: np.ndarray       # (n,) y - fitted


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Plain OLS via lstsq -- infrastructure, not the taught estimator."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# TODO(human) -- two-stage least squares, as two literal OLS regressions
# ---------------------------------------------------------------------------
# Goal: implement 2SLS exactly the way it's introduced in every
# intro-econometrics course -- as two separate, literal OLS regressions --
# before Phase 2 shows the equivalent single-matrix formula.
#
# Why this matters: two-stage least squares purges the endogenous
# regressor D of its correlation with the structural error by first
# regressing D on everything exogenous (including the instrument Z),
# keeping only the part of D that *is* explained by exogenous variation,
# and using that purified D_hat in place of D in the second-stage
# regression. This is what "using an instrument" concretely means.
#
# Steps:
#   1. First stage: regress endog (D) on the full instrument matrix
#      W = hstack([exog, instruments]) via OLS -> get D_hat = W @ pi_hat.
#   2. Second stage: regress y on hstack([exog, D_hat]) via OLS -> the
#      coefficient on the D_hat column is the 2SLS estimate of endog's
#      effect; the exog coefficients come along for free.
#   3. Compute fitted values and residuals from the *original* endog (not
#      D_hat), so residuals reflect the actual structural equation:
#      fitted = exog @ beta_exog + endog * beta_endog.
#
# Do not regress y directly on the original (unpurified) D -- that's just
# OLS, the very thing this estimator exists to fix.
# ---------------------------------------------------------------------------
def tsls_two_stage(exog: np.ndarray, endog: np.ndarray, instruments: np.ndarray, y: np.ndarray) -> TSLSFit:
    """Two-stage least squares via two explicit OLS regressions.

    `exog` is (n, kx) included exogenous regressors, `endog` is (n,) the
    single endogenous regressor, `instruments` is (n, kz) excluded
    instruments, `y` is (n,) the outcome. Returns a `TSLSFit`.
    """
    raise NotImplementedError("TODO(human): implement two-stage least squares")


def compare_ols_vs_2sls(data) -> None:
    """Fit naive OLS and two-stage 2SLS on the same dataset; print both against the truth."""
    X_ols = np.column_stack([data.exog, data.endog])
    beta_ols = _ols(X_ols, data.y)[-1]
    fit = tsls_two_stage(data.exog, data.endog, data.instruments, data.y)
    print(f"True beta:      {data.beta_true:.3f}")
    print(f"OLS (biased):   {beta_ols:.3f}")
    print(f"2SLS (Phase 1): {fit.beta_endog:.3f}")


def main() -> None:
    data = simulate_iv_linear(n=500, pi=1.0, seed=0)
    try:
        compare_ols_vs_2sls(data)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
