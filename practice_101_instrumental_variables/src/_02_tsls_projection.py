"""Phase 2 -- 2SLS as a single projection.

Phase 1 computed 2SLS as two literal regressions. This phase implements
the algebraically equivalent single-matrix formula that reference
implementations (`statsmodels`' `IV2SLS`, `linearmodels.IV2SLS`) actually
use, and validates all three routes -- two-stage, single-projection, and
`linearmodels` -- against each other.

Run standalone:
    uv run python -m src._02_tsls_projection
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from linearmodels.iv import IV2SLS

from ._01_tsls_two_stage import TSLSFit, tsls_two_stage
from .datasets import simulate_iv_linear


# TODO(human) -- 2SLS as a single projection onto the instrument space
# ---------------------------------------------------------------------------
# Goal: implement 2SLS as one matrix formula instead of two regressions --
# the same estimator, viewed as a single projection.
#
# Why this matters: let X = [exog, endog] be *all* regressors (including
# the endogenous one) and W = [exog, instruments] be the full instrument
# set (exogenous regressors instrument themselves). Phase 1's two
# regressions are algebraically identical to:
#     beta_2sls = (X' P_W X)^-1 X' P_W y
# where P_W = W (W'W)^-1 W' is the projection matrix onto the column space
# of W. Seeing 2SLS this way is what makes its generalization to
# over-identified models (more instruments than endogenous regressors --
# not needed by this practice's just-identified design) and to GMM
# obvious: 2SLS is GMM with the weighting matrix (W'W)^-1.
#
# Steps:
#   1. Build X = hstack([exog, endog.reshape(-1, 1)]) and
#      W = hstack([exog, instruments]).
#   2. Form P_W = W @ solve(W.T @ W, W.T) (an (n, n) projection matrix --
#      fine at this practice's sample sizes; production code never forms
#      P_W explicitly, but the explicit form is the clearest way to see
#      what "projection" means here).
#   3. Solve beta = solve(X.T @ P_W @ X, X.T @ P_W @ y).
#   4. Split beta into beta_exog (all but the last entry) and beta_endog
#      (the last entry, endog's coefficient), and compute fitted/resid the
#      same way as Phase 1.
#
# Do not call tsls_two_stage from inside this function -- the point is to
# arrive at the same number through the projection formula alone.
# ---------------------------------------------------------------------------
def tsls_projection(exog: np.ndarray, endog: np.ndarray, instruments: np.ndarray, y: np.ndarray) -> TSLSFit:
    """2SLS as a single projection onto the instrument space.

    Same inputs and return type as `tsls_two_stage` -- the two functions
    must agree to numerical precision on identical inputs.
    """
    raise NotImplementedError("TODO(human): implement 2SLS as a single projection")


def compare_to_linearmodels(data) -> None:
    """Fit 2SLS three ways -- two-stage, single projection, `linearmodels` -- and compare."""
    fit_two_stage = tsls_two_stage(data.exog, data.endog, data.instruments, data.y)
    fit_projection = tsls_projection(data.exog, data.endog, data.instruments, data.y)

    df = pd.DataFrame({
        "y": data.y,
        "d": data.endog,
        "z": data.instruments[:, 0],
        "const": data.exog[:, 0],
    })
    lm_fit = IV2SLS(df["y"], df[["const"]], df["d"], df[["z"]]).fit(cov_type="unadjusted")

    print(f"True beta:                  {data.beta_true:.4f}")
    print(f"2SLS (two-stage, Phase 1):   {fit_two_stage.beta_endog:.4f}")
    print(f"2SLS (projection, Phase 2):  {fit_projection.beta_endog:.4f}")
    print(f"2SLS (linearmodels):         {lm_fit.params['d']:.4f}")

    diff = abs(fit_two_stage.beta_endog - fit_projection.beta_endog)
    assert diff < 1e-8, "two-stage and single-projection 2SLS should agree to numerical precision"
    diff_lm = abs(fit_projection.beta_endog - lm_fit.params["d"])
    assert diff_lm < 1e-6, "hand-rolled 2SLS should match linearmodels.IV2SLS"


def main() -> None:
    data = simulate_iv_linear(n=500, pi=1.0, seed=0)
    try:
        compare_to_linearmodels(data)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")


if __name__ == "__main__":
    main()
