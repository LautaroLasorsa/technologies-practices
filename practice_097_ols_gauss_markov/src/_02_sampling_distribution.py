"""Phase 2 — The sampling distribution of beta-hat.

beta_hat is a random variable: it depends on the sample drawn. The
Gauss-Markov theorem is a claim about that randomness — under A1-A5,
beta_hat is unbiased and has the *smallest* variance among linear unbiased
estimators (BLUE), and that variance has a closed form. This phase makes
"beta_hat is random" concrete via Monte Carlo: redraw the dataset many
times, refit OLS every time, and look at the empirical spread of beta_hat
against the formula's prediction.

Run on its own to compare the analytical and empirical standard errors:
    uv run python -m src._02_sampling_distribution
"""
from __future__ import annotations

import numpy as np

from ._01_ols_qr import ols_via_qr
from .datasets import TRUE_BETA, load_dataset


# TODO(human) — classical (homoskedastic) OLS variance-covariance estimator
# ---------------------------------------------------------------------------
# Goal: implement the textbook Gauss-Markov variance formula
#   Var(beta_hat) = sigma_hat^2 * (X'X)^-1
# where sigma_hat^2 = RSS / (n - k) is the unbiased estimator of the error
# variance (n = observations, k = number of regressors including the
# intercept, RSS = sum of squared residuals).
#
# Why this matters: this formula is *only* valid under A1-A5 (linearity,
# strict exogeneity, no perfect collinearity, homoskedasticity, no
# autocorrelation). Phases 3-5 exist because this formula silently gives
# the wrong standard errors the moment A5 (or independence across
# observations) breaks. This phase's Monte Carlo simulation is what makes
# "silently wrong" visible: under the homoskedastic scenario, this
# formula's diagonal should match the *empirical* variance of beta_hat
# across simulated resamples almost exactly; under the other two
# scenarios (used in Phases 3-5) it won't.
#
# Steps:
#   1. sigma_hat_sq = sum(resid ** 2) / (n - k).
#   2. vcov = sigma_hat_sq * XtX_inv.
# ---------------------------------------------------------------------------
def ols_vcov_homoskedastic(X: np.ndarray, resid: np.ndarray, XtX_inv: np.ndarray) -> np.ndarray:
    """Classical OLS covariance matrix under homoskedastic, independent errors.

    Returns the (k, k) covariance matrix; `np.sqrt(np.diag(...))` gives the
    standard errors reported by every basic regression table.
    """
    raise NotImplementedError("TODO(human): implement the classical OLS vcov formula")


def simulate_beta_hat_distribution(n: int = 200, n_sims: int = 500, seed: int = 0) -> np.ndarray:
    """Monte Carlo: redraw the dataset `n_sims` times, refit OLS, collect beta_hat.

    This is the empirical sampling distribution the Gauss-Markov theorem
    makes claims about — fully scaffolded, since generating data and
    looping is not the teaching point here (implementing the estimator in
    Phase 1 is what makes this loop meaningful).
    """
    rng = np.random.default_rng(seed)
    betas = np.empty((n_sims, len(TRUE_BETA)))
    for i in range(n_sims):
        data = load_dataset("homoskedastic", n=n, seed=int(rng.integers(0, 2**31 - 1)))
        fit = ols_via_qr(data.X, data.y)
        betas[i] = fit.beta
    return betas


def main() -> None:
    data = load_dataset("homoskedastic", n=200, seed=0)
    try:
        fit = ols_via_qr(data.X, data.y)
    except NotImplementedError as e:
        print(f"(skipped — Phase 1 not implemented yet — {e})")
        return

    try:
        vcov = ols_vcov_homoskedastic(data.X, fit.resid, fit.XtX_inv)
        print(f"Analytical SE (one sample):     {np.sqrt(np.diag(vcov))}")
    except NotImplementedError as e:
        print(f"(skipped analytical vcov — {e})")

    betas = simulate_beta_hat_distribution(n=200, n_sims=500, seed=1)
    empirical_se = betas.std(axis=0, ddof=1)
    print(f"Empirical SE (Monte Carlo, 500x): {empirical_se}")


if __name__ == "__main__":
    main()
