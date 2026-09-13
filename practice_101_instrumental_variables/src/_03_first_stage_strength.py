"""Phase 3 -- First-stage strength and the weak-instrument problem.

2SLS is consistent for any nonzero instrument relevance in principle, but
at finite sample sizes a weak instrument (a first-stage coefficient close
to zero) makes 2SLS behave more and more like the very OLS estimator it
was meant to fix, and its confidence intervals stop covering the truth at
their nominal rate. The first-stage F statistic -- a simple joint test
that the excluded instrument(s) predict the endogenous regressor -- is the
standard diagnostic (Staiger & Stock, 1997; Stock & Yogo, 2005 give the
widely cited "F < 10 is a weak instrument" rule of thumb).

Run standalone:
    uv run python -m src._03_first_stage_strength
"""
from __future__ import annotations

import numpy as np

from ._02_tsls_projection import tsls_projection
from .datasets import simulate_iv_linear


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


# TODO(human) -- first-stage F statistic for excluded-instrument relevance
# ---------------------------------------------------------------------------
# Goal: implement the F test of joint significance of the excluded
# instrument(s) in the first-stage regression of `endog` on
# `[exog, instruments]`.
#
# Why this matters: relevance (Cov(Z, D) != 0) is the one IV condition
# that *can* be checked in the data, unlike the exclusion restriction. The
# first-stage F statistic tests whether the excluded instruments jointly
# add explanatory power for D beyond what `exog` alone provides -- a small
# F means the instrument barely moves D, which is exactly the regime
# where 2SLS's finite-sample bias creeps back toward OLS's (this phase's
# already-scaffolded simulation below makes that visible).
#
# Steps (classic restricted-vs-unrestricted F test):
#   1. Unrestricted first stage: regress endog on
#      W_ur = hstack([exog, instruments]); compute RSS_ur.
#   2. Restricted first stage: regress endog on W_r = exog alone; compute
#      RSS_r.
#   3. q = number of excluded instruments (columns of `instruments`);
#      k = number of columns of W_ur; n = number of observations.
#   4. F = ((RSS_r - RSS_ur) / q) / (RSS_ur / (n - k)).
# ---------------------------------------------------------------------------
def first_stage_f_stat(exog: np.ndarray, endog: np.ndarray, instruments: np.ndarray) -> float:
    """F statistic for joint significance of the excluded instrument(s) in the first stage.

    Larger is stronger; the Stock-Yogo rule of thumb treats F < 10 as a
    weak-instrument warning.
    """
    raise NotImplementedError("TODO(human): implement the first-stage F statistic")


def weak_instrument_simulation(
    pi_grid: np.ndarray, n: int = 300, n_sims: int = 200, seed: int = 0
) -> dict[str, np.ndarray]:
    """Sweep the first-stage coefficient `pi` (instrument strength); at each
    value, simulate `n_sims` fresh datasets and record 2SLS's average
    first-stage F, its bias against the true beta, its 95% CI coverage,
    and OLS's bias for comparison.

    Fully scaffolded -- the simulation loop is infrastructure; Phase 3's
    teaching content is the F statistic it calls.
    """
    rng = np.random.default_rng(seed)
    avg_f = np.empty(len(pi_grid))
    bias_2sls = np.empty(len(pi_grid))
    bias_ols = np.empty(len(pi_grid))
    coverage = np.empty(len(pi_grid))

    for i, pi in enumerate(pi_grid):
        f_stats, estimates, ols_estimates, covered = [], [], [], []
        for _ in range(n_sims):
            data = simulate_iv_linear(n=n, pi=float(pi), seed=int(rng.integers(0, 2**31 - 1)))
            f_stats.append(first_stage_f_stat(data.exog, data.endog, data.instruments))

            fit = tsls_projection(data.exog, data.endog, data.instruments, data.y)
            estimates.append(fit.beta_endog)

            # Homoskedastic 2SLS variance: sigma_hat^2 * (X_hat'X_hat)^-1,
            # with X_hat the projected [exog, endog] -- the same classical
            # formula practice 097 introduced, applied to the 2SLS fitted
            # regressors instead of the raw ones.
            X = np.column_stack([data.exog, data.endog])
            W = np.column_stack([data.exog, data.instruments])
            X_hat = W @ np.linalg.solve(W.T @ W, W.T @ X)
            sigma_hat_sq = np.sum(fit.resid ** 2) / (len(data.y) - X.shape[1])
            se_endog = np.sqrt(sigma_hat_sq * np.linalg.inv(X_hat.T @ X_hat)[-1, -1])
            ci_lo, ci_hi = fit.beta_endog - 1.96 * se_endog, fit.beta_endog + 1.96 * se_endog
            covered.append(ci_lo <= data.beta_true <= ci_hi)

            ols_estimates.append(_ols(X, data.y)[-1])

        avg_f[i] = float(np.mean(f_stats))
        bias_2sls[i] = float(np.mean(estimates) - data.beta_true)
        bias_ols[i] = float(np.mean(ols_estimates) - data.beta_true)
        coverage[i] = float(np.mean(covered))

    return {"avg_f": avg_f, "bias_2sls": bias_2sls, "bias_ols": bias_ols, "coverage": coverage}


def main() -> None:
    data = simulate_iv_linear(n=300, pi=1.0, seed=0)
    try:
        f_strong = first_stage_f_stat(data.exog, data.endog, data.instruments)
    except NotImplementedError as e:
        print(f"(skipped -- {e})")
        return

    weak = simulate_iv_linear(n=300, pi=0.05, seed=0)
    f_weak = first_stage_f_stat(weak.exog, weak.endog, weak.instruments)
    print(f"First-stage F (pi=1.00, strong instrument): {f_strong:.2f}")
    print(f"First-stage F (pi=0.05, weak instrument):   {f_weak:.2f}")


if __name__ == "__main__":
    main()
