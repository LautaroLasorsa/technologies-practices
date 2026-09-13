"""Phase 3 — Diagnosing heteroskedasticity: the Breusch-Pagan test.

Gauss-Markov assumption A5 (homoskedasticity) says the error variance is
constant across observations. It's the assumption most likely to break in
real data (variance often grows with the scale of a regressor — bigger
firms have noisier revenue, richer households have noisier spending) and,
unlike a violation of exogeneity (A2), it doesn't bias beta_hat — it just
makes Phase 2's classical standard errors wrong. This phase implements the
standard diagnostic test for it.

Run on its own to see the test correctly distinguish the two scenarios:
    uv run python -m src._03_heteroskedasticity
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from ._01_ols_qr import ols_via_qr
from .datasets import load_dataset


# TODO(human) — Breusch-Pagan LM test for heteroskedasticity
# ---------------------------------------------------------------------------
# Goal: implement the Breusch-Pagan (1979) Lagrange-Multiplier test, which
# detects whether the error variance depends on the regressors (violating
# Gauss-Markov assumption A5).
#
# Why this matters: A5 is the assumption OLS's textbook standard errors
# need most in practice. This test is the standard first diagnostic before
# reaching for the robust standard errors of Phase 4 — running HC3
# unconditionally "just in case" is common in practice, but knowing *how*
# to test for the problem you're routing around is the point here.
#
# Procedure (Wooldridge, "Introductory Econometrics", ch. 8.3):
#   1. You already have OLS residuals `resid` from fitting y on X.
#   2. Auxiliary regression: regress `resid ** 2` on the same X (reuse
#      `ols_via_qr` — it's already implemented from Phase 1).
#   3. LM statistic = n * R^2 of that auxiliary regression, where
#      R^2 = 1 - RSS_aux / TSS_aux (TSS_aux uses the *mean* of resid**2).
#   4. Under H0 (homoskedasticity), LM ~ chi2(df) with df = X.shape[1] - 1
#      (regressors excluding the intercept).
#   5. p_value = 1 - stats.chi2(df).cdf(LM) (`stats` is scipy.stats,
#      already imported above).
# ---------------------------------------------------------------------------
def breusch_pagan_lm_test(X: np.ndarray, resid: np.ndarray) -> tuple[float, float]:
    """Breusch-Pagan LM test for heteroskedasticity.

    Returns `(lm_statistic, p_value)`. A small p-value (e.g. < 0.05)
    rejects the null of homoskedasticity.
    """
    n = X.shape[0]
    k = X.shape[1]
    resid_2 = resid**2
    residual_fit = ols_via_qr(X,resid_2)
    RSS_aux = np.sum(residual_fit.resid**2)
    TSS_aux = np.var(resid_2)*n
    lm_statistic = n * (1 - RSS_aux/TSS_aux)
    p_value  = 1 - stats.chi2(n-k).cdf(lm_statistic)
    return (lm_statistic,p_value)

def main() -> None:
    for scenario in ("homoskedastic", "heteroskedastic"):
        data = load_dataset(scenario, n=300, seed=0)
        try:
            fit = ols_via_qr(data.X, data.y)
        except NotImplementedError as e:
            print(f"(skipped — Phase 1 not implemented yet — {e})")
            return
        try:
            lm, pval = breusch_pagan_lm_test(data.X, fit.resid)
            print(f"{scenario:16s} LM={lm:7.3f}  p={pval:.4f}")
        except NotImplementedError as e:
            print(f"(skipped — {e})")


if __name__ == "__main__":
    main()
