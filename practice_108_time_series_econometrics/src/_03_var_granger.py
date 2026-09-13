"""Phase 3 — VAR systems, Granger causality and impulse responses.

A univariate ARIMA model explains a series by its own past. A vector
autoregression (VAR) lets several series explain each other: every
variable is regressed on lags of *every* variable, so the system's
dynamics are estimated jointly. Two questions follow immediately.

"Does x help predict y beyond y's own past?" is **Granger causality** — a
joint F-test on the x-lag coefficients in the y equation. It is a
statement about predictive content, not about causation: Granger causality
is neither necessary nor sufficient for a causal effect, and the name is
the single most over-read term in applied time series.

"What happens to the system after a one-off shock?" is the **impulse
response function** — the dynamic path traced out by iterating the
estimated VAR forward, which is where a VAR's economic interpretation
actually lives.

Run on its own:
    uv run python -m src._03_var_granger
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR

from .datasets import generate_var_system
from .regression import add_const, build_lag_matrix, ols


@dataclass
class GrangerResult:
    """Outcome of one Granger-causality F-test."""

    f_stat: float
    p_value: float
    df_num: int  # restrictions tested (= number of excluded lag coefficients)
    df_den: int  # residual degrees of freedom of the unrestricted model
    rss_restricted: float
    rss_unrestricted: float


# TODO(human) — the Granger-causality F-test
# ---------------------------------------------------------------------------
# Goal: test whether `x` Granger-causes `y` by comparing two nested
# regressions with an F-test built from their residual sums of squares.
#
# Why this matters: this is the classic nested-model F-test, and seeing it
# assembled from two RSS values is worth more than any amount of reading
# about it — the identical machinery tests any linear restriction (lag
# exclusion, coefficient equality, joint significance) anywhere else in
# econometrics. It is also the honest way to see what Granger causality
# *is*: a claim that adding x's past to a forecasting equation reduces the
# residual variance by more than chance would, and nothing more.
#
# The two models (both on the same sample, which matters — the F-test is
# only valid if the two fits use identical rows):
#   restricted:   y_t = c + sum_j a_j*y_{t-j}                     + eps_t
#   unrestricted: y_t = c + sum_j a_j*y_{t-j} + sum_j b_j*x_{t-j} + eps_t
#
# Steps:
#   1. Build the aligned lag blocks. `build_lag_matrix(y, lags)` gives
#      (y_trimmed, y_lags); call it on `x` too and keep only the lag block
#      — x's lags are already row-aligned with y's because both series have
#      the same length and the same number of leading rows are dropped.
#   2. X_restricted   = add_const(y_lags)
#      X_unrestricted = add_const(np.column_stack([y_lags, x_lags]))
#   3. Fit both with ols(...) against the same y_trimmed.
#   4. q  = number of restrictions = `lags` (one per excluded x coefficient)
#      df_den = fit_unrestricted.n - fit_unrestricted.k
#      F = ((RSS_r - RSS_u) / q) / (RSS_u / df_den)
#   5. p-value from the upper tail: stats.f.sf(F, q, df_den).
#
# Null hypothesis: all b_j = 0, i.e. x does NOT Granger-cause y. A small
# p-value rejects that, so it is evidence that x carries predictive content
# about y's future beyond y's own history.
#
# Expected on this practice's data: the DGP has x -> y at lag 1 and no
# feedback, so testing x -> y should reject decisively while y -> x should
# not — an asymmetry that only shows up if you run the test both ways,
# which is exactly what the scaffolded `main` does.
# ---------------------------------------------------------------------------
def granger_causality_f_test(y: np.ndarray, x: np.ndarray, lags: int = 2) -> GrangerResult:
    """F-test of the null that `x` does not Granger-cause `y`.

    Compares a restricted regression of `y` on its own `lags` lags against
    an unrestricted one that adds `lags` lags of `x`. Returns a
    `GrangerResult`; a small `p_value` rejects "no Granger causality".
    """
    raise NotImplementedError("TODO(human): implement the Granger-causality F-test")


def fit_var(y: np.ndarray, x: np.ndarray, maxlags: int = 2):
    """Fit a bivariate VAR with statsmodels; columns are ordered [y, x]."""
    frame = pd.DataFrame({"y": y, "x": x})
    return VAR(frame).fit(maxlags=maxlags)


def impulse_response(results, periods: int = 12, shock: str = "x", response: str = "y"):
    """Impulse response of `response` to a one-SD orthogonalised shock in `shock`.

    Returns `(responses, lower, upper)`, each a (periods + 1,) array: the
    response path and its asymptotic 95% band. Orthogonalised means the
    shocks are Cholesky-decorrelated first, so the ordering of the columns
    in `fit_var` is itself an identifying assumption — the standard caveat
    attached to every VAR impulse response.
    """
    irf = results.irf(periods)
    names = list(results.names)
    i_shock, i_resp = names.index(shock), names.index(response)
    responses = irf.orth_irfs[:, i_resp, i_shock]
    stderr = irf.stderr(orth=True)[:, i_resp, i_shock]
    return responses, responses - 1.96 * stderr, responses + 1.96 * stderr


def compare_with_statsmodels(y: np.ndarray, x: np.ndarray, lags: int = 2) -> None:
    """Print the hand-built F-test next to statsmodels' VAR `test_causality`."""
    ours = granger_causality_f_test(y, x, lags=lags)
    ref = fit_var(y, x, maxlags=lags).test_causality("y", ["x"], kind="f")
    print(f"  ours:        F={ours.f_stat:8.3f}  p={ours.p_value:.3e}  df=({ours.df_num}, {ours.df_den})")
    print(f"  statsmodels: F={ref.test_statistic:8.3f}  p={ref.pvalue:.3e}  df={ref.df}")


def main() -> None:
    data = generate_var_system(n=300, seed=0)
    print("True VAR(1) coefficient matrix (rows = equations for [y, x]):")
    print(data.coef_matrix)
    print("  -> x Granger-causes y; y does NOT Granger-cause x\n")

    try:
        for label, (dep, indep) in (("x -> y", (data.y, data.x)), ("y -> x", (data.x, data.y))):
            res = granger_causality_f_test(dep, indep, lags=2)
            verdict = "REJECT no-causality" if res.p_value < 0.05 else "fail to reject"
            print(f"  {label}:  F={res.f_stat:7.3f}  p={res.p_value:.4f}  -> {verdict}")
    except NotImplementedError as e:
        print(f"  (skipped — {e})")
        return

    print("\nAgainst statsmodels (x -> y):")
    compare_with_statsmodels(data.y, data.x, lags=2)

    results = fit_var(data.y, data.x, maxlags=2)
    responses, lower, upper = impulse_response(results, periods=10)
    print("\nIRF of y to a one-SD shock in x (first 5 periods):")
    for h in range(5):
        print(f"  h={h}: {responses[h]:+.4f}  [{lower[h]:+.4f}, {upper[h]:+.4f}]")


if __name__ == "__main__":
    main()
