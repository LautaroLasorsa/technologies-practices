"""Phase 4 — Cointegration and the error-correction model.

Phase 1's lesson was "never regress I(1) series on each other". This phase
is the exception that makes time-series econometrics interesting: if two
I(1) series share the *same* stochastic trend, some linear combination of
them is stationary. They are then **cointegrated**, the regression in
levels is no longer spurious, and the relationship it estimates is a
genuine long-run equilibrium.

Engle & Granger (1987) turned that into a two-step recipe: estimate the
long-run relation by OLS in levels, then test its residual for a unit
root. If the residual is stationary, the Granger Representation Theorem
guarantees an **error-correction model** exists — a short-run equation in
differences in which the lagged equilibrium error pulls the system back
toward its long-run path.

Run on its own:
    uv run python -m src._04_cointegration_ecm
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint

from ._01_stationarity_unit_roots import adf_test
from .datasets import generate_cointegrated_pair
from .regression import add_const, ols


@dataclass
class EngleGrangerResult:
    """Step 1 of Engle-Granger: the long-run relation and its unit-root test."""

    beta: float  # estimated cointegrating slope
    intercept: float
    equilibrium_error: np.ndarray  # (n,) residual y - intercept - beta*x
    adf_stat: float  # ADF statistic on that residual


@dataclass
class ECMResult:
    """Step 2 of Engle-Granger: the short-run error-correction equation."""

    speed: float  # adjustment speed (coefficient on the lagged equilibrium error)
    speed_tstat: float
    short_run: float  # coefficient on the contemporaneous difference of x
    resid: np.ndarray


# TODO(human) — Engle-Granger step 1: the cointegrating regression and its ADF test
# ---------------------------------------------------------------------------
# Goal: estimate the long-run relationship between two I(1) series by OLS
# in levels, then test whether its residual is stationary.
#
# Why this matters: this is precisely the regression Phase 1 told you never
# to run — and the whole content of cointegration is that it becomes
# legitimate when, and only when, the residual turns out to be I(0).
# Running the two phases back to back is the point: the same OLS call is
# nonsense on independent random walks and meaningful on a cointegrated
# pair, and nothing but a unit-root test on the residual distinguishes the
# two cases.
#
# A remarkable property makes step 1 work: under cointegration the OLS
# estimate of beta is *super-consistent* — it converges at rate T rather
# than the usual sqrt(T), so the long-run coefficient is pinned down very
# precisely even in modest samples, and any bias from omitting short-run
# dynamics vanishes asymptotically.
#
# Steps:
#   1. Regress y on a constant and x in levels: fit = ols(add_const(x), y).
#   2. intercept = fit.beta[0]; beta = fit.beta[1].
#   3. The equilibrium error is fit.resid — the estimated y - a - b*x.
#   4. Test it for a unit root by calling this practice's own
#      adf_test(fit.resid, lags=1, regression="n"). Use regression="n"
#      (no constant): OLS residuals already have mean zero by construction,
#      so including a constant would be redundant.
#   5. Return an EngleGrangerResult carrying the ADF statistic.
#
# CRITICAL — the critical values are NOT the usual ADF ones. The residual
# being tested was itself produced by an OLS fit that *minimised* its
# variance, which biases it toward looking stationary. The correct
# reference values (Engle-Granger / MacKinnon, more negative than standard
# ADF) are what statsmodels' `coint()` reports; the scaffolded
# `compare_with_statsmodels` below prints them next to your statistic.
#
# Null hypothesis: the residual HAS a unit root, i.e. NO cointegration.
# Rejecting it is evidence that the two series are cointegrated.
# ---------------------------------------------------------------------------
def engle_granger_step1(y: np.ndarray, x: np.ndarray) -> EngleGrangerResult:
    """Estimate the cointegrating relation and ADF-test its residual.

    Regresses `y` on a constant and `x` in levels; returns the slope, the
    residual series (the equilibrium error) and the ADF statistic computed
    on that residual with `regression="n"`.
    """
    raise NotImplementedError("TODO(human): implement Engle-Granger step 1")


# TODO(human) — Engle-Granger step 2: the error-correction model
# ---------------------------------------------------------------------------
# Goal: fit the short-run dynamics, with the lagged equilibrium error from
# step 1 as a regressor, and read off the speed of adjustment.
#
# Why this matters: the ECM is what makes cointegration *useful* rather
# than merely true. It separates two things a levels regression conflates —
# the long-run equilibrium (step 1's beta) and the short-run dynamics plus
# the rate at which deviations are corrected. The Granger Representation
# Theorem says the two representations are equivalent: cointegration exists
# if and only if an ECM exists. Every "the system is out of equilibrium,
# how fast does it revert?" question — inventory vs. demand, price vs.
# fundamentals, queue length vs. service rate — is this coefficient.
#
# The equation (everything in differences except the correction term):
#     Delta y_t = c + gamma * Delta x_t + alpha * z_{t-1} + eps_t
# where z_{t-1} is the previous period's equilibrium error from step 1.
#
# Steps:
#   1. dy = np.diff(y) and dx = np.diff(x) — both length n-1, where
#      dy[i] = y[i+1] - y[i].
#   2. Align the lagged equilibrium error with those differences: row i of
#      dy corresponds to time i+1, so the term you need is z at time i,
#      i.e. eq_error[:-1].
#   3. X = add_const(np.column_stack([dx, z_lagged])), fit against dy.
#   4. short_run = coefficient on dx; speed = coefficient on z_lagged,
#      with its t-statistic from fit.tstat.
#
# How to read `speed` (alpha): it must be NEGATIVE for the system to be
# stable — a positive equilibrium error (y above its long-run value) has to
# push Delta y down. Its magnitude is the fraction of the gap closed each
# period: alpha = -0.3 means roughly 30% of a deviation is corrected per
# period. A value that is positive, or statistically indistinguishable from
# zero, means no error correction — and therefore no cointegration, which
# is a second, independent check on step 1's verdict.
# ---------------------------------------------------------------------------
def error_correction_model(y: np.ndarray, x: np.ndarray, equilibrium_error: np.ndarray) -> ECMResult:
    """Fit the ECM `d(y) ~ const + d(x) + lagged equilibrium error`.

    `equilibrium_error` is the (n,) residual from step 1. Returns an
    `ECMResult`; `speed` should be negative and significant when the pair
    is genuinely cointegrated.
    """
    raise NotImplementedError("TODO(human): implement the error-correction model")


def compare_with_statsmodels(y: np.ndarray, x: np.ndarray) -> None:
    """Print the hand-built Engle-Granger statistic next to statsmodels' `coint`."""
    ours = engle_granger_step1(y, x)
    stat, pvalue, crit = coint(y, x, trend="c", autolag=None, maxlag=1)
    print(f"  cointegrating beta (ours): {ours.beta:+.4f}")
    print(f"  ADF on residual (ours):    {ours.adf_stat:8.4f}")
    print(f"  coint() statistic:         {stat:8.4f}   (p={pvalue:.4f})")
    print(f"  Engle-Granger crit values: 1%={crit[0]:.2f}  5%={crit[1]:.2f}  10%={crit[2]:.2f}")
    print("    ^ note these are more negative than the plain ADF values -- that is the")
    print("      correction for testing a residual that OLS already variance-minimised.")


def main() -> None:
    data = generate_cointegrated_pair(n=300, seed=0, beta=2.0)
    print(f"Cointegrated pair, true beta = {data.beta_true}")
    print("Both series are I(1); the combination y - beta*x is stationary.\n")

    try:
        eg = engle_granger_step1(data.y, data.x)
    except NotImplementedError as e:
        print(f"(skipped step 1 — {e})")
        return
    print("Step 1 — long-run relation:")
    compare_with_statsmodels(data.y, data.x)
    corr = float(np.corrcoef(eg.equilibrium_error, data.equilibrium_error)[0, 1])
    print(f"  corr(estimated equilibrium error, true one): {corr:+.4f}")

    try:
        ecm = error_correction_model(data.y, data.x, eg.equilibrium_error)
    except NotImplementedError as e:
        print(f"\n(skipped step 2 — {e})")
        return
    print("\nStep 2 — error-correction model:")
    print(f"  short-run coefficient on d(x): {ecm.short_run:+.4f}")
    print(f"  adjustment speed alpha:        {ecm.speed:+.4f}  (t={ecm.speed_tstat:+.2f})")
    half_life = np.log(0.5) / np.log(1.0 + ecm.speed) if -2 < ecm.speed < 0 else float("nan")
    print(f"  implied half-life of a deviation: {half_life:.2f} periods")


if __name__ == "__main__":
    main()
