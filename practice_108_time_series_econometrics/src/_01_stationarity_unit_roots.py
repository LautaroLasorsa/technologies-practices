"""Phase 1 — Spurious regression, stationarity, and unit-root testing.

Every standard-error formula in cross-sectional econometrics rests on
observations being independent draws. Time series are not: today's value
is mostly yesterday's value. When two *unrelated* series both carry a unit
root, regressing one on the other produces a large R^2 and a huge
t-statistic anyway — Granger & Newbold (1974) called this the spurious
regression, and it is the reason "significant" means nothing until
stationarity has been established.

This phase makes that failure visible with a Monte Carlo (scaffolded),
then builds the two diagnostics that catch it: the Durbin-Watson statistic
(the residual-autocorrelation tell), and the Augmented Dickey-Fuller
regression (the formal unit-root test).

Run on its own:
    uv run python -m src._01_stationarity_unit_roots
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.stattools import durbin_watson as sm_durbin_watson
from statsmodels.tsa.stattools import adfuller, kpss

from .datasets import generate_random_walk, generate_spurious_pair, generate_trend_stationary
from .regression import add_const, build_lag_matrix, ols


# TODO(human) — the Durbin-Watson statistic
# ---------------------------------------------------------------------------
# Goal: implement the Durbin-Watson statistic, the classical test for
# first-order autocorrelation in regression residuals.
#
# Why this matters: the spurious regression's giveaway is not in the
# coefficient or the R^2 — both look great — it is in the *residuals*.
# Regressing one random walk on another leaves residuals that are
# themselves a random walk: massively positively autocorrelated. Granger &
# Newbold's (1974) famous rule of thumb is "suspect a spurious regression
# whenever R^2 > DW", and this statistic is the DW half of it. It is also
# the cheapest possible check that the independence assumption behind every
# classical standard error has failed.
#
# Formula:
#   DW = sum_{t=2..n} (e_t - e_{t-1})^2 / sum_{t=1..n} e_t^2
#
# How to read it: DW is approximately 2 * (1 - rho_hat), where rho_hat is
# the first-order autocorrelation of the residuals. So
#   DW ~ 2  -> no autocorrelation (what you want),
#   DW -> 0 -> strong *positive* autocorrelation (the spurious case),
#   DW -> 4 -> strong negative autocorrelation.
# Its range is [0, 4].
#
# Implementation note: `np.diff` gives you the numerator's differences in
# one call — no Python loop is needed.
# ---------------------------------------------------------------------------
def durbin_watson(resid: np.ndarray) -> float:
    """Durbin-Watson statistic for first-order residual autocorrelation.

    Takes a (n,) residual array, returns a float in [0, 4]; ~2 means no
    autocorrelation, near 0 means strong positive autocorrelation.
    """
    raise NotImplementedError("TODO(human): implement the Durbin-Watson statistic")


@dataclass
class ADFResult:
    """Output of one Augmented Dickey-Fuller regression."""

    stat: float  # t-ratio on the lagged-level coefficient (the ADF statistic)
    gamma: float  # the lagged-level coefficient itself
    lags: int  # number of lagged differences included
    nobs: int  # observations used by the auxiliary regression


# TODO(human) — the Augmented Dickey-Fuller test statistic
# ---------------------------------------------------------------------------
# Goal: build the ADF auxiliary regression by hand and return the t-ratio
# on the lagged-level term. This is the single most-used test in applied
# time-series work, and building it once shows that it is "just" an OLS
# t-statistic on a cleverly differenced design.
#
# Why this matters: a unit root means y_t = y_{t-1} + eps_t, i.e. shocks
# never die out. Writing the AR(1) model y_t = rho*y_{t-1} + eps_t in
# differences gives
#     Delta y_t = gamma * y_{t-1} + eps_t,  with gamma = rho - 1,
# so "is there a unit root?" becomes "is gamma = 0?". The *augmented* part
# adds p lagged differences to soak up any extra serial correlation, so the
# residual is white noise and the t-ratio is well behaved.
#
# The regression to build (for t = p+1 .. T-1):
#     Delta y_t = [const/trend] + gamma * y_{t-1}
#                 + sum_{j=1..p} delta_j * Delta y_{t-j} + eps_t
#
# Steps:
#   1. dy = np.diff(y)  -> length T-1, where dy[i] = y[i+1] - y[i].
#   2. Split dy into its own lags with the shared helper:
#        dy_trimmed, dy_lags = build_lag_matrix(dy, lags)
#      `dy_trimmed` is the dependent variable; `dy_lags` has `lags` columns.
#   3. The lagged *level* aligned with those rows is y[lags : len(y) - 1].
#   4. Stack [y_lagged, dy_lags] with np.column_stack, then add the
#      deterministic terms: for regression="c" wrap it in add_const(...);
#      for regression="ct" also append a time trend column
#      np.arange(len(dy_trimmed), dtype=float) *before* add_const.
#   5. Fit with ols(X, dy_trimmed). The ADF statistic is the t-statistic on
#      the y_lagged column — mind its position once a constant (and trend)
#      have been prepended by add_const.
#
# CRITICAL — the distribution is NOT Student's t: under the null of a unit
# root, y_{t-1} is non-stationary, so this t-ratio follows the
# Dickey-Fuller distribution, whose critical values are far more negative
# than the familiar -1.96 (about -2.86 at 5% with a constant). Never
# compare this statistic to a normal or t table; the scaffolded
# `compare_with_statsmodels` below prints statsmodels' correct critical
# values next to yours.
#
# Null hypothesis: gamma = 0, i.e. a unit root IS present. Rejecting (a
# sufficiently negative statistic) is evidence *for* stationarity.
# ---------------------------------------------------------------------------
def adf_test(y: np.ndarray, lags: int = 1, regression: str = "c") -> ADFResult:
    """Augmented Dickey-Fuller statistic from a hand-built auxiliary regression.

    `lags` is the number of lagged differences to include; `regression` is
    "c" (constant), "ct" (constant and trend) or "n" (neither). Returns an
    `ADFResult`; the statistic is the t-ratio on the lagged level.
    """
    raise NotImplementedError("TODO(human): implement the ADF auxiliary regression and its t-statistic")


def regress_levels(y: np.ndarray, x: np.ndarray):
    """OLS of `y` on a constant and `x`, in levels — the spurious regression itself."""
    return ols(add_const(x), y)


def simulate_spurious_regressions(n_sims: int = 500, n: int = 300, seed: int = 0) -> dict[str, np.ndarray]:
    """Monte Carlo: regress two fresh *independent* random walks, n_sims times.

    Returns arrays of the slope t-statistic, R^2 and Durbin-Watson statistic
    across replications. Fully scaffolded — the simulation is the evidence,
    not the exercise.
    """
    tstats, r2s, dws = np.empty(n_sims), np.empty(n_sims), np.empty(n_sims)
    for i in range(n_sims):
        pair = generate_spurious_pair(n=n, seed=seed + 7919 * i)
        fit = regress_levels(pair.y, pair.x)
        tstats[i] = fit.tstat[1]
        r2s[i] = fit.r2
        dws[i] = durbin_watson(fit.resid)
    return {"tstat": tstats, "r2": r2s, "dw": dws}


def rejection_rate(tstats: np.ndarray, crit: float = 1.96) -> float:
    """Share of replications where |t| exceeds the nominal 5% critical value."""
    return float(np.mean(np.abs(tstats) > crit))


def kpss_summary(series: np.ndarray, regression: str = "c") -> tuple[float, float]:
    """KPSS statistic and p-value from statsmodels (scaffolded reference).

    KPSS reverses ADF's null: here the null is *stationarity*, so a small
    p-value is evidence of a unit root. Running both is standard practice —
    they can agree (clear verdict), or disagree (the series is probably
    neither cleanly I(0) nor cleanly I(1), e.g. fractionally integrated).
    """
    import warnings

    with warnings.catch_warnings():  # statsmodels warns when the p-value is clipped at a table edge
        warnings.simplefilter("ignore")
        stat, pvalue, *_ = kpss(series, regression=regression, nlags="auto")
    return float(stat), float(pvalue)


def compare_with_statsmodels(y: np.ndarray, lags: int = 1, regression: str = "c") -> None:
    """Print the hand-built ADF statistic next to statsmodels' `adfuller`."""
    ours = adf_test(y, lags=lags, regression=regression)
    stat, pvalue, usedlag, nobs, crit, *_ = adfuller(
        y, maxlag=lags, regression=regression, autolag=None
    )
    print(f"  ADF (ours):        {ours.stat:8.4f}   (gamma={ours.gamma:+.4f}, nobs={ours.nobs})")
    print(f"  ADF (statsmodels): {stat:8.4f}   (p={pvalue:.4f}, nobs={nobs})")
    print(f"  critical values:   1%={crit['1%']:.2f}  5%={crit['5%']:.2f}  10%={crit['10%']:.2f}")
    print(f"  max |diff|:        {abs(ours.stat - stat):.2e}")
    assert abs(ours.stat - stat) < 1e-6, "the hand-built ADF statistic should match statsmodels"


def main() -> None:
    pair = generate_spurious_pair(n=300, seed=0)
    fit = regress_levels(pair.y, pair.x)
    print("Spurious regression of one random walk on another (true slope = 0):")
    print(f"  slope={fit.beta[1]:+.4f}  t={fit.tstat[1]:+.2f}  R^2={fit.r2:.3f}")

    try:
        dw = durbin_watson(fit.resid)
        print(f"  Durbin-Watson (ours):        {dw:.4f}")
        print(f"  Durbin-Watson (statsmodels): {sm_durbin_watson(fit.resid):.4f}")
        print(f"  Granger-Newbold rule (R^2 > DW => spurious): {fit.r2 > dw}")
    except NotImplementedError as e:
        print(f"  (skipped DW — {e})")

    print("\nUnit-root tests:")
    for label, series in (
        ("random walk   ", generate_random_walk(300, seed=0)),
        ("trend-stationary", generate_trend_stationary(300, seed=0)),
    ):
        regression = "c" if label.startswith("random") else "ct"
        print(f"\n  [{label.strip()}]  (regression={regression!r})")
        try:
            compare_with_statsmodels(series, lags=1, regression=regression)
        except NotImplementedError as e:
            print(f"  (skipped ADF — {e})")
        kstat, kp = kpss_summary(series, regression=regression)
        print(f"  KPSS:              {kstat:8.4f}   (p={kp:.4f}; null here is STATIONARITY)")


if __name__ == "__main__":
    main()
