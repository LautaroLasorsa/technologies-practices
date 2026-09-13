"""Phase 2 — Box-Jenkins identification and conditional least squares.

Once a series is stationary (Phase 1), the modelling question becomes
*which* ARMA process generated it. Box & Jenkins' answer is a three-step
loop — identify, estimate, diagnose — and the identification step reads
two pictures: the ACF and the PACF. An AR(p) has a PACF that cuts off
sharply after lag p and an ACF that decays; an MA(q) is the mirror image;
an ARMA has both decaying.

Estimation of the AR part is then remarkably plain: regress the series on
its own lags. That is "conditional least squares" — conditional because
the first p observations are used only as regressors, never as outcomes.

Run on its own:
    uv run python -m src._02_arima_identification
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

from .datasets import generate_ar_series, generate_arma_series
from .regression import add_const, build_lag_matrix, ols


@dataclass
class ARFit:
    """Conditional-least-squares fit of an AR(p) model."""

    const: float  # estimated intercept
    phi: np.ndarray  # (p,) autoregressive coefficients, phi[0] multiplies y_{t-1}
    sigma2: float  # residual variance estimate
    resid: np.ndarray  # (n - p,) residuals
    nobs: int  # rows the regression actually used


# TODO(human) — AR(p) by conditional least squares
# ---------------------------------------------------------------------------
# Goal: estimate an AR(p) model by ordinary least squares on lagged levels,
# and return the coefficients together with the residual variance.
#
# Why this matters: this is the estimator hiding inside
# `statsmodels.tsa.ar_model.AutoReg` (and inside the AR half of ARIMA's
# starting values). Seeing that an AR(p) fit is *literally* a regression on
# lagged columns demystifies the whole family — and it explains why the MA
# part is different in kind: the MA regressors are past *errors*, which are
# unobserved, so no closed-form least-squares solution exists and an
# iterative likelihood method is required instead.
#
# "Conditional" means the estimate conditions on the first p observations:
# they are used as regressors for later rows but never get a row of their
# own, so the regression has n - p rows rather than n. That is why AutoReg
# reports nobs = n - p, and why comparing coefficients across different p
# means comparing fits on slightly different samples.
#
# The model:
#     y_t = c + phi_1*y_{t-1} + ... + phi_p*y_{t-p} + eps_t
#
# Steps:
#   1. y_trimmed, lags = build_lag_matrix(y, p) — the helper aligns y with
#      its own first p lags; column j-1 of `lags` is y_{t-j}.
#   2. X = add_const(lags) to include the intercept.
#   3. fit = ols(X, y_trimmed).
#   4. Unpack: fit.beta[0] is the constant, fit.beta[1:] are phi_1..phi_p.
#   5. Estimate sigma^2 as fit.rss / (nobs - k), where k = p + 1 is the
#      number of estimated coefficients (the same degrees-of-freedom
#      correction the classical variance formula uses).
#
# Sanity expectation: on the pure AR(2) series from `datasets.py` the
# recovered phi should land near the true (0.6, -0.2), and should match
# statsmodels' `AutoReg(y, lags=p).fit()` to numerical precision — AutoReg
# computes exactly this regression.
# ---------------------------------------------------------------------------
def ar_conditional_least_squares(y: np.ndarray, p: int) -> ARFit:
    """Fit an AR(p) by OLS on lagged levels (conditional least squares).

    Takes a (n,) stationary series and the order `p`; returns an `ARFit`
    whose `phi[j]` multiplies `y_{t-(j+1)}`. Uses n - p rows.
    """
    raise NotImplementedError("TODO(human): implement AR(p) estimation by conditional least squares")


def identification_table(series: np.ndarray, nlags: int = 12) -> str:
    """ACF and PACF values with their 95% bands, as a printable table.

    The same numbers `plotting.acf_pacf_panel` draws — useful when reading
    a cut-off point off a picture is ambiguous. A value whose band excludes
    zero is 'significant' at that lag.
    """
    acf_v, acf_ci = acf(series, nlags=nlags, alpha=0.05, result_object=False)
    pacf_v, pacf_ci = pacf(series, nlags=nlags, alpha=0.05, method="ywm")
    rows = ["  lag      ACF   sig?      PACF   sig?", "  " + "-" * 38]
    for lag in range(1, nlags + 1):
        acf_sig = not (acf_ci[lag, 0] - acf_v[lag] < 0 < acf_ci[lag, 1] - acf_v[lag])
        pacf_sig = not (pacf_ci[lag, 0] - pacf_v[lag] < 0 < pacf_ci[lag, 1] - pacf_v[lag])
        rows.append(
            f"  {lag:3d}  {acf_v[lag]:+7.3f}   {'*' if acf_sig else ' '}     "
            f"{pacf_v[lag]:+7.3f}   {'*' if pacf_sig else ' '}"
        )
    return "\n".join(rows)


def arima_reference(series: np.ndarray, order: tuple[int, int, int]):
    """Fit an ARIMA(p, d, q) with statsmodels (full MLE) as the reference model."""
    return ARIMA(series, order=order).fit()


def ljung_box(resid: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box Q statistic and p-value — the Box-Jenkins 'diagnose' step.

    Null: the residuals are jointly uncorrelated up to `lags`. A large
    p-value means the fitted model has extracted the serial structure and
    left white noise behind, which is what 'this model is adequate' means.
    """
    out = acorr_ljungbox(resid, lags=[lags], return_df=True)
    return float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])


def compare_with_autoreg(y: np.ndarray, p: int) -> None:
    """Print the hand-rolled CLS fit next to statsmodels' `AutoReg`."""
    ours = ar_conditional_least_squares(y, p)
    ref = AutoReg(y, lags=p, old_names=False).fit()
    ours_all = np.concatenate([[ours.const], ours.phi])
    diff = float(np.max(np.abs(ours_all - ref.params)))
    print(f"  ours        (c, phi): {np.round(ours_all, 4)}")
    print(f"  AutoReg     (c, phi): {np.round(ref.params, 4)}")
    print(f"  max |diff|:           {diff:.2e}   (nobs ours={ours.nobs}, AutoReg={int(ref.nobs)})")
    assert diff < 1e-8, "conditional least squares should match AutoReg exactly"


def main() -> None:
    ar_data = generate_ar_series(n=400, seed=0)
    print(f"Pure AR(2), true phi = {ar_data.ar_true}")
    print(identification_table(ar_data.series, nlags=8))
    print("  (PACF should cut off after lag 2; ACF should decay)\n")
    try:
        compare_with_autoreg(ar_data.series, p=2)
    except NotImplementedError as e:
        print(f"  (skipped — {e})")

    arma_data = generate_arma_series(n=400, seed=0)
    print(f"\nARMA(2, 1), true phi = {arma_data.ar_true}, true theta = {arma_data.ma_true}")
    print(identification_table(arma_data.series, nlags=8))
    print("  (both ACF and PACF decay — the MA term is why a pure AR fit is mis-specified)")
    ref = arima_reference(arma_data.series, order=(2, 0, 1))
    stat, pvalue = ljung_box(np.asarray(ref.resid), lags=10)
    print(f"\n  ARIMA(2,0,1) MLE params: {np.round(ref.params, 4)}")
    print(f"  Ljung-Box(10) on its residuals: Q={stat:.3f}, p={pvalue:.4f} (large p = adequate)")


if __name__ == "__main__":
    main()
