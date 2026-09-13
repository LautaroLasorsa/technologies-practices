"""Synthetic time-series datasets with known ground truth.

Every dataset here comes from a data-generating process (DGP) with a
*known* true structure -- a true AR/MA order, a true VAR coefficient
matrix, a true cointegrating relationship -- so every phase can compare an
estimator's output against the ground truth instead of against nothing. A
fixed seed makes results reproducible across runs and across learners.
No TODO here: data generation is infrastructure, not the taught technique.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def generate_random_walk(n: int, seed: int, drift: float = 0.0) -> np.ndarray:
    """One I(1) random walk: y_t = drift + y_{t-1} + eps_t, y_0 = 0.

    A random walk is the textbook non-stationary process: its variance
    grows linearly with t, and it has no tendency to return to any fixed
    mean -- the opposite of everything a stationary AR/MA process does.
    """
    rng = np.random.default_rng(seed)
    steps = drift + rng.normal(scale=1.0, size=n - 1)
    return np.concatenate([[0.0], np.cumsum(steps)])


@dataclass
class SpuriousPairData:
    """Two independent random walks -- no true relationship exists between them."""

    y: np.ndarray
    x: np.ndarray

    def as_frame(self) -> pd.DataFrame:
        """Tidy pandas view, mainly for notebook `.head()` display."""
        return pd.DataFrame({"y": self.y, "x": self.x})


def generate_spurious_pair(n: int = 300, seed: int = 0) -> SpuriousPairData:
    """Two *independent* random walks generated from unrelated random seeds.

    By construction there is no relationship between `y` and `x` -- yet an
    OLS regression of one on the other will typically report a high R^2
    and a "significant" t-statistic anyway. This is the canonical
    spurious-regression demonstration (Granger & Newbold, 1974): two
    unrelated I(1) series can look strongly related purely because they
    are both trending, not because either causes the other.
    """
    y = generate_random_walk(n, seed=seed)
    x = generate_random_walk(n, seed=seed + 1000)
    return SpuriousPairData(y=y, x=x)


def generate_trend_stationary(n: int = 300, seed: int = 0, slope: float = 0.05, rho: float = 0.7) -> np.ndarray:
    """A *trend-stationary* series: deterministic trend + stationary AR(1) noise.

    Visually this is the twin of a drifting random walk -- both rise
    steadily -- but it has no unit root: detrend it and what is left is
    stationary, whereas a random walk needs *differencing*. Telling these
    two apart is exactly what the ADF/KPSS pair is for, and applying the
    wrong remedy (differencing a trend-stationary series, or detrending an
    I(1) one) leaves the residual mis-specified either way.
    """
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for t in range(1, n):
        noise[t] = rho * noise[t - 1] + rng.normal(scale=1.0)
    return slope * np.arange(n, dtype=float) + noise


# True ARMA(2, 1) coefficients, in the "y_t = phi_1 y_{t-1} + phi_2 y_{t-2}
# + eps_t + theta_1 eps_{t-1}" convention used throughout this practice.
TRUE_AR = np.array([0.6, -0.2])
TRUE_MA = np.array([0.4])


@dataclass
class ArmaSeriesData:
    """One simulated stationary ARMA series with known true coefficients."""

    series: np.ndarray
    ar_true: np.ndarray
    ma_true: np.ndarray


def generate_arma_series(n: int = 400, seed: int = 0) -> ArmaSeriesData:
    """Stationary ARMA(2, 1) series with known true AR/MA coefficients.

    Uses statsmodels' `ArmaProcess`, which expects AR/MA polynomials in
    the "lag operator" convention (`1 - phi_1*L - phi_2*L^2` for the AR
    side) -- `np.r_[1, -TRUE_AR]` converts our convention to that one.
    Burn-in is handled internally by `ArmaProcess.generate_sample`.
    """
    ar_poly = np.r_[1, -TRUE_AR]
    ma_poly = np.r_[1, TRUE_MA]
    process = ArmaProcess(ar_poly, ma_poly)
    rng = np.random.default_rng(seed)
    series = process.generate_sample(nsample=n, distrvs=rng.standard_normal)
    return ArmaSeriesData(series=series, ar_true=TRUE_AR, ma_true=TRUE_MA)


def generate_ar_series(n: int = 400, seed: int = 0) -> ArmaSeriesData:
    """Pure AR(2) series — same AR coefficients as above, no MA term.

    Conditional least squares recovers AR coefficients exactly (it is just
    OLS on lagged levels) but cannot fit an MA term, whose lagged *errors*
    are unobserved. So the AR-only series is the one with a ground truth
    the Phase 2 estimator can actually be checked against, while the
    ARMA(2, 1) series is the one whose ACF/PACF shapes make the
    identification lesson interesting.
    """
    process = ArmaProcess(np.r_[1, -TRUE_AR], np.r_[1])
    rng = np.random.default_rng(seed)
    series = process.generate_sample(nsample=n, distrvs=rng.standard_normal)
    return ArmaSeriesData(series=series, ar_true=TRUE_AR, ma_true=np.array([]))


# True bivariate VAR(1) coefficient matrix: x Granger-causes y (the (0, 1)
# entry is nonzero) but y does NOT Granger-cause x (the (1, 0) entry is
# exactly zero) -- this asymmetry is what Phase 3's Granger test should
# detect in one direction and fail to detect in the other.
TRUE_VAR_COEF = np.array([[0.5, 0.3], [0.0, 0.4]])


@dataclass
class VarSystemData:
    """A stationary bivariate VAR(1) system with a known, asymmetric
    Granger-causal structure."""

    y: np.ndarray  # (n,) Granger-caused by x
    x: np.ndarray  # (n,) Granger-causes y; not Granger-caused by y
    coef_matrix: np.ndarray  # (2, 2) true VAR(1) coefficient matrix


def generate_var_system(n: int = 300, seed: int = 0, n_burn: int = 100) -> VarSystemData:
    """Simulate the bivariate VAR(1): z_t = A @ z_{t-1} + eps_t, z = [y, x].

    `A`'s eigenvalues (0.5 and 0.4) are both inside the unit circle, so the
    system is stationary -- no unit roots to worry about here, only the
    Granger-causal structure and the impulse responses it implies.
    """
    rng = np.random.default_rng(seed)
    z = np.zeros((n + n_burn, 2))
    for t in range(1, n + n_burn):
        z[t] = TRUE_VAR_COEF @ z[t - 1] + rng.normal(scale=1.0, size=2)
    z = z[n_burn:]
    return VarSystemData(y=z[:, 0], x=z[:, 1], coef_matrix=TRUE_VAR_COEF)


@dataclass
class CointegratedPairData:
    """Two I(1) series sharing one common stochastic trend, related by a
    stationary long-run equilibrium."""

    y: np.ndarray
    x: np.ndarray
    beta_true: float
    equilibrium_error: np.ndarray  # the true stationary combination y - beta*x


def generate_cointegrated_pair(n: int = 300, seed: int = 0, beta: float = 2.0) -> CointegratedPairData:
    """Two I(1) series cointegrated by construction: `y_t = beta*x_t + u_t`
    with `u_t` a stationary AR(1) process (no unit root) and `x_t` itself
    a random walk plus noise.

    Both `y` and `x` are individually non-stationary (each inherits the
    random walk's unit root), but the specific combination `y - beta*x`
    equals the stationary `u_t` -- the textbook Engle & Granger (1987)
    setup this practice's Phase 4 recovers via the two-step procedure.
    """
    rng = np.random.default_rng(seed)
    common_trend = generate_random_walk(n, seed=seed)
    x = common_trend + rng.normal(scale=0.3, size=n)

    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.7 * u[t - 1] + rng.normal(scale=0.5)

    y = beta * x + u
    return CointegratedPairData(y=y, x=x, beta_true=beta, equilibrium_error=u)
