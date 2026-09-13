"""xy.pyplot plotting helpers shared across phases.

`xy.pyplot` is a drop-in `matplotlib.pyplot` replacement (see this
practice's CLAUDE.md, "Plotting" section) used for every plot in this
practice -- including the statsmodels-emitted ACF/PACF diagnostics, which
normally require a real matplotlib `Axes`. `statsmodels.graphics.utils.
create_mpl_ax` only imports matplotlib when `ax is None`; passed a real
`xy.pyplot.Axes` instead, it duck-types through cleanly (verified during
scaffolding -- see CLAUDE.md "Notes"), so there is no `matplotlib`
dependency anywhere in this practice. No TODO here: plotting is not the
taught technique, the estimators that feed these plots are.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xy.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def spurious_regression_plot(x: np.ndarray, y: np.ndarray, slope: float, intercept: float,
                              title: str = "Spurious regression of two independent random walks"):
    """Scatter of two independent random walks plus their (misleading) OLS fit line."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(x, y, alpha=0.6)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xs, intercept + slope * xs, linestyle="--", label="OLS fit")
    ax.set_xlabel("x (random walk)")
    ax.set_ylabel("y (independent random walk)")
    ax.set_title(title)
    ax.legend()
    return fig


def rolling_stats_plot(series: np.ndarray, window: int = 20, title: str = "Rolling mean & std"):
    """Rolling mean/std over time -- the eyeball stationarity check: a
    stationary series has a roughly flat rolling mean and std; a
    non-stationary one visibly drifts or fans out."""
    s = pd.Series(series)
    roll_mean = s.rolling(window).mean()
    roll_std = s.rolling(window).std()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.asarray(series), alpha=0.35, label="series")
    ax.plot(roll_mean.to_numpy(), label=f"rolling mean ({window})")
    ax.plot(roll_std.to_numpy(), label=f"rolling std ({window})")
    ax.set_title(title)
    ax.legend()
    return fig


def acf_pacf_plot(series: np.ndarray, lags: int = 20, title_prefix: str = ""):
    """ACF/PACF side by side, drawn by statsmodels directly onto xy.pyplot
    Axes (see module docstring) -- the standard Box-Jenkins identification
    picture: an AR(p)'s PACF cuts off after lag p, an MA(q)'s ACF cuts off
    after lag q."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    prefix = f"{title_prefix} -- " if title_prefix else ""
    plot_acf(series, ax=axes[0], lags=lags, title=f"{prefix}Autocorrelation")
    plot_pacf(series, ax=axes[1], lags=lags, method="ywm", title=f"{prefix}Partial Autocorrelation")
    return fig


def impulse_response_plot(irf: np.ndarray, ci_lower: np.ndarray, ci_upper: np.ndarray,
                           response_name: str, impulse_name: str, title: str | None = None):
    """Impulse response function with a shaded 95% confidence band."""
    horizon = np.arange(len(irf))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(horizon, ci_lower, ci_upper, alpha=0.25, label="95% CI")
    ax.plot(horizon, irf, marker="o")
    ax.axhline(0.0, linestyle="--")
    ax.set_xlabel("Horizon")
    ax.set_ylabel(f"Response of {response_name}")
    ax.set_title(title or f"Impulse response: {impulse_name} -> {response_name}")
    ax.legend()
    return fig


def series_plot(series_by_name: dict[str, np.ndarray], title: str, ylabel: str = "level"):
    """Overlay one or more series against an integer time axis.

    `series_by_name` maps a legend label to a 1-D array. Used for raw
    levels, differenced series, and the level-vs-equilibrium views.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in series_by_name.items():
        ax.plot(np.arange(len(values)), np.asarray(values), label=label)
    ax.set_xlabel("t")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    return fig


def tstat_distribution_plot(tstats: np.ndarray, crit: float = 1.96):
    """Monte Carlo distribution of |t| from regressing independent random walks.

    Under a valid null the t-statistic would be standard normal and only 5%
    of draws would land beyond +/-1.96. The gap between that 5% and what
    this histogram shows is the size distortion the spurious regression
    causes.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.abs(tstats), bins=40, alpha=0.8, label="|t| across replications")
    ax.axvline(crit, linestyle="--", label=f"nominal 5% critical value ({crit})")
    ax.set_xlabel("|t-statistic| on the spurious slope")
    ax.set_ylabel("count")
    ax.set_title("A t-statistic that rejects far too often")
    ax.legend()
    return fig


def equilibrium_error_plot(equilibrium_error: np.ndarray, title: str = "Equilibrium error (cointegrating residual)"):
    """The estimated `y - beta*x` over time, with its mean marked.

    If the pair really is cointegrated this series is stationary: it
    recrosses its mean instead of wandering away from it, which is what
    makes the error-correction interpretation meaningful.
    """
    values = np.asarray(equilibrium_error, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(np.arange(len(values)), values, label="y - beta_hat * x")
    ax.axhline(float(values.mean()), linestyle="--", label="mean")
    ax.set_xlabel("t")
    ax.set_ylabel("equilibrium error")
    ax.set_title(title)
    ax.legend()
    return fig
