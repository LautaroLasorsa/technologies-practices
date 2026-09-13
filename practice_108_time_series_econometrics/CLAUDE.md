# Practice 108 — Time Series Econometrics: ARIMA, VAR & Cointegration

## Technologies

- **statsmodels** — reference implementations (`adfuller`, `kpss`, `ARIMA`, `AutoReg`, `VAR`, `coint`) used to validate every from-scratch estimator in this practice.
- **xy** — plotting (spurious-regression scatter, rolling stats, ACF/PACF panels, impulse responses, equilibrium error).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The problem: time series break the independence assumption

Every standard-error formula in cross-sectional econometrics (practice 097) assumes
observations are independent draws. A time series is not: today's value is mostly
yesterday's value. Ignoring that produces **spurious regression** (Granger & Newbold,
1974) — regress one unrelated random walk on another and OLS reports a high R^2 and a
huge t-statistic anyway, purely because both series trend, not because either causes
the other. Nearly everything in this practice is either detecting that failure mode
(stationarity testing) or building models that respect the dependence instead of
ignoring it (ARIMA, VAR, cointegration).

### Stationarity and unit roots

A series is (weakly) **stationary** if its mean, variance, and autocovariance structure
don't depend on `t`. A **random walk** `y_t = y_{t-1} + eps_t` is the canonical
non-stationary case: it has a **unit root** (the AR polynomial `1 - phi*L` has a root at
`L = 1`, i.e. `phi = 1`), its variance grows with `t`, and shocks never decay. The
**augmented Dickey-Fuller (ADF)** test regresses `Delta y_t` on `y_{t-1}` (plus lagged
differences to soak up serial correlation) and tests `H0: gamma = 0` (a unit root)
against `H1: gamma < 0` (stationary) — but the t-ratio under `H0` follows the
non-standard *Dickey-Fuller distribution*, not Student's t, because `y_{t-1}` is itself
non-stationary under the null. **KPSS** (Kwiatkowski et al., 1992) flips the null and
alternative: `H0` is stationarity, `H1` is a unit root. Running both is standard
practice — agreement gives a confident verdict; disagreement flags a series that is
neither cleanly I(0) nor cleanly I(1) (e.g. fractionally integrated, or trend-stationary
tested with the wrong ADF specification).

### ARIMA and Box-Jenkins identification

An **ARMA(p, q)** process is `y_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t +
theta_1 eps_{t-1} + ... + theta_q eps_{t-q}`; **ARIMA(p, d, q)** applies it after
differencing a non-stationary series `d` times. Box & Jenkins' (1970) three-step
recipe — **identify, estimate, diagnose** — reads the order off the autocorrelation
function (ACF) and partial autocorrelation function (PACF): a pure AR(p) has a PACF
that cuts off sharply after lag `p` while its ACF decays gradually; a pure MA(q) is the
mirror image; a mixed ARMA has both decaying, and order selection becomes closer to
trial and error (helped by information criteria). Estimation of the AR part is
**conditional least squares** — literally an OLS regression of the series on its own
lags, conditioning on the first `p` observations as regressors rather than modeling
their distribution. The MA part cannot be estimated this way: its regressors are past
*errors*, which are never directly observed, so real implementations fall back to an
iterative (Kalman-filter) likelihood method instead. The diagnose step is the
**Ljung-Box** test on the fitted residuals — a large p-value means the model has
extracted the serial structure and left white noise behind.

### VAR, Granger causality and impulse responses

A **vector autoregression (VAR)** generalizes AR to a system: every variable is
regressed on lags of every variable, letting several series' dynamics be estimated
jointly instead of one at a time. **Granger causality** asks a narrow, purely
predictive question — "do x's lags help predict y beyond y's own past?" — answered by
a nested-model F-test comparing a restricted regression (y on its own lags) against an
unrestricted one (y on its own lags plus x's lags). It is neither necessary nor
sufficient for true causation and is the single most over-read term in applied time
series: rejecting the null means x carries predictive content about y's future, nothing
more. The **impulse response function (IRF)** is where a VAR's economic interpretation
actually lives — it traces the dynamic path implied by iterating the fitted system
forward after a one-off shock. Because a VAR's reduced-form shocks are correlated
across equations, IRFs are usually **orthogonalized** via a Cholesky decomposition
before being reported — which makes the ordering of the system's variables an
identifying assumption in its own right.

### Cointegration and the error-correction model

Phase 1's lesson is "never regress two I(1) series on each other." **Cointegration**
(Engle & Granger, 1987) is the documented exception: if two I(1) series share the same
underlying stochastic trend, some linear combination of them is stationary, and *that*
combination is a genuine long-run equilibrium rather than a spurious artifact. The
**Engle-Granger two-step test**: (1) estimate the long-run relation `y = alpha + beta*x
+ u` by OLS in levels — under cointegration this OLS estimate is *super-consistent*,
converging at rate `T` rather than the usual `sqrt(T)`; (2) ADF-test the residual `u_hat`
for a unit root, using MacKinnon's adjusted critical values (more negative than plain
ADF, because the residual was already variance-minimized by step 1's fit). If step 2
rejects, the Granger Representation Theorem guarantees an **error-correction model
(ECM)** exists: `Delta y_t = c + gamma*Delta x_t + alpha*u_hat_{t-1} + eps_t`. The
adjustment speed `alpha` must be negative for the system to be stable, and its magnitude
is the fraction of a deviation from equilibrium corrected each period.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Unit root / I(1)** | A series whose AR polynomial has a root at 1; shocks never decay; must be differenced once to become stationary. |
| **ADF test** | Regression-based unit-root test; `H0` = unit root present. |
| **KPSS test** | Complementary unit-root test with the opposite null: `H0` = stationarity. |
| **Spurious regression** | High R^2 / significant t-stat from regressing unrelated non-stationary series — a large `R^2 > DW` gap is the classic tell. |
| **Conditional least squares (CLS)** | OLS-on-lags estimation of an AR(p) model, conditioning on the first `p` observations. |
| **Granger causality** | F-test of whether `x`'s lags improve a forecast of `y` beyond `y`'s own history — predictive content, not causal effect. |
| **Impulse response function (IRF)** | The dynamic path a VAR predicts after a one-unit shock to one variable. |
| **Cointegration** | Two+ I(1) series with a stationary linear combination — a genuine long-run equilibrium. |
| **Error-correction model (ECM)** | Short-run dynamics equation whose lagged equilibrium error pulls the system back toward its long-run path. |

### Where This Fits

Time series methods are the workhorse of demand forecasting, capacity planning, and
anomaly detection — anywhere the object of interest is "this metric over time," not
independent cross-sectional units. ARIMA is the univariate baseline that modern
forecasting tools (Prophet, exponential smoothing, even most deep-learning forecasters)
are still benchmarked against; VAR and Granger causality are the standard first pass
for "which of these correlated metrics leads the others"; cointegration and the ECM are
the classical answer to "these two series must not drift apart forever" (a queue length
and a service rate, an inventory level and demand, a price and its fundamental value).
Alternatives include state-space/Kalman-filter models (a strict superset that subsumes
ARIMA and handles missing data and time-varying parameters more naturally) and, for
long panels of related series, hierarchical or global forecasting models — but the
stationarity/identification/causality vocabulary built here transfers directly to all
of them.

### References

- Granger & Newbold, 1974, "Spurious Regressions in Econometrics": <https://doi.org/10.1016/0304-4076(74)90034-7>
- Dickey & Fuller, 1979, "Distribution of the Estimators for Autoregressive Time Series with a Unit Root": <https://doi.org/10.2307/2286348>
- Kwiatkowski, Phillips, Schmidt & Shin, 1992, "Testing the Null Hypothesis of Stationarity": <https://doi.org/10.1016/0304-4076(92)90104-Y>
- Box, Jenkins, Reinsel & Ljung, *Time Series Analysis: Forecasting and Control* (5th ed.): <https://www.wiley.com/en-us/Time+Series+Analysis%3A+Forecasting+and+Control%2C+5th+Edition-p-9781118675021>
- Engle & Granger, 1987, "Co-integration and Error Correction: Representation, Estimation, and Testing": <https://doi.org/10.2307/1913236>
- statsmodels time-series docs: <https://www.statsmodels.org/stable/tsa.html>

## Description

Build the estimators behind `statsmodels`' time-series tools from scratch and validate
each against it: an ADF regression and a Durbin-Watson statistic on independent random
walks (Phase 1), an AR(p) conditional-least-squares fit read off an ACF/PACF
identification picture (Phase 2), a Granger-causality F-test and impulse responses on a
bivariate VAR (Phase 3), and the Engle-Granger two-step cointegration test plus its
error-correction model (Phase 4). Every dataset has a known ground truth, so every
estimate is checked against it directly.

### What you'll learn

1. Why "significant" means nothing on non-stationary series, and how ADF/KPSS's opposite nulls jointly diagnose that.
2. How Box-Jenkins identification reads an AR/MA order off the ACF/PACF, and why conditional least squares stops at the AR part.
3. Why Granger causality is a predictive-content test, not a causal one, and how the same nested-F machinery drives it.
4. Why regressing two I(1) series is normally wrong, and exactly what makes cointegration the documented exception.
5. How the error-correction model turns a cointegrating relationship into an actionable adjustment speed.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_time_series_econometrics.ipynb`

### Phase 1: Stationarity, Spurious Regression & Unit Roots (~25-30 min) — `src/_01_stationarity_unit_roots.py`

The spurious-regression Monte Carlo and the OLS/ADF/KPSS scaffolding are fully
implemented; the two statistics that catch a spurious fit are the exercise.

1. **TODO #1 — `durbin_watson(resid)`**: the classical first-order residual-autocorrelation statistic. ~10 lines.
2. **TODO #2 — `adf_test(y, lags, regression)`**: build the augmented Dickey-Fuller auxiliary regression by hand and return its t-ratio. ~20-25 lines.

### Phase 2: ARIMA Identification via ACF/PACF & Conditional Least Squares (~20-25 min) — `src/_02_arima_identification.py`

ACF/PACF plotting, the Ljung-Box diagnostic, and the ARIMA/AutoReg reference fits are
fully scaffolded. The teaching content is the AR estimator itself.

3. **TODO #1 — `ar_conditional_least_squares(y, p)`**: fit AR(p) by OLS on lagged levels — the estimator behind `statsmodels.tsa.ar_model.AutoReg`. ~15-20 lines.

### Phase 3: VAR, Granger Causality & Impulse Responses (~20-25 min) — `src/_03_var_granger.py`

Fitting the VAR and computing impulse responses is fully scaffolded (`statsmodels.tsa.api.VAR`). The teaching content is the causality test.

4. **TODO #1 — `granger_causality_f_test(y, x, lags)`**: compare two nested regressions' RSS with an F-test. ~15-20 lines.

### Phase 4: Cointegration — Engle-Granger & the ECM (~25-30 min) — `src/_04_cointegration_ecm.py`

5. **TODO #1 — `engle_granger_step1(y, x)`**: the cointegrating regression in levels plus an ADF test on its residual. ~15 lines.
6. **TODO #2 — `error_correction_model(y, x, equilibrium_error)`**: the short-run ECM regression, reading off the adjustment speed. ~10-15 lines.

### Phase 5: End-to-End Run (no TODO)

Run the notebook's final cells: the Verification cell exercises every phase's estimator
against its known ground truth in one pass.

### What to look for in the results

- Phase 1: the single spurious-regression replication should show a large `|t|` and
  `R^2 > DW`; the 500-replication Monte Carlo should reject the (true) null of no
  relationship far more than the nominal 5% of the time.
- Phase 1: ADF should fail to reject a unit root on the random walk and reject it on the
  trend-stationary series (with `regression="ct"`); KPSS should show the opposite
  pattern on each.
- Phase 2: the recovered AR(2) `phi` should land near the true `(0.6, -0.2)` and match
  `AutoReg` to numerical precision; the ARMA(2, 1) series' ACF and PACF should both
  decay, unlike the pure AR(2)'s sharp PACF cutoff.
- Phase 3: testing `x -> y` should reject "no Granger causality" decisively; testing
  `y -> x` should not — the asymmetry is built into the true VAR coefficient matrix.
- Phase 4: the recovered cointegrating `beta` should land near the true value; the ECM's
  `speed` should be negative and statistically significant, with a finite implied
  half-life.

## Motivation

- **AutoScheduler.AI relevance**: demand and throughput metrics are time series first —
  any "did this change actually shift the trend, or is it just this week's noise on a
  non-stationary series" question is exactly what Phase 1 guards against, and
  lead/lag questions between two operational metrics are Phase 3's Granger-causality
  test.
- **Senior → Staff differentiator**: fitting `ARIMA(...)` or `VAR(...)` and reading off a
  p-value is common; knowing why the series had to be tested for a unit root first, why
  a Granger-causality "significant" result is not a causal claim, and when a levels
  regression on two trending series is legitimate (cointegration) rather than spurious,
  is not.
- **Generalises beyond econometrics**: the nested-model F-test behind Granger causality
  is the same machinery behind any likelihood-ratio-style comparison of restricted vs.
  unrestricted models; the error-correction pattern (a system pulled back toward a
  known equilibrium at a fitted rate) reappears anywhere a control loop or a queueing
  system exhibits mean reversion.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_time_series_econometrics.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_stationarity_unit_roots` | Spurious-regression demo plus ADF/KPSS on a random walk and a trend-stationary series. |
| **Phase 2** | `uv run python -m src._02_arima_identification` | ACF/PACF identification table and CLS AR(2) fit vs. `AutoReg`. |
| **Phase 3** | `uv run python -m src._03_var_granger` | Granger-causality F-tests both directions plus the impulse response. |
| **Phase 4** | `uv run python -m src._04_cointegration_ecm` | Engle-Granger two-step test and the ECM on a cointegrated pair. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

- **Plotting: why there is no matplotlib dependency.** `statsmodels.graphics.utils.create_mpl_ax`
  (used internally by `plot_acf`/`plot_pacf`) does `fig = ax.figure` when `ax` is not
  `None` — no `isinstance` check against `matplotlib.axes.Axes`. Empirically verified
  during reconciliation (xy 0.0.7, statsmodels 0.14): passing a real `xy.pyplot.Axes`
  into `plot_acf`/`plot_pacf`, including the two-panel `subplots(1, 2)` layout
  `src/plotting.py` uses, duck-types through cleanly — the ACF/PACF stems,
  confidence bands, and titles render correctly on the `xy` axes, confirmed both by
  inspecting `ax.lines`/`ax.collections` after the call and by rendering the figure to
  PNG and visually inspecting it. `matplotlib` is therefore not a dependency of this
  practice; `src/plotting.py` uses `xy.pyplot` throughout, including for the
  statsmodels-emitted diagnostics.

## State

`not-started`
