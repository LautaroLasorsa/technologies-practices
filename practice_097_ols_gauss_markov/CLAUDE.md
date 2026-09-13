# Practice 097 — Econometrics: OLS from Scratch & the Gauss-Markov Assumptions

## Technologies

- **statsmodels** — reference OLS implementation used to validate the from-scratch estimator, and as the standard for comparison throughout econometrics tooling.
- **NumPy** (`np.linalg.qr`) — the QR decomposition that makes OLS numerically stable without ever forming `X'X`.
- **xy** — plotting (residual diagnostics, coefficient plot, sampling-distribution histogram).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### What OLS Actually Estimates

Ordinary Least Squares fits a linear model `y = X*beta + u` by choosing `beta_hat` to
minimize the sum of squared residuals `||y - X*beta||^2`. The closed-form solution is
the **normal equations**: `beta_hat = (X'X)^-1 X'y`. That's the formula every
intro-econometrics course writes down — but it's not how production software computes
it, because explicitly forming and inverting `X'X` squares the condition number of `X`:
any near-collinearity in the regressors gets numerically amplified. Real implementations
(statsmodels, R's `lm()`, scikit-learn) instead factor `X = QR` (`Q` orthonormal, `R`
upper-triangular) and solve the triangular system `R @ beta = Q.T @ y` by
back-substitution — the same answer, computed far more stably. This practice implements
both the QR route (Phase 1) and validates it against statsmodels.

### The Gauss-Markov Theorem

The Gauss-Markov theorem says OLS is **BLUE** (Best Linear Unbiased Estimator) — the
lowest-variance estimator among all linear, unbiased estimators of `beta` — *if and only
if* five assumptions hold:

| # | Assumption | Plain-language meaning |
|---|------------|-------------------------|
| A1 | Linearity in parameters | The model is `y = X*beta + u`; `beta` enters linearly (X itself can be nonlinear, e.g. `x^2`). |
| A2 | Strict exogeneity | `E[u \| X] = 0` — the errors are uncorrelated with the regressors at every observation. |
| A3 | No perfect collinearity | `X` has full column rank — no regressor is an exact linear combination of the others. |
| A4 | Spherical errors (part 1: no autocorrelation) | `Cov(u_i, u_j) = 0` for `i != j` — errors are independent across observations. |
| A5 | Spherical errors (part 2: homoskedasticity) | `Var(u_i) = sigma^2` for all `i` — constant error variance. |

Unbiasedness of `beta_hat` needs only A1-A3. A4-A5 ("spherical errors", jointly written
`Var(u \| X) = sigma^2 * I`) are what make the *classical* variance formula
`Var(beta_hat) = sigma^2 * (X'X)^-1` correct — and what this practice spends most of its
time breaking on purpose. Breaking A5 (heteroskedasticity, Phase 3-4) or the independence
half of A4 (clustering, Phase 5) never biases `beta_hat`; it only invalidates the
standard errors built on top of it, which is a subtler and more common real-world failure
than a biased point estimate.

### Robust and Clustered Standard Errors

When A5 fails, White's (1980) **sandwich estimator**
`Var(beta_hat) = (X'X)^-1 M (X'X)^-1` (with `M` built from per-observation squared
residuals instead of one global `sigma^2`) stays consistent regardless of the
heteroskedasticity's shape. **HC0** is the raw sandwich; **HC1** adds a simple degrees-of-
freedom correction; **HC2** and **HC3** additionally reweight by each observation's
leverage, with HC3 approximating the jackknife and behaving best in small samples
(MacKinnon & White, 1985). When the independence half of A4 fails instead — observations
share a within-group shock — the fix is **cluster-robust** standard errors (Liang &
Zeger, 1986): sum the score contributions within each cluster *before* squaring, so
within-cluster correlation is absorbed rather than ignored. Ignoring clustering when it's
present is one of applied econometrics' most common errors: it doesn't bias `beta_hat`,
but it can understate standard errors by a factor of 2-5x, turning noise into apparently
"significant" results.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Normal equations** | The closed-form OLS solution `beta_hat = (X'X)^-1 X'y`. |
| **QR decomposition** | `X = QR`; turns OLS into a stable triangular solve instead of a matrix inversion. |
| **BLUE** | Best Linear Unbiased Estimator — OLS's optimality guarantee under Gauss-Markov. |
| **Homoskedasticity (A5)** | Constant error variance across observations. |
| **Sandwich / HC estimator** | Heteroskedasticity-consistent covariance matrix; HC0-HC3 are variants differing in their finite-sample correction. |
| **Leverage** | `h_i`, the `i`-th diagonal of the hat matrix `X(X'X)^-1X'`; how much observation `i` can pull its own fitted value. |
| **Cluster-robust SE** | Covariance estimator that allows correlated errors within (but not across) clusters. |
| **Breusch-Pagan test** | LM test for heteroskedasticity: regress squared residuals on `X`, test `n*R^2` against `chi2`. |

### Where This Fits

OLS with robust/clustered SEs is the workhorse of applied econometrics — nearly every
other technique in this curriculum (RDD, synthetic control, panel fixed effects, IV) is
"OLS plus a trick to satisfy A2 (exogeneity) in a specific setting," with the *same*
robust/clustered SE machinery from this practice bolted on afterward. Alternatives to
OLS itself include GLS/WLS (model the heteroskedasticity instead of just correcting for
it — efficient but requires knowing its form) and quantile regression (robust to outliers
by construction, at the cost of estimating a conditional median/quantile rather than a
mean). This practice's HC/cluster machinery is exactly what `statsmodels`'
`cov_type="HC3"` / `cov_type="cluster"` compute — implementing it once here demystifies
what that keyword argument is actually doing.

### References

- Wooldridge, *Introductory Econometrics*, ch. 3 (Gauss-Markov), ch. 8 (heteroskedasticity): <https://www.cengage.com/c/introductory-econometrics-a-modern-approach-7e-wooldridge/>
- White, 1980, "A Heteroskedasticity-Consistent Covariance Matrix Estimator": <https://www.jstor.org/stable/1912934>
- MacKinnon & White, 1985, "Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties": <https://doi.org/10.1016/0304-4076(85)90158-7>
- Cameron & Miller, 2015, "A Practitioner's Guide to Cluster-Robust Inference": <https://doi.org/10.3368/jhr.50.2.317>
- statsmodels `cov_type` docs: <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.fit.html>

## Description

Build OLS from scratch via QR decomposition and validate it against `statsmodels`, then
break each Gauss-Markov assumption on purpose using synthetic data with a known true
`beta`: examine the sampling distribution of `beta_hat` via Monte Carlo, diagnose
heteroskedasticity with the Breusch-Pagan test, implement HC0-HC3 robust standard errors,
and implement cluster-robust standard errors. Every estimate is compared against a known
ground truth — that comparison is the pedagogical core.

### What you'll learn

1. Why production OLS implementations use QR decomposition instead of the textbook normal equations.
2. What the Gauss-Markov theorem actually guarantees, and what breaks when each assumption fails.
3. How to *test* for heteroskedasticity (Breusch-Pagan) instead of just assuming it.
4. How White's sandwich estimator (HC0-HC3) fixes standard errors without touching the point estimate.
5. Why clustering is a distinct problem from heteroskedasticity, and how cluster-robust SEs fix it.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_ols_gauss_markov.ipynb`

### Phase 1: OLS via QR Decomposition (~15-20 min) — `src/_01_ols_qr.py`

Synthetic data generation (`src/datasets.py`) and the statsmodels comparison harness are
fully scaffolded. The teaching content is the estimator itself.

1. **TODO #1 — `ols_via_qr(X, y)`**: implement OLS via `np.linalg.qr` + triangular solve,
   returning coefficients, fitted values, residuals, and `(X'X)^-1` (derived from `R`,
   never by inverting `X'X` directly). ~15-20 lines.

### Phase 2: Sampling Distribution of beta-hat (~15 min) — `src/_02_sampling_distribution.py`

The Monte Carlo simulation loop (redraw the dataset, refit, collect `beta_hat`) is fully
scaffolded. The teaching content is the classical variance formula it's checked against.

2. **TODO #1 — `ols_vcov_homoskedastic(X, resid, XtX_inv)`**: implement
   `Var(beta_hat) = sigma_hat^2 * (X'X)^-1`. ~5 lines.

### Phase 3: Diagnosing Heteroskedasticity (~15 min) — `src/_03_heteroskedasticity.py`

3. **TODO #1 — `breusch_pagan_lm_test(X, resid)`**: implement the Breusch-Pagan LM test
   (auxiliary regression of squared residuals on `X`, `LM = n*R^2`, compare to `chi2`).
   ~15-20 lines.

### Phase 4: Robust Standard Errors, HC0-HC3 (~20 min) — `src/_04_robust_se.py`

4. **TODO #1 — `hc_vcov(X, resid, XtX_inv, kind)`**: implement White's sandwich estimator
   with the HC0/HC1/HC2/HC3 per-observation weight variants. ~15-20 lines.

### Phase 5: Clustered Standard Errors (~15-20 min) — `src/_05_clustered_se.py`

5. **TODO #1 — `cluster_robust_vcov(X, resid, XtX_inv, cluster_id)`**: implement the CR1
   cluster-robust sandwich estimator with the standard small-sample correction. ~15-20
   lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: a coefficient plot comparing classical vs. HC3 standard
errors on the same point estimate, on the heteroskedastic dataset.

### What to look for in the results

- Phase 1's `max |diff|` against statsmodels should be on the order of `1e-10` or
  smaller — if it isn't, the QR route has a bug, not a rounding quirk.
- Phase 2: analytical and empirical SEs should agree closely on the *homoskedastic*
  scenario — that agreement is the Gauss-Markov theorem made visible.
- Phase 3: the Breusch-Pagan p-value should reject homoskedasticity for the
  heteroskedastic scenario and fail to reject it for the homoskedastic one.
- Phase 4: HC3 standard errors are typically a bit larger than HC0's on small samples —
  that gap *is* the finite-sample correction.
- Phase 5: naive (non-clustered) SEs on the clustered dataset should be visibly smaller
  than the cluster-robust ones — the standard "false confidence" failure mode.

## Motivation

- **AutoScheduler.AI relevance**: any "does this scheduling parameter actually move the
  outcome metric" question is an OLS-plus-correct-standard-errors question in disguise —
  this practice is the foundation every other causal-inference practice in this
  curriculum builds on.
- **Senior → Staff differentiator**: knowing to reach for `cov_type="HC3"` is common;
  knowing *why* it's needed, what it actually computes, and when it's still not enough
  (clustering) is not.
- **Generalises beyond econometrics**: the sandwich-estimator pattern (bread-meat-bread)
  reappears anywhere you fit a model by M-estimation and need a robust covariance —
  GLMs, GEEs, and some ML calibration methods included.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_ols_gauss_markov.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_ols_qr` | Sanity-check OLS-via-QR against statsmodels. |
| **Phase 2** | `uv run python -m src._02_sampling_distribution` | Compare analytical vs. empirical (Monte Carlo) standard errors. |
| **Phase 3** | `uv run python -m src._03_heteroskedasticity` | Run the Breusch-Pagan test on both scenarios. |
| **Phase 4** | `uv run python -m src._04_robust_se` | Compute HC0-HC3 standard errors on the heteroskedastic dataset. |
| **Phase 5** | `uv run python -m src._05_clustered_se` | Compute cluster-robust standard errors on the clustered dataset. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
