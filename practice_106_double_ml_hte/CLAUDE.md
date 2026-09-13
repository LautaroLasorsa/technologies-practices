# Practice 106 — Double ML & Heterogeneous Treatment Effects

## Technologies

- **scikit-learn** — flexible ML nuisance models (gradient-boosted trees, random forests) used as the "machine learning" in double machine learning.
- **EconML** — Microsoft's causal-ML library; used only for optional cross-checks (`LinearDML`, `CausalForestDML`) against the hand-rolled estimators this practice builds.
- **xy** — plotting (bias comparison, CATE calibration, policy-value curve).

`causalml` was dropped from this practice's dependencies — see Notes.

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The Problem: Naive ML Breaks Causal Inference

Suppose you want the effect of a treatment `D` on an outcome `Y`, adjusting for
covariates `X` with a partially linear model `Y = theta*D + g0(X) + U`. The
tempting shortcut is to plug a flexible ML model in for `g0` — fit
`g_hat(X) ~= E[Y|X]` with a gradient-boosted tree or random forest, then treat
`theta` as the coefficient on `D` in whatever is left over. This is a **naive
plug-in estimator**, and it is biased, not because the code is wrong, but
because of what flexible ML models *have* to do to generalize: regularize.

Frame it as a first-order-condition problem, the way you'd check whether a
constraint in an optimization problem is binding at a critical point. An
estimator built from a moment condition `E[psi(W; theta, eta)] = 0` (`eta`
standing for the nuisance functions, here `g0` and the propensity `m0`) is only
robust to *plugging in an estimated* `eta_hat` if the condition's sensitivity
to `eta` — its Gateaux derivative in the direction `eta_hat - eta0` — vanishes
at the truth. That property is **Neyman orthogonality**. The naive moment
`E[(Y - g0(X) - theta*D)*D] = 0` does *not* have it: differentiate with
respect to `g0` and the derivative is `-E[D * delta_g(X)]`, which is nonzero
whenever `D` and `X` are correlated (i.e., whenever there's confounding — the
normal case in observational data). So a `g_hat` regularization error of order
`n^{-1/4}` (typical for flexible ML, far slower than a parametric model's
`n^{-1/2}`) leaks straight into `theta_hat` at that same slow rate. Orthogonal
moments instead residualize **both** sides against `X`:
`E[(Y - g0(X) - theta*D)*(D - m0(X))] = 0`. Now the derivative with respect to
either nuisance vanishes at the truth to first order — estimation error in
`g_hat`/`m_hat` only enters `theta_hat`'s error at *second* order, small enough
that the `n^{-1/4}`-rate ML nuisances still deliver a `sqrt(n)`-consistent,
asymptotically normal `theta_hat`. This is exactly the same
"first-order-condition insensitivity" argument as checking whether a small
perturbation to a constraint moves an optimum: at a well-posed critical point,
first-order perturbations don't move the objective, and here they don't move
the estimating equation either.

### Cross-Fitting: Sample Splitting for Nuisance Estimation

Orthogonality alone is not enough. If `g_hat`/`m_hat` are estimated on the same
sample used to form `theta_hat`, a flexible enough model **overfits its own
training observations** — its residual for observation `i` is artificially
small precisely because the model partially memorized `Y_i`/`D_i`, and that
error is correlated with `i`'s own contribution to the moment. This is a
different failure mode from orthogonality: no amount of clever moment-writing
fixes it, because the problem is the *dependence* between the nuisance-fitting
sample and the evaluation sample, not the moment's functional form.

**Cross-fitting** is the fix, and it is exactly sample splitting: partition the
data into `K` folds, and for each fold, fit the nuisance models on the *other*
`K-1` folds and predict (out-of-fold) on the held-out one. Every residual used
in the final moment now comes from a model that never saw that observation
during training — the same logic as evaluating a model on a held-out test set,
just repeated `K` times so every observation eventually gets a held-out
prediction. Combined, orthogonality (insensitive to nuisance *bias*) and
cross-fitting (removes own-observation *overfitting*) are what let Double
Machine Learning (Chernozhukov et al., 2018) use black-box ML for nuisances
while still getting valid, `sqrt(n)`-consistent inference on `theta`.

### The Partially Linear DML Model

Putting it together: `Y = theta*D + g0(X) + U`, `D = m0(X) + V`. Cross-fit
`g_hat` (regression of `Y` on `X`) and `m_hat` (regression/classification of
`D` on `X`) out-of-fold, form residuals `Y_resid = Y - g_hat(X)` and
`D_resid = D - m_hat(X)`, then solve the orthogonal moment — which, once both
sides are residualized, collapses to an ordinary bivariate OLS slope
(Frisch-Waugh-Lovell): `theta_hat = Cov(D_resid, Y_resid) / Var(D_resid)`. This
is Robinson's (1988) semiparametric estimator, rediscovered as the DML
special case for a constant treatment effect.

### Heterogeneity: CATE, Meta-Learners, and Policy

A constant `theta` is often the wrong question — different units usually
respond differently to treatment. The **Conditional Average Treatment Effect**
`tau(x) = E[Y(1) - Y(0) | X=x]` is the heterogeneous generalization.
**Meta-learners** turn any off-the-shelf regressor into a CATE estimator by
composing predictions: the **S-learner** fits one model of `Y` on `(X, D)`
jointly and differences its predictions at `D=1` vs `D=0`; the **T-learner**
fits two separate models, one per arm, and differences them (this practice's
Phase 4); the **X-learner** additionally imputes each unit's counterfactual
using the other arm's model, then models the imputed effects directly,
performing better than T-learner when arms are imbalanced. **Causal forests**
(Wager & Athey, 2018; EconML's `CausalForestDML`) generalize random forests to
target treatment-effect heterogeneity directly, using an orthogonalized,
"honest" splitting criterion — the automatic, more expensive alternative to a
hand-picked meta-learner.

A CATE estimate is only useful if it changes a decision. Given a budget
(you can afford to treat some fraction of the population), the natural
**policy** is "treat whoever `tau_hat(x)` says benefits most, until the budget
runs out". Because you can't re-run the DGP under a new policy, its **value**
is estimated on the same observational data via inverse-propensity weighting:
weight each observation by how likely its *observed* treatment matches what
the policy *would have* assigned, so units whose actual treatment happens to
coincide with the policy's choice (weighted by the rarity of that match) stand
in for a population-wide rollout of the policy.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Neyman orthogonality** | A moment condition's first derivative with respect to the nuisance function vanishes at the truth — nuisance estimation error enters the target parameter's error only at second order. |
| **Cross-fitting** | K-fold sample splitting for nuisance estimation: fit nuisances on K-1 folds, predict on the held-out fold, so no observation's residual uses a model trained on itself. |
| **Partially linear model** | `Y = theta*D + g0(X) + U` — constant treatment effect `theta`, arbitrary nonlinear baseline `g0(X)`. |
| **Robinson's residual-on-residual regression** | Solving the orthogonal partially-linear moment reduces to OLS of `Y_resid` on `D_resid` (Frisch-Waugh-Lovell). |
| **CATE** | `tau(x) = E[Y(1)-Y(0)\|X=x]` — the treatment effect as a function of covariates, generalizing a constant `theta`. |
| **Meta-learner (S/T/X)** | A CATE estimator built by composing off-the-shelf regressors rather than a purpose-built causal model. |
| **Causal forest** | A random-forest variant with an orthogonalized, honest splitting rule that targets heterogeneity in `tau(x)` directly. |
| **Policy value** | The expected outcome under a treatment-assignment rule, estimated via inverse-propensity weighting on observational data. |

### Where This Fits

DML is "OLS-with-robust-SEs plus flexible nuisances" applied to any moment
condition, not just the partially linear model — the same orthogonalize
+ cross-fit recipe underlies IV-DML, DID-DML, and the `LinearDML`/`CausalForestDML`
classes in EconML. Alternatives to the meta-learner approach to heterogeneity
include fully Bayesian causal models (BART-based, e.g. `bcf`) and R-learner-style
direct loss minimization; the trade-off is almost always
interpretability/simplicity (meta-learners, easy to explain) vs. statistical
efficiency (causal forests, R-learner, usually tighter but more opaque).
Practice 097's Gauss-Markov/robust-SE machinery is the direct ancestor of the
variance theory referenced here — every nuisance model in this practice is
still, underneath, "fit a regression, look at its residual."

### References

- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins, 2018, "Double/Debiased Machine Learning for Treatment and Causal Parameters": <https://doi.org/10.1111/ectj.12097>
- Robinson, 1988, "Root-N-Consistent Semiparametric Regression": <https://doi.org/10.2307/1912705>
- Kunzel, Sekhon, Bickel, Yu, 2019, "Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning": <https://doi.org/10.1073/pnas.1804597116>
- Wager & Athey, 2018, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests": <https://doi.org/10.1080/01621459.2017.1319839>
- EconML documentation: <https://econml.azurewebsites.net/>

## Description

Simulate data from a partially linear model with a known, heterogeneous
treatment effect, then compare three estimators of the average effect side by
side: a naive ML plug-in, an orthogonal-but-not-cross-fit estimator, and full
DML — over 200 replications, so the bias each one carries is visible in a box
plot rather than asserted in prose. Then move from the average effect to
heterogeneity: estimate CATE with a T-learner, check its calibration against
the known `tau(x)`, and turn the CATE estimate into a budget-constrained
treatment policy, evaluating that policy's value via inverse-propensity
weighting as the budget varies.

### What you'll learn

1. Why a naive ML plug-in estimator is biased even when the ML model itself is a good predictor — and why that bias doesn't shrink away with more data.
2. Neyman orthogonality as a first-order-condition insensitivity property, and why residualizing both `Y` and `D` against `X` gives you that property.
3. Cross-fitting as sample splitting, and why it's a distinct requirement from orthogonality, not a restatement of it.
4. How to estimate CATE with a meta-learner (T-learner) and check its calibration against ground truth.
5. How to turn a CATE estimate into a treatment policy and estimate that policy's value without re-running the experiment.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_double_ml_hte.ipynb`

### Phase 1: The Naive ML Plug-In Estimator (~15 min) — `src/_01_naive_bias.py`

The synthetic DGP (`src/datasets.py`) and the nuisance model factory are fully
scaffolded. The teaching content is the naive estimator itself — implementing
it is what makes its bias concrete in Phase 3, instead of an abstract warning.

1. **TODO #1 — `naive_plugin_theta(X, D, Y, model_y)`**: fit an ML model of
   `Y` on `X` alone (ignoring `D`), then regress the residual on raw `D`
   (never residualized against `X`). ~10 lines.

### Phase 2: Cross-Fitting (~15-20 min) — `src/_02_cross_fitting.py`

2. **TODO #1 — `cross_fit_residuals(X, D, Y, model_y_factory, model_d_factory, n_folds, seed)`**:
   implement the K-fold cross-fitting loop — fit both nuisance models on
   `K-1` folds, predict out-of-fold on the held-out fold, for every fold.
   ~15-20 lines.

### Phase 3: The Partially Linear DML Moment (~20 min) — `src/_03_partially_linear_dml.py`

The comparison harness (`dml_without_crossfit`, `full_dml_theta`, and the
200-replication Monte Carlo loop) is fully scaffolded and reuses Phases 1-2 —
the teaching content is the orthogonal moment itself.

3. **TODO #1 — `dml_theta(Y_resid, D_resid)`**: solve the residual-on-residual
   regression `theta_hat = Cov(D_resid, Y_resid) / Var(D_resid)`. ~5 lines.

### Phase 4: Heterogeneity — the T-Learner (~15-20 min) — `src/_04_meta_learner.py`

4. **TODO #1 — `t_learner_cate(X, D, Y, model_factory)`**: fit one outcome
   model on the treated subset and one on the control subset, and difference
   their predictions on the full `X`. ~10 lines.

### Phase 5: From CATE to Policy (~15-20 min) — `src/_05_policy_value.py`

5. **TODO #1 — `policy_value(tau_hat, Y, D, propensity, budget)`**: rank units
   by `tau_hat`, treat the top `budget` fraction, and compute the
   inverse-propensity-weighted value of that policy. ~15-20 lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: the bias-comparison box plot (Phase 3), the
CATE calibration scatter (Phase 4), and the policy-value-vs-budget curve
(Phase 5).

### What to look for in the results

- Phase 3's box plot: the naive estimator's box should sit visibly off zero;
  "DML, no cross-fit" should be closer to zero but still biased; full DML's
  box should straddle zero most tightly. If naive and full DML look identical,
  something in Phase 1 or the orthogonal moment is wrong.
- Phase 4: the calibration scatter should track the `y=x` diagonal loosely but
  clearly, especially at the extremes of `x0` where `tau(x)` is farthest from
  its average.
- Phase 5: the CATE-targeted policy's value curve should sit at or above the
  random-assignment curve at every budget — targeting real heterogeneity
  should never do *worse* than ignoring it.

## Motivation

- **AutoScheduler.AI relevance**: "which lever should we pull, and for whom"
  is exactly a CATE-and-policy question — DML is what makes it possible to
  answer "does this scheduling parameter help" using flexible ML nuisances
  without silently corrupting the causal estimate.
- **Senior → Staff differentiator**: knowing to reach for `EconML`
  is common; knowing why the naive version of the same idea is biased, and
  what specifically fixes it (orthogonality vs. cross-fitting are two
  different fixes for two different problems), is not.
- **Generalises beyond econometrics**: the orthogonalize-then-cross-fit
  pattern reappears anywhere a nuisance function is estimated by flexible ML
  on the way to a lower-dimensional target — semiparametric statistics,
  targeted maximum likelihood estimation (TMLE), and modern off-policy
  evaluation in RL all lean on the same two ideas.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_double_ml_hte.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_naive_bias` | Naive plug-in estimate vs. the known ATE. |
| **Phase 2** | `uv run python -m src._02_cross_fitting` | Cross-fitted residuals' summary stats. |
| **Phase 3** | `uv run python -m src._03_partially_linear_dml` | Compare naive / no-cross-fit / full DML on one draw. |
| **Phase 4** | `uv run python -m src._04_meta_learner` | T-learner CATE MAE vs. the known `tau(x)`. |
| **Phase 5** | `uv run python -m src._05_policy_value` | Policy value across a grid of budgets. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

- `causalml>=0.15` was in the original dependency list but was dropped: `uv sync`
  fails building it from source with `error: Unable to find a compatible Visual
  Studio installation` (its `causalml.inference.tree.causal` module is a Cython/C++
  extension, and `uv`'s isolated build could not locate the VS 2022 toolchain
  present on this machine — see the C++ Build Toolchain note in project memory).
  `EconML>=0.17` installs cleanly (`uv sync` exit 0, `import econml` succeeds,
  version 0.17.0) and is sufficient for every optional cross-check this practice
  uses (`LinearDML`, `CausalForestDML`). The core teaching content never depended
  on either package — it is hand-rolled DML + scikit-learn — so this practice is
  fully functional without `causalml`.

## State

`not-started`
