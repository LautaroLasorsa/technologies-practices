# Practice 109 — Bayesian Causal Inference & Structural Time Series

## Technologies

- **PyMC** — probabilistic programming; specifies the Bayesian structural time series (BSTS) model.
- **nutpie** — a prebuilt, Rust-backed NUTS sampler PyMC can delegate to (`nuts_sampler="nutpie"`); used here so sampling never depends on a working system C/C++ compiler (PyTensor's default C linker is broken on some Windows setups — see `src/__init__.py`).
- **arviz** — posterior diagnostics (trace plots, summary tables) and posterior predictive checks.
- **xy** — plotting for everything we compose ourselves (observed-vs-counterfactual, cumulative effect, effect density).
- **matplotlib** — only for the two arviz diagnostic calls that need a real `matplotlib.axes.Axes` (see Plotting below).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### Two epistemologies for the same question

Practice 105 (synthetic control) and this practice ask the identical causal question —
"what would this series have done without the intervention?" — from two different
foundations. Synthetic control poses it as **constrained optimization**: find donor
weights (non-negative, summing to one) that make a weighted combination of untreated
units track the treated unit before the intervention, then project that combination
forward. The output is one counterfactual path and one effect number; uncertainty, if
reported at all, comes from a separate device bolted on afterward (placebo tests
re-running the same optimization on untreated units as if *they* were treated, and
comparing effect sizes).

Bayesian structural time series poses the same question as **posterior inference over a
generative model**: specify a state-space process for how the series evolves (a latent
level, optionally trend and seasonality, plus a regression on controls), put priors on
its parameters, and condition on the pre-intervention data. Forecasting that model
forward from the intervention point does not produce one counterfactual path — it
produces a full **posterior distribution over counterfactual paths**, because every
posterior draw of the parameters generates its own forward simulation. Uncertainty is
not a separate step; it is the direct, structural consequence of Bayesian conditioning.
The trade-off is the classic one: synthetic control needs no distributional assumptions
and is simple to audit (the donor weights are directly interpretable), but its inference
is asymptotic/permutation-based and brittle with few donors; BSTS needs a correctly
specified generative model and priors, but rewards it with an honest, principled
uncertainty quantification at every step, including the one number decision-makers
actually want: "how confident are we, and in what range, was the total impact?"

### Priors as auditable assumptions

A frequentist standard error is one number computed from one point estimate's asymptotic
sampling behavior; you cannot inspect *why* it is what it is without re-deriving the
estimator's variance. A Bayesian prior, by contrast, is a **written-down, inspectable
assumption** — "the level's day-to-day drift is unlikely to exceed roughly 1 unit"
(`sigma_level ~ HalfNormal(1.0)`) is a claim anyone can read, question, and replace. This
is not a rhetorical nicety: it means every assumption this practice's model makes about
how the target series behaves is visible in `src/_01_structural_model.py`, rather than
implicit in an estimator's derivation.

### The posterior as the object of interest

A 95% confidence interval's textbook meaning is about the *procedure*: were you to
repeat the experiment infinitely, 95% of such intervals would contain the true
parameter. It says nothing about *this* interval containing *this* parameter. A 94%
**credible interval** (arviz's default mass, following McElreath's *Statistical
Rethinking*) says exactly what people intuitively want a confidence interval to say:
given the data and the model, there is a 94% probability the parameter lies in this
interval. That shift — from a claim about a repeated procedure to a direct probability
statement about the unknown quantity — is the entire point of doing inference this way,
and it is why "the probability the effect is positive" (a single number this practice
computes directly from the posterior, Phase 3) is a coherent question to ask at all; a
frequentist p-value cannot answer it without redefinition.

### Structural time series as a counterfactual generator

A structural time series model decomposes an observed series into unobserved *state*
components evolving through time, plus a regression on control series:

| Component | Role |
|---|---|
| **Local level** | A random walk representing the series' current "baseline" — this practice's core state. |
| **Local trend** | (Not implemented here, see Description) A random walk on the *slope*, letting the level's drift itself drift. |
| **Seasonality** | (Not implemented here) Periodic components (day-of-week, month-of-year) as additional latent states. |
| **Regression** | A (static, in this practice) coefficient vector on control series that move with the target but are untouched by the intervention. |

Conditioning this model on only the pre-intervention data, then forecasting its state
forward through the post-period, produces the counterfactual: what the series would
have looked like had nothing changed. The gap between that forecast and what was
actually observed, accumulated over time, is the CausalImpact-style estimate — with a
full posterior attached at every step, because the forecast itself came from sampling,
not from solving a single equation.

### Posterior predictive checks as model criticism

A posterior over parameters can be well-calibrated relative to a *badly specified*
model — the machinery does not protect you from having written down the wrong
generative process. A posterior predictive check closes that gap: simulate new data
from the fitted model (parameters drawn from their posterior) and compare it against
the real data the model was trained on. Systematic mismatch — the simulated pre-period
series consistently missing a feature of the real one — is evidence the model itself,
not just its parameter estimates, needs revision. This is Bayesian workflow's answer to
"how do I know I can trust this," and it runs *before* the counterfactual forecast is
taken seriously (Phase 5).

### Key Concepts

| Concept | Definition |
|---|---|
| **Posterior distribution** | The updated belief over a parameter after conditioning a prior on observed data. |
| **Credible interval** | An interval containing a stated probability mass of the posterior — a direct probability statement, unlike a confidence interval. |
| **HDI (highest density interval)** | The *narrowest* interval containing a given posterior mass; coincides with the equal-tailed interval only for symmetric posteriors. |
| **Local level** | A latent random-walk state representing the series' baseline, absent trend/seasonality. |
| **Counterfactual posterior** | The distribution over "what the series would have done," obtained by forecasting the fitted state-space model forward per posterior draw. |
| **Posterior predictive check** | Simulating new data from the fitted model and comparing it to the real training data, as model criticism. |

### Where This Fits

Alternatives to this practice's local-level BSTS include: full structural time series
with trend and seasonal components (Scott & Varian's original CausalImpact R package,
and the `pymc-experimental` `statespace` module, both natural extensions of this
practice's model); synthetic control (practice 105, no distributional assumptions,
weaker uncertainty quantification); and difference-in-differences / panel
fixed-effects methods (practices 101-103, which need parallel-trends rather than a
correctly specified generative model, but cannot produce a full posterior over the
effect). BSTS-style counterfactual forecasting is the standard tool at Google
(CausalImpact) and Meta/Uber-style experimentation teams for single-unit,
time-series interventions where a proper A/B test was impossible (a product launch,
a policy change, a marketing campaign that could not be randomized).

### References

- Brodersen et al., 2015, "Inferring causal impact using Bayesian structural
  time-series models": <https://research.google/pubs/pub41854/>
- Scott & Varian, 2014, "Predicting the present with Bayesian structural time series":
  <https://www.google.com/url?q=https://people.ischool.berkeley.edu/~hal/Papers/2013/pred-present-with-bsts.pdf>
- McElreath, *Statistical Rethinking*, ch. 2-4 (credible intervals, HDI, posterior
  predictive checks): <https://xcelab.net/rm/statistical-rethinking/>
- PyMC `GaussianRandomWalk` docs: <https://www.pymc.io/projects/docs/en/stable/api/distributions/generated/pymc.GaussianRandomWalk.html>
- arviz `hdi` docs: <https://python.arviz.org/en/stable/api/generated/arviz.hdi.html>

## Description

Build a local-level Bayesian structural time series model in PyMC on synthetic
pre/post-intervention data with a **known** injected effect, sample its posterior,
forecast the counterfactual forward from the intervention point for every posterior
draw, compute the pointwise and cumulative effect posteriors, implement the highest
density interval from scratch, run a posterior predictive check, and plot the full
CausalImpact-style summary — observed vs. counterfactual, cumulative effect over time,
and the posterior density of the total effect.

### What you'll learn

1. How to specify a local-level structural time series model in PyMC (a random-walk
   latent state plus a regression on controls).
2. Why forecasting a fitted Bayesian model forward, per posterior draw, produces a full
   distribution over counterfactual paths instead of one number.
3. How to turn a counterfactual posterior into an effect posterior, and why that
   propagates uncertainty honestly compared to a delta-method standard error.
4. What the highest density interval is, why it differs from an equal-tailed interval,
   and how to compute it directly.
5. Why a posterior predictive check is necessary before trusting a Bayesian model's
   forecast — and how it differs from checking parameter convergence.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_bayesian_causal_sts.ipynb`

### Phase 1: The Structural Model (~20 min) — `src/_01_structural_model.py`

Synthetic data generation (`src/datasets.py`, a target + 2 controls sharing a latent
trend, with a known constant effect injected from day 90) is fully scaffolded. The
teaching content is the model specification itself.

1. **TODO #1 — `build_local_level_model(y_pre, X_pre)`**: specify the local-level BSTS
   model (`GaussianRandomWalk` level + static regression on controls + observation
   noise) as an un-sampled PyMC model. ~15-20 lines.

### Phase 2: Posterior Sampling & the Counterfactual Forecast (~20-25 min) — `src/_02_counterfactual.py`

The MCMC sampling harness (`run_mcmc`, a thin `pm.sample` wrapper) is fully scaffolded.
The teaching content is forecasting the fitted model's latent state forward.

2. **TODO #1 — `counterfactual_forecast(idata, X_post, rng)`**: simulate the local
   level's random walk forward through the post-period for every posterior draw, add
   each draw's regression and observation-noise contribution, returning a
   `(n_draws, n_post)` array of counterfactual paths. ~15-20 lines.

### Phase 3: The Effect Posterior (~10-15 min) — `src/_03_effect.py`

3. **TODO #1 — `cumulative_effect(y_post, counterfactual_draws)`**: compute the
   pointwise (`observed - counterfactual`) and cumulative (running sum over time)
   effect posteriors from the counterfactual draws. ~8-10 lines.

### Phase 4: The Highest Density Interval (~15 min) — `src/_04_hdi.py`

4. **TODO #1 — `hdi(samples, prob)`**: implement the highest density interval via the
   sorted-window algorithm (the same approach `arviz.hdi` uses internally), without
   calling any library HDI helper. ~10-15 lines.

### Phase 5: Posterior Predictive Checks (no TODO)

Run the notebook's `pm.sample_posterior_predictive` + `arviz.plot_ppc` cells — model
criticism before the counterfactual forecast is trusted.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: the observed-vs-counterfactual plot, the cumulative
effect plot, and the posterior density of the total effect, all shaded with the
Phase 4 HDI and compared against the known ground truth.

### What to look for in the results

- Phase 1: `model` should print three free RVs (`sigma_level`, `sigma_obs`, `beta`) plus
  the `level` random walk and the `y_obs` likelihood — if any is missing, the model is
  under-specified.
- Phase 2: `az.plot_trace` should show reasonably well-mixed chains for `sigma_obs` and
  `beta[1]`; `sigma_level` and `beta[0]` typically show a visibly noisier trace and a
  slightly elevated `r_hat` (around 1.05-1.2) at this practice's deliberately short
  chain length — that is the level/regression trade-off inherent to this DGP (both can
  partly explain the same latent trend), not a bug. If every parameter looks that noisy,
  or `r_hat` is above ~1.3 anywhere, increase `tune`/`draws` before trusting anything
  downstream.
- Phase 3: `P(total effect > 0)` should be close to 1.0 — the synthetic effect is
  positive and large relative to the noise, so the posterior should reflect that clearly.
- Phase 4: the 94% HDI on the total effect should contain (or sit very close to) the
  dataset's `true_cumulative_effect` — that containment is the ground-truth check this
  whole practice is built around.
- Phase 5: the posterior predictive band around the pre-period series should visibly
  contain the real series — systematic misses would mean the model is misspecified
  before it ever forecasts anything.

## Motivation

- **AutoScheduler.AI relevance**: any "did this scheduling change actually move the
  metric" question, asked on a single deployed system with no A/B test available, is
  exactly the BSTS/CausalImpact setting — a before/after series plus untreated control
  metrics.
- **Senior → Staff differentiator**: most engineers can compute a t-test; being able to
  say "the effect is between X and Y with 94% posterior probability, and here is the
  exact generative model that claim rests on" is a materially different, more defensible
  kind of statement to bring into a decision meeting.
- **Generalises beyond econometrics**: the "condition on pre-period, forecast forward,
  compare to observed" pattern reused here is the same shape as anomaly detection,
  demand forecasting with structural breaks, and any Bayesian time-series problem where
  the question is "did something change."

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_bayesian_causal_sts.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_structural_model` | Build and sample the local-level model; print a posterior summary. |
| **Phase 2** | `uv run python -m src._02_counterfactual` | Sample the posterior and forecast the counterfactual over the post-period. |
| **Phase 3** | `uv run python -m src._03_effect` | Compute the cumulative effect posterior and its probability of being positive. |
| **Phase 4** | `uv run python -m src._04_hdi` | Compute the HDI of a known Normal sample and sanity-check it against `arviz.hdi`. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
