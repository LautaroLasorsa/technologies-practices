# Practice 107 — Sensitivity Analysis & Partial Identification

## Technologies

- **statsmodels** — OLS/logistic regressions used to produce the short/long
  regression coefficients (Oster) and the naive point estimate the final
  plot contrasts against the identification bounds.
- **scipy.stats** — the normal approximation behind Rosenbaum's sensitivity
  p-value and the risk-ratio confidence interval feeding the E-value.
- **xy** — plotting (sensitivity contour, bound-narrowing chart, E-value display).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The gap a confidence interval doesn't cover

A regression coefficient plus its 95% confidence interval only prices in
**sampling error** — the uncertainty from having drawn one particular
sample out of all the samples the same data-generating process could have
produced. It says nothing about **identification error**: whether the
assumption that makes the coefficient causal (no unmeasured confounding,
no non-ignorable missingness, etc.) is even approximately true. A narrow
CI around a badly-identified estimate is *false precision* — it looks
confident about the wrong number. This practice is entirely about
quantifying that second, usually-ignored kind of error, using four
different methods that each attack a different flavor of "the
identifying assumption might be wrong."

### Rosenbaum bounds (matched designs)

A matched-pairs comparison assumes that, conditional on the matching
covariates, which unit in a pair got treated was as good as random.
Rosenbaum's (2002) sensitivity analysis introduces a parameter **Gamma**
— the odds by which a hidden confounder could make one matched unit more
likely than its pair partner to have been treated — and asks how large
Gamma would need to be before the matched-pair test (Wilcoxon signed-rank)
stops rejecting the null. A large critical Gamma means the result is
robust to substantial hidden bias; a Gamma near 1 means even a small
hidden confounder could overturn it.

### Oster's delta and R-max

A different signal, usable even without a matched design: how much does a
treatment coefficient move when observed controls are added? Oster (2019)
formalizes "the effect barely moved when I added controls" into a number:
**delta**, the multiple of "unobservable selection on the same scale as
observable selection" that would be needed to drive the coefficient to
zero, given a ceiling **R-max** on how much of the outcome's variance all
regressors (observed + unobservable) could jointly explain. A large delta
means an implausibly strong unobservable confounder would be needed;
delta near or below 1 is a red flag.

### E-values

Ding & VanderWeele (2016) unify the same question across methods into one
scale: the risk-ratio strength an unmeasured confounder would need with
**both** treatment and outcome, simultaneously, to fully explain away an
observed risk ratio. Unlike Rosenbaum's Gamma or Oster's delta, the
E-value needs no matched design and no regression-control structure — just
a risk ratio (and, for the harder question, a confidence-interval limit).

### Manski-style partial identification

The methods above all still assume a point estimate is at least
*computable* — they bound how much confounding it would take to be wrong.
Manski's (1990, 2003) framework instead abandons point identification when
the data structurally cannot support it (here: outcomes missing
non-ignorably) and reports the full range of values consistent with the
observed data and *only the assumptions you're willing to state*. The
widest, assumption-free version is the **worst-case bound**: with no
assumption about missing values, they could be anywhere in the outcome's
known range. Adding a **monotonicity** assumption — Lee's (2009) trimming
bounds, requiring that treatment shift selection-into-observation in only
one direction — narrows the interval, at the cost of that assumption being
untestable but often substantively plausible.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Sampling error** | Uncertainty from drawing one sample out of many the DGP could produce; what a standard CI captures. |
| **Identification error** | Uncertainty about whether the causal assumption behind an estimate holds at all; what this practice quantifies. |
| **Gamma (Rosenbaum)** | Odds ratio of hidden-bias-driven treatment assignment within a matched pair. |
| **Delta (Oster)** | Multiple of observable-scale selection an unobservable confounder needs to fully explain a coefficient. |
| **R-max** | Ceiling on the variance a fully-specified regression (observed + unobservable) could explain; Oster's usual default is `min(1.3 * R2_long, 1)`. |
| **E-value** | Minimum risk-ratio strength a confounder needs with *both* treatment and outcome to explain away an association. |
| **Worst-case (Manski) bounds** | Assumption-free bounds on a causal estimand when part of the outcome distribution is unobserved. |
| **Monotonicity / trimming (Lee) bounds** | Narrower bounds under an assumption that selection into observation moves in one direction only. |

### Where This Fits

Every method here is a *post-hoc* companion to a point estimate from
elsewhere in this curriculum (Practice 097's OLS, 098's experiments, 100's
matching/IPW, 102's fixed effects, ...) — none of them estimate a causal
effect on their own; they interrogate how much to trust one you already
have. Alternatives not covered here include instrumental-variables-based
bounds (Practice 101 sidesteps confounding differently, by finding a
source of *exogenous* variation instead of bounding around the lack of
one) and fully Bayesian sensitivity priors over the confounding strength
(Practice 109) — a more model-heavy alternative to this practice's
closed-form bounds.

### References

- Rosenbaum, P., 2002, *Observational Studies*, ch. 4: <https://link.springer.com/book/10.1007/978-1-4757-3692-2>
- Oster, E., 2019, "Unobservable Selection and Coefficient Stability: Theory and Evidence": <https://doi.org/10.1080/07350015.2016.1227711>
- Ding, P. & VanderWeele, T., 2016, "Sensitivity Analysis Without Assumptions": <https://doi.org/10.1097/EDE.0000000000000457>
- Manski, C., 1990, "Nonparametric Bounds on Treatment Effects": <https://www.jstor.org/stable/2006592>
- Lee, D., 2009, "Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects": <https://doi.org/10.1111/j.1467-937X.2009.00536.x>

## Description

Build a synthetic confounded-treatment / non-ignorably-missing dataset
with a known true ATE, then quantify identification error four ways:
Rosenbaum bounds on a matched design, Oster's delta from a short-vs-long
regression, E-values from a risk ratio, and Manski/Lee partial-
identification bounds. Every method's output is compared against the same
known-true ATE, and the final plot puts a naive point-estimate CI next to
the identification bounds so the sampling-vs-identification contrast is
visually obvious.

### What you'll learn

1. Why a confidence interval alone cannot tell you whether a causal
   assumption is right.
2. How Rosenbaum's Gamma converts "how much hidden bias would break this"
   into a single interpretable number for a matched design.
3. How Oster's delta uses coefficient movement under added controls to
   bound the same question without needing a matched design.
4. How the E-value puts confounding strength on one common (risk-ratio)
   scale, independent of method.
5. How to move from assumption-free worst-case bounds to narrower,
   assumption-justified bounds (Manski to Lee) when identification fails
   outright.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_sensitivity_partial_id.ipynb`

### Phase 1: Rosenbaum Bounds (~20 min) — `src/_01_rosenbaum_bounds.py`

Matching on observed covariates and the Wilcoxon-statistic bookkeeping are
fully scaffolded. The teaching content is the Gamma-sensitivity p-value
bound itself.

1. **TODO #1 — `rosenbaum_pvalue_bound(diffs, gamma)`**: implement the
   upper-bound p-value on the matched-pair Wilcoxon signed-rank statistic
   under hidden-bias strength `gamma`. ~15-20 lines.

### Phase 2: Oster's Delta (~20 min) — `src/_02_oster_delta.py`

Fitting the short and long regressions is fully scaffolded. The teaching
content is turning their four summary numbers into a bound.

2. **TODO #1 — `oster_delta(inputs, r_max)`**: implement Oster's delta
   formula and the bias-adjusted coefficient at `delta = 1`. ~15-20 lines.

### Phase 3: E-values (~15 min) — `src/_03_e_value.py`

The risk ratio and its confidence interval are fully scaffolded. The
teaching content is the E-value conversion itself.

3. **TODO #1 — `e_value(rr)`**: implement the Ding & VanderWeele E-value
   formula for a risk ratio (with the `rr < 1` inversion case). ~8-10 lines.

### Phase 4: Manski & Lee Bounds (~20-25 min) — `src/_04_manski_bounds.py`

4. **TODO #1 — `manski_worst_case_bounds(y_obs, response, t, y_min, y_max)`**:
   implement the assumption-free worst-case bounds on the ATE under
   non-ignorable missingness. ~15-20 lines.
5. **TODO #2 — `lee_trimming_bounds(y_obs, response, t)`**: implement Lee's
   (2009) monotone-selection trimming bounds, narrower than Manski's under
   the added monotonicity assumption. ~15-20 lines.

### Phase 5: End-to-End Run (no TODO)

Run the notebook's final cells: a naive point estimate + sampling-only CI
(fit on complete cases, ignoring the missingness problem entirely) plotted
next to the Manski and Lee identification bounds.

### What to look for in the results

- Phase 1: the critical Gamma should be well above 1 but not enormous —
  this design has real, but not overwhelming, hidden confounding baked in.
- Phase 2: `beta_star(delta=1)` should sit noticeably below `beta_long`,
  reflecting the same hidden confounder Phase 1 detected.
- Phase 3: the CI-limit E-value should always be `<=` the point-estimate
  E-value — explaining away "no effect at the CI edge" is an easier bar.
- Phase 4: Lee's bounds should sit strictly inside Manski's, and both
  should contain the true ATE — narrower assumptions should never
  *exclude* the truth, only tighten around it.
- Final plot: the naive CI is narrow and looks confident; the
  identification bounds are visibly wider — that gap is the entire point
  of this practice.

## Motivation

- **AutoScheduler.AI relevance**: any offline evaluation of a scheduling
  policy change built on observational logs (not a randomized rollout) is
  exactly this situation — a point estimate with a CI that says nothing
  about whether the logging policy's confounding has been fully accounted
  for.
- **Senior → Staff differentiator**: producing a point estimate with a CI
  is table stakes; being able to say "and here's how robust that estimate
  is to the assumption you can't test" is what separates a credible causal
  claim from a fragile one.
- **Generalises beyond econometrics**: the same sampling-vs-identification
  distinction applies to any observational ML evaluation (offline RL,
  counterfactual evaluation, A/B-test-adjacent observational studies) —
  this practice's four tools are directly reusable there.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_sensitivity_partial_id.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_rosenbaum_bounds` | Compute Rosenbaum p-value bounds and the critical Gamma on the matched design. |
| **Phase 2** | `uv run python -m src._02_oster_delta` | Compute Oster's delta and the bias-adjusted coefficient. |
| **Phase 3** | `uv run python -m src._03_e_value` | Compute the risk ratio and its point/CI-limit E-values. |
| **Phase 4** | `uv run python -m src._04_manski_bounds` | Compute Manski worst-case and Lee trimming bounds on the ATE. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
