# Practice 098 — Potential Outcomes & Randomized Experiments

## Technologies

- **NumPy** — the difference-in-means estimator, Neyman's variance formula, the randomization-inference loop, and the CUPED adjustment, all implemented directly on arrays.
- **SciPy** (`scipy.stats.norm`) — the normal-approximation quantiles behind the power-analysis / minimum-detectable-effect formulas.
- **xy** — plotting (randomization null distribution, power curve, CUPED variance comparison).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The Potential-Outcomes Framework

Rubin's (1974) potential-outcomes model defines a causal effect at the
level of a single unit: every unit `i` has **two** potential outcomes,
`Y_i(0)` (what would happen under control) and `Y_i(1)` (what would happen
under treatment). The individual treatment effect is `tau_i = Y_i(1) -
Y_i(0)`, and the quantity almost every experiment actually wants is the
**average treatment effect** `tau = E[Y_i(1) - Y_i(0)]`. The catch that
gives this framework its name for the whole field: for any given unit,
only one of `Y_i(0)`/`Y_i(1)` is ever observed — the other is
**counterfactual**, gone the moment a treatment decision is made. Holland
(1986) calls this "the fundamental problem of causal inference." This
practice's data is synthetic specifically so both potential outcomes can
be kept around in a **science table** — a view no real experiment ever
gets — making the fundamental problem visible instead of abstract.

### Why Randomization Identifies the ATE

The reason a randomized experiment can estimate `tau` at all, without any
assumption about the *shape* of the outcome distribution, is that random
assignment makes treatment status **statistically independent of the
potential outcomes**: `(Y(0), Y(1)) ⊥ T`. That independence is what turns
a merely descriptive comparison into a causal one:

```
E[Y_obs | T=1] = E[Y(1) | T=1] = E[Y(1)]      (by independence)
E[Y_obs | T=0] = E[Y(0) | T=0] = E[Y(0)]      (by independence)
=> E[Y_obs | T=1] - E[Y_obs | T=0] = E[Y(1)] - E[Y(0)] = tau
```

No observational-data estimator gets this for free — every technique
later in this curriculum (RDD, synthetic control, instrumental variables,
panel fixed effects) is best understood as "recovering this same
independence in a setting where nobody flipped a coin."

### Neyman's Variance Estimator and Its Conservatism

Neyman (1923) derived the sampling variance of the diff-in-means
estimator directly from the **design** — which fixed number of units
happen to land in the treated group — rather than from a modeling
assumption. The result decomposes into three pieces:
`Var(tau_hat) = Var(Y(1))/n1 + Var(Y(0))/n0 - Var(tau_i)/n`. The first
two terms are estimable from the two groups' sample variances; the third
is not, because `tau_i` requires both potential outcomes for the *same*
unit, which is never observed. Dropping a term that is always `>= 0`
means the resulting variance *estimator* is **weakly conservative** — it
never understates the true sampling variance, so intervals built from it
remain valid (if sometimes wider than necessary), which is why applied
work treats it as the safe default rather than a downward-biased
approximation.

### Fisher's Sharp Null and Randomization Inference

Fisher's (1935) alternative starts from a **sharp null**:
`H0: Y_i(1) = Y_i(0)` for *every* unit, not merely on average. Under that
null, the fully observed outcome vector doesn't depend on which units
were assigned to treatment — so relabeling the same fixed outcomes with
every other possible complete-randomization assignment produces an
equally likely dataset. Recomputing the test statistic under thousands of
such relabelings builds an exact null distribution with **no
distributional assumptions**, and the fraction of relabelings at least as
extreme as the one actually observed is an exact p-value. This is a
fundamentally different inferential foundation than Neyman's variance
formula — it needs no asymptotics, only the physical randomization that
was actually performed.

### Power Analysis and the Minimum Detectable Effect

Before an experiment runs, the operative question flips from "what
variance did we get" to "what effect size could we have detected." The
**minimum detectable effect (MDE)** is the smallest true effect a test
would catch with a target probability (power, conventionally 80%) at a
given significance level and sample size — the standard
normal-approximation formula inverts the same two-sample variance from
the Neyman section to answer a design question instead of an inference
question.

### CUPED

CUPED — "Controlled-experiment Using Pre-Experiment Data" (Deng, Xu, Kohavi
& Walker, 2013) — reduces variance without touching the point estimate,
by adjusting the outcome with a pre-period covariate `X` measured *before*
assignment (so it cannot itself be affected by treatment):
`Y' = Y - theta * (X - mean(X))`, with `theta = Cov(Y, X)/Var(X)` chosen to
minimize `Var(Y')`. Because `theta` is a constant applied identically to
both arms, `E[Y'] = E[Y]` — the ATE estimate on `Y'` is unbiased for the
same `tau` — but `Var(Y') <= Var(Y)` whenever `X` is correlated with `Y`,
tightening every downstream confidence interval "for free."

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Potential outcomes** | `Y_i(0)`, `Y_i(1)` — what would happen to unit `i` under control/treatment. |
| **Individual treatment effect** | `tau_i = Y_i(1) - Y_i(0)`; never jointly observable for any unit. |
| **ATE** | `tau = E[Y_i(1) - Y_i(0)]`, the population-average causal effect. |
| **Fundamental problem of causal inference** | Only one of `Y_i(0)`/`Y_i(1)` is ever observed per unit (Holland, 1986). |
| **Science table** | The (synthetic-only) table with both potential outcomes visible. |
| **SUTVA** | Stable Unit Treatment Value Assumption — no interference between units, one version of each treatment. |
| **Neyman variance** | Design-based, conservative variance of the diff-in-means estimator. |
| **Sharp null** | Fisher's null hypothesis that treatment has *zero* effect for every unit, not just on average. |
| **Randomization inference** | Exact inference via reshuffling the observed treatment labels under the sharp null. |
| **MDE** | Minimum detectable effect — smallest true effect size detectable at a given power/sample size. |
| **CUPED** | Pre-period-covariate variance reduction that leaves the point estimate unbiased. |

### Where This Fits

Potential outcomes is the conceptual foundation the rest of this
curriculum stands on: RDD, synthetic control, panel/fixed-effects
methods, and instrumental variables are all strategies for approximating
the *same* `(Y(0), Y(1)) ⊥ T` independence in settings where nobody
actually randomized. Compared to purely predictive ML, causal-inference
estimators explicitly trade off predictive accuracy for unbiasedness of a
specific contrast (`E[Y(1)] - E[Y(0)]`) — a model that predicts `Y`
extremely well can still be a terrible estimator of `tau` if it isn't
built around this independence assumption.

### References

- Rubin, 1974, "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies": <https://doi.org/10.1037/h0037350>
- Holland, 1986, "Statistics and Causal Inference": <https://doi.org/10.1080/01621459.1986.10478354>
- Imbens & Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*, ch. 6 (Neyman) & ch. 5 (Fisher): <https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB>
- Deng, Xu, Kohavi & Walker, 2013, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data": <https://doi.org/10.1145/2433396.2433413>

## Description

Simulate a randomized experiment with a full science table (both
potential outcomes known, since the data is synthetic), then work through
the estimation and inference toolkit that every A/B test in industry
ultimately reduces to: the difference-in-means ATE estimator, Neyman's
conservative variance, Fisher's exact randomization test, a
minimum-detectable-effect power calculation, and a CUPED variance
reduction using a pre-period covariate.

### What you'll learn

1. Why the potential-outcomes framework makes "correlation vs. causation" precise instead of hand-wavy, and what the fundamental problem of causal inference actually blocks you from observing.
2. Why randomization — not a large sample, not a particular outcome distribution — is what licenses treating a mean difference as a causal effect.
3. Two genuinely different ways to do valid inference on the same estimator: Neyman's variance formula vs. Fisher's randomization test.
4. How to translate "how big an effect can I detect" into a concrete sample-size number before running an experiment.
5. How a pre-period covariate can shrink an experiment's variance without touching the point estimate (CUPED).

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_potential_outcomes_rct.ipynb`

### Phase 1: The Science Table & Difference-in-Means (~15 min) — `src/_01_diff_in_means.py`

Synthetic data generation, including the full potential-outcomes science
table, is fully scaffolded in `src/datasets.py`. The teaching content is
the estimator itself — implemented on the observed outcome only, exactly
as a real experiment would have to.

1. **TODO #1 — `difference_in_means(y, treatment)`**: implement the
   sample analogue of `E[Y_obs|T=1] - E[Y_obs|T=0]`. ~5-10 lines.

### Phase 2: Neyman's Variance Estimator (~15 min) — `src/_02_neyman_variance.py`

2. **TODO #1 — `neyman_variance(y, treatment)`**: implement
   `s1^2/n1 + s0^2/n0`, the conservative design-based variance of the
   diff-in-means estimator. ~10 lines.

### Phase 3: Fisher's Sharp Null via Randomization Inference (~20 min) — `src/_03_randomization_inference.py`

3. **TODO #1 — `randomization_test(y, treatment, n_perm, rng)`**: implement
   the reshuffle-and-recompute loop under the sharp null, returning the
   observed statistic, the null distribution, and the two-sided p-value.
   ~15-20 lines.

### Phase 4: Power Analysis & Minimum Detectable Effect (~15-20 min) — `src/_04_power_analysis.py`

4. **TODO #1 — `minimum_detectable_effect(n, sigma2, alpha, power)`**:
   implement the normal-approximation MDE formula. ~8-10 lines.
5. **TODO #2 — `power_for_effect(effect, n, sigma2, alpha)`**: implement
   the achieved-power formula (the same algebra, solved in the other
   direction). ~8-10 lines.

### Phase 5: CUPED Variance Reduction (~15-20 min) — `src/_05_cuped.py`

6. **TODO #1 — `cuped_adjust(y, x)`**: implement the CUPED covariate
   adjustment `y - theta*(x - mean(x))` with the variance-minimizing
   `theta = Cov(y, x)/Var(x)`. ~10 lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: the randomization null distribution with
the observed statistic marked, the power curve over sample size, and the
CUPED before/after variance comparison, all on the same simulated
experiment.

### What to look for in the results

- Phase 1: the diff-in-means estimate should land close to the true ATE
  (`tau_true = 2.0`) — any single sample will be off by some sampling
  noise, but not by an order of magnitude.
- Phase 2: the Neyman 95% CI should cover the true ATE; re-running with a
  different seed should cover it roughly 95% of the time.
- Phase 3: the observed statistic should sit out in a tail of its own null
  distribution (low p-value) — the sharp null is false here by
  construction (`tau_true != 0`).
- Phase 4: MDE should shrink as `n` grows, roughly proportional to
  `1/sqrt(n)` — quadrupling `n` roughly halves the detectable effect.
- Phase 5: CUPED's variance reduction should be noticeably larger than
  zero, since `x` and `y` are constructed to be correlated — and the ATE
  point estimate should barely move.

## Motivation

- **AutoScheduler.AI relevance**: any "did this change actually move the
  metric" question that gets A/B tested reduces to this exact toolkit —
  diff-in-means, a defensible variance/CI, and (when traffic is scarce) a
  variance-reduction trick like CUPED to get a usable answer faster.
- **Senior → Staff differentiator**: running `scipy.stats.ttest_ind` is
  common; knowing that randomization — not the t-test's normality
  assumption — is what actually licenses the causal claim, and knowing
  when Neyman's variance is merely conservative rather than exact, is not.
- **Generalises beyond this practice**: the science-table intuition
  (unobserved counterfactuals) and the "randomization identifies the
  independence assumption" argument are the lens every other practice in
  this curriculum reuses for a setting where nobody actually randomized.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_potential_outcomes_rct.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_diff_in_means` | Sanity-check the diff-in-means ATE estimator against the true ATE. |
| **Phase 2** | `uv run python -m src._02_neyman_variance` | Compute Neyman's variance and a 95% CI for the ATE. |
| **Phase 3** | `uv run python -m src._03_randomization_inference` | Run the Fisher randomization test and report the p-value. |
| **Phase 4** | `uv run python -m src._04_power_analysis` | Compute the MDE across sample sizes and achieved power at n=250. |
| **Phase 5** | `uv run python -m src._05_cuped` | Compute the CUPED-adjusted ATE and its variance reduction. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
