# Practice 101 — Instrumental Variables & 2SLS

## Technologies

- **linearmodels** — reference `IV2SLS` implementation used to validate the from-scratch estimator, and the standard tool applied econometrics reaches for.
- **xy** — plotting (first-stage fit, weak-instrument bias/coverage, estimate-comparison plot).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The Problem IV Solves

OLS estimates `y = alpha + beta*D + u` correctly only if `D` is uncorrelated
with `u` (Gauss-Markov's strict exogeneity, A2 — see practice 097). Whenever an
unobserved factor drives both the regressor of interest `D` and the outcome
`y` — ability and schooling, motivation and job-search effort, local policy
and local outcomes — that correlation is nonzero and `beta_hat` is biased,
often substantially. **Instrumental variables** fixes this without observing
the confounder at all, by finding a variable `Z` that moves `D` but has no
direct effect on `y`. Two conditions make `Z` a valid instrument:

- **Relevance**: `Cov(Z, D) != 0` — the instrument actually predicts the
  regressor. This *is* testable (Phase 3's first-stage F statistic).
- **Exclusion restriction**: `Z` affects `y` only through `D`, never directly,
  and is uncorrelated with `u`. This is **not** testable from data alone — it
  is an assumption defended by institutional knowledge or a design argument
  (e.g. a policy that shifted `D` for reasons unrelated to `y`'s other
  determinants).

### Two-Stage Least Squares (2SLS)

With one endogenous regressor and one excluded instrument (the
"just-identified" case this practice uses throughout), 2SLS is:

1. **First stage**: regress `D` on everything exogenous, including `Z`. Keep
   the fitted values `D_hat` — the part of `D`'s variation that is explained
   by exogenous sources, purged of any correlation with `u`.
2. **Second stage**: regress `y` on `D_hat` (in place of `D`). The coefficient
   on `D_hat` is the 2SLS estimate.

That two-regression recipe is algebraically identical to a single projection:
with `X = [exog, D]` and `W = [exog, Z]`,

```
beta_2sls = (X' P_W X)^-1 X' P_W y,     P_W = W (W'W)^-1 W'
```

`P_W` projects onto the column space spanned by every exogenous variable and
instrument. Seeing 2SLS this way is what generalizes cleanly to
over-identified models (more instruments than endogenous regressors) and to
GMM: 2SLS is GMM with weighting matrix `(W'W)^-1`. This practice implements
both views (Phase 1: two regressions; Phase 2: one projection) and checks they
agree to numerical precision, then validates both against `linearmodels.IV2SLS`.

### The First-Stage F Statistic and Weak Instruments

2SLS is consistent for any nonzero relevance in the large-sample limit, but at
finite `n` a *weak* instrument (first-stage coefficient near zero) causes two
well-documented failures (Staiger & Stock, 1997; Stock & Yogo, 2005):

- **Bias toward OLS**: 2SLS's finite-sample bias grows as the instrument
  weakens, and its probability limit as relevance goes to exactly zero is
  *the OLS estimand* — the same biased number 2SLS exists to avoid.
- **Wrong coverage**: standard-error-based confidence intervals cover the true
  effect far less often than their nominal rate (this practice's Phase 3
  simulation shows both failures directly).

The first-stage F statistic — an F test of joint significance of the excluded
instrument(s) in the first-stage regression — is the standard diagnostic. The
widely cited (and widely debated) **rule of thumb is F < 10 signals a weak
instrument** (Stock & Yogo, 2005); modern practice (Lee et al., 2022) argues
the conventional 5% critical value is itself too permissive and recommends
much higher thresholds, but F < 10 remains the number every applied paper
reports.

### LATE vs. ATE

Every result above assumes a single, homogeneous causal effect — 2SLS
estimates *one number*. Real effects vary across individuals. With a **binary**
instrument and a binary treatment, and under **monotonicity** (no "defiers" —
nobody does the opposite of what the instrument nudges them toward), IV
identifies the **Local Average Treatment Effect (LATE)**: the average effect
among **compliers** — the subpopulation whose treatment status is actually
moved by the instrument (Imbens & Angrist, 1994; Angrist, Imbens & Rubin,
1996). It says nothing about **always-takers** (treated regardless of the
instrument) or **never-takers** (untreated regardless) — their effects are
never revealed by instrument-driven variation, however large or small they
truly are. The **Average Treatment Effect (ATE)**, by contrast, averages over
the *entire* population, compliers included. LATE equals ATE only when the
effect is homogeneous (the case in Phases 1-3) or compliers happen to be
representative of the whole population — neither holds in general, and this
practice's Phase 4 dataset is built so they visibly differ.

For a binary instrument, 2SLS collapses to the **Wald estimator**: the ratio
of the instrument's reduced-form effect on `y` to its first-stage effect on
`D`. This practice implements it as a plain difference-in-means ratio and
verifies it is numerically identical to Phase 2's projection-based 2SLS.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Endogeneity** | `Cov(D, u) != 0` — the regressor is correlated with the structural error; OLS is biased. |
| **Instrument (Z)** | A variable satisfying relevance (`Cov(Z, D) != 0`) and the exclusion restriction (`Z` affects `y` only through `D`). |
| **Two-stage least squares (2SLS)** | Replace `D` with its exogenous-only-explained part `D_hat`, then regress `y` on `D_hat`. |
| **First-stage F statistic** | Test of joint significance of the excluded instrument(s) predicting `D`; F < 10 is the classic weak-instrument warning. |
| **Weak instrument** | A first-stage coefficient near zero; causes 2SLS to be biased toward OLS and its CIs to under-cover. |
| **Monotonicity** | No "defiers" — the instrument never pushes anyone's treatment status in the opposite direction from everyone else's. |
| **Complier / always-taker / never-taker** | The three types a binary instrument + binary treatment sorts individuals into under monotonicity. |
| **LATE** | The average causal effect among compliers only — what IV with a binary instrument actually identifies. |
| **ATE** | The average causal effect across the entire population, compliers included. |
| **Wald estimator** | `(E[y|z=1]-E[y|z=0]) / (E[d|z=1]-E[d|z=0])` — 2SLS's closed form for a binary instrument and treatment. |

### Where This Fits

IV is the standard fix when an experiment is infeasible but a plausibly
exogenous source of variation in the treatment exists — it is the technique
behind natural-experiment papers using lottery admissions, draft lotteries, or
policy discontinuities as instruments. Alternatives include: **RCTs**
(directly randomize `D` — the gold standard when feasible, sidestepping the
exclusion restriction entirely), **RDD** (practice 104 — exploits a
discontinuous rule around a threshold, a *local* natural experiment rather
than an instrument), and **panel fixed effects** (practices 102/103 — controls
for *time-invariant* confounders instead of finding an instrument, a different
identification strategy for a different threat). IV's distinctive cost is the
exclusion restriction: unlike relevance, it cannot be tested, only argued for
— which is why applied IV papers spend most of their space defending it, not
running the two regressions.

### References

- Angrist & Pischke, *Mostly Harmless Econometrics*, ch. 4 (Instrumental Variables in Theory and Practice): <https://www.mostlyharmlesseconometrics.com/>
- Imbens & Angrist, 1994, "Identification and Estimation of Local Average Treatment Effects": <https://doi.org/10.2307/2951620>
- Angrist, Imbens & Rubin, 1996, "Identification of Causal Effects Using Instrumental Variables": <https://doi.org/10.2307/2291629>
- Staiger & Stock, 1997, "Instrumental Variables Regression with Weak Instruments": <https://doi.org/10.2307/2171753>
- Stock & Yogo, 2005, "Testing for Weak Instruments in Linear IV Regression": <https://doi.org/10.1017/CBO9780511614491.006>
- `linearmodels.IV2SLS` docs: <https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.model.IV2SLS.html>

## Description

Implement 2SLS from scratch two ways — as two literal OLS regressions, then
as a single projection onto the instrument space — and validate both against
`linearmodels.IV2SLS`. Diagnose instrument strength with the first-stage F
statistic and watch 2SLS's bias converge toward OLS's as the instrument
weakens, via a Monte Carlo sweep. Switch to a binary encouragement design with
known complier types and implement the Wald estimator, showing IV recovers the
LATE — not the ATE — and confirming the Wald ratio is the same number as 2SLS.

### What you'll learn

1. Why 2SLS needs both relevance and an (untestable) exclusion restriction, and what each buys you.
2. 2SLS as two literal regressions, and as one projection matrix formula — and why they're the same estimator.
3. How to diagnose a weak instrument with the first-stage F statistic, and what actually breaks when it's low.
4. Why IV with a binary instrument identifies the LATE, not the ATE, and what "complier" means concretely.
5. That the Wald estimator and 2SLS are the same number under a binary instrument, not two different techniques.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_instrumental_variables.ipynb`

### Phase 1: Two-Stage Least Squares, as Two Literal Regressions (~15-20 min) — `src/_01_tsls_two_stage.py`

Synthetic data generation (`src/datasets.py`) and the naive-OLS comparison are
fully scaffolded. The teaching content is the estimator itself.

1. **TODO #1 — `tsls_two_stage(exog, endog, instruments, y)`**: implement 2SLS
   as two explicit OLS regressions (first stage purifies `D`, second stage
   regresses `y` on `D_hat`). ~15-20 lines.

### Phase 2: 2SLS as a Single Projection (~15-20 min) — `src/_02_tsls_projection.py`

The `linearmodels.IV2SLS` comparison harness is fully scaffolded. The teaching
content is the projection-matrix formula.

2. **TODO #1 — `tsls_projection(exog, endog, instruments, y)`**: implement
   `beta = (X'P_W X)^-1 X'P_W y` directly, without calling Phase 1's estimator.
   ~15-20 lines.

### Phase 3: First-Stage Strength & the Weak-Instrument Problem (~15-20 min) — `src/_03_first_stage_strength.py`

The weak-instrument Monte Carlo sweep (`weak_instrument_simulation`) is fully
scaffolded — it calls your F statistic and Phase 2's 2SLS estimator in a loop.
The teaching content is the F statistic itself.

3. **TODO #1 — `first_stage_f_stat(exog, endog, instruments)`**: implement the
   restricted-vs-unrestricted F test for joint significance of the excluded
   instrument(s). ~15-20 lines.

### Phase 4: LATE vs. ATE — the Wald Estimator (~15-20 min) — `src/_04_late_wald.py`

The binary encouragement-design data generation (with known complier types,
true ATE, and true LATE) and the bootstrap standard-error helper are fully
scaffolded. The teaching content is the ratio estimator itself.

4. **TODO #1 — `wald_estimator(y, d, z)`**: implement the difference-in-means
   ratio for a binary instrument and binary treatment. ~5-10 lines.

### Phase 5: End-to-End Run (no TODO)

Run the notebook's final cells: a coefficient-style plot comparing OLS, the
Wald/2SLS estimate, the true ATE, and the true LATE on the same binary
dataset.

### What to look for in the results

- Phase 1/2: 2SLS should land close to the true `beta`, visibly closer than
  naive OLS, which should be biased by a wide margin given how strongly the
  confounder enters both equations.
- Phase 2: the two-stage estimate, the projection estimate, and
  `linearmodels.IV2SLS` should all agree to within `1e-6` or tighter — if they
  don't, one of the two hand-rolled routes has a bug.
- Phase 3: as the instrument weakens (F drops well below 10), 2SLS's bias
  should visibly grow and converge toward OLS's bias; CI coverage should fall
  well below the nominal 95%.
- Phase 4: the Wald/2SLS estimate should land close to the true LATE, and
  clearly *away* from the true ATE — the always-takers' larger effect never
  shows up, by construction.
- Phase 4: recasting the binary data as IV arrays and calling
  `tsls_projection` should reproduce the Wald estimate exactly — same
  estimator, different formula.

## Motivation

- **AutoScheduler.AI relevance**: any "does this scheduling knob actually move
  the outcome, or are both driven by something else we don't observe" question
  is exactly the endogeneity problem IV exists for — the difference between
  "the parameter changed and the metric moved" and "the parameter caused the
  metric to move."
- **Senior → Staff differentiator**: reaching for `IV2SLS` is common; knowing
  which population parameter it actually estimates (LATE, not ATE, the moment
  effects are heterogeneous), and being able to defend an exclusion
  restriction rather than just assert one, is not.
- **Generalises beyond econometrics**: "purify a variable by regressing out
  what's endogenous about it, then use the purified version" is the same
  logic behind control-function approaches to endogenous treatment
  assignment, propensity-score residualization, and orthogonalized ML
  estimators (double/debiased ML).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_instrumental_variables.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_tsls_two_stage` | Sanity-check two-stage 2SLS against naive OLS and the truth. |
| **Phase 2** | `uv run python -m src._02_tsls_projection` | Validate two-stage, single-projection, and `linearmodels.IV2SLS` against each other. |
| **Phase 3** | `uv run python -m src._03_first_stage_strength` | Compute the first-stage F statistic on a strong and a weak instrument. |
| **Phase 4** | `uv run python -m src._04_late_wald` | Compute the Wald/LATE estimate against the true ATE and LATE. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
