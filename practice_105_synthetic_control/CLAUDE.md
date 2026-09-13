# Practice 105 — Synthetic Control & Synthetic DiD

## Technologies

- **pysyncon** — reference implementation of Abadie, Diamond & Hainmueller's synthetic control method, used to validate the from-scratch weight optimizer.
- **SciPy** (`scipy.optimize.minimize`, `trust-constr`) — the constrained quadratic program that fits donor weights on the probability simplex.
- **xy** — plotting (treated-vs-synthetic, gap, placebo spaghetti).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The problem: one treated unit, no natural control

Difference-in-differences, regression discontinuity, and panel fixed effects
all lean on *some* untreated comparison group observed alongside the
treated one. Synthetic control (Abadie & Gardeazabal 2003; Abadie, Diamond
& Hainmueller 2010) is for the case where that comparison group doesn't
exist: exactly **one** unit (a state, a country, a firm) is treated, and no
single untreated unit looks enough like it, pre-treatment, to serve as a
credible counterfactual. Instead of picking one imperfect comparison unit,
the method **builds** one: a weighted average of several untreated "donor"
units, with the weights chosen so the weighted average tracks the treated
unit's own pre-treatment outcome path as closely as possible.

### Building the synthetic control: an OR framing

Let `y1_pre` be the treated unit's pre-treatment outcome vector (`T0`
periods) and `Y0_pre` be the donor pool's outcomes over the same periods,
one column per donor (`T0 x J`). The weights are the solution to

```
minimize_w   || y1_pre - Y0_pre @ w ||^2
subject to   w_j >= 0  for all j
             sum_j w_j = 1
```

This is a **quadratic program (QP) over the probability simplex** — the
same feasible region that shows up in portfolio optimization (long-only
weights summing to 1) and finite-mixture models. It is worth naming
explicitly: an OR background makes the two structural payoffs of the
simplex constraint immediate rather than mysterious.

1. **No extrapolation.** A convex combination of donors can never predict,
   at any single time point, a value outside the range the donors
   themselves span at that time point. An unconstrained (or negative-
   weight) least-squares fit has no such guarantee — it can and will
   extrapolate outside the data whenever that improves the fit, which is
   exactly the kind of "fit" that shouldn't be trusted as a counterfactual.
2. **Interpretable weights.** Because weights are non-negative and sum to
   one, they read directly as a mixture: "this synthetic California is 60%
   Nevada, 20% Montana, 20% a blend of the rest" is a sentence you can say
   out loud to a non-technical stakeholder. An OLS fit's coefficients admit
   no such reading (they aren't a mixture; they can be negative or exceed
   one).

The full Abadie et al. method **nests** this problem inside an outer one:
this practice's `w`-problem sits inside an outer optimization over a second
weight vector `V`, which decides how much each *predictor* (not each unit)
matters when judging pre-treatment fit — a nested-optimization structure
(inner: unit weights; outer: predictor weights) that is itself a
recognizable OR pattern (bilevel optimization). See "What this practice
simplifies" below for exactly what this practice keeps and skips.

### What this practice simplifies

The published method matches on a handful of *aggregated* pre-treatment
predictors (average income, average retail price, ...) plus the full
pre-treatment outcome path, with the outer `V`-problem choosing how much
weight each predictor category gets. This practice's exercises implement
the **outcome-path-only** variant: match purely on the treated unit's
entire pre-treatment trajectory, giving every pre-period equal weight (the
inner `w`-problem above, with the outer `V`-problem fixed rather than
optimized). This is a real, commonly used simplification in the
literature — not a corner cut for convenience — and it is what lets the
hand-rolled QP stay a single, cleanly stated simplex optimization instead
of a nested bilevel one, while still validating meaningfully against
`pysyncon` (configured the same way; see `src/_02_synthetic_weights.py`).

### The donor-pool selection problem

The weights can only be as good as the units they're allowed to choose
from. A donor that itself received a similar treatment during the sample
window, or that is structurally too different from the treated unit to
ever be a plausible ingredient, will either bias the fit (if included and
contaminated) or simply go unused (if included but irrelevant — its weight
converges to zero, which is a *correct* outcome, not a failure). This
practice's data already reflects that decision: the classic Prop 99 dataset
excludes several US states that ran their own large-scale tobacco-control
programs during 1970-2000 (see `src/datasets.py`). Choosing the donor pool
is a substantive, pre-registered-ideally judgment call — the optimizer
cannot rescue a badly chosen pool by down-weighting a contaminated donor
towards (but never exactly) zero.

### Placebo / permutation inference

There is no textbook standard error for an estimate built from one treated
unit — there's no sampling distribution to invoke. Abadie, Diamond &
Hainmueller's answer is a **permutation test**: re-run the identical
method on every donor in turn, pretending each one was treated when it
wasn't, and see where the true unit's post/pre-treatment RMSPE ratio falls
against that donor-generated null distribution. If California's ratio
isn't unusual compared to what an untreated donor can produce just from
noise, the "effect" isn't distinguishable from what the method would
manufacture out of nothing.

### Synthetic DiD: reconciling synthetic control with the DiD tradition

Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg,
Imbens & Wager 2021) keeps synthetic control's simplex-constrained *unit*
weights, but changes two things: it adds a second, analogous set of *time*
weights (which pre-treatment periods matter most, mirroring the unit
side), and it estimates the effect as a **difference-in-differences** on
top of the weighted average — matching *trends*, the way DiD does, rather
than matching *levels* the way plain synthetic control does. The result
inherits DiD's classical two-way-fixed-effects estimating equation, just
computed on a reweighted panel instead of a simple average — which is why
it can be read as "synthetic control's donor-selection machinery, bolted
onto DiD's estimator," reconciling the two traditions rather than
replacing either.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Treated unit** | The single unit that received the intervention (California). |
| **Donor pool** | The candidate untreated units the synthetic control can be built from. |
| **Simplex constraint** | `w_j >= 0`, `sum_j w_j = 1` — weights form a convex combination. |
| **RMSPE** | Root-mean-squared prediction error; the fit-quality metric, in outcome units. |
| **Gap** | Treated minus synthetic, at every time period — the estimated dynamic effect. |
| **Placebo test** | Re-running the method on every donor as if it were treated, to build a null distribution. |
| **Permutation p-value** | The true unit's rank (by post/pre RMSPE ratio) among the placebo distribution. |
| **Synthetic DiD** | Synthetic control's unit weights + DiD's trend-matching estimator + time weights. |

### Where This Fits

Synthetic control sits alongside DiD and RDD as one of the field's core
"there's no randomized experiment, so make the counterfactual credible some
other way" tools — but it is the one built specifically for **N=1 treated
unit**, where DiD's parallel-trends assumption has no natural comparison
group to test it against. Alternatives include simple case studies (no
formal counterfactual at all — the problem this method exists to fix),
matching methods (pick one similar unit rather than blend many — usually a
worse pre-treatment fit), and generalized synthetic control / interactive
fixed-effects models (Xu 2017), which handle **many** treated units with
staggered timing, a setting this single-treated-unit method doesn't cover.

### References

- Abadie & Gardeazabal, 2003, "The Economic Costs of Conflict: A Case Study of the Basque Country": <https://doi.org/10.1257/000282803321455188>
- Abadie, Diamond & Hainmueller, 2010, "Synthetic Control Methods for Comparative Case Studies": <https://doi.org/10.1198/jasa.2009.ap08746>
- Abadie, 2021, "Using Synthetic Controls: Feasible and Bias-Reduced Uses": <https://doi.org/10.1257/jel.20191450>
- Arkhangelsky, Athey, Hirshberg, Imbens & Wager, 2021, "Synthetic Difference-in-Differences": <https://doi.org/10.1257/aer.20190159>
- `pysyncon` documentation: <https://sdfordham.github.io/pysyncon/>

## Description

Build a synthetic California as a simplex-weighted average of 38 other US
states, matched on its pre-1989 per-capita cigarette-sales trajectory, then
use the gap between California and its synthetic counterpart to estimate
the effect of Proposition 99. Validate the hand-rolled weight optimizer
against `pysyncon`, then run a placebo/permutation test across every donor
to judge whether the estimated effect is unusual or just estimation noise.

### What you'll learn

1. Why the weights in synthetic control are constrained to the simplex, and what that constraint buys you (no extrapolation, interpretable weights).
2. How to frame and solve synthetic control's weight-fitting step as a constrained quadratic program.
3. Why the donor pool matters as much as the optimization — a badly chosen pool can't be fixed by better weights.
4. How to run and interpret a placebo/permutation test when there's no natural standard error.
5. How synthetic DiD reconciles synthetic control's unit-weighting idea with the DiD tradition.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_synthetic_control.ipynb`

### Phase 1: Pre-treatment Fit Loss (~10 min) — `src/_01_fit_loss.py`

Data loading (`src/datasets.py`) is fully scaffolded. The teaching content
is the fit-quality metric everything else in this practice is built on.

1. **TODO #1 — `rmspe(actual, synthetic)`**: root-mean-squared prediction
   error between two trajectories. ~5-8 lines.

### Phase 2: Simplex-Constrained Weight Optimization (~25-30 min) — `src/_02_synthetic_weights.py`

The core estimator. `fit_synthetic_control` (orchestration) and
`compare_to_pysyncon` (validation) are fully scaffolded.

2. **TODO #1 — `solve_synthetic_weights(Y0_pre, y1_pre)`**: fit
   simplex-constrained donor weights via `scipy.optimize.minimize`
   (`trust-constr`, needed here since donors typically outnumber
   pre-treatment periods, making the loss surface rank-deficient — see the
   TODO comment for why `SLSQP` isn't reliable on this shape). ~15-20 lines.

### Phase 3: The Gap and Post-Treatment Effect (~10-15 min) — `src/_03_gap_effect.py`

3. **TODO #1 — `compute_gap(data, y_synthetic)`**: the treated-minus-
   synthetic gap series and its post-treatment average. ~8-12 lines.

### Phase 4: Placebo/Permutation Inference (~20-25 min) — `src/_04_placebo_inference.py`

The leave-one-out placebo loop (re-fit the synthetic control with every
donor as the placebo-treated unit) is fully scaffolded — it reuses Phases
1-2 directly.

4. **TODO #1 — `permutation_p_value(true_ratio, placebo_ratios)`**:
   rank-based permutation p-value from a list of post/pre RMSPE ratios.
   ~5-8 lines.

### Phase 5: End-to-End Run (no TODO)

Run the notebook's final cells: the treated-vs-synthetic plot, the gap
plot, and the placebo spaghetti plot, plus a discussion of synthetic DiD as
the reconciliation with the DiD tradition.

### What to look for in the results

- Phase 2: the optimized synthetic control's pre-treatment RMSPE should be
  clearly lower than the naive equal-weighted donor average's — that gap
  *is* what the optimization bought you. `compare_to_pysyncon` should land
  in the same neighborhood, not necessarily the same weights.
- Phase 3: the gap should hover near zero before 1989 (that's the fit
  check) and turn negative afterward — Prop 99 is associated with *lower*
  cigarette sales than the synthetic counterfactual.
- Phase 4: California's post/pre RMSPE ratio should rank as one of the most
  extreme among the ~39 units (true + placebos) — a low permutation
  p-value is the formal version of "the effect looks real, not like noise
  a random donor could have produced."

## Motivation

- **AutoScheduler.AI relevance**: "we rolled out change X in one
  warehouse/region and nowhere else — did it work?" is exactly the N=1
  treated-unit setting synthetic control was built for, and it is common
  in operational rollouts where a full randomized rollout isn't feasible.
- **Senior → Staff differentiator**: knowing DiD and RDD covers the cases
  with a natural comparison group; knowing what to do when there isn't one
  is the less commonly held half of the causal-inference toolkit.
- **Generalises beyond econometrics**: simplex-constrained weight fitting
  (a QP with the same feasible region as portfolio optimization) reappears
  anywhere a weighted blend of reference examples needs to stay
  interpretable and non-extrapolating — ensembling, benchmark-matching, and
  index replication all lean on the same constraint set.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_synthetic_control.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_fit_loss` | Sanity-check RMSPE against a naive donor-average baseline. |
| **Phase 2** | `uv run python -m src._02_synthetic_weights` | Fit synthetic-control weights and cross-check against `pysyncon`. |
| **Phase 3** | `uv run python -m src._03_gap_effect` | Compute the gap and the post-treatment average effect. |
| **Phase 4** | `uv run python -m src._04_placebo_inference` | Run the full placebo test and report the permutation p-value. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

### Dataset provenance

`data/prop99.csv` is the classic Abadie, Diamond & Hainmueller (2010)
California Proposition 99 panel — annual state-level per-capita cigarette
sales (`cigsale`) plus several covariates (`lnincome`, `beer`, `retprice`,
`age15to24`), 39 US states (California + 38 donors, already restricted to
states without a large tobacco-control program of their own), 1970-2000.
It is **not** bundled in `pysyncon` (unlike its R counterpart, the `Synth`
package, which ships `data(synth.smoking)`), so it is vendored here as a
CSV rather than loaded from a package, per this curriculum's dataset
policy. Sourced from the public replication repository
[Matt2371/Proposition99](https://github.com/Matt2371/Proposition99)
(`Data/data.csv`), which itself republishes the original Abadie et al.
study data — this is real data, not a synthetic substitute. Some
covariate columns (`lnincome`, `beer`) have missing values in the years
the original study didn't collect them; this practice only uses `cigsale`
(see "What this practice simplifies" above), so those gaps don't affect
any exercise.

## State

`not-started`
