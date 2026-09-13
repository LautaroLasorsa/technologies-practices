# Practice 103 — Difference-in-Differences & the Staggered Adoption Problem

## Technologies

- **pyfixest** — fast fixed-effects regression (the modern Python `reghdfe`-equivalent), used for the naive two-way-fixed-effects (TWFE) regression and the event-study specification.
- **differences** — an independent Python implementation of the Callaway-Sant'Anna group-time ATT estimator, used only as an external cross-check on this practice's hand-rolled version.
- **xy** — plotting (event-study plot, Goodman-Bacon decomposition scatter, method-comparison plot).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### The Canonical 2x2 and Parallel Trends

Difference-in-differences estimates a treatment effect by comparing the *change* in an
outcome for a treated group against the change for an untreated (control) group over the
same period — the "double difference." Its identifying assumption is **parallel trends**:
absent treatment, the treated group's outcome would have moved in parallel with the
control group's. Under that assumption, the control group's observed change is a valid
stand-in for the treated group's unobserved counterfactual change, and the double
difference isolates the treatment effect. The standard way to *test* the plausibility of
parallel trends is the **event-study specification**: replace the single post-treatment
dummy with one dummy per relative-time bin (periods before/after treatment), and check
that the *pre*-period coefficients are close to zero — if the treated group was already
diverging from the control group before treatment started, parallel trends is suspect.

### Why Staggered Adoption Breaks Two-Way Fixed Effects

The 2x2 case generalizes badly the moment units are treated at *different* times
(staggered adoption) and effects are **dynamic** — they change with how long a unit has
been treated. Goodman-Bacon (2021) proves that the naive two-way-fixed-effects (TWFE)
regression `y ~ treated | unit + time` — the regression almost every applied paper ran on
staggered panels before ~2020 — is not one clean 2x2 DiD. It is a **weighted average of
every pairwise 2x2 comparison** between cohorts, including comparisons where an
**already-treated cohort serves as the control group** for a later-treated one. That
comparison is contaminated: the "control" cohort's own change over the window bundles in
whatever its own treatment effect did between those two periods. If effects grow over
time (as in this practice's synthetic DGP), that growth gets subtracted from the later
cohort's estimated effect as if it were a generic time trend — and the resulting 2x2 can
be badly wrong, including strongly negative, even when every true effect in the data is
positive. De Chaisemartin & D'Haultfœuille (2020) formalize a related, sharper result:
under treatment-effect heterogeneity, the TWFE coefficient can be written as a weighted
sum of the underlying (unit, time) treatment effects with **some strictly negative
weights** — meaning TWFE can be non-monotonic in the treatment effects it is supposedly
averaging. Both papers describe the same underlying pathology (already-treated units
contaminating later comparisons) from different angles: Goodman-Bacon decomposes the
*regression coefficient* into 2x2 DiDs (whose own weights are non-negative, but whose
*estimates* can be pathological); de Chaisemartin & D'Haultfœuille decompose it into
*treatment effects* (whose weights can be literally negative).

### The Repair: Callaway-Sant'Anna Group-Time ATT

Callaway & Sant'Anna (2021) replace the single pooled regression with **one clean 2x2 per
(cohort, period) cell**: `ATT(g, t)`, the average effect on the cohort first treated at
time `g`, observed at calendar time `t`, always compared against a group that is
genuinely untreated at both endpoints (never-treated, or "not-yet-treated" in a fuller
version of the estimator). Because every cell is its own self-contained 2x2, no
already-treated unit can ever sneak in as a contaminated control — the Goodman-Bacon
failure mode is structurally impossible. The `ATT(g, t)` grid is then **aggregated**
along whichever axis is useful: by relative event time (an event-study path, average
effect at "periods since treatment") or into one overall number (a cohort-size-weighted
average across every post-treatment cell).

### Why This Practice Hand-Rolls the Estimator

Python's staggered-DiD tooling is real but immature relative to R's `did` /
`didimputation` / `DIDmultiplegt` packages, which remain the field's reference
implementations. The `differences` package (used here only as a cross-check) implements
Callaway-Sant'Anna reasonably, but is newer and far less battle-tested than its R
counterparts. Implementing `ATT(g, t)` by hand on top of Phase 1's 2x2 estimator is
therefore the point of this practice, not a workaround for a missing library — it is also
the only way to *see* why the estimator is immune to Phase 3's failure, instead of trusting
a library's claim that it is.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Parallel trends** | The identifying assumption of DiD: absent treatment, treated and control groups' outcomes would move in parallel. |
| **Event-study specification** | A regression with one dummy per relative-time bin instead of a single post dummy; used to test pre-trends. |
| **Two-way fixed effects (TWFE)** | `y ~ treated \| unit + time` — the naive staggered-DiD regression that Goodman-Bacon shows is a weighted average of many 2x2s. |
| **Forbidden comparison** | A 2x2 DiD where the "control" group is already treated, contaminating the estimate with the control's own treatment-effect path. |
| **Goodman-Bacon decomposition** | The theorem that TWFE equals a weighted average of every pairwise cohort 2x2 DiD (Goodman-Bacon, 2021). |
| **ATT(g, t)** | Callaway-Sant'Anna's group-time average treatment effect on the treated: the effect on cohort `g` observed at time `t`. |
| **Event-time aggregation** | Averaging `ATT(g, t)` across cohorts within the same relative event time `t - g`. |

### Where This Fits

DiD-with-staggered-timing is arguably the most consequential correction in applied
econometrics of the last decade: a huge share of published policy-evaluation papers
(minimum wage, healthcare mandates, corporate-governance rules) used naive TWFE on
staggered rollouts, and the 2020-2021 literature (Goodman-Bacon; de Chaisemartin &
D'Haultfœuille; Callaway & Sant'Anna; Sun & Abraham; Borusyak et al.) showed many of those
estimates cannot be trusted at face value. Alternatives to the Callaway-Sant'Anna approach
include Sun & Abraham's interaction-weighted estimator (similar idea, different
aggregation), Borusyak et al.'s imputation estimator (impute untreated potential outcomes
directly), and de Chaisemartin & D'Haultfœuille's estimator (handles a wider class of
"switchers," including units that revert to untreated). All of them share the same
diagnosis this practice teaches: **never let an already-treated unit be any other unit's
control.**

### References

- Goodman-Bacon, 2021, "Difference-in-Differences with Variation in Treatment Timing": <https://doi.org/10.1016/j.jeconom.2021.03.014>
- Callaway & Sant'Anna, 2021, "Difference-in-Differences with Multiple Time Periods": <https://doi.org/10.1016/j.jeconom.2020.12.001>
- de Chaisemartin & D'Haultfœuille, 2020, "Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects": <https://doi.org/10.1257/aer.20181169>
- Baker, Larcker & Wang, 2022, "How Much Should We Trust Staggered Difference-in-Differences Estimates?": <https://doi.org/10.1016/j.jfineco.2022.01.004>
- `differences` package docs: <https://bernardodionisi.github.io/differences/>

## Description

Build the canonical 2x2 DiD and its event-study extension, then simulate a staggered-
adoption panel with known, heterogeneous, dynamic treatment effects and watch a naive
TWFE regression return an estimate outside the range of every true effect — with the
wrong sign. Decompose that failure with a hand-rolled Goodman-Bacon decomposition, then
repair it by hand-rolling the Callaway-Sant'Anna group-time ATT(g,t) estimator on top of
`pyfixest`/the practice's own 2x2 primitive, aggregating it into an event-study path and
an overall ATT, and cross-checking the result against the `differences` package.

### What you'll learn

1. What the canonical 2x2 DiD estimates, and how the event-study specification tests
   parallel trends via pre-trend coefficients.
2. Why staggered adoption plus dynamic treatment effects breaks two-way fixed effects —
   concretely, via a DGP where TWFE returns a negative estimate despite every true effect
   being positive.
3. How to decompose a TWFE coefficient into the pairwise 2x2 comparisons behind it
   (Goodman-Bacon), and see exactly which comparisons are the pathological ones.
4. How Callaway-Sant'Anna's group-time ATT(g,t) sidesteps the failure by construction,
   and how to aggregate it into an event-study path and a single overall effect.
5. Where Python's staggered-DiD tooling stands relative to R's, and why implementing the
   estimator by hand is worth doing even when a package exists.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_did_staggered.ipynb`

### Phase 1: The Canonical 2x2 DiD (~15 min) — `src/_01_two_by_two_did.py`

Data loading is fully scaffolded. The teaching content is the double-difference
estimator every later phase reuses.

1. **TODO #1 — `did_2x2(df)`**: implement the 2x2 double-difference estimator by hand —
   four cell means, one subtraction of differences, plus its standard error. ~15-20 lines.

### Phase 2: The Event-Study Specification (~15-20 min) — `src/_02_event_study.py`

Fitting the event-study regression (`pyfixest`, one dummy per relative-time bin) is fully
scaffolded. The teaching content is turning its raw output into something usable.

2. **TODO #1 — `event_study_coefficients(model)`**: extract a tidy, sorted table of
   (relative time, estimate, se, CI) from the fitted model, splicing back in the omitted
   reference period. ~15-20 lines.

### Phase 3: TWFE Under Staggered Timing & the Goodman-Bacon Decomposition (~25-30 min) — `src/_03_twfe_and_bacon.py`

The staggered-adoption DGP (`src/datasets.py`) and the naive TWFE fit are fully
scaffolded. The teaching content is decomposing that one coefficient into the pairwise
comparisons behind it.

3. **TODO #1 — `goodman_bacon_decompose(df)`**: for every pair of cohorts, build the
   pairwise 2x2 sub-frame, call Phase 1's `did_2x2` on it, label the comparison "clean" or
   "forbidden," and weight it by sample size. ~20-25 lines.

### Phase 4: The Repair — Callaway-Sant'Anna ATT(g,t) (~20-25 min) — `src/_04_att_gt.py`

Looping over every (cohort, period) cell is fully scaffolded. The teaching content is one
cell's estimator.

4. **TODO #1 — `att_gt(df, g, t, control_group)`**: compute one group-time ATT — cohort
   `g`'s DiD from its last pre-treatment period to `t`, against the never-treated group —
   by building the right sub-frame and calling Phase 1's `did_2x2`. ~15-20 lines.

### Phase 5: Aggregating ATT(g,t) (~15-20 min) — `src/_05_aggregate_att.py`

5. **TODO #1 — `aggregate_event_study(att_gt_df)`**: cohort-size-weighted average of
   `ATT(g,t)` within each relative event time. ~10-15 lines.
6. **TODO #2 — `aggregate_overall_att(att_gt_df)`**: cohort-size-weighted average across
   every (g,t) cell, into one overall ATT with a standard error. ~10 lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: the hand-rolled overall ATT is cross-checked against the
`differences` package's independent Callaway-Sant'Anna implementation on the exact same
panel, and both are plotted alongside the naive TWFE estimate and the true overall ATT.

### What to look for in the results

- Phase 1's hand-computed double difference should match the `treat:post` regression
  coefficient to numerical precision — same estimator, two ways of computing it.
- Phase 2: pre-period event-study coefficients should be small and centered near zero —
  parallel trends holds by construction in this DGP.
- Phase 3: the naive TWFE coefficient should fall **outside** `[true_min, true_max]` —
  in this practice's calibration, it is even negative-signed. The decomposition's
  "forbidden" rows should be visibly the negative outliers; "clean" rows should cluster
  near the true effects.
- Phase 4: every `ATT(g,t)` should closely match its ground-truth value — the whole
  point of estimating cell-by-cell instead of pooling.
- Phase 5-6: the hand-rolled overall ATT should land close to both the true overall ATT
  and the `differences` package's independent estimate — three numbers that agree with
  each other while the naive TWFE estimate is the outlier.

## Motivation

- **AutoScheduler.AI relevance**: any "we rolled out a scheduling policy change to
  different sites at different times, did it help" question is a staggered-DiD question —
  getting this wrong (naive TWFE) is one of the most common real-world analysis mistakes,
  and knowing to check for it is a genuine differentiator.
- **Senior → Staff differentiator**: recognizing "we have staggered rollout data" as a
  red flag, and knowing which repair to reach for (and why), is rarer than knowing DiD
  exists at all.
- **Generalises beyond econometrics**: "don't let an already-treated unit contaminate
  another unit's control group" is a special case of a much more general lesson about
  contamination in any before/after comparison with rolling deployment — A/B tests with
  staggered ramp-up, feature-flag rollouts, and canary deployments all have the same
  failure mode under the hood.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_did_staggered.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_two_by_two_did` | Sanity-check the 2x2 DiD against a regression. |
| **Phase 2** | `uv run python -m src._02_event_study` | Print the fitted event-study coefficients and pre-trend check. |
| **Phase 3** | `uv run python -m src._03_twfe_and_bacon` | Print the naive TWFE estimate and its Goodman-Bacon decomposition. |
| **Phase 4** | `uv run python -m src._04_att_gt` | Compute every ATT(g,t) cell and compare against ground truth. |
| **Phase 5** | `uv run python -m src._05_aggregate_att` | Aggregate ATT(g,t) into an event-study path and an overall ATT. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
