# Practice 102 — Panel Data & Fixed Effects

## Technologies

- **pyfixest** — high-performance fixed-effects estimation (`reghdfe`-style absorption), used to validate the hand-rolled estimators and as the speed benchmark on a large panel.
- **linearmodels** — `PanelOLS`, the standard `statsmodels`-adjacent panel-data estimator, used as a second independent validation source.
- **xy** — plotting (coefficient plot, CI-coverage plot).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### What Repeated Observations Buy You

A single cross-section can never separate a regressor's effect from any
unobserved trait of the unit that happens to correlate with it — a firm's
management quality, a county's culture, a person's ability. Panel data
(the same units observed over multiple periods) makes that trait
*visible* in a specific sense: because it doesn't change over time, it
shows up identically in every one of a unit's own observations, and can
therefore be subtracted out using only that unit's own data, with no need
to ever measure or name it. That's the entire promise of fixed effects:
control for *everything* time-invariant about a unit, observed or not,
at the cost of only being able to speak to what varies *within* a unit
over time.

### The Within (Demeaning) Transformation

The **within** (or **fixed-effects**) estimator implements that promise
mechanically: subtract each unit's own mean from `y` and from every
regressor, then run OLS on what's left. Because a time-invariant quantity
equals its own unit mean in every period, demeaning removes it *exactly*,
whatever its value — no need to estimate it. The same operation is
merciless about what else it removes: any regressor that doesn't vary
within a unit has zero within-unit variation after demeaning, so its
coefficient is not identified. The intercept is the most extreme case of
this — a column of ones has no within-unit variation at all, so it
disappears along with everything else time-invariant.

### Frisch-Waugh-Lovell: Why Demeaning Works

Demeaning isn't a special trick invented for panel data — it's one
instance of a fully general fact about multiple regression, the
**Frisch-Waugh-Lovell (FWL) theorem** (Frisch & Waugh, 1933; generalized
by Lovell, 1963): in a regression of `y` on `[X_focal, W]`, the
coefficient on `X_focal` is identical to the coefficient obtained by (1)
regressing `y` on `W` and keeping the residual, (2) regressing `X_focal`
on `W` and keeping the residual, and (3) regressing residual-on-residual.
"Partialling out" `W` and then regressing what's left over recovers
exactly what including `W` directly would have given `X_focal`. Demeaning
by unit is FWL with `W` set to a full set of unit dummy columns: the
projection onto that dummy-variable subspace *is* the group-mean
projection. This is why the within estimator gives the identical answer
to "least squares dummy variables" (LSDV — literally including one dummy
per unit) without ever estimating a single dummy coefficient, which
matters enormously once a panel has thousands of units.

### Fixed vs. Random Effects, and the Hausman Logic

The within estimator only needs the idiosyncratic error to be uncorrelated
with the regressors *given* the unit effect — it makes no assumption
about whether the unit effect itself is correlated with the regressors,
because it never estimates the unit effect at all. The **random-effects**
(RE) estimator asks for more: it treats the unit effect as an unobserved
random draw, uncorrelated with every regressor, and exploits that extra
assumption to estimate more efficiently via GLS (using *both* within- and
between-unit variation, rather than throwing between-unit variation away).
If the assumption is true, RE is more efficient than FE; if it's false —
if the unit effect correlates with a regressor, exactly the scenario this
practice's DGP builds in — RE is *inconsistent*, while FE stays
consistent regardless. The **Hausman (1978) specification test** operationalizes
the choice: under H0 (RE's extra assumption holds), FE and RE are both
consistent, so they should differ only by sampling noise; a large,
statistically significant `FE - RE` gap is evidence the unit effect is
correlated with the regressors, and therefore evidence to prefer FE.

### Two-Way Fixed Effects

A single set of unit dummies protects against time-invariant confounders.
It does nothing about a **common shock that varies over time** — a macro
trend, a policy change, a business cycle — which can still confound a
regressor whose own variation trends over the same period (in this
practice, `x2`'s rollout intensity grows over time alongside a genuine
time effect). Adding a second set of dummies, one per period, removes
that too: **two-way fixed effects** subtracts unit means, period means,
and adds back the grand mean once (since it was subtracted twice). For a
*balanced* panel — every unit observed in every period — that closed-form
double-demeaning is exact; an unbalanced panel needs iterative alternating
projections instead (Guimaraes & Portugal, 2010), which is what
`pyfixest`'s `reghdfe`-style absorption actually implements under the
hood and why it, not hand-rolled iteration, is the tool of choice once
panels get large or ragged.

### Clustering at the Level of Treatment Assignment

Fixing the point estimate's bias says nothing about whether its standard
errors are trustworthy. If the regressor of real interest is assigned at
a level coarser than the unit — a state policy applying to every county
in the state, a firm-wide change applying to every worker — and the error
term has any correlation within that coarser group (a state-level shock,
a firm-level shock), then treating every row as independent information
overstates how much independent information there actually is. The fix
is **cluster-robust** standard errors (Liang & Zeger, 1986): sum each
cluster's score contributions *before* squaring them in the sandwich
estimator's "meat," so within-cluster correlation is absorbed instead of
ignored. Bertrand, Duflo & Mullainathan (2004) made the point impossible
to ignore for differences-in-differences designs specifically: ignoring
clustering when the treatment varies at a group level can understate
standard errors by a factor of several, turning noise into apparently
"significant" results. The rule of thumb this practice's simulation makes
concrete: **cluster at the level the regressor of interest actually
varies at**, not at the (finer) level the data happens to be recorded at.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Within (demeaning) transformation** | Subtract each unit's own mean from `y` and `X`; the fixed-effects estimator. |
| **LSDV** | "Least squares dummy variables" — including one dummy per unit directly; algebraically identical to the within estimator. |
| **Frisch-Waugh-Lovell (FWL) theorem** | A subset of regressors' coefficient can be recovered by partialling out every other regressor from both `y` and that subset, then regressing residual on residual. |
| **Fixed effects (FE)** | Allows the unit effect to correlate with regressors; consistent regardless, at the cost of dropping time-invariant regressors. |
| **Random effects (RE)** | Assumes the unit effect is uncorrelated with regressors; more efficient if true, inconsistent if false. |
| **Hausman test** | Compares FE vs. RE estimates; a large gap is evidence against RE's extra assumption. |
| **Two-way fixed effects** | Unit *and* period dummies (or double-demeaning), removing time-invariant confounders and common time shocks together. |
| **Cluster-robust SE** | Covariance estimator that allows correlated errors within (but not across) clusters. |
| **Level of treatment assignment** | The (often coarser-than-unit) group a regressor of interest is actually assigned at — the level standard errors must be clustered at. |

### Where This Fits

Panel fixed effects is "OLS plus a trick to satisfy strict exogeneity by
differencing away time-invariant confounders" — the same family this
whole curriculum's other techniques belong to (RDD exploits a
discontinuity instead, synthetic control constructs a counterfactual unit
instead, IV instruments around the endogeneity instead). Alternatives to
FE within panel data itself include random effects (more efficient, but
only if its extra assumption holds — see Hausman above) and first-
differencing (algebraically different from demeaning outside of the
two-period case, but solving the same problem). The cluster-robust
machinery here is exactly what `linearmodels`' `cov_type="clustered"` and
`pyfixest`'s `vcov={"CRV1": ...}` compute — this practice demystifies what
those keyword arguments are actually doing, the same way practice 097
demystified `cov_type="HC3"`.

### References

- Frisch, R. & Waugh, F.V., 1933, "Partial Time Regressions as Compared with Individual Trends": <https://doi.org/10.2307/1907330>
- Lovell, M.C., 1963, "Seasonal Adjustment of Economic Time Series": <https://doi.org/10.2307/2283327>
- Hausman, J.A., 1978, "Specification Tests in Econometrics": <https://doi.org/10.2307/1913827>
- Guimaraes, P. & Portugal, P., 2010, "A Simple Feasible Alternative Procedure to Estimate Models with High-Dimensional Fixed Effects": <https://doi.org/10.1177/1536867X1001000308>
- Bertrand, M., Duflo, E. & Mullainathan, S., 2004, "How Much Should We Trust Differences-in-Differences Estimates?": <https://doi.org/10.1162/003355304772839588>
- Cameron, A.C. & Miller, D.L., 2015, "A Practitioner's Guide to Cluster-Robust Inference": <https://doi.org/10.3368/jhr.50.2.317>
- `pyfixest` docs: <https://s3alfisc.github.io/pyfixest/>
- `linearmodels` `PanelOLS` docs: <https://bashtage.github.io/linearmodels/panel/models.html>

## Description

Build a synthetic balanced panel with a known true `beta`, a unit effect
correlated with a regressor, and a time effect correlated with another
regressor's rollout. Implement the one-way within (demeaning) estimator
and validate it against `pyfixest` and `linearmodels.PanelOLS`; implement
the two-way (unit + time) demeaning for the same validation; prove
demeaning is a special case of the Frisch-Waugh-Lovell theorem; implement
cluster-robust standard errors and run a Monte Carlo simulation showing
nominal 95% confidence intervals covering far less than 95% of the time
when clustering is ignored. Finish with a coefficient plot (pooled OLS vs.
within vs. two-way FE vs. truth), a coverage plot (naive vs. clustered),
and a speed benchmark against `pyfixest`/`linearmodels` on a much larger
panel.

### What you'll learn

1. Why demeaning by unit removes time-invariant confounders exactly, and why it kills time-invariant regressors (including the intercept) along with them.
2. Why the Frisch-Waugh-Lovell theorem is the *reason* demeaning gives the same answer as including unit dummies directly.
3. The fixed-vs-random-effects trade-off and the logic behind the Hausman test.
4. Why a time trend correlated with a regressor's rollout defeats one-way FE, and how two-way FE fixes it.
5. Why standard errors must be clustered at the level a regressor of interest is actually assigned at — and what happens to CI coverage when they aren't.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_panel_fixed_effects.ipynb`

### Phase 1: The Within Transformation (~15-20 min) — `src/_01_within_transform.py`

Synthetic panel generation (`src/datasets.py`) and the pyfixest/linearmodels
comparison harness are fully scaffolded. The teaching content is the
demeaning transformation itself.

1. **TODO #1 — `within_demean(y, X, unit_id)`**: subtract each unit's own
   mean from `y` and from every column of `X`. ~10-15 lines.

### Phase 2: Two-Way Fixed Effects (~15-20 min) — `src/_02_two_way_demean.py`

2. **TODO #1 — `two_way_demean(y, X, unit_id, time_id)`**: implement the
   closed-form double-demeaning formula for a balanced panel (subtract
   unit and period means, add back the grand mean once). ~10-15 lines.

### Phase 3: Frisch-Waugh-Lovell (~15 min) — `src/_03_fwl_demonstration.py`

3. **TODO #1 — `fwl_partial_out(y, X_focal, W)`**: implement the general
   partialling-out routine (regress `y` and `X_focal` on `W`, take
   residuals, regress residual on residual); check it reproduces Phase
   1's within-estimator beta exactly when `W` is a unit-dummy matrix.
   ~10-15 lines.

### Phase 4: Clustering (~20 min) — `src/_04_cluster_vcov.py`

The Monte Carlo coverage simulation is fully scaffolded. The teaching
content is the cluster-robust variance formula it's built on.

4. **TODO #1 — `cluster_robust_vcov(X, resid, XtX_inv, cluster_id)`**:
   implement the CR1 cluster-robust sandwich estimator with the standard
   small-sample correction. ~15-20 lines.

### Phase 5: End-to-End Run (no TODO)

Run the notebook's final cells: a coefficient plot comparing pooled OLS,
one-way within, and two-way FE against the truth; a coverage plot
(naive vs. clustered CIs); and a speed benchmark of the hand-rolled
within estimator against `pyfixest` and `linearmodels.PanelOLS` on a
~100k-row panel.

### What to look for in the results

- Phase 1: pooled OLS's `beta_x1` should be visibly off from the true
  value; the within estimator's should match `pyfixest`/`linearmodels`
  to within `1e-6`.
- Phase 2: the one-way within estimator's `beta_x2` should still be a
  bit off (the time-trend confound); two-way FE should close that gap.
- Phase 3: the FWL route (partialling out unit dummies) should match
  Phase 1's within-estimator beta to within `1e-6` — that agreement
  *is* the theorem made visible.
- Phase 4: cluster-robust standard errors should be visibly larger than
  naive ones on `beta_x2` — the standard "false confidence" failure mode.
- Phase 5's coverage plot: naive CI coverage should sit well below 95%;
  clustered coverage should sit close to it. The speed benchmark should
  show `pyfixest` well ahead of both the hand-rolled estimator and
  `linearmodels` on the large panel.

## Motivation

- **AutoScheduler.AI relevance**: any "does this scheduling parameter
  actually move the outcome metric, net of everything stable about a
  warehouse/route/carrier we can't measure" question is a panel-fixed-
  effects question — this practice is the natural extension of practice
  097's OLS foundations to repeated-observations data, which is the
  overwhelmingly common shape of real operational data.
- **Senior → Staff differentiator**: knowing to add `| unit_id` to a
  `pyfixest` formula is common; knowing that it's throwing away all
  between-unit variation on purpose, why that's the right trade for a
  suspected confounder, and when to cluster instead of (or alongside)
  absorbing is not.
- **Generalises beyond econometrics**: the Frisch-Waugh-Lovell partialling-
  out pattern reappears anywhere a model is fit with a large nuisance
  parameter block (fixed effects, seasonal dummies, batch effects in
  bioinformatics) that's more efficient to project out than to estimate
  directly.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_panel_fixed_effects.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_within_transform` | Validate the within estimator against pyfixest and linearmodels. |
| **Phase 2** | `uv run python -m src._02_two_way_demean` | Validate the two-way estimator against pyfixest and linearmodels. |
| **Phase 3** | `uv run python -m src._03_fwl_demonstration` | Check the FWL partialling-out route matches the within estimator. |
| **Phase 4** | `uv run python -m src._04_cluster_vcov` | Compare naive vs. cluster-robust SEs, then run the coverage simulation. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
