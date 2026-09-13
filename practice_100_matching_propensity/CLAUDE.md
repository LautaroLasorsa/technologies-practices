# Practice 100 — Matching & Propensity Scores

## Technologies

- **causaldata** — ships the real LaLonde/Dehejia-Wahba NSW and CPS-1 datasets used as this practice's centerpiece.
- **scikit-learn** — the propensity-score (logistic regression) and outcome-regression (linear regression) models the estimators are built on.
- **xy** — plotting (balance/Love plot, propensity-score overlap densities, estimate-comparison plot).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### Selection on Observables

Randomized experiments guarantee `E[Y(1) - Y(0) | X] = E[Y | T=1, X] - E[Y | T=0, X]` by
construction — treatment assignment is independent of the potential outcomes. Outside an
experiment, that independence has to be *assumed*: **unconfoundedness** (a.k.a.
ignorability, selection-on-observables), `(Y(0), Y(1)) ⊥ T | X` — conditional on a rich
enough covariate set `X`, treatment assignment is "as good as random." Every technique in
this practice is a different way of exploiting that assumption once it's granted; none of
them can *test* it, which is why this practice's centerpiece — the LaLonde replication — is
built to show what happens when adjusting for observables still isn't enough.

### The Propensity Score and Its Balancing Property

The propensity score `e(X) = P(T=1 | X)` is the probability of treatment given covariates.
Rosenbaum & Rubin (1983) proved two things that make everything downstream work: (1) if
unconfoundedness holds given `X`, it also holds given `e(X)` alone — a single scalar
summarizes all of `X` for adjustment purposes; (2) `e(X)` is a **balancing score** —
conditioning on it balances the *entire* covariate vector between treatment arms, exactly
as randomization would have. That second property is what licenses matching or weighting on
a single number instead of on a high-dimensional `X` directly.

### Matching, IPW, and AIPW — Three Ways to Use `e(X)`

| Method | Idea | Failure mode |
|---|---|---|
| **Nearest-neighbor / caliper matching** | Pair each treated unit with the control(s) closest on `e(X)`; discard pairs too far apart (the caliper). | Discards unmatched units — the estimate no longer targets the full population, only the "matchable" region. |
| **Inverse probability weighting (IPW)** | Reweight every unit by `1/e(X)` (treated) or `1/(1-e(X))` (control) so the reweighted sample looks randomized (Horvitz & Thompson, 1952). The Hajek/self-normalized form divides by the summed weights per arm instead of `n`. | Extreme weights when `e(X)` is near 0 or 1 for some units — a handful of poorly-overlapped units can dominate the estimate (this practice's LaLonde replication hits exactly this). |
| **Augmented IPW (AIPW) / doubly robust** | Start from an outcome-regression prediction `mu1(X) - mu0(X)`, then add an IPW-weighted correction built from each unit's *actual* outcome. Consistent if *either* the propensity model or the outcome model is correctly specified (Robins, Rotnitzky & Zhao, 1994). | Still needs reasonable overlap — if `e(X)` is degenerate, the correction term inherits IPW's instability even though the outcome model provides some cushioning. |

### Overlap / Common Support

All three methods implicitly assume **overlap**: `0 < e(X) < 1` for every unit in the
population of interest — every covariate profile must have *some* chance of appearing in
either treatment arm. When treated and control groups come from genuinely different
populations (this practice's NSW-vs-CPS comparison), some units have `e(X)` near 0 or 1:
matching silently drops them (via the caliper), while IPW/AIPW keep them and let their huge
weights distort the estimate. Checking overlap — via a propensity-score density plot split
by arm — is a mandatory diagnostic before trusting *any* of these estimators, not an optional
extra.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Unconfoundedness / selection on observables** | `(Y(0), Y(1)) ⊥ T \| X` — treatment is as-good-as-random once `X` is accounted for. Untestable; the core assumption every method here relies on. |
| **Propensity score `e(X)`** | `P(T=1 \| X)`; a balancing score (Rosenbaum & Rubin, 1983). |
| **Standardized mean difference (SMD)** | `(mean_treat - mean_control) / sqrt((var_treat + var_control)/2)` — a scale-free covariate-imbalance metric; `\|SMD\| < 0.1` is the conventional "balanced" threshold (Austin, 2011). |
| **Caliper** | Maximum allowed `\|e(X)\|` distance for a match to be accepted; `0.2 * sd(logit(e(X)))` is the standard default (Rosenbaum & Rubin, 1985). |
| **Hajek (self-normalized) IPW** | An IPW estimator that divides by the sum of weights within each arm rather than by `n` — lower finite-sample variance than the raw Horvitz-Thompson form. |
| **Doubly robust / AIPW** | An estimator consistent if *either* the propensity model or the outcome model is correctly specified, not both. |
| **Overlap / common support** | The region of `X` where `0 < e(X) < 1` meaningfully — where adjustment is possible at all. |

### Where This Fits

Selection-on-observables methods (this practice) are what you reach for when there's no
randomization, no instrument (IV), and no discontinuity (RDD) to exploit — only a claim that
"we observed enough to adjust for confounding." That claim is strictly weaker than a design-
based identification strategy, which is exactly why the LaLonde literature became the
standard cautionary tale: LaLonde (1986) showed that a wide range of then-standard
econometric adjustments, applied to a non-experimental comparison group, failed to recover
a known experimental benchmark; Dehejia & Wahba (1999, 2002) argued propensity-score methods
do much better *given adequate overlap and covariates*; Smith & Todd (2005) pushed back,
showing the results are sensitive to specification and comparison-group choice. This
practice's own replication lands in the middle of that debate on purpose (see "What to look
for in the results").

### References

- Rosenbaum & Rubin, 1983, "The Central Role of the Propensity Score in Observational Studies for Causal Effects": <https://doi.org/10.1093/biomet/70.1.41>
- Rosenbaum & Rubin, 1985, "Constructing a Control Group Using Multivariate Matched Sampling Methods That Incorporate the Propensity Score": <https://doi.org/10.1080/00031305.1985.10479383>
- Robins, Rotnitzky & Zhao, 1994, "Estimation of Regression Coefficients When Some Regressors Are Not Always Observed": <https://doi.org/10.1080/01621459.1994.10476818>
- LaLonde, 1986, "Evaluating the Econometric Evaluations of Training Programs with Experimental Data": <https://www.jstor.org/stable/1806062>
- Dehejia & Wahba, 2002, "Propensity Score-Matching Methods for Nonexperimental Causal Studies": <https://doi.org/10.1162/003465302317331982>
- Austin, 2011, "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies": <https://doi.org/10.1080/00273171.2011.568786>

## Description

Build propensity-score, matching, IPW, and AIPW estimators from scratch, applied to a real
replication of LaLonde (1986): the randomized NSW job-training experiment gives a known-good
experimental ATE (~$1,794), and a naive comparison against the CPS-1 observational comparison
group (a completely different population) is off by roughly $10,000. The practice builds up,
phase by phase, the tools to try to close that gap — and shows, honestly, how far each one
gets.

### What you'll learn

1. Why the propensity score is a *balancing score*, and what that buys you over matching on raw covariates.
2. How caliper matching trades off sample size (dropped units) against bias reduction.
3. How IPW reweights the *whole* sample instead of discarding any of it — and why that makes it sensitive to poor overlap.
4. Why AIPW is "doubly robust," and what that guarantee does and doesn't protect against.
5. Why checking overlap (propensity-score density by arm) is not optional — and what it looks like when it fails.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_matching_propensity.ipynb`

### Phase 1: Propensity Score Estimation (~15 min) — `src/_01_propensity_score.py`

Data loading (`src/datasets.py`) is fully scaffolded — it loads the real NSW/CPS-1 data via
`causaldata`. The teaching content is the propensity model itself.

1. **TODO #1 — `fit_propensity_score(X, treat)`**: fit a logistic regression on standardized
   covariates and return `P(treat=1 | X)` for every unit. ~10-15 lines.

### Phase 2: Covariate Balance (~10-15 min) — `src/_02_balance.py`

2. **TODO #1 — `standardized_mean_diff(x, treat, weights=None)`**: implement the SMD balance
   metric, with an optional weighted-mean path for post-adjustment balance checks. ~10-15
   lines.

### Phase 3: Nearest-Neighbor / Caliper Matching (~20 min) — `src/_03_matching.py`

3. **TODO #1 — `nearest_neighbor_match(ps_treated, ps_control, y_treated, y_control)`**:
   implement 1:1 nearest-neighbor matching on the propensity score with a caliper, and
   compute the resulting ATT. ~15-20 lines.

### Phase 4: Inverse Probability Weighting (~15 min) — `src/_04_ipw.py`

4. **TODO #1 — `ipw_ate(y, treat, propensity)`**: implement the Hajek-normalized IPW
   estimator of the ATE. ~10 lines.

### Phase 5: Doubly-Robust (AIPW) Estimation (~15-20 min) — `src/_05_aipw.py`

Outcome-regression fitting (`fit_outcome_regressions`) is scaffolded — plumbing, not the
taught technique.

5. **TODO #1 — `aipw_ate(y, treat, propensity, mu1, mu0)`**: implement the AIPW pseudo-outcome
   and average it into the doubly-robust ATE estimate. ~10-15 lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: every method's estimate side by side with the experimental
benchmark (`estimate_comparison_plot`), plus Love plots showing balance before/after IPW and
after matching.

### What to look for in the results

- Phase 1: the propensity-score overlap plot shows real, severe non-overlap — NSW treated
  units cluster at moderate-to-high scores, CPS-1 controls pile up near 0. This is the root
  cause of everything that follows.
- Phase 2: raw SMDs are large (several are above 1.0 in absolute value) — NSW and CPS-1 are
  not comparable populations on observables alone.
- Phase 3: matching (with its caliper silently dropping badly-overlapped units) recovers an
  ATT close to the experimental benchmark — matching's willingness to discard units is,
  here, a feature.
- Phase 4: IPW's ATE stays badly biased, closer to the naive estimate than to the
  experimental benchmark. This is **expected, not a bug**: roughly 1 in 10 treated units has
  a propensity score under 0.01, producing IPW weights above 100x that dominate the weighted
  average — a textbook illustration of why the overlap assumption matters and why IPW without
  trimming is fragile on this specific comparison.
- Phase 5: AIPW partially cushions IPW's instability (the outcome-regression term pulls the
  estimate back toward the outcome model whenever a unit's weight would otherwise dominate),
  landing between IPW and matching, but still with meaningful remaining bias — the "doubly
  robust" guarantee is about *consistency* given at least one correct model, not immunity to
  poor overlap in a finite sample.
- The overall pattern — matching >> AIPW > IPW, and none of them fully free of bias — is the
  same tension the LaLonde/Dehejia-Wahba/Smith-Todd literature has argued over for decades;
  this practice reproduces it in miniature rather than papering over it.

## Motivation

- **AutoScheduler.AI relevance**: "did this scheduling policy change actually help" questions
  almost never come with a randomized rollout attached — propensity/matching/IPW is the
  standard toolkit for making an observational comparison as credible as the data allows, and
  for knowing when it still isn't credible enough.
- **Senior → Staff differentiator**: knowing `sklearn.LogisticRegression` and calling it a
  "propensity score" is common; knowing what balancing property that buys you, why overlap
  can silently break every estimator built on it, and being able to show — not just assert —
  that failure mode is not.
- **Generalises beyond econometrics**: the same three-estimator family (match / reweight /
  doubly-robust-combine) reappears in off-policy evaluation for bandits and RL, uplift
  modeling, and any observational A/B analysis where randomization wasn't possible.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_matching_propensity.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_propensity_score` | Fit the propensity score model and print its range. |
| **Phase 2** | `uv run python -m src._02_balance` | Print raw covariate SMDs on the observational comparison. |
| **Phase 3** | `uv run python -m src._03_matching` | Run caliper matching and print the matched ATT. |
| **Phase 4** | `uv run python -m src._04_ipw` | Compute the IPW ATE. |
| **Phase 5** | `uv run python -m src._05_aipw` | Compute the AIPW (doubly-robust) ATE. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

- **Dataset verification**: the installed `causaldata` package (0.1.5, verified by installing
  it and inspecting `pkgutil.iter_modules`) ships `nsw_mixtape` (the Dehejia-Wahba
  experimental NSW subsample, 185 treated + 260 control, both randomized) and `cps_mixtape`
  (the CPS-1 non-experimental comparison group, 15,992 units). It does **not** ship a PSID
  comparison group in this version — there is no `psid`/`psid_mixtape` module among its ~34
  datasets. This practice therefore uses CPS-1 as its one observational comparison group
  rather than the PSID/CPS pair the classic LaLonde tables report side by side. Verified
  ground truth from the loaded data: experimental ATE = $1,794.34 (matches the textbook
  Dehejia-Wahba figure exactly), naive NSW-treated-vs-CPS-1 diff = -$8,497.52.
- **`xy` workaround — Love/balance plot**: `xy` has an open upstream bug in low-cardinality
  categorical-axis handling (reflex-dev/xy#471). `love_plot`/`estimate_comparison_plot`
  (`src/plotting.py`) avoid it entirely by plotting against a plain `np.arange(...)` numeric
  y-position and attaching covariate/method names via `ax.set_yticklabels(...)` instead of
  passing the names as plotted data — the same pattern `practice_097`'s `coefficient_plot`
  already uses for its x-axis.
- **`xy` workaround — propensity-score overlap densities**: `xy` has no native KDE/density
  mark. `overlap_density_plot` computes the two groups' kernel density estimates itself with
  `scipy.stats.gaussian_kde` over a shared grid, then hands `xy` the resulting curves as two
  plain numeric `line` series.

## State

`not-started`
