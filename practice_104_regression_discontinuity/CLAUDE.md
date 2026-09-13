# Practice 104 — Regression Discontinuity Design

## Technologies

- **rdrobust** — the standard local-polynomial RDD package; used to validate the from-scratch local linear estimator and as the reference for bandwidth selection.
- **xy** — plotting (the canonical RDD figure, a bandwidth-sensitivity plot, the McCrary density plot).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### What RDD Identifies

Regression discontinuity design (RDD) exploits an arbitrary threshold rule:
whenever treatment assignment is a known, deterministic (or probability-
shifting) function of a **running variable** `X` crossing a **cutoff** `c`,
units just below and just above the cutoff are — in the limit as the
window shrinks to zero — as good as randomly assigned. It is the closest
observational design gets to a randomized experiment, because the
"treatment or not" decision at the threshold is not the unit's choice. The
canonical example (Lee, 2008, used throughout this practice): in a
first-past-the-post election, winning is a deterministic function of vote
margin crossing 0 — a candidate who wins by 0.1 points and one who loses by
0.1 points are, absent precise control over the vote count, comparable in
every other respect.

### Sharp vs. Fuzzy RDD

| Design | Treatment rule | Estimator |
|--------|----------------|-----------|
| **Sharp** | `D = 1{X >= c}` — deterministic | Jump in `E[Y\|X]` at `c` |
| **Fuzzy** | `P(D=1\|X)` merely jumps at `c` — imperfect compliance | Wald ratio: jump in `E[Y\|X]` / jump in `E[D\|X]`, at `c` |

Fuzzy RDD is structurally an instrumental-variables problem: "crossing the
cutoff" is used as an instrument for actual treatment take-up, exactly the
way randomized *assignment* instruments for treatment *receipt* in a trial
with imperfect compliance (Hahn, Todd & Van der Klaauw, 2001). The Wald
ratio recovers a **local average treatment effect (LATE)** for compliers
near the cutoff — units whose treatment status is genuinely determined by
which side of the cutoff they land on.

### Local Linear Regression, Not Global Polynomials

An RDD estimate only needs the two conditional means `E[Y|X=c^-]` and
`E[Y|X=c^+]` — the limits from each side. The textbook temptation is to fit
one global polynomial (quartic, quintic...) over the *entire* running-
variable range and read off its value at the cutoff. Gelman & Imbens (2019,
"Why High-Order Polynomials Should Not Be Used in Regression Discontinuity
Designs") show this is a bad idea: a high-order polynomial lets data far
from the cutoff — which carries no information about the local jump —
distort the extrapolation right at it, and the implied observation weights
can even be *negative* far from the cutoff. The modern standard (Hahn,
Todd & Van der Klaauw, 2001; Imbens & Lemieux, 2008) is **local linear
regression**: fit a simple line to each side, using only a bandwidth-wide
window of data around the cutoff, weighted so nearby points count more
than distant ones. This practice implements exactly that estimator (Phase
1) and never fits a global polynomial.

### Bandwidth Selection and the Bias-Variance Tradeoff

The bandwidth is the single most consequential choice in RDD estimation,
because it is a direct bias-variance tradeoff: a wide bandwidth pulls in
more data (lower variance in `tau_hat`) but also more curvature a local
*linear* fit cannot represent (higher bias, since the true conditional mean
isn't linear that far out); a narrow bandwidth does the reverse. Two
data-driven selectors dominate practice:

- **Imbens & Kalyanaraman (2012, "IK")** — a plug-in selector: estimate the
  curvature and the residual variance near the cutoff from pilot
  regressions, then plug them into the bandwidth formula that minimizes
  asymptotic mean squared error.
- **Calonico, Cattaneo & Titiunik (2014, "CCT")** — refines IK by pairing
  the point estimate with a **bias-corrected** estimate and a **robust**
  variance formula that accounts for the bias correction itself, which is
  what makes `rdrobust`'s default confidence intervals valid even when the
  point estimate is somewhat biased.

Phase 3 implements a **simplified** plug-in rule-of-thumb (ROT) bandwidth
in the spirit of IK's pilot stage — same ingredients (curvature, noise,
sample-size scaling), a single pooled quartic pilot fit instead of IK's
multi-stage, boundary-corrected pilot bandwidths, and CCT's bias correction
is described here but not implemented. Treat the result as a reasonable
starting bandwidth to sanity-check against a sensitivity plot and against
`rdrobust`'s own (fully MSE-optimal) choice — not as a byte-for-byte
reproduction of either published algorithm.

### Manipulation and Balance Checks

RDD's core identifying assumption — units cannot precisely manipulate `X`
to land on their preferred side of `c` — is fundamentally untestable
directly, but it has testable *implications*:

- **McCrary (2008) density test**: if units are sorting themselves across
  the cutoff, the running variable's **density** should show a jump at
  `c`, even when nothing else does. Phase 4 implements a simplified
  version (binned, count-weighted local linear density regression on each
  side + a Wald z-test) rather than McCrary's original local-likelihood
  procedure with adaptive bandwidth selection — same question, less
  machinery.
- **Covariate balance placebo checks**: a covariate measured *before* the
  cutoff cannot be caused by crossing it. Running the same local linear
  estimator with a pre-treatment covariate as the "outcome" should show no
  jump; a jump signals either manipulation or an unrelated policy change
  coinciding with the same threshold.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Running variable** | The continuous variable whose value determines treatment eligibility (e.g. vote margin, test score). |
| **Cutoff** | The threshold value of the running variable at which treatment assignment (or its probability) jumps. |
| **Sharp RDD** | Treatment is a deterministic function of crossing the cutoff. |
| **Fuzzy RDD** | Crossing the cutoff only shifts the *probability* of treatment. |
| **Local linear regression** | A separate weighted linear fit on each side of the cutoff, using only a bandwidth-wide window — the recommended RDD estimator. |
| **Bandwidth** | The window width around the cutoff used by the local linear fit; the bias-variance tradeoff's control knob. |
| **Wald ratio** | Fuzzy RDD's IV-style estimator: reduced-form jump in Y divided by first-stage jump in treatment take-up. |
| **McCrary density test** | Tests whether the running variable's density is continuous at the cutoff — evidence for or against manipulation. |
| **Covariate balance / placebo check** | Re-running the RDD estimator on a pre-treatment covariate; a jump there signals a design problem. |

### Where This Fits

RDD is one of the "design-based" identification strategies (alongside
difference-in-differences, synthetic control, and IV) that trade the
strong, often implausible assumptions of a global regression-adjustment
approach for a much weaker, local one — at the cost of only identifying an
effect *at the cutoff*, not an average effect for the whole population.
Compared to IV, RDD's "instrument" (which side of the cutoff a unit falls
on) is usually far more credible than an externally argued exclusion
restriction; compared to a randomized experiment, RDD needs no
intervention at all, only a pre-existing threshold rule — which is also
its main limitation: the estimate is a **local** effect, and generalizing
it beyond units near the cutoff is itself an assumption, not a result.

### References

- Lee, 2008, "Randomized Experiments from Non-random Selection in U.S. House Elections": <https://www.princeton.edu/~davidlee/wp/RDrand.pdf>
- Imbens & Kalyanaraman, 2012, "Optimal Bandwidth Choice for the Regression Discontinuity Estimator": <https://doi.org/10.1093/restud/rdr043>
- Calonico, Cattaneo & Titiunik, 2014, "Robust Nonparametric Confidence Intervals for Regression-Discontinuity Designs": <https://doi.org/10.3982/ECTA11757>
- Gelman & Imbens, 2019, "Why High-Order Polynomials Should Not Be Used in Regression Discontinuity Designs": <https://doi.org/10.1080/07350015.2017.1366909>
- McCrary, 2008, "Manipulation of the Running Variable in the Regression Discontinuity Design": <https://doi.org/10.1016/j.jeconom.2007.05.005>
- Hahn, Todd & Van der Klaauw, 2001, "Identification and Estimation of Treatment Effects with a Regression-Discontinuity Design": <https://doi.org/10.1111/1468-0262.00183>
- `rdrobust` documentation: <https://rdpackages.github.io/rdrobust/>

## Description

Implement sharp and fuzzy RDD from scratch — a kernel-weighted local
linear estimator, a plug-in bandwidth selector, and a Wald-ratio fuzzy
estimator — validated against `rdrobust` on synthetic data with a known
jump, then applied to the real Lee (2008) US House elections dataset to
recover the Democratic incumbency-advantage effect. Manipulation and
covariate-balance checks make the design's identifying assumption
falsifiable rather than merely asserted.

### What you'll learn

1. Why RDD estimates a jump via two local linear extrapolations, not a naive difference in group means or a global polynomial fit.
2. The difference between sharp and fuzzy RDD, and why the fuzzy case is structurally an IV/Wald problem.
3. What the RDD bandwidth choice actually trades off, and how a plug-in selector estimates that tradeoff from data.
4. How to test the untestable-in-principle "no manipulation" assumption via the McCrary density test and covariate balance checks.
5. How to validate a hand-rolled nonparametric estimator against a trusted reference implementation (`rdrobust`) at a fixed bandwidth.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_regression_discontinuity.ipynb`

### Phase 1: Sharp RDD — The Local Linear Estimator (~20 min) — `src/_01_local_linear_rdd.py`

Synthetic data generation (`src/datasets.py`) and the `rdrobust` comparison
harness are fully scaffolded. The teaching content is the estimator itself.

1. **TODO #1 — `local_linear_rdd(running, outcome, cutoff, bandwidth, kernel)`**:
   implement the two-sided, triangular-kernel-weighted local linear RDD
   estimator via weighted least squares. ~20-25 lines.

### Phase 2: The Canonical RDD Plot — Binned Means (~15 min) — `src/_02_binned_means.py`

2. **TODO #1 — `binned_means(running, outcome, cutoff, bin_width)`**:
   implement cutoff-anchored, evenly spaced binning of the running
   variable and compute the mean outcome per bin. ~15-20 lines.

### Phase 3: Bandwidth Selection (~20 min) — `src/_03_bandwidth_selection.py`

3. **TODO #1 — `rule_of_thumb_bandwidth(running, outcome, cutoff)`**:
   implement a simplified plug-in (ROT) bandwidth selector from a pilot
   quartic fit's curvature and residual variance. ~15-20 lines.

### Phase 4: Manipulation and Balance Checks (~10 min) — `src/_04_diagnostics.py`

Fully scaffolded, no TODO — the McCrary density test and the covariate
balance placebo check both reuse machinery from Phases 1-3. The teaching
content here is running and interpreting them, on the real Lee (2008) data
and on a synthetic "manipulated" scenario built to fail the check on
purpose.

### Phase 5: Fuzzy RDD — The Wald Ratio (~15 min) — `src/_05_fuzzy_rdd.py`

4. **TODO #1 — `fuzzy_rdd_wald(running, outcome, treatment, cutoff, bandwidth)`**:
   implement the Wald-ratio fuzzy RDD estimator by calling
   `local_linear_rdd` twice (once for the outcome, once for treatment
   take-up) and dividing. ~10-15 lines.

### Phase 6: End-to-End Run (no TODO)

Run the notebook's final cells: the full pipeline (ROT bandwidth, local
linear estimate, `rdrobust` cross-check, canonical plot) applied to the
real Lee (2008) data, recovering the Democratic incumbency-advantage
estimate.

### What to look for in the results

- Phase 1's `|diff|` against `rdrobust`'s "Conventional" estimate (same
  bandwidth, same triangular kernel) should be on the order of `1e-6` or
  smaller — this compares the *same* procedure, not a bias-corrected one.
- Phase 3: the bandwidth-sensitivity plot should show a reasonably stable
  plateau around the selected bandwidth; wild swings would mean the result
  is bandwidth-fragile.
- Phase 4: the McCrary test should fail to reject continuity on the real
  Lee (2008) data (consistent with Lee's and McCrary's own published
  findings — candidates cannot precisely control their vote margin) and
  should reject it on the synthetic "manipulated" scenario. The covariate
  balance check should show a jump small relative to Phase 1's real effect.
- Phase 5: the Wald estimate should land close to the DGP's true `tau`,
  noticeably closer than the raw (uncorrected-for-compliance) reduced-form
  jump alone.
- Phase 6: the Lee (2008) incumbency-advantage estimate should land in the
  neighborhood of 0.07-0.09 (7-9 percentage points), consistent with Lee
  (2008) and Imbens & Kalyanaraman (2012)'s published estimates.

## Motivation

- **AutoScheduler.AI relevance**: any policy with a hard threshold (an
  eligibility score, a capacity limit, an SLA breach line) creates a
  natural RDD around that threshold — this is the tool for asking "did
  crossing this line actually change the outcome" without needing an
  experiment.
- **Senior → Staff differentiator**: knowing to call `rdrobust` is common;
  knowing why local linear beats a global polynomial, what the bandwidth
  is actually trading off, and how to falsify the design's core assumption
  is not.
- **Generalises beyond econometrics**: the "arbitrary threshold as a
  natural experiment" pattern reappears in A/B-adjacent settings —
  eligibility cutoffs, score-based gating, capacity limits — anywhere a
  rule, not a choice, decides who's treated.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_regression_discontinuity.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_local_linear_rdd` | Sanity-check the local linear RDD estimator against `rdrobust`. |
| **Phase 2** | `uv run python -m src._02_binned_means` | Print binned means either side of the cutoff. |
| **Phase 3** | `uv run python -m src._03_bandwidth_selection` | Compute the ROT bandwidth and compare it to `rdrobust`'s optimal choice. |
| **Phase 4** | `uv run python -m src._04_diagnostics` | Run the McCrary density test (Lee 2008 vs. a manipulated synthetic scenario) and a covariate balance check. |
| **Phase 5** | `uv run python -m src._05_fuzzy_rdd` | Compute the fuzzy RDD Wald estimate on the synthetic fuzzy scenario. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

### Dataset provenance — `data/lee2008.csv`

Real data: Lee (2008), "Randomized Experiments from Non-random Selection in
U.S. House Elections," *Journal of Econometrics* 142(2):675-697. 6,558 US
House elections, 1946-1998. Columns: `x` = Democratic vote-share margin at
election *t-1* (centered so the cutoff is exactly 0: positive means the
Democrat won the previous election), `y` = Democratic vote share at
election *t*. This is the dataset used throughout the RDD literature
(Imbens & Kalyanaraman, 2012; the `RDHonest`, `rddtools`, `rdcqr`, and
`RATest` R packages all ship it under different names — `lee08`, `house`,
`lee`, `lee2008`). Vendored here from the public, citable mirror at
[LOST-STATS.github.io](https://raw.githubusercontent.com/LOST-STATS/LOST-STATS.github.io/master/Model_Estimation/Data/Regression_Discontinuity_Design/house.csv)
(the "Library of Statistical Techniques", an open teaching-materials
project) since the dataset is not bundled in any Python package and the
original Stata files (Mostly Harmless Econometrics data archive) require R
tooling to convert. Row count (6,558) and summary statistics were verified
against the published description before vendoring.

### Synthetic data — `src/datasets.py :: load_dataset`

Every `load_dataset(...)` scenario ("sharp", "fuzzy", "manipulated") is
**synthetic**, generated from a known data-generating process with a
known true jump `TRUE_TAU = 4.0` — clearly not real-world data, used only
so every estimator can be checked against ground truth before touching
Lee (2008).

## State

`not-started`
