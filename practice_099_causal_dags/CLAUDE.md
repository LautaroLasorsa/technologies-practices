# Practice 099 — Causal DAGs & Identification Strategies

## Technologies

- **DoWhy** — identification API (`CausalModel.identify_effect`) that searches a given DAG for valid adjustment sets via the backdoor/frontdoor criteria, used to cross-check the from-scratch checkers built in this practice.
- **causal-learn** — the PC constraint-based structure-discovery algorithm, used to see what (and what not) can be recovered from data alone without a known graph.
- **xy** — plotting, including the DAGs themselves (see "DAG drawing" below).

## Stack

Python 3.11+ (uv), Jupyter.

## Theoretical Context

### Graphs as the Language of Identification

A causal DAG encodes assumptions, not data: every arrow is a claim ("this variable
causes that one") the analyst is willing to defend, and every *missing* arrow is
just as much a claim (no direct effect). Once those assumptions are fixed, the
question "can I estimate the effect of D on Y from observational data, and if so,
how?" becomes a **graphical** question — answerable by tracing paths through the
DAG, without needing to see the data at all. This is Pearl's causal-graph
framework: identification is decided by the graph's structure; estimation (the
actual number-crunching) is what happens after identification says "yes, and here's
the adjustment set to use."

### D-Separation and Conditional Independence

**D-separation** is the graphical criterion for reading conditional independencies
directly off a DAG: two nodes are d-separated by a set Z if every path between them
is *blocked* given Z. A path is blocked at a node depending on that node's role on
the path:

| Structure | Path shape | Blocked when... | Unblocked when... |
|---|---|---|---|
| **Chain** (mediator) | `A -> M -> B` | Z contains M | Z does not contain M |
| **Fork** (confounder) | `A <- Z -> B` | Z contains Z | Z does not contain Z |
| **Collider** | `A -> C <- B` | Z does **not** contain C (or any descendant of C) | Z contains C or a descendant of C |

The collider row is the one every other row's intuition gets backwards on first
encounter: conditioning on a chain or fork node *blocks* the path (good, if it's
confounding you want removed); conditioning on a collider *opens* it (bad — it
manufactures dependence between two variables that were independent). This
practice implements the check algorithmically (moralize the ancestral graph, test
separation in the undirected result — Lauritzen et al., 1990) rather than
classifying every path by hand, because the moralization route is what scales to
graphs too large to eyeball.

### The Backdoor and Frontdoor Criteria

The **backdoor criterion** (Pearl, 1993) formalizes "control for confounders, not
mediators or colliders" as a graphical test: a set Z is a valid adjustment set for
(D, Y) iff Z contains no descendant of D, and Z blocks every path into D that isn't
part of D's own causal effect (a "backdoor path" — one that starts with an arrow
*into* D). Removing D's outgoing edges and checking d-separation of D and Y given Z
in the result is exactly this test.

The **frontdoor criterion** handles the case where a valid backdoor set doesn't
exist (e.g. an unmeasured confounder of D and Y directly) but a mediator M fully
carries D's effect and is itself unconfounded with Y given D: the effect is then
identified as (effect of D on M) x (effect of M on Y adjusted for D), even though
no covariate set alone blocks the backdoor path. It's a much rarer applicable
pattern in practice than the backdoor criterion, which is why this practice's
hands-on estimator work focuses on backdoor adjustment, and frontdoor is covered
conceptually and via DoWhy's identification output rather than a from-scratch
estimator.

### Three Structures, One Shape, Opposite Rules

The pedagogical core of this practice is that **confounder, mediator, and
collider** all look like the same three-node shape (X - middle - Y) but demand
opposite actions:

- **Confounder** (`Z -> D`, `Z -> Y`): condition on it. It's the only way to close
  the backdoor path `D <- Z -> Y`.
- **Mediator** (`D -> M -> Y`): don't, if the *total* effect is what you want.
  Conditioning on M blocks part of the causal path itself, leaving only D's
  *direct* effect.
- **Collider** (`D -> C <- Y`): never condition on it (or any descendant of it).
  D and Y can be entirely unrelated causally, yet conditioning on their common
  effect C manufactures an association between them — bias created from nothing.

**M-bias** is the sharpest version of the collider trap: a variable M that *looks*
like a pre-treatment confounder (correlated with both D and Y) but is structurally
a collider of two unmeasured causes (`U1 -> D`, `U1 -> M`, `U2 -> M`, `U2 -> Y`).
An analyst who adds M "just to be safe" opens the collider path and biases an
estimate that was unbiased without it — the standard "controlling for more is not
safer" failure mode, and the reason **selection bias** (sample-selecting on a
collider, e.g. a survey that only reaches respondents above some threshold, rather
than regression-adjusting for it) produces the identical bias through a different
mechanism: selecting on a variable is d-separation-equivalent to conditioning on
it.

### Key Concepts

| Concept | Definition |
|---|---|
| **D-separation** | Graphical criterion: X and Y are d-separated by Z iff every path between them is blocked given Z. |
| **Backdoor path** | A path from treatment to outcome that starts with an arrow *into* the treatment — represents confounding, not the causal effect. |
| **Backdoor criterion** | Z is a valid adjustment set iff it contains no descendant of the treatment and blocks every backdoor path. |
| **Frontdoor criterion** | Identifies an effect via a fully-mediating, unconfounded-with-outcome variable, when no backdoor set exists. |
| **Confounder** | A common cause of treatment and outcome (`Z -> D`, `Z -> Y`); must be adjusted for. |
| **Mediator** | An intermediate cause on the causal path (`D -> M -> Y`); adjusting for it blocks part of the effect. |
| **Collider** | A common effect of two variables (`A -> C <- B`); adjusting for it (or selecting on it) creates bias. |
| **M-bias** | Bias induced by conditioning on a variable that is a collider of two unmeasured causes, despite looking like a plausible confounder. |
| **Structure discovery** | Learning a DAG's (equivalence class of) structure from data alone, e.g. via the PC algorithm's conditional-independence tests. |

### Where This Fits

Every technique elsewhere in this curriculum (RDD, synthetic control, panel fixed
effects, IV — practices 100-109) is a strategy for satisfying the backdoor
criterion (or working around its failure) in a specific empirical setting; this
practice is the graphical vocabulary all of them are stated in. The main
alternative to DAG-based reasoning is the potential-outcomes / Rubin causal model
framework (unconfoundedness, SUTVA, propensity scores) — mathematically equivalent
for a large class of problems, but expressed in terms of counterfactuals rather
than graphs. DAGs tend to make conditioning mistakes (mediators, colliders,
M-bias) visually obvious in a way that potential-outcomes notation does not,
which is precisely why this practice leads with them.

### DAG Drawing Decision

Every DAG in this practice (3-5 nodes, fixed pedagogical shapes) is drawn with
`xy.pyplot` alone: `ax.scatter` for nodes at hand-picked coordinates
(`src/datasets.py::LAYOUTS`), `ax.text` for labels, and `ax.annotate("", ...,
arrowprops=...)` for directed edges — all part of `xy.pyplot`'s
matplotlib-compatible 2-D Axes surface (see `02-answer-xy.md`), so no graph-layout
library or mermaid/markdown fallback is needed. This works because these graphs
are small and their layout is already known (a triangle or a diamond) — it would
not scale to an automatically-laid-out graph with dozens of nodes, which is the
scenario the frozen scaffold spec's mermaid/markdown fallback is aimed at.

### References

- Pearl, 1993, "Comment: Graphical Models, Causality and Intervention" (backdoor criterion): <https://doi.org/10.1214/ss/1177010894>
- Lauritzen, Dawid, Larsen & Leimer, 1990, "Independence Properties of Directed Markov Fields" (d-separation via moralization): <https://doi.org/10.1002/net.3230200503>
- Greenland, Pearl & Robins, 1999, "Causal Diagrams for Epidemiologic Research" (M-bias, selection bias): <https://www.jstor.org/stable/3703997>
- Spirtes & Glymour, 1991, "An Algorithm for Fast Recovery of Sparse Causal Graphs" (PC algorithm): <https://doi.org/10.1080/08839519108927921>
- DoWhy documentation, identification: <https://www.pywhy.org/dowhy/main/user_guide/causal_tasks/identifying_causal_effect/index.html>
- causal-learn documentation, PC algorithm: <https://causal-learn.readthedocs.io/en/latest/search_methods_index/Constraint-based%20causal%20discovery%20methods/PC.html>

## Description

Implement d-separation and the backdoor criterion from scratch over a small
graph representation, then use them to explain why a confounder, a mediator, and
a collider demand opposite treatment under conditioning — including simulated
M-bias and selection bias, where adding a "safe-looking" control variable makes a
correct model worse. Cross-check the from-scratch reasoning against `DoWhy`'s
identification API on the same graphs, then run `causal-learn`'s PC algorithm to
see what structure (and what ambiguity) can be recovered from data alone.

### What you'll learn

1. How d-separation turns "which conditional independencies does this DAG imply" into a checkable graph algorithm instead of a case-by-case path classification.
2. How the backdoor criterion formalizes "adjust for confounders, not mediators or colliders" as one graphical test.
3. Why a confounder, a mediator, and a collider need opposite adjustment decisions despite looking like the same three-node shape.
4. How M-bias and selection bias make a correct model worse by conditioning on (or selecting on) a disguised collider.
5. What a structure-discovery algorithm can and cannot recover from data alone, and why a known DAG still matters even with it.

## Instructions

### Phase 0: Setup (~5-10 min)

1. `uv sync`
2. `uv run nbstripout --install`
3. `uv run jupyter lab notebooks/_01_causal_dags.ipynb`

### Phase 1: D-Separation (~20 min) — `src/_01_dsep.py`

The parent-lookup and ancestor-walk helpers are fully scaffolded. The teaching
content is the moralize-and-separate algorithm itself.

1. **TODO #1 — `is_d_separated(graph, x, y, z)`**: implement d-separation via the
   moralized ancestral graph (build the ancestral subgraph over `{x, y} u z`,
   moralize it, drop directions, then check reachability with `z` removed).
   ~20-25 lines.

### Phase 2: The Backdoor Criterion (~15 min) — `src/_02_backdoor.py`

The descendant-walk and edge-removal helpers are fully scaffolded. The teaching
content is combining them with Phase 1's checker into the two-part criterion.

2. **TODO #1 — `is_valid_backdoor_set(graph, treatment, outcome, z)`**: implement
   the backdoor criterion (no descendant of `treatment` in `z`, and `z`
   d-separates `treatment` from `outcome` once `treatment`'s outgoing edges are
   removed). ~15 lines.

### Phase 3: Three Structures, One Estimator (~20 min) — `src/_03_structures.py`

The confounder/mediator/collider datasets (`src/datasets.py`) and the
scenario-comparison harness are fully scaffolded. The teaching content is the one
estimator every comparison in this practice reuses.

3. **TODO #1 — `estimate_effect_by_adjustment(df, treatment, outcome, adjust_for)`**:
   implement the regression-adjustment effect estimator (OLS coefficient on
   `treatment` from `outcome ~ treatment + adjust_for`). ~10-15 lines.

### Phase 4: M-Bias and Selection Bias (~15-20 min) — `src/_04_mbias_selection.py`

The M-bias dataset, the selection-bias-by-filtering demo, and the end-to-end
comparison are fully scaffolded. The teaching content is measuring how much a
collider (M-bias's M, or a plain collider) moves the estimate when conditioned on.

4. **TODO #1 — `collider_stratification_bias(df, treatment, outcome, collider)`**:
   compute the unadjusted and collider-adjusted estimates (via Phase 3's
   estimator) and their difference. ~10 lines.

### Phase 5: Cross-Check with DoWhy (no TODO) — `src/_05_dowhy_identification.py`

Run DoWhy's `CausalModel.identify_effect` on the same confounder/mediator/collider
graphs and compare its identified estimand to what Phases 1-2's from-scratch
checkers already told you.

### Phase 6: Structure Discovery with causal-learn (no TODO) — `src/_06_causal_learn_discovery.py`

Run the PC algorithm on simulated data with no graph given, and see that the
confounder/mediator/collider skeletons are indistinguishable from conditional
independence tests alone — the reason a known DAG (not just data) is needed for
identification.

### Phase 7: End-to-End Run (no TODO)

Run the notebook's final cells: the DAGs for all four scenarios drawn side by
side, the estimate-vs-truth plot across adjustment sets (the headline
"controlling for more is not safer" figure), and the collider-stratification bias
magnitude plot comparing the collider and M-bias scenarios.

### What to look for in the results

- Phase 1-2: every check in each module's `main()` should print `OK`, not
  `MISMATCH` — a mismatch means the algorithm, not the expected answer, is wrong.
- Phase 3: the confounder scenario's correctly-adjusted estimate should land near
  its true ATE; the mediator scenario's unadjusted estimate should land near the
  true *total* effect while its M-adjusted estimate lands near the smaller direct
  effect only; the collider scenario's unadjusted estimate should be near 0 while
  its C-adjusted estimate is visibly nonzero.
- Phase 4: both the M-bias and collider scenarios should show `bias != 0` when
  conditioning on the collider/M-bias node, even though the *unadjusted* estimate
  in both cases is already close to the truth — the adjusted one is the one that
  went wrong.
- Phase 5: DoWhy's identified estimand should name the same adjustment set (or
  "no valid backdoor set" for the collider case) that Phases 1-2 computed by hand.
- Phase 6: the PC algorithm should recover *an* undirected skeleton connecting the
  right variables, but should not be expected to orient every edge correctly or
  to distinguish the three structures from each other without extra assumptions.

## Motivation

- **AutoScheduler.AI relevance**: "does changing this scheduling knob actually
  improve the outcome, or are we both being driven by a third factor" is exactly
  the confounder-vs-collider judgment call this practice makes precise —
  especially dangerous when a "control variable" someone adds to a regression is
  secretly a collider or a mediator.
- **Senior -> Staff differentiator**: knowing to "control for confounders" is
  common; knowing that adding a plausible-looking covariate can silently make an
  estimate worse (M-bias, selection bias) — and being able to check *which* case
  you're in graphically instead of by intuition — is not.
- **Generalises beyond econometrics**: the same three-structure reasoning governs
  observational ML feature selection (is this feature a collider on the label?),
  A/B test interpretation (is this segment a mediator of the treatment?), and any
  "let's just add this covariate" decision in an analysis pipeline.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Setup** | `uv sync` | Install Python dependencies. |
| | `uv run nbstripout --install` | Register the git filter that strips notebook outputs on commit. |
| | `uv run jupyter lab notebooks/_01_causal_dags.ipynb` | Open the practice notebook. |
| **Phase 1** | `uv run python -m src._01_dsep` | Run the worked d-separation checks on all four scenario graphs. |
| **Phase 2** | `uv run python -m src._02_backdoor` | Run the worked backdoor-criterion checks. |
| **Phase 3** | `uv run python -m src._03_structures` | Compare correct vs. wrong adjustment on the confounder/mediator/collider datasets. |
| **Phase 4** | `uv run python -m src._04_mbias_selection` | Compute collider-stratification and selection-bias magnitudes. |
| **Phase 5** | `uv run python -m src._05_dowhy_identification` | Print DoWhy's identified estimand for each scenario. |
| **Phase 6** | `uv run python -m src._06_causal_learn_discovery` | Run the PC algorithm's structure discovery on simulated data. |
| **Cleanup** | `python clean.py` | Remove caches, checkpoints, venv, generated outputs. |

## Notes

_(populated during the practice.)_

## State

`not-started`
