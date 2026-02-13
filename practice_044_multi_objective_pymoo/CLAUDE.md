# Practice 044: Multi-Objective Optimization — pymoo

## Technologies

- **pymoo** — Multi-objective Optimization in Python. Framework providing state-of-the-art MOO algorithms (NSGA-II, NSGA-III, MOEA/D), benchmark problems (ZDT, DTLZ), performance indicators (hypervolume, IGD), and decision-making tools (pseudo-weights, high trade-off points). Developed by the Computational Optimization and Innovation (COIN) Lab at Michigan State University.
- **NumPy** — Array operations for decision variables, objective values, and Pareto front manipulation.
- **Matplotlib** — Visualization of 2D/3D Pareto fronts, convergence curves, and decision-space plots.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- pymoo >= 0.6 (algorithms, problems, indicators, MCDM)
- NumPy >= 1.26, Matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### What pymoo Is and the Problem It Solves

**pymoo** is a comprehensive Python framework for multi-objective optimization. It provides production-ready implementations of the most important evolutionary multi-objective algorithms, benchmark test problems, performance metrics, and multi-criteria decision-making tools — all in a unified API.

The problem pymoo solves: implementing MOO algorithms from scratch (as in practice 038 with NSGA-II in Rust) is educational but impractical for real applications. pymoo gives you battle-tested implementations with correct operator tuning, performance optimizations, and a rich ecosystem of utilities. You focus on problem formulation and analysis rather than algorithm internals.

### Recap: Multi-Objective Optimization Formulation

The general multi-objective optimization problem (MOP):

```
minimize    F(x) = (f_1(x), f_2(x), ..., f_M(x))
subject to  g_j(x) <= 0,     j = 1, ..., J     (inequality constraints)
            h_k(x) = 0,      k = 1, ..., K     (equality constraints)
            x_l <= x <= x_u                      (bound constraints)
```

Where `x` is a vector of `n` decision variables, `M` is the number of objectives, and `J + K` is the number of constraints.

### Recap: Pareto Dominance and Pareto Front

From practice 038 (implemented from scratch in Rust):

- **Pareto dominance:** Solution A dominates B (A ≺ B) iff A is at least as good on all objectives and strictly better on at least one.
- **Pareto optimal set:** The set of all solutions not dominated by any other feasible solution.
- **Pareto front (PF):** The image of the Pareto optimal set in objective space — the boundary of achievable trade-offs.

The goal of multi-objective optimization is to find a set of solutions that: (1) converges to the true Pareto front, and (2) is well-distributed across the front (diversity).

### NSGA-II: Non-dominated Sorting Genetic Algorithm II

NSGA-II (Deb et al., 2002) is the most widely used multi-objective evolutionary algorithm. You implemented it from scratch in practice 038. Key mechanisms:

1. **Non-dominated sorting:** Partition the population into fronts F_1, F_2, ... where F_1 contains all non-dominated solutions, F_2 the non-dominated solutions after removing F_1, etc.
2. **Crowding distance:** Within each front, measure how isolated each solution is — the sum of normalized distances to its nearest neighbors on each objective. Higher crowding distance = more isolated = more valuable for diversity.
3. **Selection:** Prefer solutions from lower-ranked fronts. Among solutions in the same front, prefer higher crowding distance.

NSGA-II works well for **2-3 objectives**. It degrades for many-objective problems (4+) because almost all solutions become non-dominated (the proportion of non-dominated solutions grows exponentially with the number of objectives), making non-dominated sorting and crowding distance ineffective at distinguishing quality.

### NSGA-III: Reference-Point-Based Selection for Many-Objective Problems

NSGA-III (Deb & Jain, 2014) replaces crowding distance with **reference-point-based selection** to handle many-objective problems (3+ objectives):

1. **Reference directions:** A set of well-distributed reference points on the unit simplex in objective space. These define "targets" for diversity maintenance. For M objectives with p partitions, the Das-Dennis method generates C(M+p-1, p) uniformly spaced points.
2. **Normalization:** The current population's objective values are normalized using ideal (minimum per objective) and nadir (maximum per objective) points, mapping them to [0, 1].
3. **Association:** Each population member is associated with its closest reference direction (by perpendicular distance).
4. **Niche-preserving selection:** When filling the next generation, prefer solutions associated with reference directions that have the fewest associated members. This ensures uniform spread across all reference directions.

Key difference from NSGA-II: NSGA-III explicitly maintains diversity through reference directions rather than relying on crowding distance in objective space. This scales to 10+ objectives where crowding distance fails.

### MOEA/D: Multi-Objective Evolutionary Algorithm by Decomposition

MOEA/D (Zhang & Li, 2007) takes a fundamentally different approach from NSGA-II/III. Instead of evolving a single population with Pareto-based selection, it **decomposes the multi-objective problem into N scalar subproblems** and optimizes them simultaneously:

1. **Weight vectors:** Define N weight vectors λ_1, ..., λ_N uniformly distributed on the unit simplex. Each weight vector defines a scalar subproblem.
2. **Decomposition approaches:**
   - **Weighted sum:** `g(x|λ) = Σ λ_i * f_i(x)` — simple but cannot find solutions on non-convex parts of the PF.
   - **Tchebycheff:** `g(x|λ) = max_i { λ_i * |f_i(x) - z*_i| }` — where z* is the ideal point. Can find solutions on non-convex fronts. The most commonly used approach.
   - **PBI (Penalty-based Boundary Intersection):** `g(x|λ) = d_1 + θ * d_2` — where d_1 is the distance along the weight vector direction and d_2 is the perpendicular distance. Balances convergence and diversity.
3. **Neighborhood:** Each subproblem has a neighborhood of T closest subproblems (by weight vector distance). Mating partners are selected from neighbors, and offspring replace the worst in the neighborhood.

MOEA/D is typically faster than NSGA-III because it replaces Pareto-based sorting with simple scalar comparisons. However, it requires careful setting of neighborhood size and decomposition method.

### Reference Directions: Das-Dennis and Energy-Based

Reference directions are critical for both NSGA-III and MOEA/D. Two main generation methods:

**Das-Dennis (structured simplex lattice):**
- Places points on a regular grid on the M-dimensional unit simplex.
- For M objectives and p partitions: generates C(M+p-1, p) points.
- Simple and deterministic, but the number of points grows combinatorially with M and p.
- Works well for 2-5 objectives. For higher dimensions, two-layer approaches (inner + outer) are used.

**Riesz s-Energy (energy-based):**
- Distributes N points on the unit simplex by minimizing the Riesz s-energy (repulsive force analogy).
- Allows arbitrary N (not constrained to C(M+p-1, p) values).
- Produces well-distributed points for any dimensionality.
- More flexible but requires iterative optimization to generate.

### Performance Metrics

How to compare different algorithms or parameter settings:

| Metric | Measures | Needs True PF? | Direction |
|--------|----------|-----------------|-----------|
| **Hypervolume (HV)** | Volume of objective space dominated by solutions, bounded by a reference point | No (but needs ref point) | Higher = better |
| **Generational Distance (GD)** | Average distance from each obtained solution to the nearest true PF point | Yes | Lower = better |
| **Inverted GD (IGD)** | Average distance from each true PF point to the nearest obtained solution | Yes | Lower = better |
| **IGD+** | Modified IGD using Pareto-compliant distance | Yes | Lower = better |
| **Spread (Delta)** | Measures distribution uniformity of solutions | Yes | Lower = better |

**Hypervolume** is the only Pareto-compliant metric that does not require the true Pareto front — it is the gold standard but computationally expensive (exponential in the number of objectives).

**IGD** is the most practical metric when the true PF is known (benchmark problems). It measures both convergence (solutions close to PF) and diversity (covers the entire PF). A solution set with low IGD is both converged and well-spread.

### Constraint Handling in Evolutionary MOO

pymoo uses the **constraint violation** approach:
- Constraints are defined as `g(x) <= 0` (inequality) and `h(x) = 0` (equality).
- The total constraint violation `CV(x) = Σ max(0, g_j(x)) + Σ |h_k(x)|` measures infeasibility.
- **Feasible solutions always dominate infeasible ones.** Among infeasible solutions, prefer the one with smaller constraint violation.

In pymoo, you specify constraints via `n_ieq_constr` and write constraint values to `out["G"]` in the `_evaluate` method. Negative values of G indicate satisfaction (g <= 0 is satisfied), positive values indicate violation.

### Decision Making: From Pareto Front to a Single Solution

After finding the Pareto front, a decision-maker must select one solution. pymoo provides several approaches:

- **Pseudo-weights:** For each solution on the PF, compute a "pseudo-weight" vector that indicates the relative importance implied by that solution's trade-off position. To select a solution, provide desired weights and find the closest match. This answers: "which PF solution best matches my preference?"
- **High trade-off points:** Identify solutions where the PF has the highest curvature (the "knee" points). These represent the best compromise — small changes in one objective cause large changes in another. Mathematically, these are points where the local slope of the PF changes most rapidly.
- **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution):** Rank solutions by their geometric distance to the ideal point (best on all objectives) and worst point (worst on all objectives). The best solution is closest to ideal and farthest from worst.

### pymoo Architecture

pymoo follows a modular architecture:

```
pymoo API
├── Problem Layer
│   ├── ElementwiseProblem ........ Evaluate one solution at a time (simpler)
│   ├── Problem ................... Evaluate entire population at once (faster)
│   └── get_problem("zdt1") ...... Load benchmark problems
│
├── Algorithm Layer
│   ├── NSGA2 .................... 2-3 objectives, crowding distance
│   ├── NSGA3 .................... Many-objective, reference directions
│   ├── MOEAD .................... Decomposition-based
│   └── MixedVariableGA .......... Mixed integer/continuous/choice variables
│
├── Minimize
│   ├── minimize(problem, algorithm, termination) ... Main entry point
│   └── Result ................... res.F (objectives), res.X (variables)
│
├── Indicators
│   ├── Hypervolume .............. from pymoo.indicators.hv
│   ├── IGD ...................... from pymoo.indicators.igd
│   └── GD ....................... from pymoo.indicators.gd
│
├── Reference Directions
│   └── get_reference_directions("das-dennis", n_obj, n_partitions)
│
└── Decision Making (MCDM)
    ├── PseudoWeights ............ from pymoo.mcdm.pseudo_weights
    └── HighTradeoffPoints ....... from pymoo.mcdm.high_tradeoff
```

**Key API patterns:**
- Problems inherit from `ElementwiseProblem`, implement `_evaluate(self, x, out, *args, **kwargs)`.
- Objectives go to `out["F"]` (list/array of M values), constraints to `out["G"]` (list/array of J values).
- `minimize(problem, algorithm, termination, seed=..., verbose=...)` returns a `Result` object.
- `result.F` = objective values of Pareto-optimal solutions, `result.X` = corresponding decision variables.
- Performance indicators: instantiate with reference data, call `.do(F)` to compute.

### Benchmark Test Problems

**ZDT family** (Zitzler, Deb, Thiele) — bi-objective problems:
- **ZDT1:** Convex Pareto front. 30 variables. PF: f_2 = 1 - sqrt(f_1).
- **ZDT3:** Disconnected Pareto front (5 segments). 30 variables. Tests algorithm's ability to find disconnected regions.

**DTLZ family** (Deb, Thiele, Lautmanns, Zitzler) — scalable to any number of objectives:
- **DTLZ2:** Spherical Pareto front. n_var = n_obj + k - 1 (typically k=10). PF lies on the unit sphere: Σ f_i^2 = 1. Standard benchmark for many-objective algorithms.

### Where pymoo Fits in the Ecosystem

```
Multi-Objective Optimization
├── From-scratch implementations (educational)
│   └── Practice 038 — NSGA-II in Rust (this repo)
│
├── Python frameworks (production)
│   ├── pymoo ............... Most comprehensive MOO framework ← this practice
│   ├── Platypus ............ Pure Python MOO algorithms
│   ├── DEAP ................ General evolutionary computation
│   └── jMetalPy ............ Port of Java jMetal framework
│
├── Solver-based (exact for linear/convex MOO)
│   ├── CVXPY ............... Scalarized convex MOO (practice 041a)
│   ├── Pyomo ............... Epsilon-constraint method (practice 040b)
│   └── OR-Tools ............ Weighted-sum LP/MIP
│
└── Commercial/Specialized
    ├── modeFRONTIER ........ Engineering MOO (CAE integration)
    └── MATLAB gamultiobj ... MATLAB's NSGA-II implementation
```

pymoo's advantages: comprehensive algorithm collection, built-in benchmarks, performance indicators, decision-making tools, and active maintenance. It is the standard choice for MOO research and applications in Python.

## Description

Apply pymoo's multi-objective optimization framework to solve bi-objective, many-objective, and constrained problems. Progress from basic NSGA-II usage through advanced algorithms (NSGA-III, MOEA/D) to constrained engineering design and decision-making analysis.

### What you'll build

1. **NSGA-II on benchmarks** — Define a custom bi-objective problem, run NSGA-II, extract and plot the Pareto front against the analytical solution.
2. **Many-objective with NSGA-III & MOEA/D** — Solve a 3-objective DTLZ2 problem with reference-direction-based algorithms and compare results.
3. **Constrained engineering design** — Formulate a constrained multi-objective problem with inequality constraints and mixed variables.
4. **Performance analysis & decision making** — Compute hypervolume and IGD, track convergence, and apply TOPSIS/pseudo-weights to select final solutions.

## Instructions

### Phase 1: pymoo Basics — Problem Definition & NSGA-II (~25 min)

**File:** `src/phase1_nsga2_basics.py`

This phase teaches the fundamental pymoo workflow: define a problem by subclassing `ElementwiseProblem`, configure NSGA-II, run `minimize()`, and extract the Pareto front. You will implement a custom bi-objective problem and compare the obtained front against the known analytical Pareto front of ZDT1.

**What you implement:**
- `CustomBiObjectiveProblem` — Subclass `ElementwiseProblem`, define `_evaluate` with two conflicting objectives and bound constraints. This is the core pymoo pattern — every problem you'll ever solve starts here.
- `run_nsga2_on_custom(pop_size, n_gen)` — Instantiate NSGA-II with given population size and generations, call `minimize()`, return the result. This teaches the algorithm-problem-minimize triangle.
- `run_nsga2_on_zdt1(pop_size, n_gen)` — Same workflow on the built-in ZDT1 problem. Compare obtained PF with the analytical PF from `problem.pareto_front()`.

**Why it matters:** This is the entry point for all pymoo usage. The `ElementwiseProblem` pattern, `minimize()` call, and `result.F`/`result.X` access are used in every subsequent phase. Comparing with analytical PF teaches how to evaluate solution quality visually.

### Phase 2: Advanced Algorithms — NSGA-III & MOEA/D (~25 min)

**File:** `src/phase2_advanced_algorithms.py`

This phase introduces algorithms designed for many-objective problems (3+ objectives). NSGA-II's crowding distance degrades with more objectives because almost all solutions become non-dominated. NSGA-III and MOEA/D address this differently: NSGA-III uses reference-point association, MOEA/D decomposes into scalar subproblems.

**What you implement:**
- `run_nsga3_on_dtlz2(n_partitions, n_gen)` — Generate Das-Dennis reference directions for 3 objectives, configure NSGA-III, solve DTLZ2. The reference directions define the target distribution on the Pareto front.
- `run_moead_on_dtlz2(n_partitions, n_gen, n_neighbors)` — Configure MOEA/D with Tchebycheff decomposition and neighborhood size, solve the same DTLZ2 problem. Compare the result distribution with NSGA-III.

**Why it matters:** Many-objective optimization is a distinct challenge from bi-objective. Reference directions are the key concept — they appear in NSGA-III, MOEA/D, and most modern many-objective algorithms. Understanding how Das-Dennis points tile the simplex and how algorithms use them for diversity is essential for applying MOO beyond toy problems.

### Phase 3: Constraints & Mixed Variables (~25 min)

**File:** `src/phase3_constraints_mixed.py`

This phase adds real-world complexity: inequality constraints that restrict the feasible region, and mixed-variable problems where some decision variables are integers. The key concept is constraint handling in evolutionary algorithms — feasible solutions always dominate infeasible ones, and among infeasible solutions, lower constraint violation is preferred.

**What you implement:**
- `WeldedBeamProblem` — Define the classic welded beam design problem as a custom `ElementwiseProblem` with 4 continuous design variables, 2 objectives (cost, deflection), and 4 inequality constraints (stress, deflection, buckling, geometric). This is a standard engineering benchmark.
- `run_constrained_optimization(pop_size, n_gen)` — Solve the welded beam problem with NSGA-II, extract feasible Pareto-optimal solutions, analyze constraint satisfaction.

**Why it matters:** Real engineering problems are always constrained. Learning how pymoo handles constraints (via `n_ieq_constr` and `out["G"]`) and how to verify constraint satisfaction in the Pareto front is critical for practical applications.

### Phase 4: Performance Metrics & Decision Making (~25 min)

**File:** `src/phase4_metrics_decisions.py`

This phase teaches how to quantitatively evaluate MOO results and how to select a single solution from the Pareto front. You will compute hypervolume and IGD metrics, track convergence across generations, and apply decision-making methods (pseudo-weights, high trade-off points) to choose a final solution.

**What you implement:**
- `compute_quality_metrics(F, pf_true, ref_point)` — Compute hypervolume and IGD for a solution set. Hypervolume measures dominated volume (higher = better), IGD measures coverage of the true PF (lower = better).
- `track_convergence(problem, algorithm, n_gen)` — Run the algorithm with a callback that records HV at each generation. Return the convergence history for plotting.
- `select_solution_pseudo_weights(F, weights)` — Use pymoo's `PseudoWeights` to find the Pareto-optimal solution that best matches the decision-maker's preference weights.
- `find_knee_points(F)` — Use pymoo's `HighTradeoffPoints` to identify knee points on the Pareto front — solutions where the trade-off slope changes most rapidly.

**Why it matters:** Metrics are how you compare algorithms, tune parameters, and validate results. Without them, you are just looking at scatter plots. Decision making closes the loop: MOO gives you a front, but you ship one solution. Pseudo-weights and knee-point analysis are the standard approaches for making that final selection.

## Motivation

After implementing NSGA-II from scratch in Rust (practice 038), this practice shows how to use a production-grade multi-objective optimization framework. pymoo is the standard Python MOO library, used in engineering design, machine learning hyperparameter tuning, supply chain optimization, and research. It provides algorithms (NSGA-II, NSGA-III, MOEA/D), benchmarks (ZDT, DTLZ), metrics (hypervolume, IGD), and decision-making tools — everything needed for a complete MOO workflow.

This practice covers the full cycle: problem definition, algorithm selection, result evaluation with metrics, and final solution selection — the skills needed to apply MOO in any real-world context.

## Commands

All commands are run from the `practice_044_multi_objective_pymoo/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (pymoo, NumPy, Matplotlib) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_nsga2_basics.py` | Phase 1: Custom problem definition, NSGA-II, Pareto front vs analytical |
| `uv run python src/phase2_advanced_algorithms.py` | Phase 2: NSGA-III and MOEA/D on 3-objective DTLZ2, 3D Pareto front |
| `uv run python src/phase3_constraints_mixed.py` | Phase 3: Constrained welded beam problem, constraint satisfaction analysis |
| `uv run python src/phase4_metrics_decisions.py` | Phase 4: Hypervolume, IGD, convergence tracking, pseudo-weights, knee points |

## References

- [pymoo Documentation](https://pymoo.org/) — Official documentation, tutorials, and API reference
- [pymoo Getting Started — Part II: Multi-objective Optimization](https://pymoo.org/getting_started/part_2.html) — Problem definition and NSGA-II tutorial
- [pymoo Getting Started — Part III: Decision Making](https://pymoo.org/getting_started/part_3.html) — Pseudo-weights, high trade-off analysis
- [pymoo Getting Started — Part IV: Convergence Analysis](https://pymoo.org/getting_started/part_4.html) — Performance indicators and convergence tracking
- [pymoo Problem Definition](https://pymoo.org/problems/definition.html) — ElementwiseProblem, Problem, _evaluate API
- [pymoo NSGA-II](https://pymoo.org/algorithms/moo/nsga2.html) — NSGA-II algorithm configuration
- [pymoo NSGA-III](https://pymoo.org/algorithms/moo/nsga3.html) — NSGA-III with reference directions
- [pymoo MOEA/D](https://pymoo.org/algorithms/moo/moead.html) — Decomposition-based algorithm
- [pymoo Reference Directions](https://pymoo.org/misc/reference_directions.html) — Das-Dennis and energy-based methods
- [pymoo Performance Indicators](https://pymoo.org/misc/indicators.html) — HV, GD, IGD, IGD+
- [pymoo Constraint Handling](https://www.pymoo.org/constraints/index.html) — Inequality and equality constraints
- [pymoo MCDM](https://www.pymoo.org/mcdm/index.html) — Multi-criteria decision making
- [pymoo Mixed Variables](https://pymoo.org/customization/mixed.html) — MixedVariableGA, Choice, Integer, Real
- [Deb et al. (2002) — NSGA-II Paper](https://doi.org/10.1109/4235.996017) — Original NSGA-II algorithm
- [Deb & Jain (2014) — NSGA-III Paper](https://doi.org/10.1109/TEVC.2013.2281535) — Reference-point-based NSGA-III
- [Zhang & Li (2007) — MOEA/D Paper](https://doi.org/10.1109/TEVC.2007.892759) — Decomposition-based MOEA/D
- [Blank & Deb (2020) — pymoo Paper](https://doi.org/10.1109/ACCESS.2020.2990567) — pymoo framework paper

## State

`not-started`
