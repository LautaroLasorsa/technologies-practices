# Practice 038: Multi-Objective Optimization — NSGA-II from Scratch

## Technologies

- **Rust** — Systems language with ownership model, zero-cost abstractions
- **rand 0.8** — Random number generation for population initialization, crossover, mutation

## Stack

- Rust (cargo)

## Theoretical Context

### What Multi-Objective Optimization Is

**Multi-objective optimization (MOO)** is the problem of simultaneously optimizing two or more conflicting objective functions. Unlike single-objective optimization, where there is one clear best solution, MOO produces a **set** of trade-off solutions because improving one objective typically worsens another.

The general formulation:

```
minimize    (f_1(x), f_2(x), ..., f_M(x))
subject to  x in S  (feasible set)
```

Where `x` is a decision vector, `f_1, ..., f_M` are M objective functions, and `S` is the feasible region defined by constraints.

**Examples of conflicting objectives in practice:**
- **Engineering design:** minimize weight AND maximize strength (lighter structures are weaker)
- **Machine learning:** minimize training loss AND minimize model complexity (more complex models fit better)
- **Supply chain:** minimize cost AND minimize delivery time (faster shipping costs more)
- **Finance:** maximize return AND minimize risk (higher returns require more risk)
- **HFT:** minimize latency AND maximize throughput (processing more data takes longer)

The key insight: there is no single solution that simultaneously minimizes all objectives. Instead, we seek the set of optimal trade-offs.

### Pareto Dominance

**Pareto dominance** is the fundamental ordering relation for comparing solutions in multi-objective optimization.

**Definition:** Solution A **dominates** solution B (written A ≺ B) if and only if:
1. A is **at least as good** as B on **every** objective: `f_i(A) <= f_i(B)` for all i
2. A is **strictly better** than B on **at least one** objective: `f_j(A) < f_j(B)` for some j

If neither A dominates B nor B dominates A, they are **non-dominated** with respect to each other (incomparable). This happens when A is better on some objectives but worse on others — they represent different trade-offs.

**Example (minimize both f1 and f2):**
- A = (1, 5), B = (3, 7): A dominates B (better on both)
- A = (1, 5), C = (2, 3): Neither dominates — A is better on f1, C is better on f2
- A = (1, 5), D = (1, 5): Neither dominates — equal on all (A does not strictly improve any)

Pareto dominance induces a **partial order** on the solution space — not all pairs of solutions are comparable. This is fundamentally different from single-objective optimization where the objective function induces a total order.

### Pareto Front (Pareto-Optimal Set)

The **Pareto front** (or Pareto-optimal set) is the set of all feasible solutions that are **not dominated by any other feasible solution**. In objective space, it forms a curve (2D) or surface (3D+) representing the best achievable trade-offs.

```
f2 |
   |  x          (dominated — some Pareto solution beats it on both objectives)
   |   \
   |    *---*     (* = Pareto front — no solution is better on ALL objectives)
   |         \
   |          *
   |           \
   |            *
   |_____________\___ f1
```

**Properties of the Pareto front:**
- Every solution on the front is optimal in the sense that no other solution improves one objective without worsening another.
- The front can be **convex** (ZDT1), **non-convex** (ZDT2), or **disconnected** (ZDT3).
- The **shape** of the Pareto front reveals the nature of the trade-off: a steep section means large f2 cost for small f1 gain, and vice versa.
- In practice, we find a finite approximation of the front. The goal is to find a set that is (1) close to the true front and (2) well-spread across it.

### Scalarization Approaches (and Their Limitations)

Before NSGA-II, the common approach to MOO was **scalarization** — converting multiple objectives into a single objective:

**Weighted Sum Method:**
```
minimize    w_1 * f_1(x) + w_2 * f_2(x) + ... + w_M * f_M(x)
```
Choose weights, solve a single-objective problem, get one Pareto-optimal point. Repeat with different weights.

**Limitations:**
- Each run produces only ONE point on the Pareto front — need many runs with different weights.
- **Cannot find non-convex Pareto regions.** The weighted sum method can only find solutions on the convex hull of the Pareto front. For ZDT2 (concave front), entire regions are inaccessible regardless of weights.
- Weight-to-Pareto-point mapping is nonlinear and unpredictable — uniformly spaced weights do NOT produce uniformly spaced Pareto points.

**Epsilon-Constraint Method:**
```
minimize    f_1(x)
subject to  f_2(x) <= epsilon_2, ..., f_M(x) <= epsilon_M
```
Optimize one objective, constrain the others. Can find non-convex points, but requires choosing epsilon values carefully and running once per desired point.

**Why evolutionary algorithms (NSGA-II) are preferred:**
- Find an entire approximation of the Pareto front in a **single run**.
- Population-based: maintain many solutions simultaneously, naturally exploring different trade-offs.
- No convexity assumptions — work on any Pareto front shape.
- Naturally parallelizable (evaluate population members independently).

### NSGA-II (Non-dominated Sorting Genetic Algorithm II)

NSGA-II, introduced by Deb, Pratap, Agarwal, and Meyarivan in 2002, is the **most cited multi-objective evolutionary algorithm**. It remains the standard baseline and is widely used in practice due to its simplicity, efficiency, and effectiveness.

**Key innovations over NSGA (1995):**
1. **Fast non-dominated sorting** — O(MN^2) instead of O(MN^3)
2. **Crowding distance** — replaces sharing function (no sigma parameter to tune)
3. **Elitism** — parent+offspring combined selection guarantees non-dominated solutions survive

#### 1. Fast Non-Dominated Sorting

The population is partitioned into **fronts** F_1, F_2, F_3, ...:
- **F_1** (rank 0): all non-dominated solutions in the population
- **F_2** (rank 1): all solutions dominated ONLY by solutions in F_1
- **F_3** (rank 2): all solutions dominated only by F_1 and F_2
- And so on until every solution is assigned a front

**Algorithm:**
```
For each individual p:
    S_p = set of individuals that p dominates
    n_p = count of individuals that dominate p

F_1 = { p : n_p == 0 }   (non-dominated individuals)

k = 1
While F_k is not empty:
    F_{k+1} = {}
    For each p in F_k:
        For each q in S_p:
            n_q -= 1
            If n_q == 0:
                F_{k+1}.add(q)
    k += 1
```

**Complexity:** O(M * N^2) where M = number of objectives, N = population size. The pairwise comparison is O(M) per pair, and there are O(N^2) pairs.

#### 2. Crowding Distance

Within each front, solutions are ranked by **crowding distance** — a measure of how isolated a solution is in objective space. Higher crowding distance means the solution is in a less populated region of the front, contributing to diversity.

**Algorithm for one front:**
```
1. Initialize cd[i] = 0 for all i in front
2. For each objective m:
   a. Sort front by objective m
   b. cd[first] = cd[last] = INFINITY   (boundary solutions always preserved)
   c. For i = 1 to len-2:
      cd[i] += (f_m[i+1] - f_m[i-1]) / (f_m_max - f_m_min)
```

The crowding distance is the sum of normalized distances to the nearest neighbors on each objective. It approximates the size of the largest cuboid enclosing the solution without including any other solution (the "crowding" around it).

**Why it matters:** Without crowding distance, the algorithm would converge to a few clustered points on the Pareto front. Crowding distance ensures the population spreads across the entire front, giving the decision-maker a comprehensive view of trade-offs.

#### 3. Selection Operator (Crowded Tournament)

Binary tournament selection using **crowded comparison**:
- If solutions have **different ranks**: prefer lower rank (closer to Pareto front)
- If solutions have **same rank**: prefer higher crowding distance (more isolated)

This creates selection pressure toward (1) better Pareto fronts and (2) diverse spread within each front.

#### 4. Elitism via (mu + lambda) Strategy

The key to NSGA-II's effectiveness:
```
1. Parent population P (size N)
2. Create offspring Q (size N) via selection + crossover + mutation
3. Combine R = P union Q (size 2N)
4. Non-dominated sort R into fronts F_1, F_2, ...
5. Fill next generation P' by adding entire fronts F_1, F_2, ... until adding the next front would exceed N
6. For the last partial front: compute crowding distance, sort by CD descending, take the top individuals to fill exactly N
```

This ensures: (a) good solutions from the parent generation survive (elitism), (b) the best 2N → N selection uses both Pareto quality and diversity.

### Genetic Operators for Real-Valued Problems

**Simulated Binary Crossover (SBX):**
SBX (Deb & Agrawal, 1995) creates offspring that have the same spread as binary-coded single-point crossover. It uses a distribution index `eta_c` — larger values produce children closer to parents.

```
For each variable i:
    u = random(0, 1)
    if u <= 0.5:
        beta = (2*u)^(1/(eta_c+1))
    else:
        beta = (1/(2*(1-u)))^(1/(eta_c+1))
    child1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
    child2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
    Clip to bounds.
```

**Polynomial Mutation:**
Per-variable mutation with distribution index `eta_m`:

```
For each variable i with probability p_m:
    u = random(0, 1)
    if u < 0.5:
        delta = (2*u)^(1/(eta_m+1)) - 1
    else:
        delta = 1 - (2*(1-u))^(1/(eta_m+1))
    x[i] = x[i] + delta * (upper[i] - lower[i])
    Clip to bounds.
```

### ZDT Test Problems

The ZDT (Zitzler-Deb-Thiele) test suite is the standard benchmark for bi-objective optimization. All have 30 decision variables, 2 objectives, and known true Pareto fronts:

| Problem | Pareto Front Shape | True Front | Difficulty |
|---------|-------------------|------------|------------|
| **ZDT1** | Convex | f2 = 1 - sqrt(f1) | Baseline — easy for most algorithms |
| **ZDT2** | Non-convex (concave) | f2 = 1 - f1^2 | Weighted sum fails here |
| **ZDT3** | Disconnected | f2 = 1 - sqrt(f1) - f1*sin(10*pi*f1) | Multiple disconnected segments |

All ZDT problems have the form:
```
f_1(x) = x_1
f_2(x) = g(x_2, ..., x_n) * h(f_1, g)
```
Where `g` captures distance from the Pareto front (optimal when g=1), and `h` determines the front shape.

### Quality Metrics

**Hypervolume (HV):**
The volume (area in 2D) of objective space dominated by the Pareto front approximation, bounded by a reference point. Higher is better. The ONLY metric that is **Pareto-compliant** — a front A with higher HV than front B is guaranteed to not be dominated by B.

```
2D Hypervolume:
    Sort front by f1 ascending
    HV = sum of rectangles from each point to reference point
```

**Inverted Generational Distance (IGD):**
Average distance from each point on the true Pareto front to the nearest point on the approximation. Lower is better. Measures both convergence and diversity but requires knowing the true front.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Multi-objective optimization** | Simultaneously optimizing multiple conflicting objectives |
| **Pareto dominance** | Solution A dominates B if A is no worse on all objectives and strictly better on at least one |
| **Pareto front** | Set of all non-dominated solutions — the optimal trade-off surface |
| **Non-dominated sorting** | Partition population into fronts: F1 (non-dominated), F2 (dominated only by F1), etc. |
| **Crowding distance** | Diversity metric: sum of normalized distances to nearest neighbors per objective |
| **Crowded comparison** | Selection criterion: prefer (1) lower rank, then (2) higher crowding distance |
| **Elitism** | Combine parent+offspring, select best N from 2N using fronts + crowding |
| **SBX crossover** | Simulated Binary Crossover for real-valued variables |
| **Polynomial mutation** | Bounded perturbation of real-valued variables |
| **Hypervolume** | Volume of objective space dominated by the Pareto front — higher is better |
| **IGD** | Average distance from true front to approximation — lower is better |
| **ZDT test suite** | Standard 2-objective test problems with known Pareto fronts |
| **Distribution index** | Parameter controlling spread of offspring around parents (SBX: eta_c, mutation: eta_m) |

### Where NSGA-II Fits in the Multi-Objective Landscape

```
Multi-Objective Optimization
├── Classical (scalarization)
│   ├── Weighted Sum (only convex fronts)
│   └── Epsilon-Constraint (one point per run)
├── Evolutionary (population-based, one run → entire front)
│   ├── NSGA-II (2002) — fast sorting + crowding distance ← THIS PRACTICE
│   ├── NSGA-III (2014) — reference-point based (many objectives, M > 3)
│   ├── MOEA/D (2007) — decomposition into scalar subproblems
│   └── SPEA2 (2001) — strength-based fitness + archive
├── Indicator-based
│   └── SMS-EMOA, IBEA — optimize hypervolume directly
└── Libraries
    └── pymoo (Python), jMetal (Java), DEAP (Python), platypus (Python)
```

NSGA-II is the baseline that everything else is compared against. Understanding it is prerequisite to NSGA-III, MOEA/D, and any production multi-objective work.

## Description

Implement NSGA-II from scratch in Rust: non-dominated sorting, crowding distance, tournament selection with Pareto rank, and the full evolutionary loop. Test on ZDT benchmark problems and measure Pareto front quality using the hypervolume metric.

### What you'll build

1. **Non-dominated sorting** — Rank a population into Pareto fronts using Deb's fast algorithm
2. **Crowding distance** — Compute diversity metric within each front
3. **Full NSGA-II** — Complete evolutionary loop with tournament selection, SBX crossover, polynomial mutation, and elitist (mu+lambda) selection
4. **Benchmark suite** — Run NSGA-II on ZDT1, ZDT2, ZDT3, compute hypervolume, and visualize Pareto fronts (text-based)

## Instructions

### Phase 1: Non-Dominated Sorting (~25 min)

**File:** `src/nondominated_sort.rs`

This phase teaches Pareto dominance and the fast non-dominated sorting algorithm — the core ranking mechanism that replaces single-objective comparison. You implement the O(MN^2) algorithm that partitions a population into fronts.

**What you implement:**
- `fast_nondominated_sort()` — Deb's fast algorithm: compute domination counts and dominated sets, then extract fronts layer by layer.

**Why it matters:** Non-dominated sorting is called every generation to rank the combined parent+offspring population. Its correctness determines whether the algorithm converges to the true Pareto front. Understanding the front structure (F1, F2, ...) is essential for understanding why NSGA-II works — selection pressure comes from front rank.

### Phase 2: Crowding Distance (~20 min)

**File:** `src/crowding.rs`

This phase teaches the diversity preservation mechanism. Without crowding distance, NSGA-II would converge to a few clustered points. You implement the per-front crowding computation and the crowded comparison operator.

**What you implement:**
- `compute_crowding_distance()` — For each objective, sort the front, assign infinity to boundaries, compute normalized neighbor distances for middle solutions.
- `crowded_comparison()` — Compare two individuals by (rank ascending, crowding distance descending).

**Why it matters:** Crowding distance is what makes NSGA-II find a spread-out Pareto front rather than a single cluster. The crowded comparison operator combines convergence pressure (rank) with diversity pressure (crowding), and is used in both tournament selection and the truncation step.

### Phase 3: Full NSGA-II (~35 min)

**File:** `src/nsga2.rs`

This phase assembles everything into the complete NSGA-II algorithm. Non-dominated sorting, crowding distance, and comparison are provided. You implement the genetic operators (tournament selection, SBX crossover, polynomial mutation) and the main evolutionary loop with elitist selection.

**What you implement:**
- `tournament_selection()` — Binary tournament using crowded comparison.
- `sbx_crossover()` — Simulated Binary Crossover for real-valued decision variables.
- `polynomial_mutation()` — Per-variable bounded mutation.
- `nsga2()` — The complete NSGA-II main loop: initialize, evolve, combine, sort, truncate.

**Why it matters:** The evolutionary loop is where all components work together. The elitist selection (combine 2N, keep best N) is NSGA-II's key advantage over NSGA. Understanding the loop structure is essential for modifying or extending the algorithm (custom operators, constraint handling, adaptive parameters).

### Phase 4: Benchmarks & Visualization (~25 min)

**File:** `src/benchmarks.rs`

This phase teaches how to evaluate multi-objective algorithm quality. You implement the hypervolume indicator — the gold standard metric — and run NSGA-II on the ZDT test suite to observe how different Pareto front shapes affect algorithm performance.

**What you implement:**
- `hypervolume_2d()` — Compute the 2D hypervolume indicator for a Pareto front approximation.

**Why it matters:** Hypervolume is the ONLY Pareto-compliant metric — it rewards both convergence and diversity. Understanding it is essential for comparing algorithms and tuning parameters. Running on ZDT1/2/3 shows that NSGA-II handles convex, concave, and disconnected fronts, unlike weighted sum approaches.

## Motivation

NSGA-II is the most widely used multi-objective optimization algorithm. Understanding Pareto dominance, crowding distance, and evolutionary multi-objective optimization is essential for:

- **Engineering optimization**: most real problems have multiple conflicting objectives (cost vs performance, weight vs strength, latency vs throughput).
- **Machine learning**: hyperparameter optimization with multiple metrics (accuracy vs inference time), neural architecture search, multi-task learning trade-offs.
- **Operations research**: scheduling with cost and time objectives, supply chain with service level and inventory cost, portfolio optimization with return and risk.
- **Algorithm design**: NSGA-II concepts (non-dominated sorting, crowding) appear in NSGA-III, MOEA/D, and multi-objective reinforcement learning.
- **Career relevance**: MOO knowledge is expected in optimization roles at trading firms, robotics companies, and any team that designs systems with conflicting requirements.

## Commands

All commands are run from the `practice_038_multi_objective_nsga2/` folder root.

### Build

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all binaries (fetches `rand` crate on first run) |
| `cargo build --release` | Compile with optimizations (faster evolution for Phase 4 benchmarking) |
| `cargo check` | Fast type-check without producing binaries |

### Run — Phase 1: Non-Dominated Sorting

| Command | Description |
|---------|-------------|
| `cargo run --bin phase1_nondominated_sort` | Run Phase 1: sort a sample population into Pareto fronts, verify ranking |

### Run — Phase 2: Crowding Distance

| Command | Description |
|---------|-------------|
| `cargo run --bin phase2_crowding` | Run Phase 2: compute crowding distance within fronts, print sorted results |

### Run — Phase 3: Full NSGA-II

| Command | Description |
|---------|-------------|
| `cargo run --bin phase3_nsga2` | Run Phase 3: full NSGA-II on ZDT1, print Pareto front every 10 generations |

### Run — Phase 4: Benchmarks

| Command | Description |
|---------|-------------|
| `cargo run --bin phase4_benchmarks` | Run Phase 4: NSGA-II on ZDT1/2/3, compute hypervolume, visualize fronts |
| `cargo run --release --bin phase4_benchmarks` | Run Phase 4 with optimizations for accurate timing and larger populations |

## References

- [Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II](https://ieeexplore.ieee.org/document/996017) — Original NSGA-II paper
- [Zitzler, E., Deb, K., & Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: Empirical results](https://doi.org/10.1162/106365600568202) — ZDT test suite
- [Deb, K. (2001). Multi-Objective Optimization using Evolutionary Algorithms](https://www.wiley.com/en-us/Multi-Objective+Optimization+using+Evolutionary+Algorithms-p-9780471873396) — Definitive textbook on MOEA
- [Coello Coello, C.A. (2007). Evolutionary Algorithms for Solving Multi-Objective Problems](https://link.springer.com/book/10.1007/978-0-387-36797-2) — Comprehensive reference on evolutionary MOO
- [pymoo: Multi-Objective Optimization in Python](https://pymoo.org/) — Python library implementing NSGA-II, NSGA-III, and many others

## State

`not-started`
