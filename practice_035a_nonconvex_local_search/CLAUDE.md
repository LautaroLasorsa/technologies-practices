# Practice 035a: Non-convex Optimization — Local Search & Simulated Annealing

**State:** `not-started`

## Technologies

- **Rust** — Systems programming language
- **`rand` crate** — Random number generation (Gaussian noise, uniform sampling)

## Stack

Rust

## Theoretical Context

### Non-convex Optimization

In convex optimization, any local minimum is the global minimum — gradient descent always finds the best solution. **Non-convex optimization** is fundamentally harder: the objective function has **multiple local minima**, and there is **no polynomial-time algorithm guaranteed to find the global optimum** for general non-convex problems.

Most real-world optimization problems are non-convex: scheduling, routing, chip placement, protein folding, neural network training. The landscape looks like a mountain range with many valleys of varying depth. An optimizer can get trapped in a shallow valley (local minimum) and never discover the deepest valley (global minimum).

**Key challenge:** How do we escape local optima to explore better solutions?

### Local Search

Local search is the simplest approach to optimization:

1. Start from some solution `x`
2. Define a **neighborhood** `N(x)` — the set of solutions "close" to `x`
3. Move to a neighbor that improves the objective
4. Repeat until no improving neighbor exists (stuck at a local optimum)

The neighborhood structure is the most critical design choice. For continuous problems, neighbors might be points within some radius. For combinatorial problems like TSP, neighbors might be tours differing by a single swap.

### Hill Climbing (Greedy Local Search)

The simplest local search strategy:

- **Always accept improving moves**, reject non-improving ones
- Also called "steepest descent" (minimization) or "steepest ascent" (maximization)
- Guaranteed to converge to a local optimum (the objective never worsens)
- **Fatal flaw:** gets permanently stuck at the first local optimum it finds

Think of a blind person trying to find the lowest point in hilly terrain by always walking downhill — they'll reach the bottom of the nearest valley, which may not be the deepest valley.

### Multi-start Local Search

Simple idea to partially overcome local optima:

1. Run hill climbing from many **random starting points**
2. Return the best result across all runs

If there are `k` basins of attraction and you run `n` independent starts, the probability of missing the global optimum basin decreases roughly as `(1 - 1/k)^n`. With enough restarts, you're likely to land in the global basin at least once.

**Limitations:**
- No guarantee: might always miss the global optimum's basin
- No learning: each restart is independent, doesn't use information from previous runs
- Wasteful: spends equal effort on good and bad basins

### Simulated Annealing (SA)

**Inspired by metallurgical annealing:** When metal is heated to high temperature, atoms move freely. As the metal cools slowly, atoms settle into a low-energy crystalline structure (global minimum of energy). If cooled too fast, the metal becomes amorphous (stuck in a local minimum).

SA translates this into optimization:

- **Temperature `T`** controls randomness
- At each step, generate a random neighbor and compute `ΔE = f(neighbor) - f(current)`:
  - If `ΔE < 0` (improvement): **always accept**
  - If `ΔE ≥ 0` (worsening): **accept with probability `P = exp(-ΔE / T)`**

**Temperature dynamics:**
- **High T** → `exp(-ΔE/T) ≈ 1` → accept almost anything → **exploration** (random walk)
- **Low T** → `exp(-ΔE/T) ≈ 0` → accept only improvements → **exploitation** (hill climbing)
- **T → 0** → SA degenerates to pure hill climbing

This acceptance criterion is the **Metropolis criterion**, from the Metropolis-Hastings algorithm in statistical physics. It samples states from the Boltzmann distribution at temperature T.

### Temperature Schedule (Cooling Schedule)

How temperature decreases over time is crucial:

| Schedule | Formula | Properties |
|----------|---------|------------|
| **Geometric** | `T(k+1) = α · T(k)` | Most common. α ∈ [0.9, 0.999]. Simple, effective. |
| **Linear** | `T(k) = T₀ - k · (T₀/N)` | Simple but can hit T=0 early. |
| **Logarithmic** | `T(k) = c / ln(k+1)` | Theoretically optimal (converges to global optimum in probability). Impractically slow. |
| **Adaptive** | Adjusted based on acceptance rate | Maintains target acceptance ratio (e.g., 40-50%). |

**Practical guidance:**
- `T₀` should be high enough that ~80% of moves are accepted initially
- `α = 0.95` is a common starting point; tune between 0.9 (fast) and 0.999 (slow)
- Slow cooling → better solutions but more iterations needed
- Fast cooling → converges quickly but to worse solutions

### Convergence Theory

**Theorem (Hajek, 1988):** If the cooling schedule satisfies `T(k) ≥ c / ln(k)` for sufficiently large `c`, then SA converges to the global optimum in probability as `k → ∞`.

This is a theoretical guarantee only — logarithmic cooling is impractically slow (essentially infinite time). In practice, geometric cooling with enough iterations works well for most problems. SA is a **heuristic**: it usually finds very good solutions, but provides no optimality guarantee in finite time.

### Neighborhood Structure

The choice of neighborhood determines what SA can explore:

| Problem Type | Common Neighborhoods |
|-------------|---------------------|
| Continuous | Gaussian perturbation, uniform ball |
| Permutations (TSP) | 2-opt (reverse segment), 3-opt, or-opt (move segment) |
| Binary | Flip single bit, flip k bits |
| Scheduling | Swap two jobs, move job to different position |

**Design principles:**
- **Connectivity:** Any solution must be reachable from any other via a sequence of moves
- **Locality:** Small moves should produce small changes in objective value
- **Efficiency:** Generating and evaluating a neighbor should be fast

### Applications

SA is one of the most widely-used metaheuristics:

- **TSP and vehicle routing** — 2-opt and 3-opt neighborhoods
- **VLSI placement** — placing circuit components to minimize wire length
- **Scheduling** — job shop, resource allocation
- **Protein folding** — finding minimum-energy conformations
- **Image processing** — denoising, segmentation
- **Machine learning** — hyperparameter optimization, neural architecture search

### Comparison: SA vs Alternatives

| Method | Escapes Local Optima? | Memory | Theoretical Guarantee | Practical Quality |
|--------|----------------------|--------|----------------------|-------------------|
| Hill climbing | No | O(1) | Local optimum only | Poor on multimodal |
| Multi-start | Partially (by sampling) | O(1) | Probabilistic | Moderate |
| **Simulated Annealing** | **Yes (via temperature)** | **O(1)** | **Global with log cooling** | **Very good** |
| Genetic algorithms | Yes (via population) | O(pop_size) | None formal | Good |
| Gradient descent + restarts | Partially | O(1) | Local optimum per run | Good on smooth |

### Key Concepts

| Concept | Definition |
|---------|------------|
| Local minimum | Point where all neighbors have equal or worse objective value |
| Global minimum | The best solution across the entire search space |
| Neighborhood N(x) | Set of solutions reachable from x by a single move |
| Metropolis criterion | Accept worse solution with probability exp(-ΔE/T) |
| Temperature T | Controls exploration vs exploitation tradeoff |
| Cooling schedule | How T decreases over time (geometric, linear, logarithmic) |
| Rastrigin function | Standard non-convex test function with many local minima |
| 2-opt | TSP neighborhood: reverse a segment of the tour |
| Basin of attraction | Set of starting points from which local search converges to a given local minimum |
| Acceptance ratio | Fraction of proposed moves that are accepted |

## Description

Implement hill climbing, multi-start local search, and simulated annealing on two problem types:

1. **Continuous optimization** — Rastrigin function (a standard test function with ~10^n local minima in n dimensions, global minimum of 0 at the origin)
2. **Combinatorial optimization** — Traveling Salesman Problem (TSP) with 2-opt neighborhood

The progression shows why SA's temperature-based acceptance is powerful: hill climbing gets stuck, multi-start helps but is wasteful, and SA systematically balances exploration and exploitation.

## Instructions

### Phase 1: Hill Climbing on Rastrigin

**What you learn:** The fundamental limitation of greedy local search — it always gets stuck in the nearest local optimum.

The Rastrigin function `f(x) = 10n + Σ[x_i² - 10·cos(2πx_i)]` has a global minimum of 0 at the origin, but is covered in local minima spaced roughly 1 unit apart. Hill climbing from a random start will almost never find the global optimum.

**Exercise:** Implement `hill_climbing()` — the greedy local search loop. Generate a neighbor, accept it only if it improves the objective, and repeat until `max_iter`.

Run on 2D and 5D Rastrigin. Observe that the result depends entirely on the starting point, and is almost never near 0.

### Phase 2: Multi-start Hill Climbing

**What you learn:** Running hill climbing from many random starting points improves results by sampling more basins of attraction, but is inherently limited — it doesn't learn from previous runs.

**Exercise:** Implement `multi_start_hill_climbing()` — run hill climbing from `n_starts` random initial points within given bounds, return the best result found.

Compare single-start vs 10, 50, and 200 starts. Watch how more starts generally find better solutions, but with diminishing returns.

### Phase 3: Simulated Annealing (Continuous)

**What you learn:** The SA acceptance criterion — accepting worse solutions with probability `exp(-ΔE/T)` — enables escape from local optima. The cooling schedule controls how quickly SA transitions from exploration to exploitation.

**Exercise:** Implement `simulated_annealing()` — the full SA loop with Metropolis criterion and geometric cooling.

Experiment with different cooling rates (`α = 0.9, 0.99, 0.999`). Observe that slow cooling finds better solutions. Compare SA results with multi-start hill climbing.

### Phase 4: SA for TSP (Combinatorial)

**What you learn:** SA works equally well on discrete/combinatorial problems. The 2-opt neighborhood is the standard move for TSP. This demonstrates that the SA framework is problem-agnostic — only the neighborhood operator changes.

**Exercise:** Implement `two_opt_swap()` (reverse a tour segment) and `sa_tsp()` (SA with 2-opt neighborhood).

Generate 20 random cities, construct an initial tour with nearest-neighbor heuristic, then improve it with SA. Print the improvement percentage.

## Motivation

Simulated Annealing is arguably the most widely-used metaheuristic in operations research and combinatorial optimization. Understanding local search and temperature-based acceptance is foundational for:

- All metaheuristics (tabu search, genetic algorithms, ant colony)
- Practical optimization in scheduling, routing, and placement
- Understanding the exploration-exploitation tradeoff that appears everywhere in optimization and ML
- HFT/quantitative finance: portfolio optimization, parameter tuning

SA also connects to statistical physics (Boltzmann distribution, Metropolis algorithm) and complexity theory (NP-hard problems where exact methods fail).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Build | `cargo build` | Compile all binaries |
| Phase 1 | `cargo run --bin phase1_hill_climbing` | Hill climbing on Rastrigin — observe local optima traps |
| Phase 2 | `cargo run --bin phase2_multi_start` | Multi-start hill climbing — compare with single start |
| Phase 3 | `cargo run --bin phase3_sa_continuous` | Simulated annealing on Rastrigin — compare cooling rates |
| Phase 4 | `cargo run --bin phase4_sa_tsp` | SA with 2-opt for TSP — combinatorial optimization |
