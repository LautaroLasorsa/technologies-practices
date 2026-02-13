# Practice 035b: Non-convex Optimization — Evolutionary Algorithms

## Technologies

- **Rust** — Systems language with ownership model, zero-cost abstractions
- **rand 0.8** — Random number generation (uniform, index selection, coin flips)

## Stack

- Rust (cargo)

## Theoretical Context

### Evolutionary Algorithms: Population-Based Optimization

**Evolutionary Algorithms (EAs)** are a family of metaheuristic optimization methods inspired by biological evolution. Instead of maintaining a single candidate solution and improving it (like gradient descent or simulated annealing), EAs maintain a **population** of candidate solutions and evolve them over generations through selection, recombination, and mutation.

The core loop shared by all EAs:

```
1. INITIALIZE a population of random candidate solutions
2. EVALUATE fitness of each individual
3. SELECT parents based on fitness (better individuals more likely chosen)
4. CREATE offspring via recombination (crossover) and mutation
5. REPLACE some/all of the population with offspring
6. REPEAT from step 2 until termination criterion met
```

EAs are **gradient-free** (black-box) optimizers: they only need a fitness function `f(x) -> R`, not its derivative. This makes them applicable to discontinuous, noisy, multi-modal, and combinatorial objective functions where gradient-based methods fail.

### Genetic Algorithm (GA)

The **Genetic Algorithm** (Holland, 1975; Goldberg, 1989) is the original and most widely known evolutionary algorithm. It represents solutions as **chromosomes** — typically binary strings or real-valued vectors — and evolves them through biologically-inspired operators.

**Representation:** Each individual is a chromosome, a vector of **genes**. For a binary knapsack problem, a chromosome is a bit vector `[1, 0, 1, 1, 0, ...]` where each gene indicates whether an item is included. For continuous optimization, genes are real numbers representing coordinates.

**Selection** determines which individuals become parents. The key principle: **fitter individuals should be more likely to reproduce, but not guaranteed** — this maintains diversity and avoids premature convergence.

| Selection Method | Mechanism | Pros | Cons |
|-----------------|-----------|------|------|
| **Tournament selection** | Pick `k` random individuals, select the best among them | Simple, adjustable pressure via `k`, parallelizable | Doesn't use global ranking |
| **Roulette wheel** | Probability proportional to fitness: `p_i = f_i / sum(f)` | Classic, intuitive | Fails with negative fitness, dominated by super-fit individuals |
| **Rank-based** | Sort by fitness, probability proportional to rank | Handles fitness scaling, more uniform pressure | Requires sorting O(n log n) |

**Tournament selection** with k=2 or k=3 is the most common in practice — it is simple, efficient, and the selection pressure is easily controlled.

**Crossover** (recombination) combines two parents to produce offspring. This is the primary exploration operator — it combines building blocks from different solutions.

| Crossover Type | Mechanism | Use Case |
|---------------|-----------|----------|
| **Single-point** | Choose random point `p`, child1 = parent1[0..p] + parent2[p..], child2 = vice versa | Simple, good for positionally-linked genes |
| **Two-point** | Choose points `p1, p2`, swap the middle segment | More mixing than single-point |
| **Uniform** | For each gene, coin flip to choose from parent1 or parent2 | Maximum mixing, good for independent genes |

**Mutation** introduces random perturbation to individual genes. For binary chromosomes: **bit-flip** — each gene flips with probability `mutation_rate` (typically 1/chromosome_length to 0.05). For real-valued: add Gaussian noise. Mutation is the primary diversity-maintenance operator — without it, the population can lose alleles permanently.

**Elitism:** Copy the best `k` individuals (typically k=1 or k=2) unchanged into the next generation. This **guarantees monotonic improvement** — the best fitness never decreases. Without elitism, a lucky crossover might be lost. With elitism, the best solution survives.

**GA pseudocode (generational model):**

```
population = random_init(pop_size)
for gen in 0..max_generations:
    fitnesses = [fitness(ind) for ind in population]
    elite = best k individuals (elitism)
    new_population = elite.copy()
    while len(new_population) < pop_size:
        parent1 = tournament_select(population, fitnesses)
        parent2 = tournament_select(population, fitnesses)
        if rand() < crossover_rate:
            child1, child2 = crossover(parent1, parent2)
        else:
            child1, child2 = parent1.clone(), parent2.clone()
        mutate(child1, mutation_rate)
        mutate(child2, mutation_rate)
        new_population.push(child1)
        new_population.push(child2)
    population = new_population[0..pop_size]
return best individual ever seen
```

**GA is particularly well-suited for combinatorial problems** (knapsack, TSP, scheduling) where the discrete structure of the chromosome maps naturally to the problem structure.

### Differential Evolution (DE)

**Differential Evolution** (Storn & Price, 1997) is a population-based optimizer specifically designed for **continuous optimization**. It is remarkably simple — just three operations: mutation, crossover, selection — yet consistently ranks among the top performers on continuous benchmark problems.

**Key insight:** DE's mutation operator uses **vector differences from the population itself** as perturbation directions. This automatically scales the search: when the population is spread out (early search), mutations are large; as the population converges, mutations shrink. No external step-size parameter adaptation is needed.

**DE/rand/1 mutation** (the standard variant):

```
For target vector x_i, pick three distinct random individuals a, b, c (none equal to i):
    v_i = x_a + F * (x_b - x_c)
```

Where `F` (scale factor, typically 0.5-0.9) controls the amplification of the differential variation `(x_b - x_c)`. The vector `v_i` is called the **mutant vector** or **donor vector**.

**Binomial crossover:**

```
For each dimension j:
    if rand() < CR or j == j_rand:   (j_rand ensures at least one component from mutant)
        trial_j = v_j  (take from mutant)
    else:
        trial_j = x_j  (keep from target)
```

`CR` (crossover rate, typically 0.5-1.0) controls how many components come from the mutant vector. `j_rand` is a randomly chosen dimension that always takes the mutant value — this guarantees the trial vector differs from the target in at least one dimension.

**Selection** is greedy per-individual: the trial vector replaces the target **only if it has equal or better fitness**. This one-to-one competition is what makes DE stable — no individual can get worse.

```
if f(trial_i) <= f(x_i):  (for minimization)
    x_i = trial_i
```

**DE pseudocode:**

```
population = random_init(pop_size, bounds)
for gen in 0..max_generations:
    for i in 0..pop_size:
        a, b, c = pick 3 random distinct indices != i
        v = population[a] + F * (population[b] - population[c])   # mutation
        trial = binomial_crossover(population[i], v, CR)           # crossover
        trial = clip(trial, bounds)                                # respect bounds
        if f(trial) <= f(population[i]):                           # selection
            population[i] = trial
return best individual
```

**Why DE often outperforms GA on continuous problems:**
- The differential mutation naturally adapts step sizes to the population spread
- No separate mutation distribution to tune (no sigma, no temperature)
- The greedy selection ensures monotonic per-individual improvement
- Fewer hyperparameters: just `F`, `CR`, and `pop_size`

### Particle Swarm Optimization (PSO)

**Particle Swarm Optimization** (Kennedy & Eberhart, 1995) models a swarm of particles flying through the search space. Each particle has a **position** (candidate solution) and a **velocity** (direction and speed of movement). Particles are attracted toward their own best-known position (**personal best**, `pbest`) and the swarm's best-known position (**global best**, `gbest`).

**Velocity update:**

```
v_i(t+1) = w * v_i(t)                           # inertia (keep moving)
          + c1 * r1 * (pbest_i - x_i(t))        # cognitive (own memory)
          + c2 * r2 * (gbest - x_i(t))          # social (swarm knowledge)
```

Where:
- `w` (inertia weight, typically 0.4-0.9): controls how much of the previous velocity is retained. High `w` = more exploration, low `w` = more exploitation.
- `c1` (cognitive coefficient, typically 1.5-2.0): attraction toward personal best.
- `c2` (social coefficient, typically 1.5-2.0): attraction toward global best.
- `r1, r2`: independent uniform random numbers in [0, 1] per dimension, providing stochasticity.

**Position update:**

```
x_i(t+1) = x_i(t) + v_i(t+1)
```

**PSO pseudocode:**

```
for each particle i:
    x_i = random position in bounds
    v_i = random small velocity
    pbest_i = x_i
gbest = argmin f(pbest_i) over all i

for iter in 0..max_iterations:
    for each particle i:
        for each dimension d:
            r1, r2 = rand(), rand()
            v_i[d] = w * v_i[d]
                   + c1 * r1 * (pbest_i[d] - x_i[d])
                   + c2 * r2 * (gbest[d] - x_i[d])
        x_i = x_i + v_i
        clip x_i to bounds
        if f(x_i) < f(pbest_i):
            pbest_i = x_i
            if f(x_i) < f(gbest):
                gbest = x_i
return gbest
```

**Strengths of PSO:**
- Very few hyperparameters (w, c1, c2)
- No crossover or mutation operators to design
- Fast convergence on unimodal and moderately multimodal problems
- Easy to parallelize (particles are independent within an iteration)

**Weaknesses:**
- Can converge prematurely on highly multimodal problems (all particles collapse to one point)
- No guarantee of escaping local optima once the swarm converges
- Sensitive to inertia weight: too high = never converges, too low = premature convergence

### CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**CMA-ES** (Hansen & Ostermeier, 2001) is widely considered the **state-of-the-art for continuous black-box optimization** up to ~100 dimensions. It is NOT implemented in this practice but is important to understand conceptually.

CMA-ES maintains a **multivariate Gaussian distribution** N(m, sigma^2 * C) over the search space:
- `m`: the mean (center of the search distribution)
- `sigma`: the global step size
- `C`: the covariance matrix (encodes variable correlations and per-variable scaling)

Each generation:
1. **Sample** lambda offspring from the Gaussian: `x_k ~ N(m, sigma^2 * C)`
2. **Evaluate** fitness of each offspring
3. **Select** the best mu offspring
4. **Update** the mean `m` toward the center of mass of the best offspring
5. **Adapt** the covariance matrix `C` to favor directions that produced good offspring
6. **Adapt** the step size `sigma` using a cumulative path-length control

**Why CMA-ES is powerful:**
- Learns the local geometry of the fitness landscape through covariance adaptation
- Handles rotated, ill-conditioned functions where axes-aligned methods fail
- Invariant to rotations and translations of the search space
- Step-size adaptation prevents premature convergence AND stagnation

**Why not implemented here:** CMA-ES requires eigendecomposition of the covariance matrix (O(n^3) per generation), which adds significant complexity. It is best learned as a library (e.g., `cmaes` crate) after understanding the simpler methods.

### Algorithm Comparison

| Property | GA | DE | PSO | CMA-ES |
|----------|----|----|-----|--------|
| **Best domain** | Combinatorial (binary, permutation) | Continuous (moderate dim) | Continuous (moderate dim) | Continuous (up to ~100D) |
| **Representation** | Binary strings, permutations, real vectors | Real vectors | Real vectors | Real vectors (Gaussian) |
| **Key operator** | Crossover (recombination) | Differential mutation | Velocity update | Covariance adaptation |
| **Selection** | Tournament/roulette (probabilistic) | Greedy 1-to-1 | Implicit (pbest/gbest) | Truncation (best mu of lambda) |
| **Hyperparameters** | pop_size, crossover_rate, mutation_rate, tournament_size | pop_size, F, CR | pop_size, w, c1, c2 | pop_size (lambda), rest auto-adapted |
| **Adaptive** | No (static rates) | Partially (step auto-scales) | Partially (inertia decay) | Yes (covariance + step size) |
| **Convergence guarantee** | None | None | None | None (but very reliable in practice) |
| **Typical pop size** | 50-200 | 5*D to 10*D | 20-50 | 4 + floor(3*ln(D)) |

### Convergence and Diversity

No EA provides convergence guarantees in the mathematical sense (unlike gradient descent on convex functions). However, population-based search has a fundamental advantage over single-point methods: **diversity**. A diverse population explores multiple regions of the search space simultaneously, reducing the chance of all search effort being trapped in one local optimum.

Key diversity mechanisms:
- **GA**: Crossover recombines solutions from different regions; mutation prevents allele loss
- **DE**: Differential perturbation direction is inherently diverse (different vectors point differently)
- **PSO**: Stochastic velocity updates + personal bests maintain some diversity (but weakest of the three)

**Population size trade-off**: Larger populations = more diversity = better exploration, but slower per-generation computation and slower convergence. Rule of thumb: use the smallest population that consistently finds good solutions.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Population** | A set of candidate solutions maintained simultaneously |
| **Fitness** | Objective function value of a candidate (higher = better for maximization) |
| **Generation** | One iteration of the EA loop: evaluate, select, recombine, mutate, replace |
| **Chromosome** | The representation of a candidate solution (bit string, real vector, etc.) |
| **Gene** | A single element of a chromosome (one bit, one real number) |
| **Selection** | Choosing parents based on fitness (tournament, roulette, rank) |
| **Crossover** | Combining two parents to produce offspring (single-point, uniform, etc.) |
| **Mutation** | Random perturbation of individual genes |
| **Elitism** | Copying the best individual(s) unchanged to the next generation |
| **Tournament selection** | Pick k random individuals, return the fittest |
| **Differential mutation** | v = x_a + F*(x_b - x_c) — mutation using population differences |
| **Binomial crossover** | Per-component coin flip to choose mutant vs target component |
| **Particle** | A candidate solution in PSO with position, velocity, and personal best |
| **Inertia weight (w)** | Controls how much previous velocity is retained in PSO |
| **Personal best (pbest)** | Best position a particle has visited |
| **Global best (gbest)** | Best position any particle in the swarm has visited |
| **CMA-ES** | Adapts a multivariate Gaussian to sample increasingly better solutions |
| **Scale factor (F)** | DE parameter controlling differential perturbation magnitude |
| **Crossover rate (CR)** | Probability of taking a component from the mutant/donor vector |

### Test Functions for Continuous Optimization

| Function | Formula (per dimension) | Global minimum | Character |
|----------|------------------------|----------------|-----------|
| **Rastrigin** | `f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))` | f(0,...,0) = 0 | Highly multimodal, regular grid of local minima |
| **Ackley** | `f(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e` | f(0,...,0) = 0 | Multimodal with a large basin near the global optimum |
| **Schwefel** | `f(x) = 418.9829n - sum(x_i * sin(sqrt(abs(x_i))))` | f(420.9687,...) = 0 | Deceptive: global optimum far from next-best local optima |

## Description

Implement three evolutionary optimization algorithms in Rust: a **Genetic Algorithm** for binary knapsack, **Differential Evolution** for continuous optimization, and **Particle Swarm Optimization** on the same continuous problems. Compare all three on standard benchmark functions.

### What you'll build

1. **Genetic Algorithm** — Tournament selection, single-point crossover, bit-flip mutation, elitism. Applied to a binary knapsack problem.
2. **Differential Evolution** — DE/rand/1 mutation, binomial crossover, greedy selection. Applied to Rastrigin and Ackley functions.
3. **Particle Swarm Optimization** — Velocity update with inertia, cognitive, and social components. Applied to the same continuous benchmarks.
4. **Comparison** — Run all methods on the same continuous problems, compare convergence speed and solution quality.

## Instructions

### Phase 1: Genetic Algorithm for Binary Knapsack (~30 min)

**File:** `src/genetic.rs`

This phase teaches the core GA operators: tournament selection, crossover, mutation, and the generational loop. The binary knapsack is the ideal first problem because the chromosome representation (bit vector = which items to include) is natural and intuitive.

**What you implement:**
- `tournament_selection()` — The fundamental selection operator. Pick k random individuals from the population, return the one with the highest fitness. This is how "survival of the fittest" works in practice: not deterministic, but probabilistic — fitter individuals are more likely to be chosen.
- `single_point_crossover()` — The primary exploration operator. Choose a random crossover point, swap the tails of two parents. This is how GAs combine building blocks from different solutions to create potentially better offspring.
- `mutate()` — The diversity maintenance operator. Flip each bit with small probability. Without mutation, the population can permanently lose alleles (all 0s or all 1s at a position), trapping the search.
- `genetic_algorithm()` — The main GA loop tying everything together: evaluate fitness, apply elitism, fill next generation via selection + crossover + mutation.

**Why it matters:** GA is the foundational evolutionary algorithm. Every other EA is a variation of this structure. Understanding selection pressure, crossover exploitation, and mutation exploration is prerequisite knowledge for DE, PSO, and CMA-ES.

### Phase 2: Differential Evolution (~25 min)

**File:** `src/differential_evolution.rs`

This phase teaches DE's elegant continuous optimization approach. The key insight is that mutation direction and scale come from the population itself (vector differences), not from a separate distribution.

**What you implement:**
- `de_mutation()` — DE/rand/1: pick 3 distinct random individuals, compute v = x_a + F*(x_b - x_c). This is the operator that gives DE its name and its power. The differential (x_b - x_c) automatically encodes the scale and direction of population spread.
- `de_crossover()` — Binomial crossover: for each dimension, choose mutant or target component with probability CR. The j_rand trick ensures the trial always differs from the target. This is how DE balances exploration (many mutant components) vs exploitation (few mutant components).
- `differential_evolution()` — The DE loop: for each individual, generate trial via mutation + crossover, keep the better of trial vs target. The greedy per-individual selection ensures monotonic improvement.

**Why it matters:** DE is the default first choice for continuous black-box optimization in practice. It requires minimal tuning (just F and CR), has no gradient requirement, and consistently performs well. Many real-world engineering optimization problems are solved with DE.

### Phase 3: Particle Swarm Optimization (~25 min)

**File:** `src/pso.rs`

This phase teaches swarm intelligence. Unlike GA/DE which use explicit selection and recombination, PSO uses velocity-based movement where particles are attracted toward known good positions.

**What you implement:**
- `update_velocity()` — The PSO velocity formula: v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x). Three components: inertia (momentum), cognitive (memory), social (swarm). Each component contributes differently to the search dynamics.
- `pso()` — The PSO loop: update velocities, move particles, update personal and global bests. Unlike GA/DE, there is no explicit "selection" — the guidance comes from pbest/gbest attraction.

**Why it matters:** PSO is widely used in engineering optimization, neural network training (hyperparameter search), and robotics (swarm coordination). Its simplicity makes it easy to implement and modify. Understanding PSO also provides intuition for other swarm methods (ant colony optimization, firefly algorithm).

### Phase 4: Comparison (~20 min)

**File:** `src/comparison.rs`

This phase teaches how to empirically compare optimization algorithms. You run GA (adapted for continuous representation), DE, and PSO on the same benchmark functions and compare results.

**What you implement:**
- `run_comparison()` — Execute all three algorithms on the same function, collect best fitness, number of function evaluations, and wall-clock time. Print a comparison table. This teaches the importance of fair comparison: same function evaluation budget, same dimensionality, same random seeds for reproducibility.

**Why it matters:** In practice, choosing the right optimizer for a problem requires empirical comparison. No single algorithm dominates all problems (No Free Lunch theorem). Learning to run controlled experiments and interpret results is essential for applied optimization.

## Motivation

Evolutionary algorithms are the **default tools for black-box optimization** — problems where the objective function is a black box (no gradients, possibly discontinuous, noisy, multimodal). This covers a vast range of real-world problems:

- **Engineering design**: Aerodynamic shape optimization, circuit design, structural optimization
- **Machine learning**: Hyperparameter tuning, neural architecture search, neuroevolution
- **Finance**: Portfolio optimization with complex constraints, strategy parameter tuning
- **Operations research**: When exact MIP formulations are intractable, EAs provide good heuristic solutions
- **Scheduling**: Job-shop scheduling, resource allocation with non-linear objectives

DE is particularly important as it is **state-of-the-art for continuous problems** up to moderate dimensions and requires minimal tuning. Understanding GA, DE, and PSO provides the conceptual foundation for the entire field of metaheuristic optimization, including hybrid methods and modern variants.

## Commands

All commands are run from the `practice_035b_nonconvex_evolutionary/` folder root.

### Build

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all binaries (fetches `rand` crate on first run) |
| `cargo build --release` | Compile with optimizations (important for Phase 4 timing comparison) |
| `cargo check` | Fast type-check without producing binaries |

### Run — Phase 1: Genetic Algorithm

| Command | Description |
|---------|-------------|
| `cargo run --bin phase1_genetic` | Run GA on binary knapsack, print best solution per generation |

### Run — Phase 2: Differential Evolution

| Command | Description |
|---------|-------------|
| `cargo run --bin phase2_de` | Run DE on 10D Rastrigin and 10D Ackley |

### Run — Phase 3: Particle Swarm Optimization

| Command | Description |
|---------|-------------|
| `cargo run --bin phase3_pso` | Run PSO on 10D Rastrigin and 10D Ackley |

### Run — Phase 4: Comparison

| Command | Description |
|---------|-------------|
| `cargo run --bin phase4_comparison` | Run all algorithms on same problems, print comparison table |
| `cargo run --release --bin phase4_comparison` | Run comparison with optimizations for accurate timing |

## References

- [Holland, J.H. (1975). Adaptation in Natural and Artificial Systems](https://mitpress.mit.edu/9780262581110/) — Original Genetic Algorithm book
- [Goldberg, D.E. (1989). Genetic Algorithms in Search, Optimization, and Machine Learning](https://www.pearson.com/en-us/subject-catalog/p/genetic-algorithms-in-search-optimization-and-machine-learning/P200000003381) — Classic GA textbook
- [Storn, R. & Price, K. (1997). Differential Evolution](https://link.springer.com/article/10.1023/A:1008202821328) — Original DE paper in Journal of Global Optimization
- [Kennedy, J. & Eberhart, R. (1995). Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — Original PSO paper (ICNN 1995)
- [Hansen, N. (2016). The CMA Evolution Strategy: A Tutorial](https://arxiv.org/abs/1604.00772) — Comprehensive CMA-ES reference
- [Das, S. & Suganthan, P.N. (2011). Differential Evolution: A Survey](https://ieeexplore.ieee.org/document/5601760) — DE variants and applications survey
- [Poli, R. et al. (2007). Particle Swarm Optimization: An Overview](https://link.springer.com/article/10.1007/s11721-007-0002-0) — PSO survey in Swarm Intelligence journal

## State

`not-started`
