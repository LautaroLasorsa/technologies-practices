# Practice 043: Global Optimization — scipy & Optuna

## Technologies

- **scipy.optimize** — SciPy's optimization module providing local optimizers (`minimize` with Nelder-Mead, BFGS, L-BFGS-B, trust methods), global optimizers (`differential_evolution`, `dual_annealing`, `shgo`, `basinhopping`), and constrained optimization support.
- **Optuna** — Bayesian hyperparameter optimization framework with define-by-run API, Tree-structured Parzen Estimator (TPE) sampler, CMA-ES sampler, multi-objective support, and trial pruning.
- **NumPy** — Array operations and test function evaluation.
- **Matplotlib** — Visualization of convergence curves, optimization landscapes, and Pareto fronts.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- scipy >= 1.12 (optimization algorithms)
- Optuna >= 3.5 (Bayesian optimization, multi-objective)
- NumPy >= 1.26, Matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### What Global Optimization Is and the Problem It Solves

**Global optimization** seeks the absolute best solution over an entire search domain, in contrast to **local optimization** which finds a nearby optimum relative to a starting point. The distinction matters when the objective function has multiple local minima (is **multimodal**).

A local optimizer like gradient descent applied to a multimodal function will converge to whichever basin of attraction contains the starting point — potentially a poor local minimum far from the global optimum. Global optimization methods use strategies (population diversity, stochastic perturbation, surrogate models) to escape local basins and explore the full landscape.

```
Local optimizer: gradient descent from x0 → nearest local min (may be terrible)
Global optimizer: explores many basins → finds global min (or near-global)
```

**When you need global optimization:**
- The objective function is multimodal (multiple local minima)
- No convexity guarantees (unlike CVXPY problems from practice 041a)
- Black-box functions: no gradient available or gradient is unreliable
- Hyperparameter tuning: discrete/continuous/conditional parameter spaces
- Engineering design: simulation-based objectives with complex landscapes

### Local vs Global Optimization Landscape

Consider a 2D function like Rastrigin: `f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i))`. It has a global minimum at the origin but is covered in a regular grid of local minima. A gradient-based optimizer started at a random point will almost certainly converge to the wrong local minimum.

**Local optimization** assumes:
- The objective is smooth (or at least locally smooth)
- A good initial guess is available
- Convergence to a nearby stationary point is acceptable

**Global optimization** makes no such assumptions but pays a cost in:
- Many more function evaluations
- No convergence guarantees (heuristic methods) or exponential worst-case (exact methods)
- Sensitivity to algorithm parameters (population size, temperature, etc.)

### Derivative-Free Local Methods

**Nelder-Mead (Downhill Simplex):** Maintains a simplex of n+1 points in n-dimensional space. At each step, the worst vertex is reflected through the centroid of the remaining vertices. The simplex expands, contracts, or shrinks based on the function values at these trial points. No gradient needed — only function comparisons. Convergence is slow (linear) but robust to noise and discontinuities.

**Powell's method:** Performs sequential 1D line searches along conjugate directions. Each iteration searches along each coordinate direction, then replaces the direction of largest decrease. Converges superlinearly for smooth functions. No gradient needed.

### Gradient-Based Local Methods

**BFGS (Broyden-Fletcher-Goldfarb-Shanno):** A quasi-Newton method that builds an approximation to the inverse Hessian using only gradient information. At each step: `x_{k+1} = x_k - alpha_k * H_k * grad f(x_k)`, where `H_k` is updated via the BFGS formula from successive gradient differences. Converges superlinearly. Memory: O(n^2) for the Hessian approximation.

**L-BFGS-B (Limited-memory BFGS with Bounds):** Stores only the last m gradient differences (typically m=10) instead of the full Hessian approximation. Memory: O(mn) instead of O(n^2). Supports simple box bounds on variables. The go-to method for large-scale smooth optimization with bounds.

**Trust-region methods (trust-constr, trust-ncg, trust-krylov):** Instead of choosing a step direction and then a step size (line search), trust-region methods define a region around the current point where the quadratic model is trusted, then find the best step within that region. More robust than line search methods near saddle points and in ill-conditioned problems.

### Population-Based Global Methods

**Differential Evolution (DE):** Maintains a population of candidate solutions. For each member, a mutant vector is created by adding weighted differences of randomly selected population members to a third member. The mutant is crossed with the original to create a trial vector. If the trial is better, it replaces the original. Key parameters: population size (NP), mutation factor (F in [0,2]), crossover rate (CR in [0,1]). DE is highly effective for continuous multimodal problems and is embarrassingly parallel.

Mutation strategies in scipy:
- `best1bin`: mutant = best + F*(r1 - r2) — exploitative
- `rand1bin`: mutant = r1 + F*(r2 - r3) — explorative
- `currenttobest1bin`: mutant = current + F*(best - current) + F*(r1 - r2) — balanced

**CMA-ES (Covariance Matrix Adaptation Evolution Strategy):** Maintains a multivariate Gaussian distribution over the search space. Samples candidate solutions, evaluates them, and updates the mean, covariance matrix, and step size to maximize the probability of generating good solutions. The covariance matrix adapts to the local landscape curvature — effectively learning a second-order model without computing gradients. State-of-the-art for continuous black-box optimization in dimensions 2-100.

### Simulated Annealing Variants

**Classical Simulated Annealing:** Generates random perturbations and accepts worse solutions with probability `exp(-delta_f / T)`, where T (temperature) decreases over time. At high T, almost any move is accepted (exploration). At low T, only improvements are accepted (exploitation). The cooling schedule determines the exploration/exploitation tradeoff.

**Dual Annealing** (scipy's `dual_annealing`): Combines classical simulated annealing with a local search. Uses the **Generalized Simulated Annealing (GSA)** framework with a distorted Cauchy-Lorentz visiting distribution (controlled by parameter `visit`). The visiting distribution has heavier tails than Gaussian, enabling larger jumps to escape deep basins. After each SA phase, a local optimizer refines the best point found. Key parameters:
- `initial_temp`: Starting temperature (default 5230.0). Higher = wider initial exploration.
- `visit`: Controls the tail heaviness of the visiting distribution (default 2.62). Values near 3 give very heavy tails (more global exploration).
- `restart_temp_ratio`: When T drops to `initial_temp * restart_temp_ratio`, temperature is reset (reannealing). Default 2e-5.

**Basin-Hopping:** A two-phase method that alternates random perturbation with local minimization. From the current local minimum, take a random step, run a local optimizer, then accept or reject the new local minimum via a Metropolis criterion. This effectively transforms the energy landscape into a set of "basins" and hops between them. Particularly effective for molecular geometry optimization and cluster problems.

### Bayesian Optimization and TPE

**Bayesian Optimization** is a sequential model-based approach to optimizing expensive black-box functions. It maintains a **surrogate model** of the objective function and uses an **acquisition function** to decide where to evaluate next.

**Tree-structured Parzen Estimator (TPE)** — Optuna's default sampler:

Instead of modeling `p(y|x)` (the objective as a function of parameters) like Gaussian Process BO, TPE models:
- `l(x) = p(x | y < y*)` — distribution of parameters in "good" trials (below threshold y*)
- `g(x) = p(x | y >= y*)` — distribution of parameters in "bad" trials

The acquisition function becomes `l(x) / g(x)` — sample points that are likely in the good distribution and unlikely in the bad distribution. TPE uses kernel density estimation to model l(x) and g(x) independently for each parameter, making it naturally handle:
- Mixed parameter types (float, int, categorical)
- Conditional parameters (parameter B only exists when parameter A = "option1")
- High-dimensional spaces (independent KDE per parameter, no covariance matrix)

TPE is less sample-efficient than GP-based BO in low dimensions but scales much better to high-dimensional and mixed-type spaces.

### Hyperparameter Optimization as Black-Box Optimization

Hyperparameter tuning (learning rate, batch size, architecture choices) is a **black-box optimization** problem:
- The objective (validation loss) has no analytical gradient with respect to hyperparameters
- Each evaluation is expensive (training a model)
- The search space is mixed (continuous, integer, categorical, conditional)
- The landscape is noisy (different random seeds give different results)

This makes it a natural application for Bayesian optimization (Optuna's TPE) rather than grid search or random search. Optuna's define-by-run API lets you express conditional search spaces naturally in Python code, unlike grid/random search which require flat parameter grids.

### Multi-Objective Optimization in Black-Box Settings

When optimizing multiple conflicting objectives (e.g., accuracy vs latency, precision vs recall), there is no single optimal solution — instead, there is a **Pareto front** of solutions where no objective can be improved without degrading another.

Optuna supports multi-objective optimization via `create_study(directions=["minimize", "maximize"])`. The TPE sampler is extended to handle multiple objectives by maintaining separate l(x)/g(x) models per objective and using a multi-objective acquisition function. After optimization, `study.best_trials` returns the set of Pareto-optimal trials.

### Pruning Strategies

In iterative optimization (e.g., training a neural network epoch by epoch), **pruning** stops unpromising trials early to save compute. Optuna supports:

- **MedianPruner**: Prune a trial at step s if its intermediate value is worse than the median of intermediate values of previous trials at the same step. Simple and effective.
- **SuccessiveHalvingPruner**: Allocate resources (steps) in rounds. In each round, keep only the top fraction of trials and double their budget. Based on the Successive Halving algorithm.
- **HyperbandPruner**: Runs multiple brackets of Successive Halving with different initial budgets. Handles the budget/breadth tradeoff automatically.

Pruning requires the objective function to call `trial.report(value, step)` at intermediate steps and check `trial.should_prune()`.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Multimodal function** | Function with multiple local minima — local optimization fails to find the global optimum |
| **Basin of attraction** | Region of initial points from which a local optimizer converges to the same local minimum |
| **Derivative-free** | Optimization using only function values, no gradients (Nelder-Mead, Powell, DE) |
| **Quasi-Newton** | Methods that approximate the Hessian using gradient information only (BFGS, L-BFGS) |
| **Population-based** | Methods maintaining a set of candidate solutions that evolve together (DE, CMA-ES) |
| **Simulated annealing** | Probabilistic method accepting worse solutions with decreasing probability (temperature) |
| **Bayesian optimization** | Sequential optimization using a surrogate model + acquisition function |
| **TPE** | Tree-structured Parzen Estimator — models good/bad parameter distributions separately |
| **Acquisition function** | Function that balances exploration vs exploitation when choosing next evaluation point |
| **Pareto front** | Set of solutions where no objective can be improved without worsening another |
| **Pruning** | Early stopping of unpromising trials based on intermediate results |
| **Surrogate model** | Cheap approximation of the expensive objective function (GP, TPE, random forest) |
| **Cooling schedule** | How temperature decreases in simulated annealing — controls exploration/exploitation |
| **Crossover (CR)** | In DE, probability of inheriting components from the mutant vector |
| **Mutation factor (F)** | In DE, scale factor for the difference vectors used in mutation |

### Where Global Optimization Fits in the Ecosystem

```
Optimization Methods
├── Exact (convex, guaranteed global optimum)
│   ├── LP/QP solvers .............. HiGHS, OSQP (practices 032a, 040a)
│   ├── Conic solvers .............. CLARABEL, SCS, MOSEK (practice 041a/b)
│   └── MIP solvers ............... HiGHS, CP-SAT (practices 034a, 040a, 042a)
│
├── Local (smooth, gradient-based or derivative-free)
│   ├── scipy.optimize.minimize ... Nelder-Mead, BFGS, L-BFGS-B, trust-constr ← Phase 1
│   └── NLopt ..................... Large collection of local NLP solvers
│
├── Global (heuristic, population/stochastic-based)
│   ├── scipy.optimize ............ DE, dual_annealing, shgo, basinhopping ← Phase 2
│   ├── pymoo ..................... Multi-objective EA (NSGA-II, NSGA-III) (practice 044)
│   └── Custom implementations .... SA, GA, PSO (practices 035a/b, 038)
│
└── Black-box / Bayesian
    ├── Optuna .................... TPE, CMA-ES, multi-objective, pruning ← Phases 3-4
    ├── Ax (Meta) ................. GP-based BO
    ├── SMAC3 ..................... Random forest surrogate
    └── Hyperopt .................. TPE (predecessor to Optuna)
```

scipy.optimize provides the classical algorithmic toolkit (both local and global). Optuna adds the Bayesian optimization layer specifically designed for expensive black-box objectives with mixed parameter types. Together, they cover the full spectrum of continuous global optimization needs in Python.

## Description

Apply local and global optimization methods from scipy.optimize and Bayesian optimization from Optuna to multimodal test functions and synthetic black-box problems. Progress from classical local methods through population-based global methods to Bayesian hyperparameter optimization and multi-objective optimization.

### What you'll build

1. **Classical local optimization** — Compare Nelder-Mead, BFGS, and L-BFGS-B on the Rosenbrock function with bounds and constraints. Visualize convergence paths.
2. **Global optimization methods** — Apply differential evolution, dual annealing, SHGO, and basin-hopping to multimodal test functions (Rastrigin, Ackley, Schwefel). Compare success rates and convergence.
3. **Optuna basics** — Create an Optuna study with TPE sampler to optimize a synthetic objective with mixed parameter types. Configure pruning and visualize optimization history.
4. **Optuna advanced** — Multi-objective optimization with Pareto front visualization, conditional search spaces, and sampler comparison.

## Instructions

### Phase 1: scipy.optimize — Classical Local Methods (~25 min)

**File:** `src/phase1_scipy_classical.py`

This phase teaches the fundamental scipy.optimize workflow for local optimization. You'll minimize the Rosenbrock function — a classic non-convex test function with a narrow curved valley — using three different methods. The key learning: derivative-free methods (Nelder-Mead) are robust but slow; gradient-based methods (BFGS, L-BFGS-B) are fast but need smooth functions; bounded/constrained optimization requires specific methods.

**What you implement:**
- `optimize_unconstrained(x0, method)` — Set up `scipy.optimize.minimize` with a given method and starting point, capture the convergence history via callback, return the result.
- `optimize_bounded(x0)` — Add box bounds to the Rosenbrock optimization using L-BFGS-B (the only built-in method supporting simple bounds directly). Compare the bounded vs unbounded solution.
- `optimize_constrained(x0)` — Add a nonlinear constraint (e.g., `x[0]^2 + x[1]^2 <= 1.5`) using `scipy.optimize.minimize` with method `trust-constr` or `SLSQP`. This demonstrates how constraints change the optimal point.

**Why it matters:** Local optimization is the foundation. Even global methods use local optimizers internally (dual_annealing and basinhopping run `minimize` after each global step). Understanding convergence behavior, method selection, and constraint handling is prerequisite knowledge for everything that follows.

### Phase 2: scipy.optimize — Global Methods (~25 min)

**File:** `src/phase2_scipy_global.py`

This phase applies scipy's four global optimization algorithms to multimodal test functions where local methods fail. Each algorithm uses a fundamentally different strategy to escape local minima: population evolution (DE), temperature-based acceptance (dual annealing), topological decomposition (SHGO), and perturbation + local search (basinhopping).

**What you implement:**
- `optimize_differential_evolution(func, bounds)` — Configure and run `differential_evolution` with specified strategy, population size, and tolerance. Record the convergence callback.
- `optimize_dual_annealing(func, bounds)` — Configure and run `dual_annealing` with appropriate temperature and visit parameters.
- `optimize_shgo(func, bounds)` — Run `shgo` (simplicial homology global optimization) which finds ALL local minima via topological methods.
- `optimize_basinhopping(func, x0)` — Run `basinhopping` from a starting point with appropriate step size and temperature.

**Why it matters:** Choosing the right global optimizer depends on the problem structure. DE is robust for high-dimensional continuous problems. Dual annealing is effective for very rugged landscapes. SHGO can enumerate all local minima (unique among scipy methods). Basinhopping excels when local minima form a structured hierarchy (common in chemistry/physics).

### Phase 3: Optuna — Bayesian Hyperparameter Optimization (~25 min)

**File:** `src/phase3_optuna_basics.py`

This phase introduces Optuna's define-by-run API for black-box optimization. Instead of specifying the search space upfront (as in scipy), you define it inline during each trial. You'll create a study, define an objective with mixed parameter types, configure pruning, and compare TPE against random sampling.

**What you implement:**
- `objective(trial)` — Define a trial objective that uses `trial.suggest_float`, `trial.suggest_int`, and `trial.suggest_categorical` to sample parameters, evaluates a synthetic objective, reports intermediate values for pruning, and returns the final value.
- `run_study(sampler, pruner, n_trials)` — Create an Optuna study with specified sampler and pruner, run optimization, return the study for analysis.

**Why it matters:** Optuna's define-by-run API is the modern standard for hyperparameter optimization. The TPE sampler is more sample-efficient than random/grid search. Pruning saves compute by stopping bad trials early. Understanding these concepts is essential for any ML workflow that involves model selection or hyperparameter tuning.

### Phase 4: Optuna — Multi-Objective & Advanced Features (~25 min)

**File:** `src/phase4_optuna_advanced.py`

This phase covers multi-objective optimization and conditional search spaces — two features that make Optuna production-ready. You'll optimize two conflicting objectives simultaneously, implement conditional parameter spaces (where some parameters only exist when another parameter takes a certain value), and visualize the Pareto front.

**What you implement:**
- `multi_objective(trial)` — Define a trial with two conflicting objectives (e.g., accuracy proxy vs latency proxy). Return both values as a tuple.
- `conditional_objective(trial)` — Define a trial where some parameters depend on the value of other parameters (e.g., if `optimizer == "sgd"`, suggest `momentum`; if `optimizer == "adam"`, suggest `beta1`, `beta2`). This demonstrates Optuna's natural support for tree-structured search spaces.
- `run_multi_objective_study(n_trials)` — Create a multi-objective study with `directions=["minimize", "minimize"]`, optimize, extract the Pareto front, and visualize it.

**Why it matters:** Real optimization problems often have multiple conflicting objectives. The Pareto front reveals the tradeoff surface, letting decision-makers choose the best compromise. Conditional search spaces are ubiquitous in ML (different model architectures have different hyperparameters). Optuna handles both naturally — features that grid search and scipy cannot express.

## Motivation

Global optimization and hyperparameter tuning are ubiquitous in:
- **Machine Learning**: model selection, architecture search, learning rate scheduling
- **Engineering design**: simulation-based optimization with complex landscapes
- **Operations Research**: non-convex objective functions arising from real-world constraints
- **Finance**: portfolio optimization with non-convex risk measures, strategy parameter tuning

After implementing metaheuristics from scratch (practices 035a/b — SA, genetic algorithms) and using convex optimization tools (practice 041a — CVXPY), this practice bridges the gap to production-ready black-box optimization. scipy.optimize provides the classical algorithmic toolkit, while Optuna adds modern Bayesian optimization with features specifically designed for ML workflows (mixed types, conditional spaces, pruning, multi-objective).

Understanding when to use local vs global methods, population-based vs surrogate-based approaches, and how to properly configure these tools is essential for any optimization practitioner.

## Commands

All commands are run from the `practice_043_global_optimization/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (scipy, Optuna, NumPy, Matplotlib) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_scipy_classical.py` | Phase 1: Local optimization — Nelder-Mead, BFGS, L-BFGS-B on Rosenbrock with bounds and constraints |
| `uv run python src/phase2_scipy_global.py` | Phase 2: Global optimization — DE, dual annealing, SHGO, basinhopping on multimodal test functions |
| `uv run python src/phase3_optuna_basics.py` | Phase 3: Optuna basics — TPE sampler, mixed parameter types, pruning, optimization history |
| `uv run python src/phase4_optuna_advanced.py` | Phase 4: Optuna advanced — multi-objective Pareto front, conditional search spaces, sampler comparison |

## References

- [scipy.optimize — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/optimize.html) — Full API reference for all optimization functions
- [scipy.optimize.minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) — Local minimization with method selection
- [scipy.optimize.differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html) — Population-based global optimizer
- [scipy.optimize.dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html) — Generalized simulated annealing
- [scipy.optimize.shgo](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.shgo.html) — Simplicial homology global optimization
- [scipy.optimize.basinhopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) — Basin-hopping algorithm
- [Optuna Documentation](https://optuna.readthedocs.io/en/stable/) — Official Optuna docs
- [Optuna TPE Sampler](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) — Tree-structured Parzen Estimator
- [Optuna Visualization](https://optuna.readthedocs.io/en/stable/reference/visualization/index.html) — Built-in plotting functions
- [Optuna Multi-Objective Example](https://github.com/optuna/optuna-examples/blob/main/basic/quadratic_multi_objective.py) — Official multi-objective example
- [Optuna MedianPruner](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html) — Median stopping rule for pruning

## State

`not-started`
