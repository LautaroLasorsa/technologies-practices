# Practice 047: Bayesian Optimization

## Technologies

- **scikit-learn (GaussianProcessRegressor)** — Gaussian Process regression for building surrogate models with configurable kernels (RBF, Matern, etc.), posterior mean/variance prediction, and kernel hyperparameter optimization via marginal likelihood.
- **scikit-optimize (skopt)** — Sequential model-based optimization library providing `gp_minimize` for GP-based Bayesian optimization, built on top of scikit-learn.
- **Optuna** — Bayesian hyperparameter optimization framework with Tree-structured Parzen Estimator (TPE) sampler for comparison against GP-based BO.
- **NumPy** — Array operations, linear algebra, and random number generation.
- **SciPy** — Optimization of acquisition functions (`minimize`), statistical functions (normal CDF/PDF).
- **Matplotlib** — Visualization of GP posteriors, acquisition functions, convergence curves, and Pareto fronts.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- scikit-learn >= 1.4 (Gaussian Process regression, kernels)
- scikit-optimize >= 0.10 (gp_minimize)
- Optuna >= 3.5 (TPE comparison)
- NumPy >= 1.26, SciPy >= 1.12, Matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### What Bayesian Optimization Is and the Problem It Solves

**Bayesian Optimization (BO)** is a sequential strategy for optimizing **expensive black-box functions**. Unlike gradient-based or population-based methods (practices 033a, 035a), BO builds a probabilistic **surrogate model** of the objective function and uses it to decide where to evaluate next. The key insight: it is much cheaper to evaluate the surrogate than the true objective, so BO can make intelligent decisions about where to sample with far fewer function evaluations than methods that treat the objective as a simple oracle.

**When BO is appropriate:**
- **Expensive evaluations**: Each function evaluation takes minutes, hours, or dollars (e.g., training a neural network, running a physics simulation, conducting a lab experiment)
- **No gradient available**: The objective is a black box — you get `f(x)` but not `nabla f(x)`
- **Low to moderate dimensions**: BO works best in ~2-20 dimensions; above ~50 it struggles
- **Smooth-ish landscape**: The objective should have some structure that a GP can capture (not pure noise)

**Comparison with other black-box methods:**
```
Method              | Function evals | Dimensionality | Type
--------------------|----------------|----------------|------------------
Grid Search         | O(k^d)         | Very low (d<5) | Exhaustive
Random Search       | O(budget)      | Any            | No model
Bayesian Opt (GP)   | O(10-100)      | Low (d<20)     | Sequential, model-based
TPE (Optuna)        | O(100-1000)    | Medium (d<100) | Sequential, model-based
CMA-ES              | O(1000+)       | Medium (d<100) | Population-based
Differential Evol.  | O(1000+)       | Medium (d<100) | Population-based
```

BO is the method of choice when each evaluation is expensive enough that you can afford the overhead of fitting a surrogate model and optimizing an acquisition function between evaluations.

### Gaussian Processes: Prior Over Functions

A **Gaussian Process (GP)** is a distribution over functions. Any finite collection of function values `[f(x_1), ..., f(x_n)]` follows a multivariate Gaussian distribution. A GP is fully specified by:

- **Mean function** `m(x)`: the expected value of f(x). Often assumed to be zero or a constant (the GP learns the mean from data).
- **Covariance function (kernel)** `k(x, x')`: determines how correlated `f(x)` and `f(x')` are. The kernel encodes our assumptions about the function's smoothness, periodicity, and amplitude.

Given `n` observed points `X = [x_1, ..., x_n]` with values `y = [y_1, ..., y_n]`, the GP posterior at a new point `x*` is:

```
mu(x*)    = k(x*, X) @ [K(X,X) + sigma_n^2 * I]^{-1} @ y
sigma^2(x*) = k(x*, x*) - k(x*, X) @ [K(X,X) + sigma_n^2 * I]^{-1} @ k(X, x*)
```

Where:
- `K(X, X)` is the n x n kernel matrix with `K_{ij} = k(x_i, x_j)`
- `sigma_n^2` is observation noise variance
- `mu(x*)` is the posterior mean (best estimate of f(x*))
- `sigma^2(x*)` is the posterior variance (uncertainty about f(x*))

**Key property**: The GP is exact at observed points (zero variance if noise-free) and reverts to the prior far from data (high variance). This is exactly what BO needs: confident predictions near known points, honest uncertainty far from them.

### Kernels (Covariance Functions)

The kernel is the most important modeling choice in a GP. It determines what kinds of functions the GP considers plausible.

**RBF (Squared Exponential / Radial Basis Function):**
```
k(x, x') = sigma_f^2 * exp(-||x - x'||^2 / (2 * l^2))
```
- `sigma_f^2`: signal variance (amplitude of function variations)
- `l`: length scale (how far apart inputs must be to be uncorrelated)
- Produces **infinitely differentiable** (very smooth) functions
- Default choice when you expect a smooth objective
- Can be too smooth for rugged landscapes (over-smooths real variations)

**Matern 3/2:**
```
k(x, x') = sigma_f^2 * (1 + sqrt(3)*r/l) * exp(-sqrt(3)*r/l)    where r = ||x - x'||
```
- Produces **once-differentiable** functions
- More realistic for many physical and engineering problems
- The default in many BO libraries (scikit-optimize uses Matern 5/2)
- Better at capturing sharp features than RBF

**Matern 5/2:**
```
k(x, x') = sigma_f^2 * (1 + sqrt(5)*r/l + 5*r^2/(3*l^2)) * exp(-sqrt(5)*r/l)
```
- Produces **twice-differentiable** functions
- Balances smoothness (RBF) and roughness (Matern 3/2)
- The most common kernel in Bayesian optimization practice
- Recommended default by Snoek et al. (2012) in "Practical Bayesian Optimization of Machine Learning Algorithms"

**Kernel composition**: Kernels can be added and multiplied to express complex structure:
- `k1 + k2`: function is sum of two independent processes (one smooth, one rough)
- `k1 * k2`: function has properties of both kernels (e.g., periodic * RBF = locally periodic)
- `ConstantKernel * Matern + WhiteKernel`: standard BO kernel (signal + noise)

### Kernel Hyperparameters and Marginal Likelihood

Kernel hyperparameters (length scale `l`, signal variance `sigma_f^2`, noise `sigma_n^2`) are typically optimized by maximizing the **log marginal likelihood**:

```
log p(y | X, theta) = -1/2 * y^T @ K_y^{-1} @ y - 1/2 * log|K_y| - n/2 * log(2*pi)
```

Where `K_y = K(X,X) + sigma_n^2 * I` and `theta = {l, sigma_f, sigma_n}`.

This balances:
- **Data fit** (first term): how well the GP explains the observations
- **Complexity penalty** (second term): simpler models (larger length scales) are preferred
- This is automatic Occam's razor — no cross-validation needed

scikit-learn's `GaussianProcessRegressor` optimizes this internally via `optimizer="fmin_l_bfgs_b"` with `n_restarts_optimizer` random restarts to avoid local optima.

### Acquisition Functions

Given the GP posterior `(mu(x), sigma(x))` and the current best observation `f(x+) = min(y_1, ..., y_n)`, acquisition functions decide where to evaluate next by trading off **exploration** (high uncertainty) and **exploitation** (low predicted value).

**Probability of Improvement (PI):**
```
PI(x) = Phi((f(x+) - mu(x) - xi) / sigma(x))
```
Where `Phi` is the standard normal CDF, `xi >= 0` is a trade-off parameter.
- Returns the probability that `f(x)` will be better than `f(x+) - xi`
- Pure exploitation when `xi = 0`: only cares about probability, not magnitude of improvement
- Tends to over-exploit: samples near the current best, missing potentially better regions
- Simple to compute and understand, but rarely used in practice due to myopia

**Expected Improvement (EI):**
```
EI(x) = (f(x+) - mu(x) - xi) * Phi(Z) + sigma(x) * phi(Z)
        where Z = (f(x+) - mu(x) - xi) / sigma(x)
```
Where `phi` is the standard normal PDF.
- The **expected amount of improvement** over the current best
- First term: exploitation (how much better the mean is than current best)
- Second term: exploration (proportional to uncertainty)
- When `sigma(x) = 0`: EI = max(f(x+) - mu(x), 0) (pure exploitation)
- When `mu(x) >> f(x+)`: EI is driven by the sigma term (explores uncertain regions)
- The most popular acquisition function — good balance without much tuning
- `xi` parameter controls exploration: higher xi = more exploration

**Upper Confidence Bound (UCB / GP-UCB):**
```
UCB(x) = mu(x) - kappa * sigma(x)    (for minimization)
```
- `kappa > 0` controls exploration-exploitation tradeoff
- Linear combination: subtract predicted value (exploitation) + add uncertainty bonus (exploration)
- Theoretical guarantees: Srinivas et al. (2010) prove sublinear regret with `kappa = sqrt(2 * log(t * d^2 * pi^2 / (6 * delta)))` where t = iteration, d = dimension, delta = failure probability
- In practice, `kappa = 2.0` is a good default; increase for more exploration
- Simpler to implement and reason about than EI

### The Bayesian Optimization Loop

```
1. Initialize: Evaluate f(x) at a few random points (Latin hypercube or Sobol)
2. Fit GP: Condition the GP on all observed (x, y) pairs, optimize kernel hyperparameters
3. Maximize acquisition: Find x_next = argmax alpha(x) over the domain
   (This is a cheap inner optimization — use L-BFGS-B with multiple restarts)
4. Evaluate: y_next = f(x_next)   ← The expensive step
5. Update: Add (x_next, y_next) to observations
6. Repeat steps 2-5 until budget exhausted
```

**Initialization**: Typically 5-10 random points (or `2*d` where d = dimension). Too few initial points = poor GP fit and erratic early behavior. Too many = wasted budget on random exploration.

**GP fitting**: O(n^3) in the number of observations due to matrix inversion. This is fine for BO (n is typically 10-200) but limits scalability to very large datasets.

**Acquisition optimization**: Must be done carefully — acquisition functions are themselves multimodal. Use L-BFGS-B from multiple random restarts (e.g., 20-50 restarts). This is cheap because it only evaluates the GP, not the true objective.

### Thompson Sampling as Alternative to Acquisition Functions

Instead of computing an acquisition function, **Thompson Sampling** draws a random function from the GP posterior and optimizes it:

```
1. Sample f_t ~ GP(mu, K)   (a single function draw from the posterior)
2. x_next = argmin f_t(x)    (optimize the sample, not the posterior)
3. Evaluate y_next = f(x_next)
4. Update GP
```

Thompson Sampling naturally balances exploration and exploitation:
- In uncertain regions, posterior samples vary widely, so different samples will explore different areas
- In well-explored regions, samples are similar, so it exploits the known good areas
- No acquisition function hyperparameters to tune (kappa, xi)

In practice, Thompson Sampling is approximated by drawing from the GP at a finite set of points and optimizing the resulting function.

### Constrained Bayesian Optimization

When the objective has unknown constraints `c_j(x) <= 0` that are also expensive to evaluate:

**Feasibility approach**: Model each constraint with a separate GP:
```
p(feasible | x) = product_j Phi(-mu_cj(x) / sigma_cj(x))
```
Then multiply the acquisition function by the feasibility probability:
```
alpha_constrained(x) = EI(x) * p(feasible | x)
```

This is called **Expected Feasible Improvement (EFI)** — it drives sampling toward regions that are both improving AND likely feasible. Points that are uncertain about feasibility are explored (because `p(feasible)` contributes to exploration), while clearly infeasible regions are avoided.

### Batch Bayesian Optimization

When you can evaluate `q` points in parallel (e.g., multiple GPUs, multiple lab experiments):

**Kriging Believer**: After selecting `x_1` via the acquisition function, "fantasize" its outcome as `y_1 = mu(x_1)` (the GP mean), update the GP with this hallucinated observation, then select `x_2` from the updated GP. Repeat q times. This is a greedy sequential heuristic — fast but may under-explore.

**Local Penalization**: After selecting `x_1`, add a penalty around `x_1` to the acquisition function to prevent nearby selections. The penalty is a Gaussian bump centered at `x_1` with width proportional to the GP length scale.

**qEI (q-Expected Improvement)**: The theoretically correct approach — compute the joint expected improvement of all q points simultaneously. Requires O(q!) computation in the naive case, but Monte Carlo approximations (as in BoTorch) make it tractable.

### High-Dimensional BO Challenges

Standard GP-based BO degrades above ~20 dimensions because:
- The GP posterior becomes uninformative (curse of dimensionality)
- Acquisition function optimization becomes harder (multimodal in high-d)
- Kernel hyperparameter optimization has more local optima

**Solutions:**
- **Random embeddings (REMBO)**: Project from high-d to low-d random subspace, run BO in the subspace. Works when the function has low effective dimensionality.
- **Trust regions (TuRBO)**: Maintain a local trust region that expands/contracts based on optimization progress. Run BO only within the trust region.
- **Additive models**: Assume the kernel decomposes as `k(x,x') = sum_g k_g(x_g, x'_g)` where each `k_g` operates on a small subset of dimensions.
- **Use TPE instead**: Optuna's TPE handles dimensions independently (no covariance matrix), scaling much better to 100+ dimensions.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Surrogate model** | Cheap probabilistic approximation of the expensive objective (GP, random forest, TPE) |
| **Gaussian Process** | Distribution over functions defined by mean function and kernel; provides posterior mean + variance |
| **Kernel (covariance function)** | Determines GP smoothness and correlation structure; RBF, Matern 3/2, Matern 5/2 |
| **Length scale** | Kernel hyperparameter controlling how quickly correlation decays with distance |
| **Marginal likelihood** | Probability of observations under the GP model; optimized to set kernel hyperparameters |
| **Acquisition function** | Utility function balancing exploration/exploitation to select next evaluation point |
| **Expected Improvement (EI)** | Expected amount of improvement over current best; most popular acquisition function |
| **Probability of Improvement (PI)** | Probability of beating current best; tends to over-exploit |
| **Upper Confidence Bound (UCB)** | Mean minus kappa*sigma; linear exploration-exploitation tradeoff with theoretical guarantees |
| **Thompson Sampling** | Draw random function from GP posterior and optimize it; parameter-free alternative |
| **Kriging Believer** | Batch BO heuristic: fantasize outcomes as GP mean, select points sequentially |
| **Expected Feasible Improvement** | EI * P(feasible); acquisition function for constrained BO |
| **Regret** | Difference between best possible value and best found; BO achieves sublinear regret |

### Where Bayesian Optimization Fits in the Ecosystem

```
Black-Box Optimization Methods
|
+-- Model-free
|   +-- Grid Search .............. Exhaustive, exponential in d
|   +-- Random Search ............ Better than grid (Bergstra & Bengio 2012)
|   +-- Evolutionary (DE, GA) .... Population-based, 1000+ evals (practices 035a/b)
|   +-- CMA-ES ................... Adaptive covariance, 100-10000 evals
|
+-- Model-based (surrogate-assisted)
    +-- GP-based BO .............. This practice (phases 1-3)
    |   +-- scikit-optimize ...... gp_minimize
    |   +-- BoTorch .............. PyTorch-based, qEI, high-dimensional
    |   +-- Ax (Meta) ............ Production BO platform
    |
    +-- TPE ...................... Optuna, Hyperopt (practice 043 phase 3-4)
    |
    +-- Random Forest ........... SMAC3 (tree-based surrogate)
    |
    +-- Neural Network .......... DeepHyper, BOHB
```

GP-based BO is the gold standard for low-dimensional expensive optimization. TPE (Optuna) scales better to high dimensions and mixed parameter types. Both are sequential model-based methods — they share the "fit surrogate, optimize acquisition, evaluate" loop but differ in the surrogate model.

## Description

Build Bayesian Optimization from the ground up: start by understanding Gaussian Process regression as a surrogate model, then implement acquisition functions (EI, PI, UCB) that guide sampling, combine them into a complete BO loop, and finally tackle advanced extensions including constraints, batch suggestions, and high-dimensional challenges.

### What you'll build

1. **GP Surrogate Modeling** — Fit Gaussian Processes to noisy observations of 1D test functions, visualize posterior mean and uncertainty bands, experiment with kernel choices and hyperparameters.
2. **Acquisition Functions** — Implement EI, PI, and UCB from the GP posterior, visualize them over the domain, and observe how each balances exploration vs exploitation.
3. **Full BO Loop** — Implement a manual BO loop (evaluate -> fit GP -> optimize acquisition -> repeat) and compare with scikit-optimize's `gp_minimize` on benchmark functions.
4. **Advanced BO** — Constrained BO with feasibility GP, batch suggestions via kriging believer, and comparison with Optuna's TPE on mixed-variable problems.

## Instructions

### Phase 1: Gaussian Process Surrogate (~25 min)

**File:** `src/phase1_gp_surrogate.py`

This phase teaches Gaussian Process regression as the foundation of Bayesian optimization. You'll fit GPs to sampled points from test functions, visualize how the posterior mean and uncertainty evolve as data points are added, and experiment with different kernels. The key learning: the GP posterior provides not just predictions but calibrated uncertainty estimates, which is exactly what acquisition functions need to decide where to sample next.

**What you implement:**
- `fit_gp(X_train, y_train, kernel)` — Create and fit a scikit-learn GaussianProcessRegressor with a specified kernel, returning the fitted model. You'll configure noise handling (alpha parameter), kernel hyperparameter optimization (n_restarts_optimizer), and normalization.
- `predict_with_uncertainty(gp, X_test)` — Use the fitted GP to predict mean and standard deviation at test points. This is the GP posterior that acquisition functions will consume.
- `experiment_kernels(X_train, y_train, X_test)` — Fit GPs with RBF, Matern(nu=1.5), and Matern(nu=2.5) kernels to the same data, compare posterior predictions and learned hyperparameters.

**Why it matters:** The GP is the engine of Bayesian optimization. Its posterior mean provides the current best estimate of the objective, and its posterior variance quantifies where we are uncertain — exactly the two quantities that acquisition functions combine. Understanding how kernel choice affects the posterior (smoother vs rougher, over-confident vs well-calibrated) directly impacts BO performance.

### Phase 2: Acquisition Functions (~25 min)

**File:** `src/phase2_acquisition.py`

This phase implements the three major acquisition functions from the GP posterior and visualizes how they choose the next evaluation point. You'll see how PI over-exploits, EI balances well, and UCB's behavior depends on kappa. The key learning: the acquisition function IS the strategy of BO — the GP is just the data structure; the acquisition function makes the decisions.

**What you implement:**
- `probability_of_improvement(mu, sigma, f_best, xi)` — Implement PI from the GP posterior mean and variance, using the standard normal CDF.
- `expected_improvement(mu, sigma, f_best, xi)` — Implement EI with both the exploitation term (Phi) and exploration term (phi).
- `upper_confidence_bound(mu, sigma, kappa)` — Implement UCB (negated for minimization) as mu - kappa * sigma.

**Why it matters:** Acquisition functions embody the exploration-exploitation tradeoff that makes BO sample-efficient. Implementing them from scratch (rather than using a library) forces understanding of how the GP posterior's two outputs (mean, variance) are combined. The visualization reveals that different acquisition functions have different "personalities" — PI is greedy, EI is balanced, UCB is tunable.

### Phase 3: Full Bayesian Optimization Loop (~25 min)

**File:** `src/phase3_bo_loop.py`

This phase combines the GP and acquisition functions into a complete BO loop. You'll implement the full cycle (evaluate -> fit GP -> maximize acquisition -> evaluate) manually, then compare your implementation against scikit-optimize's `gp_minimize` on the same benchmark functions. The key learning: the BO loop is conceptually simple but the details matter — initialization strategy, acquisition optimization quality, and GP fitting affect convergence dramatically.

**What you implement:**
- `bayesian_optimization_loop(objective, bounds, n_init, n_iter, acquisition)` — The complete manual BO loop: Latin hypercube initialization, GP fitting at each step, acquisition function maximization via scipy.optimize, and convergence tracking. Returns the optimization history.
- `compare_with_skopt(objective, bounds, n_calls)` — Run scikit-optimize's `gp_minimize` on the same objective and compare convergence with your manual implementation.

**Why it matters:** Implementing the loop yourself reveals the engineering decisions that BO libraries make: How many initial random points? How many restarts for acquisition optimization? When to refit kernel hyperparameters? Comparing with `gp_minimize` shows where your manual implementation matches or diverges from a production library, building intuition for when to use libraries vs custom implementations.

### Phase 4: Advanced BO — Constraints & High Dimensions (~25 min)

**File:** `src/phase4_advanced_bo.py`

This phase extends BO to constrained optimization (where the feasible region is unknown and expensive to evaluate), batch suggestions (selecting multiple points to evaluate in parallel), and mixed-variable problems. You'll implement constrained BO by modeling constraints with separate GPs, batch suggestions via kriging believer, and compare GP-based BO against Optuna's TPE on a mixed-variable problem.

**What you implement:**
- `constrained_bo_loop(objective, constraint, bounds, n_init, n_iter)` — Constrained BO: fit separate GPs for objective and constraint, compute Expected Feasible Improvement (EI * P(feasible)), and optimize. Track both objective value and constraint satisfaction.
- `kriging_believer_batch(gp, acquisition_func, bounds, batch_size)` — Batch suggestion: select batch_size points sequentially, fantasizing each selected point's outcome as the GP mean before selecting the next.
- `compare_bo_vs_tpe(objective, param_space, n_trials)` — Compare GP-based BO (skopt) against Optuna's TPE on a problem with mixed continuous/integer/categorical variables.

**Why it matters:** Real optimization problems have constraints, parallelism opportunities, and mixed variable types. Constrained BO with feasibility GPs is the standard approach in engineering design. Kriging believer is the simplest batch BO method and reveals how "fantasizing" outcomes enables parallelism. The TPE comparison shows that GP-based BO is not always best — TPE handles high dimensions and mixed types more naturally.

## Motivation

Bayesian Optimization is the state-of-the-art method for optimizing expensive black-box functions in low to moderate dimensions. It is central to:

- **Machine Learning**: Hyperparameter tuning (the original motivation for Spearmint, Hyperopt, Optuna), neural architecture search, AutoML pipelines
- **Engineering Design**: Simulation-based optimization where each evaluation is a CFD run, FEA simulation, or materials experiment
- **Drug Discovery**: Molecular property optimization where each evaluation requires lab synthesis and testing
- **Robotics**: Controller parameter tuning where each evaluation is a physical experiment
- **Operations Research**: Tuning heuristic/metaheuristic parameters (e.g., SA temperature schedule, GA crossover rate)

After practice 043 (scipy global optimization + Optuna), this practice goes deeper into the GP-based branch of surrogate-assisted optimization. Understanding GP posteriors, acquisition functions, and the BO loop from scratch provides the theoretical foundation to use libraries like scikit-optimize, BoTorch, and Ax effectively, and to extend BO to constrained, batch, and high-dimensional settings.

**References:**
- [Practical Bayesian Optimization of Machine Learning Algorithms (Snoek et al. 2012)](https://arxiv.org/abs/1206.2944)
- [A Tutorial on Bayesian Optimization (Brochu et al. 2010)](https://arxiv.org/abs/1012.2599)
- [Gaussian Processes for Machine Learning (Rasmussen & Williams 2006)](http://www.gaussianprocess.org/gpml/)
- [Taking the Human Out of the Loop (Shahriari et al. 2016)](https://ieeexplore.ieee.org/document/7352306)
- [scikit-learn GaussianProcessRegressor docs](https://scikit-learn.org/stable/modules/gaussian_process.html)
- [scikit-optimize docs](https://scikit-optimize.github.io/stable/)
- [Acquisition Functions in Bayesian Optimization](https://ekamperi.github.io/machine%20learning/2021/06/11/acquisition-functions.html)
- [Bayesian Optimization — Cornell Optimization Wiki](https://optimization.cbe.cornell.edu/index.php?title=Bayesian_Optimization)

## Commands

All commands are run from the `practice_047_bayesian_optimization/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (scikit-learn, scikit-optimize, Optuna, NumPy, SciPy, Matplotlib) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_gp_surrogate.py` | Phase 1: Gaussian Process surrogate — fit GPs, visualize posterior mean + uncertainty, compare kernels |
| `uv run python src/phase2_acquisition.py` | Phase 2: Acquisition functions — implement EI, PI, UCB, visualize exploration-exploitation tradeoff |
| `uv run python src/phase3_bo_loop.py` | Phase 3: Full BO loop — manual BO implementation vs scikit-optimize gp_minimize |
| `uv run python src/phase4_advanced_bo.py` | Phase 4: Advanced BO — constrained BO, kriging believer batch, GP-BO vs TPE comparison |

## State

`not-started`
