# Practice 033a: Convex Optimization -- First-Order Methods

## Technologies

- **C++17** -- Modern C++ with structured bindings, lambdas, auto
- **Eigen** -- Header-only linear algebra library (via CMake FetchContent)
- **CMake 3.16+** -- Build system with FetchContent for dependency management

## Stack

- C++17
- Eigen 3.4 (fetched via CMake FetchContent)

## Theoretical Context

First-order methods are optimization algorithms that use only gradient information (first derivatives) to find minima. They are the workhorse of modern optimization: gradient descent powers ML training (SGD), signal processing, and large-scale OR. Understanding their convergence theory is essential for diagnosing slow training, choosing step sizes, and selecting appropriate solvers.

### Convex Sets and Convex Functions

A **convex set** C is one where, for any two points x, y in C, the entire line segment between them lies in C:

    lambda * x + (1 - lambda) * y in C,  for all lambda in [0, 1]

Examples of convex sets:
- **Polyhedra**: {x : Ax <= b} -- intersections of half-spaces (e.g., feasible regions of LPs)
- **Balls**: {x : ||x - c|| <= r} -- Euclidean balls, L-infinity boxes
- **Cones**: {x : x >= 0} -- the non-negative orthant; second-order cones {(x,t) : ||x|| <= t}
- **Hyperplanes and half-spaces**: {x : a^T x = b} and {x : a^T x <= b}

A function f is **convex** if its domain is a convex set and:

    f(lambda * x + (1 - lambda) * y)  <=  lambda * f(x) + (1 - lambda) * f(y)

for all x, y in dom(f) and lambda in [0, 1]. Geometrically, the function lies below any chord connecting two points on its graph.

A function is **strictly convex** if the inequality is strict for x != y, and **strongly convex with parameter mu > 0** if f(x) - (mu/2)||x||^2 is convex. Strong convexity implies a unique global minimum and faster convergence.

Examples of convex functions:
- **Quadratic**: f(x) = 0.5 * x^T Q x + q^T x, when Q is positive semidefinite
- **Norms**: ||x||_1, ||x||_2, ||x||_inf
- **Log-sum-exp**: log(sum(exp(a_i^T x + b_i)))
- **Negative log**: -log(x) on x > 0

### Why Convexity Matters

The fundamental theorem of convex optimization: **every local minimum of a convex function is a global minimum.** This means:
- Gradient descent is guaranteed to find the optimum (no getting stuck in local minima)
- There is rich convergence theory with provable rates
- Duality theory provides certificates of optimality (KKT conditions)
- Efficient algorithms exist for problems with millions of variables

Non-convex optimization (neural network training, combinatorial optimization) inherits many techniques from convex optimization but loses the global optimality guarantee.

### Gradient Descent

The simplest first-order method. Starting from x_0, iterate:

    x_{k+1} = x_k - alpha * gradient(f(x_k))

where alpha > 0 is the **step size** (or **learning rate** in ML). The gradient points in the direction of steepest ascent, so we move in the opposite direction.

**Convergence rates** depend on the function class:
- **Convex, L-Lipschitz gradient**: f(x_k) - f(x*) <= O(1/k) -- sublinear convergence
- **Strongly convex (parameter mu), L-Lipschitz gradient**: f(x_k) - f(x*) <= O(rho^k) where rho = (kappa - 1)/(kappa + 1) -- linear (exponential) convergence
- Here kappa = L/mu is the **condition number**; larger kappa means slower convergence

For a quadratic f(x) = 0.5 * x^T Q x + q^T x:
- gradient(f) = Q*x + q
- L = largest eigenvalue of Q (= lambda_max)
- mu = smallest eigenvalue of Q (= lambda_min, if Q is positive definite)
- kappa = lambda_max / lambda_min

### Step Size Strategies

The step size alpha critically affects convergence:

**Fixed step size**: alpha = 1/L guarantees convergence for L-smooth convex functions. Too large (alpha > 2/L) causes divergence; too small wastes iterations.

**Backtracking line search (Armijo condition)**: Adaptively find a good step size at each iteration. Start with a large alpha, then shrink until:

    f(x - alpha * grad) <= f(x) - c1 * alpha * ||grad||^2

where c1 in (0, 1) is typically 1e-4. Parameters: initial alpha (e.g., 1.0), shrink factor beta (e.g., 0.5).

**Exact line search**: Minimize f(x - alpha * grad) over alpha >= 0. Possible analytically for quadratics: alpha* = ||grad||^2 / (grad^T Q grad). Impractical for general functions.

### Projected Gradient Descent

For **constrained** optimization (minimize f(x) subject to x in C), we modify gradient descent by projecting back onto the feasible set after each step:

    x_{k+1} = project_C(x_k - alpha * gradient(f(x_k)))

where project_C(z) = argmin_{x in C} ||x - z||^2 is the Euclidean projection onto C. This is efficient when projection is cheap:
- **Box constraints** {l <= x <= u}: element-wise clamp, O(n)
- **L2-ball** {||x - c|| <= r}: normalize if outside, O(n)
- **Simplex** {x >= 0, sum(x) = 1}: O(n log n) sorting-based algorithm
- **Polyhedra**: requires solving a QP (expensive in general)

Projected GD converges at the same rate as unconstrained GD when the projection is exact.

### Subgradient Methods

For **non-smooth** convex functions (like f(x) = ||x||_1), the gradient may not exist everywhere. A **subgradient** g at x satisfies:

    f(y) >= f(x) + g^T (y - x)  for all y

The subgradient method replaces the gradient with a subgradient and uses a **diminishing step size** (e.g., alpha_k = 1/sqrt(k)):

    x_{k+1} = x_k - alpha_k * g_k

Convergence is slower: O(1/sqrt(k)) for convex functions. Proximal methods (Practice 033b) provide a better alternative.

### Momentum Methods

Vanilla gradient descent can oscillate on ill-conditioned problems (high kappa). Momentum methods accelerate convergence by accumulating velocity.

**Polyak's heavy ball method**:

    x_{k+1} = x_k - alpha * gradient(f(x_k)) + beta * (x_k - x_{k-1})

The momentum term beta * (x_k - x_{k-1}) carries information from previous steps, damping oscillations.

**Nesterov's accelerated gradient (NAG)**: Achieves the optimal convergence rate for first-order methods on smooth convex functions:

    y_0 = x_0,  t_0 = 1
    x_{k+1} = y_k - (1/L) * gradient(f(y_k))
    t_{k+1} = (1 + sqrt(1 + 4 * t_k^2)) / 2
    y_{k+1} = x_{k+1} + ((t_k - 1) / t_{k+1}) * (x_{k+1} - x_k)

Convergence rates:
- **Convex**: O(1/k^2) vs O(1/k) for vanilla GD -- a quadratic speedup
- **Strongly convex**: O(((sqrt(kappa) - 1) / (sqrt(kappa) + 1))^k) vs O(((kappa - 1) / (kappa + 1))^k)

For kappa = 1000: vanilla GD effectively needs ~1000 iterations, Nesterov ~31 (sqrt(1000) ~ 31.6).

### KKT Conditions

The **Karush-Kuhn-Tucker (KKT) conditions** are necessary (and sufficient for convex problems) for optimality of constrained problems. For minimize f(x) s.t. g_i(x) <= 0, h_j(x) = 0:

1. **Stationarity**: gradient(f(x*)) + sum(lambda_i * gradient(g_i(x*))) + sum(nu_j * gradient(h_j(x*))) = 0
2. **Primal feasibility**: g_i(x*) <= 0, h_j(x*) = 0
3. **Dual feasibility**: lambda_i >= 0
4. **Complementary slackness**: lambda_i * g_i(x*) = 0 -- either the constraint is active (g_i = 0) or its multiplier is zero

KKT conditions are the bridge between primal optimization and duality theory. They also inform first-order algorithms: projected GD converges to a KKT point.

### Condition Number and Convergence Speed

The **condition number** kappa = L / mu (ratio of the largest to smallest curvature of the function) determines convergence speed:

| kappa | Problem Type | GD Convergence | Nesterov Convergence |
|-------|-------------|----------------|---------------------|
| 1 | Perfectly conditioned | 1 iteration | 1 iteration |
| 10 | Well-conditioned | ~10 iterations | ~3 iterations |
| 100 | Moderate | ~100 iterations | ~10 iterations |
| 1000 | Ill-conditioned | ~1000 iterations | ~32 iterations |
| 10^6 | Very ill-conditioned | ~10^6 iterations | ~1000 iterations |

**Preconditioning** (transforming variables to reduce kappa) is a key technique in practice. For quadratics, this amounts to choosing a good coordinate system.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Convex set** | A set where any line segment between two points stays inside |
| **Convex function** | f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y) |
| **Strong convexity** | Curvature bounded below by mu > 0; implies unique global minimum |
| **Lipschitz gradient** | ||grad f(x) - grad f(y)|| <= L * ||x - y||; L bounds curvature above |
| **Condition number** | kappa = L / mu; ratio of max to min curvature; controls convergence speed |
| **Gradient descent** | x_{k+1} = x_k - alpha * grad f(x_k); simplest first-order method |
| **Backtracking line search** | Adaptive step size via Armijo sufficient decrease condition |
| **Projected GD** | GD + project onto feasible set; for constrained convex problems |
| **Nesterov acceleration** | Momentum-based GD achieving optimal O(1/k^2) rate |
| **KKT conditions** | Necessary+sufficient optimality conditions for convex constrained problems |
| **Subgradient** | Generalization of gradient for non-smooth convex functions |

### Ecosystem Context

First-order methods are the backbone of:
- **Machine learning**: SGD, Adam, AdaGrad are all gradient descent variants; understanding convergence theory helps tune learning rates
- **Signal processing**: Compressed sensing, image reconstruction (total variation minimization)
- **Operations research**: Large-scale LP relaxations, Lagrangian relaxation methods
- **Control**: Model predictive control solves convex QPs at each time step

Libraries implementing these methods: PyTorch/TensorFlow (SGD variants), CVXPY/Mosek (general convex), Optim.jl (Julia). This practice builds them from scratch to understand the theory before using black-box solvers (Practice 041a covers CVXPY).

**Alternatives and trade-offs**:
- **Second-order methods** (Newton, L-BFGS): faster convergence (quadratic), but O(n^2) or O(n^3) per step vs O(n) for first-order
- **Proximal methods**: handle non-smooth terms efficiently (Practice 033b)
- **Interior point methods**: polynomial-time convergence, but expensive per iteration and hard to warm-start

## Description

Implement gradient descent variants from scratch on well-known convex functions (quadratic, Rosenbrock-like, LASSO-style). Experiment with step sizes, observe convergence rates, implement projected gradient descent for constrained problems, and compare vanilla GD with Nesterov acceleration on ill-conditioned problems.

All implementations use Eigen for linear algebra, printing convergence logs to stdout for analysis.

## Instructions

### Phase 1: Gradient Descent (~25 min)

**What you learn:** The core GD algorithm, the critical role of step size, and how the Lipschitz constant L governs convergence.

1. Read `src/gradient_descent.cpp`. The `QuadraticFunction` struct and helpers are provided.
2. **TODO(human):** Implement `gradient_descent()` -- the core GD loop. This is the most fundamental optimization algorithm: iterate x = x - alpha * gradient(x) until convergence.
3. Experiment with different step sizes on a well-conditioned (kappa=10) and ill-conditioned (kappa=100) quadratic.
4. Observe: what happens when alpha > 2/L? When alpha << 1/L?
5. Key insight: for quadratics, the optimal fixed step is alpha = 1/L, and convergence slows linearly with condition number.

### Phase 2: Backtracking Line Search (~25 min)

**What you learn:** Adaptive step size selection using the Armijo sufficient decrease condition, which eliminates the need to know L.

1. Read `src/line_search.cpp`. A general function interface and a smooth test function are provided.
2. **TODO(human):** Implement `backtracking_line_search()` -- the Armijo backtracking algorithm that automatically finds a good step size.
3. **TODO(human):** Implement `gd_with_line_search()` -- gradient descent using your backtracking line search instead of a fixed step.
4. Compare convergence: fixed-step GD (with various alpha) vs line-search GD on the same problem.
5. Key insight: line search is more robust -- it works without knowing L and adapts to local curvature.

### Phase 3: Projected Gradient Descent (~25 min)

**What you learn:** How to handle constraints by projecting back onto the feasible set after each gradient step.

1. Read `src/projected_gd.cpp`. Constraint set structures are provided.
2. **TODO(human):** Implement `project_box()` -- projection onto a box constraint (element-wise clamp).
3. **TODO(human):** Implement `project_l2_ball()` -- projection onto an L2-ball.
4. **TODO(human):** Implement `projected_gradient_descent()` -- GD with projection after each step.
5. Observe how the constraint changes the solution compared to unconstrained optimization.
6. Key insight: projected GD converges at the same rate as unconstrained GD when projection is cheap.

### Phase 4: Nesterov Accelerated Gradient (~25 min)

**What you learn:** How momentum accelerates convergence from O(1/k) to O(1/k^2), especially dramatic on ill-conditioned problems.

1. Read `src/nesterov.cpp`. An ill-conditioned quadratic (kappa=1000) is provided.
2. **TODO(human):** Implement `nesterov_accelerated()` -- Nesterov's method with the momentum sequence.
3. Run both vanilla GD and Nesterov on the same ill-conditioned problem. Compare iterations to converge.
4. Observe the dramatic speedup: ~1000 iterations (GD) vs ~32 iterations (Nesterov) for kappa=1000.
5. Key insight: Nesterov is provably optimal among first-order methods for smooth convex functions.

## Motivation

- **Foundation for ML**: SGD, Adam, and all modern optimizers are gradient descent variants. Understanding convergence theory (step sizes, condition numbers, momentum) directly helps diagnose slow training and tune hyperparameters.
- **First-order = scalable**: These methods are O(n) per iteration, making them the only viable option for problems with millions of variables (deep learning, large-scale OR relaxations).
- **Bridges theory and practice**: Implementing from scratch reveals why certain step sizes diverge, why ill-conditioned problems are hard, and why Nesterov acceleration is not magic but provable mathematics.
- **Gateway to OR solvers**: Lagrangian relaxation, ADMM, and interior point methods all build on first-order gradient concepts. This practice is foundational for the entire OR/Optimization track.

## Commands

All commands are run from the `practice_033a_convex_first_order/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches Eigen via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all four phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_gradient_descent 2>&1"` | Build Phase 1: Gradient descent with fixed step size |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_line_search 2>&1"` | Build Phase 2: Backtracking line search |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_projected_gd 2>&1"` | Build Phase 3: Projected gradient descent |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_nesterov 2>&1"` | Build Phase 4: Nesterov accelerated gradient |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\phase1_gradient_descent.exe` | Run Phase 1: Gradient descent experiments |
| `build\Release\phase2_line_search.exe` | Run Phase 2: Line search comparison |
| `build\Release\phase3_projected_gd.exe` | Run Phase 3: Projected gradient descent |
| `build\Release\phase4_nesterov.exe` | Run Phase 4: Nesterov vs vanilla GD |

## State

`not-started`
