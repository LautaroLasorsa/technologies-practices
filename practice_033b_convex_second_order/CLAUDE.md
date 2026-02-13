# Practice 033b: Convex Optimization -- Proximal & Second-Order Methods

## Technologies

- **C++17** -- Modern C++ with structured bindings, lambdas, auto
- **Eigen 3.4** -- Header-only linear algebra library (via CMake FetchContent)
- **CMake 3.16+** -- Build system with FetchContent for dependency management

## Stack

- C++17
- Eigen 3.4 (fetched via CMake FetchContent)

## Theoretical Context

Practice 033a covered first-order methods: gradient descent, line search, projected GD, and Nesterov acceleration. These methods use only gradient information and are O(n) per iteration but converge slowly (O(1/k) for convex, O(1/k^2) for accelerated). This practice introduces methods that go beyond vanilla gradients: **proximal operators** for non-smooth problems, **ADMM** for decomposable problems, and **second-order methods** (Newton, L-BFGS) that exploit curvature information for dramatically faster convergence.

### Proximal Operator

The **proximal operator** of a function f with parameter alpha > 0 is:

    prox_{alpha*f}(v) = argmin_x { f(x) + (1 / (2*alpha)) * ||x - v||^2 }

Intuitively, it finds the point that **minimizes f while staying close to v**. It is a "denoising" step: given a noisy input v, it balances minimizing f against not moving too far. The parameter alpha controls the trade-off -- larger alpha gives more weight to minimizing f.

Why proximal operators matter:
- For **f(x) = lambda * ||x||_1** (L1 norm): prox = **soft-thresholding**, which shrinks components toward zero. This is how LASSO produces sparse solutions.
- For **f = indicator function of a convex set C** (0 if x in C, infinity otherwise): prox = **Euclidean projection** onto C. So projected GD from 033a is a special case of proximal methods.
- For **f = 0** (trivial): prox = identity. So vanilla GD is also a special case.

Closed-form proximal operators exist for many important functions (L1, L2, box constraints, nuclear norm, group lasso). When the proximal operator is cheap to evaluate, proximal methods are extremely efficient.

The **Moreau decomposition** connects a function's proximal operator to its conjugate: prox_f(v) + prox_{f*}(v) = v, where f* is the Fenchel conjugate. This is the proximal analog of the decomposition "projection onto C + projection onto C^perp = identity."

### Proximal Gradient Descent

For **composite optimization** problems of the form:

    minimize  f(x) + g(x)

where f is smooth (has Lipschitz gradient) and g is convex but possibly **non-smooth** (e.g., L1 regularization, constraints), **proximal gradient descent** alternates a gradient step on f with a proximal step on g:

    x_{k+1} = prox_{alpha*g}( x_k - alpha * grad_f(x_k) )

This generalizes three important algorithms:
- **g = 0**: reduces to vanilla gradient descent
- **g = indicator of C**: reduces to projected gradient descent (project onto C after each step)
- **g = lambda * ||x||_1**: reduces to **ISTA** (Iterative Shrinkage-Thresholding Algorithm), which solves LASSO

Convergence: O(1/k) on objective value, same as gradient descent. With acceleration (FISTA), achieves O(1/k^2).

**Step size**: alpha must satisfy alpha <= 1/L where L is the Lipschitz constant of grad_f. Can also use backtracking line search.

The **LASSO** (Least Absolute Shrinkage and Selection Operator) problem is the canonical application:

    minimize  (1/2) * ||A*x - b||^2  +  lambda * ||x||_1

The smooth part is f(x) = (1/2) * ||Ax - b||^2 with gradient A^T(Ax - b). The non-smooth part g(x) = lambda * ||x||_1 has a closed-form proximal operator: component-wise soft-thresholding.

**Soft-thresholding operator** (proximal of alpha * lambda * ||.||_1):

    S_{t}(v)_i = sign(v_i) * max(|v_i| - t, 0)

where t = alpha * lambda. This shrinks each component toward zero by t. Components with |v_i| <= t become exactly zero, producing **sparsity** -- the key feature of L1 regularization.

### ADMM (Alternating Direction Method of Multipliers)

**ADMM** solves problems of the form:

    minimize  f(x) + g(z)
    subject to  A*x + B*z = c

by forming the **augmented Lagrangian** and alternating minimization over x and z. The three steps per iteration are:

    x_{k+1} = argmin_x { f(x) + (rho/2) * ||A*x + B*z_k - c + u_k||^2 }   (x-update)
    z_{k+1} = argmin_z { g(z) + (rho/2) * ||A*x_{k+1} + B*z - c + u_k||^2 } (z-update)
    u_{k+1} = u_k + A*x_{k+1} + B*z_{k+1} - c                               (dual update)

where u is the scaled dual variable and rho > 0 is the penalty parameter.

ADMM's power is **decomposition**: it splits hard problems into simpler subproblems. Each subproblem involves only f or g (not both), and can often be solved in closed form:
- If f is quadratic, x-update is a linear system solve
- If g is an indicator function, z-update is a projection
- If g is L1, z-update is soft-thresholding

**Convergence**: ADMM converges under mild conditions (f, g convex, A, B full column rank). It is not the fastest method but is extremely versatile and robust. The penalty parameter rho affects speed but not convergence -- larger rho emphasizes constraint satisfaction.

**Stopping criteria**: monitor the **primal residual** r = Ax + Bz - c (constraint violation) and **dual residual** s = rho * A^T B (z_{k+1} - z_k) (optimality violation). Stop when both are small.

Applications of ADMM:
- **Distributed optimization**: split data across machines, each solves a local subproblem
- **Consensus optimization**: multiple agents agree on a global solution
- **LASSO** (alternative to ISTA), basis pursuit, total variation denoising
- **Model fitting with constraints**: regularized regression, sparse inverse covariance estimation
- **Control**: model predictive control, robust control

For our practice, we use the **consensus form**: min f(x) + g(z) s.t. x = z, where f is a quadratic objective and g is an indicator function for box constraints.

### Newton's Method

**Newton's method** uses second-order information (the Hessian matrix) to achieve **quadratic convergence** near the optimum:

    x_{k+1} = x_k - [H(x_k)]^{-1} * g(x_k)

where H = nabla^2 f is the Hessian (matrix of second partial derivatives) and g = nabla f is the gradient. Equivalently, at each step Newton minimizes a **local quadratic approximation**:

    f(x_k + dx) approx f(x_k) + g^T dx + (1/2) dx^T H dx

Setting the gradient of this approximation to zero gives H * dx = -g, the **Newton equation**.

**Quadratic convergence**: near the optimum, the error squares at each step:

    ||x_{k+1} - x*|| <= C * ||x_k - x*||^2

This is dramatically faster than linear convergence. If the error is 10^{-3}, the next iteration gives 10^{-6}, then 10^{-12}. In practice, Newton converges in 5-10 iterations regardless of problem size or conditioning -- but each iteration costs O(n^3) for solving the linear system.

**Newton decrement**: lambda^2 = g^T H^{-1} g = -g^T dx is a measure of proximity to the optimum. When lambda^2/2 < epsilon, we are within epsilon of the optimal value. This provides a natural stopping criterion.

**Damped Newton**: far from the optimum, the pure Newton step may overshoot or even increase the objective. **Damped Newton** uses a **backtracking line search** on the Newton direction:

    x_{k+1} = x_k + t * dx    where t <= 1 is found by backtracking

The line search ensures global convergence (any starting point), while retaining quadratic convergence near the optimum.

**Cost comparison** with first-order methods:
- GD: O(n) per iteration, O(kappa) iterations -> O(n * kappa) total
- Newton: O(n^3) per iteration (Hessian solve), O(log log(1/epsilon)) iterations
- For n=100, kappa=1000: GD ~ 100,000 flops/iter * 1000 iters vs Newton ~ 1,000,000 flops/iter * 10 iters
- Newton wins when n is moderate and kappa is large

### L-BFGS (Limited-Memory BFGS)

**BFGS** (Broyden-Fletcher-Goldfarb-Shanno) is a **quasi-Newton** method: it approximates the inverse Hessian using only gradient information, avoiding the O(n^3) cost of Newton. It maintains an n x n matrix B_k that approximates H^{-1}, updated each iteration using the rank-2 formula:

    B_{k+1} = (I - rho * s_k * y_k^T) * B_k * (I - rho * y_k * s_k^T) + rho * s_k * s_k^T

where s_k = x_{k+1} - x_k, y_k = g_{k+1} - g_k, and rho = 1 / (y_k^T s_k).

**L-BFGS** (limited-memory BFGS) avoids storing the full n x n matrix by keeping only the last **m** pairs (s_k, y_k) and computing the matrix-vector product B_k * g implicitly via the **two-loop recursion**:

    // First loop: backward through history
    q = g
    for i = m-1, ..., 0:
        rho_i = 1 / (y_i^T s_i)
        alpha_i = rho_i * s_i^T q
        q = q - alpha_i * y_i

    // Initial Hessian scaling
    r = gamma * q    where gamma = (s_{m-1}^T y_{m-1}) / (y_{m-1}^T y_{m-1})

    // Second loop: forward through history
    for i = 0, ..., m-1:
        beta = rho_i * y_i^T r
        r = r + s_i * (alpha_i - beta)

    direction = -r

This requires only O(m*n) storage and O(m*n) per iteration instead of O(n^2). Typical m = 5-20 works well in practice.

**Convergence**: L-BFGS achieves **superlinear convergence** -- faster than linear (GD) but slower than quadratic (Newton). In practice, it often converges nearly as fast as Newton at a fraction of the cost.

L-BFGS is the **workhorse of large-scale smooth optimization**:
- Default optimizer in scipy.optimize.minimize
- Used in LibLinear, Vowpal Wabbit, and many ML libraries
- Standard choice for logistic regression, CRFs, and other smooth ML objectives
- Effective for n = 10^3 to 10^7 variables

### Convergence Comparison

| Method | Per-iteration cost | Convergence rate | Best for |
|--------|-------------------|-----------------|----------|
| Gradient descent | O(n) | O(1/k) convex, linear strongly cvx | Very large n, cheap gradient |
| Nesterov accelerated | O(n) | O(1/k^2) convex, optimal first-order | Large n, ill-conditioned |
| Proximal GD (ISTA) | O(n) + prox cost | O(1/k) | Non-smooth regularizers (L1) |
| ADMM | O(n^3) or O(n) per subproblem | O(1/k) | Decomposable/constrained problems |
| Newton | O(n^3) | Quadratic (local) | Small-medium n, high precision |
| L-BFGS | O(m*n) | Superlinear | Medium-large n, smooth problems |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Proximal operator** | prox_{alpha*f}(v) = argmin { f(x) + (1/2alpha)||x-v||^2 }; "denoising" step |
| **Soft-thresholding** | Proximal of L1 norm; shrinks toward zero, produces sparsity |
| **ISTA** | Proximal GD for LASSO: gradient step on smooth part + soft-threshold |
| **Composite optimization** | min f(x) + g(x) where f smooth, g non-smooth |
| **ADMM** | Split hard problems into easy subproblems via augmented Lagrangian |
| **Primal/dual residual** | ADMM convergence metrics: constraint violation + optimality gap |
| **Newton's method** | Second-order: uses Hessian for quadratic convergence |
| **Newton decrement** | lambda^2 = g^T H^{-1} g; proximity measure to optimum |
| **Damped Newton** | Newton + line search for global convergence |
| **Quasi-Newton** | Approximate Hessian from gradient differences (BFGS family) |
| **L-BFGS** | Limited-memory BFGS; O(mn) per iteration, m pairs in memory |
| **Two-loop recursion** | Efficient O(mn) algorithm for L-BFGS direction computation |
| **Superlinear convergence** | Faster than linear, slower than quadratic; L-BFGS regime |

### Ecosystem Context

Proximal and second-order methods are foundational across optimization and ML:

- **Proximal methods** are the backbone of sparse modeling: LASSO, elastic net, group lasso, nuclear norm minimization. Libraries: CVXPY, scikit-learn (coordinate descent variant), proximal.jl
- **ADMM** enables distributed optimization at scale: split data across machines, solve local subproblems, reach consensus. Used in OSQP (QP solver), SCS (conic solver), and distributed ML frameworks
- **Newton's method** is the engine inside interior point solvers (MOSEK, Gurobi's barrier method, ECOS). These are the fastest solvers for medium-scale LP/QP/SOCP/SDP
- **L-BFGS** is the default for large-scale smooth optimization: scipy.optimize.minimize, LibLinear, CRF libraries, and many ML training pipelines

**Trade-offs**:
- First-order (GD, proximal): cheap iterations, many iterations needed, handle non-smooth terms natively
- Second-order (Newton): expensive iterations, very few iterations needed, only for smooth problems
- Quasi-Newton (L-BFGS): middle ground -- moderate cost, fast convergence, smooth problems only
- ADMM: universal "splitting" framework, moderate convergence speed, excellent for structured/distributed problems

## Description

Implement proximal gradient descent (ISTA for LASSO), ADMM for box-constrained quadratic programming, Newton's method with Hessian computation, and L-BFGS with the two-loop recursion. Compare convergence on benchmark problems to see the dramatic difference between first-order and second-order methods.

All implementations use Eigen for linear algebra, printing convergence logs to stdout for analysis.

## Instructions

### Phase 1: Proximal Gradient / ISTA (~20 min)

**What you learn:** How proximal operators handle non-smooth regularizers like L1, and how ISTA solves LASSO (the most important sparse optimization problem in statistics/ML).

1. Read `src/proximal_gd.cpp`. The LASSO problem is set up with a synthetic sparse signal recovery scenario (A matrix, b vector, lambda). The objective function and smooth gradient are provided.
2. **TODO(human):** Implement `soft_threshold()` -- the proximal operator for the L1 norm. This single function is what produces sparsity in LASSO solutions. Understanding it is the key insight of this phase.
3. **TODO(human):** Implement `ista()` -- the proximal gradient loop that alternates a gradient step on the smooth part with soft-thresholding on the L1 part.
4. Run and observe: the solution x should be sparse (many exact zeros). Compare the recovered signal with the true sparse signal.
5. Key insight: L1 regularization + proximal operator = automatic variable selection. The strength lambda controls the sparsity level.

### Phase 2: ADMM (~20 min)

**What you learn:** How ADMM decomposes constrained optimization into simple subproblems, and how primal/dual residuals track convergence.

1. Read `src/admm.cpp`. A box-constrained QP is set up: minimize quadratic objective subject to lower <= x <= upper.
2. **TODO(human):** Implement `admm_x_update()` -- solve the linear system (P + rho*I)x = -q + rho*(z - u). This is the "minimize augmented Lagrangian over x" step.
3. **TODO(human):** Implement `admm_z_update()` -- project x + u onto the box constraints. This is the "minimize over z" step, which is just clamping.
4. **TODO(human):** Implement `admm_solve()` -- the full ADMM loop with primal/dual residual convergence check.
5. Compare the ADMM solution with the unconstrained optimum. Observe how constraints push the solution to the boundary.
6. Key insight: ADMM's power is decomposition -- each subproblem is trivial (linear solve + projection), but together they solve the constrained problem.

### Phase 3: Newton's Method (~20 min)

**What you learn:** Second-order optimization with Hessian information achieves quadratic convergence -- error squares each iteration.

1. Read `src/newton.cpp`. A logistic regression problem is set up with synthetic classification data. The function evaluates objective, gradient, and Hessian.
2. **TODO(human):** Implement `newton_method()` -- pure Newton with LDLT solve for the Newton direction. Observe the Newton decrement as convergence metric.
3. **TODO(human):** Implement `damped_newton()` -- Newton with backtracking line search for global convergence.
4. Compare iteration counts: GD vs Newton on the same logistic regression problem. Newton should converge in ~6-10 iterations vs hundreds for GD.
5. Key insight: quadratic convergence means 10^{-3} -> 10^{-6} -> 10^{-12} in three steps. The cost is O(n^3) per iteration for the Hessian solve.

### Phase 4: L-BFGS (~20 min)

**What you learn:** How to approximate second-order information cheaply using only gradient differences, achieving near-Newton convergence at GD-like cost.

1. Read `src/lbfgs.cpp`. Generalized Rosenbrock and logistic regression test functions are provided, along with a gradient history buffer.
2. **TODO(human):** Implement `lbfgs_direction()` -- the two-loop recursion that computes H_k * g in O(mn) time. This is the algorithmic core of L-BFGS.
3. **TODO(human):** Implement `lbfgs_solve()` -- full L-BFGS with Wolfe line search, using your two-loop recursion for the search direction.
4. Compare GD vs Newton vs L-BFGS on 50-dimensional Rosenbrock. L-BFGS should match or beat Newton in wall-clock time while using far less memory.
5. Key insight: L-BFGS is the sweet spot between first-order and second-order methods. It is the default choice for large-scale smooth optimization.

## Motivation

- **Proximal methods** handle non-smooth regularizers (L1, nuclear norm, group lasso) that are ubiquitous in statistics and ML for producing sparse, interpretable models. ISTA/FISTA are the workhorses of compressed sensing and high-dimensional statistics.
- **ADMM** enables distributed and decomposable optimization at scale. It is the foundation of modern solvers like OSQP and SCS, and is used in distributed ML, consensus optimization, and model predictive control.
- **Newton's method** is the engine inside interior point solvers (MOSEK, Gurobi barrier). Understanding it is essential for understanding why these solvers achieve high precision in few iterations.
- **L-BFGS** is the default optimizer for large-scale smooth problems (logistic regression, CRFs, NLP). It is the most important practical optimization algorithm after SGD.
- Together, these methods cover the full spectrum of convex optimization: smooth, non-smooth, constrained, unconstrained, small-scale, large-scale.

## Commands

All commands are run from the `practice_033b_convex_second_order/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches Eigen via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all four phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_proximal_gd 2>&1"` | Build Phase 1: Proximal gradient / ISTA |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_admm 2>&1"` | Build Phase 2: ADMM |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_newton 2>&1"` | Build Phase 3: Newton's method |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_lbfgs 2>&1"` | Build Phase 4: L-BFGS |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\phase1_proximal_gd.exe` | Run Phase 1: ISTA for LASSO (sparse signal recovery) |
| `build\Release\phase2_admm.exe` | Run Phase 2: ADMM for box-constrained QP |
| `build\Release\phase3_newton.exe` | Run Phase 3: Newton's method vs GD on logistic regression |
| `build\Release\phase4_lbfgs.exe` | Run Phase 4: L-BFGS vs Newton vs GD comparison |

## State

`not-started`
