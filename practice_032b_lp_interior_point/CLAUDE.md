# Practice 032b: LP — Duality & Interior Point Methods

## Technologies

- **C++17** — Modern C++ with structured bindings, std::optional, constexpr-if
- **Eigen** — Header-only linear algebra library (matrices, solvers, decompositions) fetched via CMake FetchContent
- **CMake 3.16+** — Build system with FetchContent for dependency management

## Stack

- C++17, CMake 3.16+

## Theoretical Context

### LP Duality

Every linear program (the **primal**) has a companion linear program called its **dual**. Together they form a *primal-dual pair* that reveals deep structural information about the optimization problem.

#### Constructing the Dual from the Primal

Given a primal LP in standard form:

```
Primal (P):
  minimize   c^T x
  subject to Ax >= b
              x >= 0
```

The dual is constructed mechanically:

```
Dual (D):
  maximize   b^T y
  subject to A^T y <= c
              y >= 0
```

**Mechanical rules for construction:**

| Primal (minimization)        | Dual (maximization)          |
|------------------------------|------------------------------|
| Objective coefficients `c`   | RHS of dual constraints      |
| RHS values `b`               | Dual objective coefficients  |
| Constraint matrix `A`        | Transposed: `A^T`            |
| `>=` constraint              | Dual variable `y_i >= 0`     |
| `<=` constraint              | Dual variable `y_i <= 0`     |
| `=` constraint               | Dual variable `y_i` free     |
| Variable `x_j >= 0`          | Dual constraint `<=`         |
| Variable `x_j <= 0`          | Dual constraint `>=`         |
| Variable `x_j` free          | Dual constraint `=`          |

The dual of the dual is the primal (the relationship is symmetric).

#### Weak Duality Theorem

For any feasible primal solution `x` and any feasible dual solution `y`:

```
b^T y  <=  c^T x     (for primal min / dual max)
```

**Interpretation:** Every feasible dual solution provides a lower bound on the primal optimal value. Every feasible primal solution provides an upper bound on the dual optimal value. This gives a "certificate of optimality": if you find primal `x*` and dual `y*` with `c^T x* = b^T y*`, both must be optimal.

#### Strong Duality Theorem

If the primal has an optimal solution `x*`, then the dual also has an optimal solution `y*`, and:

```
c^T x*  =  b^T y*
```

**Conditions:** Strong duality holds whenever both the primal and dual are feasible (guaranteed for standard LP by linear programming theory, more subtle in nonlinear optimization where constraint qualifications like Slater's condition are needed).

#### Complementary Slackness

At optimality, the primal and dual solutions satisfy:

```
x_i * s_i = 0    for all i   (where s_i is the dual slack for constraint i)
y_j * t_j = 0    for all j   (where t_j is the primal slack for constraint j)
```

**Economic meaning:**
- If a primal variable `x_i > 0` (we produce some of product i), then the corresponding dual constraint must be tight (the "reduced cost" is zero — the product is priced exactly at its marginal cost).
- If a resource constraint has slack (`t_j > 0`, i.e., we have leftover resource), then its shadow price is zero (`y_j = 0` — an additional unit of that resource has no value).

Complementary slackness is a necessary and sufficient condition for optimality (given primal and dual feasibility).

#### Economic Interpretation: Shadow Prices

The dual variable `y_j` associated with the j-th constraint is the **shadow price** (or marginal value) of the j-th resource:

```
y_j  =  dZ* / db_j
```

This means: if you increase the right-hand side `b_j` by one unit (i.e., you get one more unit of resource j), the optimal objective value improves by approximately `y_j`.

**Example:** In a factory LP where constraint 3 is "machine hours <= 100" and `y_3 = 15`, then adding one more machine hour would improve profit by $15. If machine time costs $10/hour to rent, it's worth renting more. If it costs $20/hour, it's not.

#### Sensitivity Analysis

Sensitivity analysis studies how the optimal solution changes when problem data is perturbed:

- **RHS ranging:** For each constraint `b_j`, find the range `[b_j^L, b_j^U]` over which the current optimal basis remains optimal. Within this range, the optimal value changes linearly at rate `y_j` (the shadow price).
- **Cost ranging:** For each objective coefficient `c_i`, find the range over which the current basis remains optimal. Outside this range, a different variable enters the basis.
- **Connection to duality:** Sensitivity analysis is fundamentally about understanding the dual solution. Shadow prices ARE dual variables.

### Interior Point Methods

#### Motivation

The simplex method (practice 032a) moves along vertices (extreme points) of the feasible polytope. While fast in practice, it has *exponential worst-case* complexity (Klee-Minty cube). Interior point methods (IPMs) were the first *polynomial-time* algorithms for LP, starting with Khachiyan's ellipsoid method (1979) and Karmarkar's projective method (1984).

Modern IPMs are based on the **primal-dual path-following** approach and are competitive with simplex for large-scale LPs. Every major commercial solver (Gurobi, CPLEX, MOSEK) implements both simplex and IPM.

#### Barrier Method

The key idea: instead of enforcing `x >= 0` as hard constraints, add a **logarithmic barrier** that penalizes approaching the boundary:

```
minimize   c^T x  -  mu * SUM_i ln(x_i)
subject to Ax = b
```

The barrier term `-mu * SUM_i ln(x_i)` goes to +infinity as any `x_i -> 0`, keeping iterates strictly inside the feasible region (hence "interior point"). As `mu -> 0`, the barrier vanishes and the solution approaches the true LP optimum.

#### The Central Path

For each value of `mu > 0`, the barrier problem has a unique optimal solution `x(mu)`. The set of all such solutions `{x(mu) : mu > 0}` forms the **central path** — a smooth curve through the interior of the feasible region that converges to an optimal vertex as `mu -> 0`.

The central path is characterized by the KKT conditions of the barrier problem:

```
A^T lambda + s = c          (dual feasibility)
A x = b                      (primal feasibility)
X S e = mu * e               (complementarity / centrality)
x > 0, s > 0                (strict positivity)
```

where `X = diag(x)`, `S = diag(s)`, and `e = (1,1,...,1)^T`.

#### Newton Step for the KKT System

At each iteration, we linearize the KKT system and solve for `(dx, dlambda, ds)`:

```
| 0    A^T   I  | | dx      |   | c - A^T lambda - s     |
| A    0     0  | | dlambda | = | b - Ax                 |
| S    0     X  | | ds      |   | mu*e - XSe             |
```

This is a 3x3 block linear system. In practice, it's reduced to a smaller system via elimination:
1. From row 3: `ds = X^{-1}(mu*e - XSe - S*dx)`
2. Substitute into row 1 to get a system in `(dx, dlambda)` only
3. Further reduce to the **normal equations**: `(A D^2 A^T) dlambda = rhs` where `D^2 = X S^{-1}`

The normal equations are symmetric positive definite, so Cholesky factorization works.

#### Path-Following Algorithm

```
1. Start with (x, lambda, s) > 0 (strictly feasible)
2. Set mu = x^T s / n  (duality gap / n)
3. Repeat:
   a. Set target: mu_target = sigma * mu  (sigma in (0,1), e.g., 0.2)
   b. Solve Newton system for (dx, dlambda, ds) with mu = mu_target
   c. Line search: find max alpha in (0,1] such that (x + alpha*dx, s + alpha*ds) > 0
   d. Update: x += alpha*dx, lambda += alpha*dlambda, s += alpha*ds
   e. Recompute mu = x^T s / n
   f. Stop if mu < tolerance
```

**Convergence:** Path-following IPMs converge in `O(sqrt(n) * log(1/epsilon))` iterations, each requiring an `O(n^3)` linear solve. Total complexity is polynomial.

#### Simplex vs Interior Point: Comparison

| Aspect | Simplex | Interior Point |
|--------|---------|----------------|
| **Path** | Vertex to vertex along edges | Through interior of feasible region |
| **Worst-case** | Exponential | Polynomial: `O(sqrt(n) * log(1/eps))` |
| **Practice** | Fast for small/medium LPs | Competitive for large LPs (>10k vars) |
| **Warm-start** | Excellent (reuse basis) | Poor (needs interior starting point) |
| **Sensitivity** | Basis information directly available | Requires extra work |
| **Degeneracy** | Can cycle (needs Bland's rule) | No cycling issues |
| **Sparsity** | Exploits via revised simplex | Exploits via sparse Cholesky |
| **Exact solution** | Visits exact vertex | Converges to interior, needs "crossover" |

### Key Concepts Table

| Concept | Definition |
|---------|------------|
| **Primal-Dual Pair** | Every LP has a companion dual LP; solving one gives information about the other |
| **Weak Duality** | Dual objective <= Primal objective (for min); any feasible dual gives a lower bound |
| **Strong Duality** | At optimality, primal and dual objectives are equal |
| **Complementary Slackness** | `x_i * s_i = 0`: active constraints pair with positive variables |
| **Shadow Price** | Dual variable `y_j` = marginal value of resource j = `dZ*/db_j` |
| **Sensitivity Analysis** | How optimal value/solution changes with parameter perturbation |
| **Logarithmic Barrier** | `-mu * SUM ln(x_i)`: penalty that keeps iterates strictly positive |
| **Central Path** | Family of barrier-optimal solutions parameterized by `mu > 0` |
| **Newton Step** | Solve linearized KKT system for search direction `(dx, dlambda, ds)` |
| **Duality Gap** | `x^T s`: measures distance to optimality, driven to zero by IPM |
| **Normal Equations** | Reduced system `(A D^2 A^T) dlambda = rhs` for efficient Newton solve |
| **Crossover** | Post-processing step to find an exact vertex solution from IPM's interior point |

## Description

Implement LP duality verification, sensitivity analysis, and a basic barrier/interior point solver. Connect the theory (complementary slackness, shadow prices) to concrete computations. This practice bridges the gap between the algebraic simplex method (032a) and the analytical machinery that underpins modern LP solvers.

### What you'll build

1. **Dual constructor** — mechanically build the dual LP from a primal, verify weak/strong duality
2. **Complementary slackness checker** — verify optimality conditions, interpret shadow prices
3. **Sensitivity analyzer** — perturb RHS, observe optimal value change, relate to dual variables
4. **Barrier method solver** — log-barrier interior point solver with Newton steps using Eigen

## Instructions

### Phase 1: Dual Construction & Duality Verification (~25 min)

**Concepts:** The dual LP, mechanical construction rules, weak and strong duality.

1. **Study the LP and DualLP structs** — understand the representation of primal (min c^T x, Ax >= b, x >= 0) and dual (max b^T y, A^T y <= c, y >= 0).
2. **TODO(human): `construct_dual()`** — Given a primal LP, construct the dual by applying the mechanical rules: transpose A, swap roles of c and b, flip constraint direction. This teaches the duality transformation at the most fundamental level — you need to internalize the rule table to understand why dual variables correspond to primal constraints.
3. **TODO(human): `verify_weak_duality()`** — Given primal and dual objective values, check that the weak duality inequality holds. Simple but reinforces the theoretical guarantee.
4. **Verify with hardcoded known solutions** — the main() provides a primal LP with known optimal, you construct the dual and verify both weak and strong duality hold.

### Phase 2: Complementary Slackness & Shadow Prices (~25 min)

**Concepts:** Complementary slackness conditions, economic interpretation of dual variables.

1. **Study the production planning example** — a factory LP with named resources (labor, material, machine hours) and products. The known optimal primal and dual solutions are provided.
2. **TODO(human): `check_complementary_slackness()`** — Verify that x_i * s_i = 0 and y_j * t_j = 0 (within tolerance) for all primal variables and dual variables. This connects the algebraic condition to the economic intuition: if a resource has slack, its marginal value is zero.
3. **TODO(human): `interpret_shadow_prices()`** — Print the economic interpretation of each dual variable: which resource it corresponds to, its shadow price, and what that price means for decision-making.
4. **Observe** — which resources are binding (fully used)? Which have slack? How does this match the shadow prices?

### Phase 3: Sensitivity Analysis (~25 min)

**Concepts:** RHS perturbation, shadow price validation, allowable ranges.

1. **Study the solved LP** — a small LP with known optimal basis, primal solution, and dual solution.
2. **TODO(human): `sensitivity_rhs()`** — For each constraint, perturb b_i by +/- delta, re-solve the LP (using the provided simple solver), compute the rate of change dZ/db_i, and compare to the dual variable y_i. This empirically validates the theoretical result that shadow prices equal rates of change.
3. **TODO(human): `allowable_range()`** — For a given constraint, find the range of b_i values where the current optimal basis remains feasible (and hence optimal). This teaches the limits of shadow price validity.
4. **Compare** — verify that numerical dZ/db_i matches the dual variable y_i within the allowable range.

### Phase 4: Barrier Method / Interior Point (~30 min)

**Concepts:** Logarithmic barrier, central path, Newton step for KKT system, path-following.

1. **Study the BarrierState struct** — holds current iterate (x, lambda, s), barrier parameter mu, and problem data.
2. **TODO(human): `barrier_objective()`** — Compute c^T x - mu * SUM ln(x_i). This is the penalized objective that keeps x strictly positive.
3. **TODO(human): `newton_step()`** — Assemble and solve the KKT system for (dx, dlambda, ds). This is the core computation: you'll build the 3x3 block matrix and use Eigen's linear solve. Understanding this system is key to understanding how IPMs work.
4. **TODO(human): `solve_barrier()`** — Outer loop: start from a feasible interior point, repeatedly reduce mu, take Newton steps with line search, check convergence. This ties together all the pieces.
5. **Test** — solve the same LPs from Phase 1 and verify agreement with known optimal values.

## Motivation

- **Duality is the theoretical backbone** of optimization. KKT conditions, sensitivity analysis, decomposition methods (Lagrangian relaxation, Benders, Dantzig-Wolfe), and pricing in column generation all rely on duality theory.
- **Interior point methods** are used by every commercial solver (Gurobi, CPLEX, MOSEK) alongside simplex. Understanding both gives insight into why solvers choose one vs the other (simplex for warm-starting, IPM for large dense problems).
- **Shadow prices and sensitivity** are what business stakeholders actually care about — not just "what's the optimal solution" but "how much is an extra unit of X worth?" and "how robust is this solution?"
- **Foundation for nonlinear optimization** — the barrier method generalizes directly to convex optimization (practice 033), second-order cone programming, and semidefinite programming.

## Commands

All commands are run from the `practice_032b_lp_interior_point/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches Eigen via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all four phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_dual 2>&1"` | Build Phase 1: Dual construction & duality verification |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_complementary 2>&1"` | Build Phase 2: Complementary slackness & shadow prices |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_sensitivity 2>&1"` | Build Phase 3: Sensitivity analysis |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_barrier 2>&1"` | Build Phase 4: Barrier / interior point method |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\phase1_dual.exe` | Run Phase 1: Dual construction & duality verification |
| `build\Release\phase2_complementary.exe` | Run Phase 2: Complementary slackness & shadow prices |
| `build\Release\phase3_sensitivity.exe` | Run Phase 3: Sensitivity analysis |
| `build\Release\phase4_barrier.exe` | Run Phase 4: Barrier / interior point method |

## State

`not-started`
