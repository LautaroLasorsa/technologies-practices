# Practice 041b: Conic Programming — SOCP & SDP

## Technologies

- **CVXPY** — Python-embedded modeling language for convex optimization. Supports disciplined convex programming (DCP) with automatic cone transformations for LP, QP, SOCP, SDP, and exponential cone problems.
- **SCS** — Splitting Conic Solver. First-order method for large-scale cone programs including SDP. Default CVXPY solver for problems with semidefinite constraints.
- **NumPy / SciPy** — Numerical arrays and linear algebra for problem data construction.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- CVXPY >= 1.5 (modeling layer + DCP verification)
- SCS (SDP-capable solver, installed as CVXPY dependency)
- NumPy >= 1.26
- SciPy >= 1.12
- uv (package manager)

## Theoretical Context

### The Conic Optimization Hierarchy

Convex optimization problems can be classified by the type of **cones** they use in their constraints. Each level strictly generalizes the previous one, adding expressive power at the cost of harder solvers:

```
LP  ⊂  SOCP  ⊂  SDP  ⊂  General Convex

LP:    polyhedra (linear inequalities)
SOCP:  second-order cones (norms, quadratics)
SDP:   semidefinite cones (matrix inequalities)
```

- **LP (Linear Program):** `min c^T x` s.t. `Ax <= b`. Constraints define a polyhedron. Solved by simplex or interior point in polynomial time. (Practiced in 040a.)
- **SOCP (Second-Order Cone Program):** Adds constraints of the form `||Ax + b||_2 <= c^T x + d`. This is a "cone" because the set `{(x, t) : ||x|| <= t}` is a cone in R^{n+1}. Includes LP and QP as special cases. Interior point methods solve SOCP in polynomial time.
- **SDP (Semidefinite Program):** Adds constraints of the form `F(x) = F_0 + x_1 F_1 + ... + x_n F_n ≽ 0` (positive semidefinite). The PSD cone generalizes the nonneg orthant (LP) and second-order cone (SOCP). SDP is the most powerful class of tractable convex optimization.
- **General convex:** Exponential cones, power cones, etc. Handled by fewer solvers and often harder in practice.

**Why the hierarchy matters:** If your problem fits SOCP, use an SOCP solver (faster, more reliable) rather than reducing it to a general SDP. CVXPY automatically detects the minimal cone type needed.

### Second-Order Cone Program (SOCP)

**Standard form:**

```
minimize    c^T x
subject to  ||A_i x + b_i||_2  <=  c_i^T x + d_i,    i = 1, ..., m
            Fx = g
```

The constraint `||u||_2 <= t` defines a **second-order cone** (also called Lorentz cone or ice-cream cone). Geometrically, it's a cone in (n+1)-dimensional space.

**Special cases:**
- **LP:** If all `A_i = 0`, the SOCP reduces to linear constraints.
- **QP (Quadratic Program):** `min x^T P x + q^T x` s.t. `Ax <= b` can be rewritten as SOCP by introducing `t >= ||P^{1/2} x||`.
- **QCQP:** Quadratically constrained QP is also SOCP-representable.

**Key SOCP applications:**
- **Chebyshev center:** Largest ball inscribed in a polytope. The radius constraint `a_i^T x + r * ||a_i|| <= b_i` is SOCP.
- **Robust optimization:** Worst-case analysis over ellipsoidal uncertainty sets leads to norm constraints.
- **Facility location / minimax:** Minimizing the maximum distance (or sum of distances) to a set of points.
- **Portfolio optimization:** Mean-variance with risk constraints `||Sigma^{1/2} w|| <= sigma_max`.

### Semidefinite Program (SDP)

**Standard form:**

```
minimize    c^T x
subject to  F_0 + x_1 F_1 + ... + x_n F_n  ≽  0   (positive semidefinite)
            Ax = b
```

where `F_i` are symmetric matrices and `≽ 0` means all eigenvalues are nonneg (the matrix is PSD).

Equivalently, with a matrix variable:

```
minimize    trace(C @ X)
subject to  X ≽ 0
            trace(A_i @ X) = b_i,   i = 1, ..., m
```

**Why SDP is powerful:** The PSD constraint `X ≽ 0` is equivalent to requiring that `v^T X v >= 0` for all vectors `v`. This is an infinite family of linear constraints (one for each direction `v`), which is why SDP is so expressive.

**Key SDP applications:**
- **Max-Cut relaxation (Goemans-Williamson, 1995):** Replace binary variables `x_i in {-1, +1}` with a PSD matrix `X` where `X_ii = 1`. The SDP relaxation gives a 0.878-approximation to the NP-hard Max-Cut problem. This was a breakthrough result — the first proof that a polynomial-time algorithm can approximate an NP-hard problem with a known worst-case guarantee.
- **Control theory:** Lyapunov stability analysis, H-infinity control, LMI (Linear Matrix Inequality) formulations.
- **Sensor network localization:** Determine positions from noisy distance measurements via SDP relaxation of the quadratic constraints.
- **Covariance estimation:** Minimum-volume ellipsoid, covariance matrix completion, graphical lasso.
- **Combinatorial optimization:** Graph coloring, stable set, partition problems — all have SDP relaxations that often outperform LP relaxations.

### Matrix Variables in CVXPY

CVXPY has native support for matrix variables and PSD constraints:

```python
import cvxpy as cp

# Symmetric matrix variable
X = cp.Variable((n, n), symmetric=True)

# PSD constraint (semidefinite)
constraints = [X >> 0]           # X is positive semidefinite
constraints += [X[i, i] == 1]   # fix diagonal entries

# Matrix atoms
cp.trace(X)          # trace of matrix
cp.log_det(X)        # log-determinant (concave on PSD cone)
cp.lambda_min(X)     # minimum eigenvalue (concave)
cp.lambda_max(X)     # maximum eigenvalue (convex)
cp.norm(X, "nuc")    # nuclear norm (sum of singular values, convex)
cp.norm(X, "fro")    # Frobenius norm
```

**DCP rules for SDP:**
- `X >> 0` is a convex constraint (affine expression in the PSD cone).
- `cp.log_det(X)` is concave, so `cp.Maximize(cp.log_det(X))` is valid.
- `cp.lambda_min(X) >= t` is a convex constraint (lambda_min is concave).

### SOCP Examples in Detail

**Chebyshev center:** Given a polytope `{x : Ax <= b}`, the largest inscribed ball has center `x_c` and radius `r`:

```
maximize    r
subject to  a_i^T x_c + r * ||a_i||_2 <= b_i,  for all i
            r >= 0
```

This is SOCP because each constraint involves `r * ||a_i||` — a product of a scalar variable and a constant norm (which is bilinear in the original variables but becomes linear after substitution since `||a_i||` is a known constant).

**Robust least squares:** When the data matrix `A` is uncertain (`A = A_nom + Delta`, `||Delta|| <= epsilon`), the worst-case residual is:

```
minimize  max_{||Delta||<=epsilon}  ||(A_nom + Delta)x - b||_2
```

By triangle inequality: `||(A_nom + Delta)x - b|| <= ||A_nom x - b|| + ||Delta x|| <= ||A_nom x - b|| + epsilon * ||x||`. The worst case is achieved, so the robust formulation is:

```
minimize  ||A_nom x - b||_2 + epsilon * ||x||_2
```

This is a sum of norms (SOCP). The `epsilon * ||x||` term acts as Tikhonov regularization — the robust formulation automatically penalizes large `x` to limit sensitivity to perturbations.

### SDP Examples in Detail

**Max-Cut SDP relaxation:** Given a weighted graph with adjacency matrix `W`:
1. Compute Laplacian: `L = diag(W @ 1) - W`
2. The exact Max-Cut is `max (1/4) * sum_{(i,j)} w_ij (1 - x_i x_j)` with `x_i in {-1, +1}`
3. Note that `x_i x_j = X_ij` where `X = x x^T`. Since `X = x x^T`, we have `X ≽ 0` and `rank(X) = 1`.
4. **Relax** by dropping the rank-1 constraint: `maximize (1/4) trace(L @ X)` s.t. `X ≽ 0, X_ii = 1`.
5. **Round** using random hyperplane: decompose `X = V^T V` (Cholesky), generate random `r ~ N(0, I)`, set `x_i = sign(V_i^T r)`.

The Goemans-Williamson theorem proves that the expected cut value is at least `0.878 * OPT`. This is the best known polynomial-time approximation for Max-Cut.

**Minimum volume ellipsoid (MVEE / Lowner-John):** Given points `p_1, ..., p_m`, find the smallest ellipsoid `{x : ||Ax + b|| <= 1}` containing all of them:

```
maximize    log det(A)      (equivalent to minimizing volume, since vol ~ 1/det(A))
subject to  ||A p_i + b|| <= 1,  for all i
```

This combines `log_det` (SDP/exponential cone) with SOCP constraints. CVXPY handles the mixed cone structure automatically.

### Solver Considerations

| Solver | License | SDP Support | Method | Notes |
|--------|---------|-------------|--------|-------|
| **SCS** | MIT | Yes | First-order (ADMM) | Default for SDP in CVXPY. Good for large-scale, lower accuracy. |
| **CLARABEL** | Apache 2.0 | Yes | Interior point | Good accuracy, open-source alternative to MOSEK. |
| **MOSEK** | Commercial | Yes | Interior point | Fastest and most reliable SDP solver. Free academic license. |
| **ECOS** | GPL | SOCP only | Interior point | Fast for SOCP, no SDP support. |

SCS is the default choice for this practice: it's open-source, handles all cone types, and is installed automatically with CVXPY. For higher accuracy on SDP problems, increase iterations: `prob.solve(solver=cp.SCS, max_iters=10000)`.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Cone** | A set `K` where `x in K` implies `alpha * x in K` for all `alpha >= 0`. Convex cones are additionally convex. |
| **Second-order cone** | `{(x, t) : \|\|x\|\|_2 <= t}`, also called Lorentz cone or ice-cream cone |
| **PSD cone** | `{X : X = X^T, v^T X v >= 0 for all v}`, the cone of positive semidefinite matrices |
| **SOCP** | Optimization with second-order cone constraints: `\|\|Ax + b\|\| <= c^T x + d` |
| **SDP** | Optimization with semidefinite constraints: `F(x) ≽ 0` (matrix inequality) |
| **Chebyshev center** | Center of the largest inscribed ball in a convex set |
| **Robust optimization** | Optimizing the worst case over an uncertainty set |
| **Max-Cut** | NP-hard graph partitioning problem; SDP relaxation gives 0.878-approximation |
| **Goemans-Williamson** | 1995 result: random hyperplane rounding of Max-Cut SDP achieves 0.878 * OPT |
| **Graph Laplacian** | `L = D - W` where `D = diag(W @ 1)`. Encodes graph structure for spectral methods. |
| **MVEE** | Minimum Volume Enclosing Ellipsoid (Lowner-John ellipsoid) |
| **log det** | Log-determinant function. Concave on PSD matrices. Measures ellipsoid volume. |
| **Random hyperplane rounding** | Rounding SDP solution to integer: `sign(V^T r)` where `r ~ N(0, I)` |
| **DCP (Disciplined Convex Programming)** | CVXPY's rule system: convex objective, convex constraints, using atoms with known curvature |

## Description

Model and solve SOCP and SDP problems using CVXPY: Chebyshev center of a polytope (SOCP), robust least squares under matrix uncertainty (SOCP), Max-Cut SDP relaxation with Goemans-Williamson rounding (SDP), and minimum volume enclosing ellipsoid (SDP). Each phase introduces a conic problem class and demonstrates CVXPY's matrix variable and cone constraint support.

### What you'll build

1. **Chebyshev center (SOCP)** — Find the largest ball inscribed in a polytope defined by halfplanes.
2. **Robust least squares (SOCP)** — Solve a least-squares problem that is robust to perturbations in the data matrix.
3. **Max-Cut SDP relaxation (SDP)** — Approximate the NP-hard Max-Cut problem via semidefinite relaxation with random hyperplane rounding.
4. **Minimum volume ellipsoid (SDP)** — Find the smallest ellipsoid enclosing a set of 2D points.

## Instructions

### Phase 1: Chebyshev Center — SOCP (~20 min)

**File:** `src/phase1_chebyshev.py`

This phase teaches SOCP formulation on a geometric problem with a clean visual interpretation. The Chebyshev center of a polytope is the center of the largest inscribed ball — useful in robust optimization (finding the "safest" point inside a feasible region) and computational geometry.

**What you implement:**
- `chebyshev_center(A, b)` — Create CVXPY variables for center and radius, formulate the SOCP constraints that ensure the ball lies inside each halfplane, maximize the radius, solve, and return the center and radius.

**Why it matters:** This is the simplest non-trivial SOCP. The constraint `a_i^T x + r * ||a_i|| <= b_i` shows how norms enter optimization naturally — the ball must clear every halfplane by its full radius in the direction of the halfplane normal. Understanding this "budget for the worst direction" intuition is key to all robust optimization.

### Phase 2: Robust Least Squares — SOCP (~20 min)

**File:** `src/phase2_robust_ls.py`

This phase connects SOCP to a practical machine learning concern: what happens when your data matrix has measurement errors? Standard least squares (`min ||Ax - b||`) can overfit to noise in `A`. The robust formulation automatically introduces regularization proportional to the uncertainty level.

**What you implement:**
- `robust_least_squares(A_nom, b, epsilon)` — Formulate the robust LS problem as `minimize ||A_nom x - b|| + epsilon * ||x||`, solve with CVXPY, and compare with the ordinary least squares solution.

**Why it matters:** The robust formulation `||Ax - b|| + epsilon * ||x||` is mathematically identical to Tikhonov regularization (ridge regression). This reveals a deep connection: regularization is not just a statistical trick — it's the mathematically correct response to data uncertainty. The SOCP perspective makes this precise.

### Phase 3: Max-Cut SDP Relaxation — SDP (~30 min)

**File:** `src/phase3_maxcut.py`

This phase introduces SDP through the most celebrated application in combinatorial optimization. Max-Cut is NP-hard, but the SDP relaxation (Goemans-Williamson, 1995) solves a continuous relaxation in polynomial time and rounds to a partition that is at least 0.878 times optimal in expectation. This was a landmark result in approximation algorithms.

**What you implement:**
- `maxcut_sdp(adjacency_matrix)` — Create a symmetric PSD matrix variable, add diagonal-equals-one constraints, maximize `trace(L @ X)`, solve the SDP, then round using random hyperplane rounding. Compare the SDP bound with the rounded cut value.

**Why it matters:** This is the canonical example of SDP's power: it solves (approximately) a problem that no polynomial-time exact algorithm is known for. The PSD constraint `X ≽ 0` and `X_ii = 1` together encode "X is a correlation matrix," which relaxes the binary constraint `x_i in {-1, +1}`. Understanding this relaxation technique applies to graph coloring, Max-SAT, community detection, and many other combinatorial problems.

### Phase 4: Minimum Volume Ellipsoid — SDP (~20 min)

**File:** `src/phase4_mvee.py`

This phase combines SOCP constraints (point containment) with an SDP objective (`log_det`). The minimum volume enclosing ellipsoid is a fundamental object in computational geometry, robust statistics, and anomaly detection.

**What you implement:**
- `min_volume_ellipsoid(points)` — Create matrix variable `A` and vector variable `b` defining the ellipsoid `{x : ||Ax + b|| <= 1}`, constrain all points to be inside, maximize `log_det(A)`, solve, and return the ellipsoid parameters.

**Why it matters:** `log_det` maximization is the canonical SDP problem beyond simple matrix inequalities. It appears in experiment design (D-optimal), Gaussian maximum likelihood, and information theory. The MVEE also illustrates CVXPY's ability to handle mixed cone problems (SOCP containment + SDP/exponential cone objective) seamlessly.

## Motivation

SOCP and SDP are the frontier of tractable convex optimization — the most expressive problem classes that can still be solved in polynomial time by interior-point methods. SDP relaxations solve NP-hard problems approximately with provable guarantees: the Goemans-Williamson 0.878-approximation for Max-Cut is a foundational result in theoretical computer science.

Understanding conic programming unlocks the full power of CVXPY: after LP (040a) and basic convex optimization (041a), this practice completes the toolkit. In practice, many problems that look nonlinear or combinatorial have clean SDP/SOCP formulations that solvers handle efficiently — recognizing these formulations is a high-leverage skill.

Applications span operations research (robust optimization, facility location), machine learning (kernel learning, metric learning, covariance estimation), control theory (LMI-based controller design), signal processing (beamforming, sensor arrays), and combinatorial optimization (Max-Cut, graph coloring, stable set).

## Commands

All commands are run from the `practice_041b_conic_programming/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (CVXPY, NumPy, SciPy, SCS) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_chebyshev.py` | Phase 1: Chebyshev center — largest inscribed ball in a polytope (SOCP) |
| `uv run python src/phase2_robust_ls.py` | Phase 2: Robust least squares — handling uncertain data matrices (SOCP) |
| `uv run python src/phase3_maxcut.py` | Phase 3: Max-Cut SDP relaxation — Goemans-Williamson approximation (SDP) |
| `uv run python src/phase4_mvee.py` | Phase 4: Minimum volume ellipsoid — smallest enclosing ellipsoid (SDP) |

## References

- [CVXPY Documentation](https://www.cvxpy.org/) — Official docs with DCP rules, atoms, and examples
- [Boyd & Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — Chapter 4 (SOCP), Chapter 4.6 (SDP), free PDF
- [Goemans & Williamson (1995)](https://doi.org/10.1145/227683.227684) — Improved approximation algorithms for MAX CUT and MAX 2SAT
- [SCS Solver](https://www.cvxgrp.org/scs/) — Splitting Conic Solver documentation
- [Ben-Tal & Nemirovski, Lectures on Modern Convex Optimization](https://www2.isye.gatech.edu/~nemiMDCourse/) — Comprehensive treatment of conic programming
- [MOSEK Modeling Cookbook](https://docs.mosek.com/modeling-cookbook/index.html) — Excellent practical guide to conic formulations
- [Todd (2001), Semidefinite Optimization](https://people.orie.cornell.edu/miketodd/sdp.pdf) — Survey of SDP theory and applications

## State

`not-started`
