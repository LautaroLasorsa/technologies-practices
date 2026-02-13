# Practice 041a: Convex Optimization — CVXPY

## Technologies

- **CVXPY** — Domain-specific language for convex optimization embedded in Python. Natural math-like syntax with automatic convexity verification via DCP rules.
- **SCS** — Splitting Conic Solver. First-order ADMM-based solver. Handles all cone types (LP, SOCP, SDP, exponential). Scales to large problems but lower accuracy than IPMs.
- **ECOS** — Embedded Conic Solver. Interior-point method for SOCP. Fast and accurate for small-to-medium problems. Legacy default in CVXPY < 1.5.
- **CLARABEL** — Modern interior-point solver written in Rust. Default solver in CVXPY >= 1.5. Handles LP, QP, SOCP, SDP, exponential cones. Faster and more numerically stable than ECOS.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- CVXPY >= 1.5 (modeling layer + CLARABEL/SCS bundled)
- NumPy >= 1.26, SciPy >= 1.12 (data generation)
- uv (package manager)

## Theoretical Context

### What CVXPY Is and the Problem It Solves

**CVXPY** is a Python-embedded domain-specific language (DSL) for convex optimization. It lets you formulate optimization problems using natural mathematical syntax:

```python
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.norm(A @ x - b, 2) + lambd * cp.norm(x, 1)))
prob.solve()
```

The key innovation: CVXPY **automatically verifies that your problem is convex** before sending it to a solver. It does this via Disciplined Convex Programming (DCP) — a compositional ruleset that checks curvature at every node of the expression tree. If you accidentally write a non-convex problem, CVXPY raises `DCPError` at construction time, not at solve time. This prevents the silent wrong-answer failures that plague general-purpose nonlinear solvers.

CVXPY then automatically transforms the verified convex problem into a standard conic form (LP, SOCP, SDP, or exponential cone program) and dispatches it to an appropriate solver. You never write cone constraints manually.

### How CVXPY Works Internally

1. **Expression tree construction**: When you write `cp.norm(A @ x - b, 2) + lambd * cp.norm(x, 1)`, CVXPY builds a tree of `Atom` nodes. Each atom has a known curvature (convex/concave/affine) and sign (positive/negative/unknown).

2. **DCP verification**: CVXPY walks the expression tree and checks that composition rules are satisfied at every node. If the overall objective is convex (for minimization) or concave (for maximization), and all constraints are convex sets, the problem is DCP-compliant.

3. **Canonicalization**: The verified problem is rewritten into a standard conic form. For example, `cp.norm(x, 2) <= t` becomes a second-order cone constraint `||x||_2 <= t`. This happens via graph implementations — each atom knows how to express itself in conic form.

4. **Solver dispatch**: CVXPY selects a solver based on the cone types present. CLARABEL (default) handles most problems. SCS handles everything including SDPs. The solver receives the problem in standard form: `min c^T x s.t. Ax + s = b, s in K` where `K` is a product of cones.

5. **Solution extraction**: After solving, CVXPY maps the conic solution back to the original variables. `x.value` gives the optimal value as a NumPy array. Constraint dual values are accessible via `constraint.dual_value`.

### Disciplined Convex Programming (DCP) Rules

DCP is the **compositional type system** that makes CVXPY work. Every expression has two properties:

| Property | Values | Meaning |
|----------|--------|---------|
| **Curvature** | convex, concave, affine, unknown | How the expression curves |
| **Sign** | positive, negative, unknown | Sign of the expression's values |

**Composition rules** (the key insight):

| Outer function | Inner expression | Result |
|---------------|-----------------|--------|
| convex & nondecreasing | convex | convex |
| convex & nonincreasing | concave | convex |
| concave & nondecreasing | concave | concave |
| concave & nonincreasing | convex | concave |
| affine | any | same as inner |

**Arithmetic rules:**

| Operation | Rule |
|-----------|------|
| `convex + convex` | convex |
| `concave + concave` | concave |
| `affine + anything` | same as anything |
| `nonneg_const * convex` | convex |
| `neg_const * convex` | concave |

**Problem rules:**
- Minimize a **convex** objective
- Maximize a **concave** objective
- Constraints: `convex <= concave`, `affine == affine`

If any rule is violated, CVXPY raises `DCPError`. This is a feature, not a limitation: it forces you to reformulate in a provably correct way.

### Atoms: Built-in Functions with Known Curvature

CVXPY provides a library of **atoms** — functions with verified curvature properties. Key atoms used in this practice:

| Atom | Curvature | Sign | Description |
|------|-----------|------|-------------|
| `cp.sum_squares(x)` | convex | positive | `||x||_2^2` — sum of squared entries |
| `cp.norm(x, 1)` / `cp.norm1(x)` | convex | positive | `||x||_1` — sum of absolute values |
| `cp.norm(x, 2)` / `cp.norm2(x)` | convex | positive | `||x||_2` — Euclidean norm |
| `cp.quad_form(x, P)` | convex (if P PSD) | depends | `x^T P x` — quadratic form |
| `cp.sum(x)` | affine | depends | Sum of entries |
| `cp.diff(x)` | affine | unknown | First differences: `x[1]-x[0], x[2]-x[1], ...` |
| `cp.maximum(x, y)` | convex | depends | Elementwise maximum |
| `cp.log(x)` | concave | unknown | Elementwise logarithm |
| `cp.exp(x)` | convex | positive | Elementwise exponential |
| `cp.multiply(c, x)` | affine (c const) | depends | Elementwise multiplication by constant |

The curvature of an atom determines where it can appear in a DCP problem. Convex atoms can appear in the objective (minimization) or in `<= ` constraints. Concave atoms can appear in the objective (maximization) or in `>=` constraints.

### Problem Types CVXPY Solves

CVXPY handles any problem that can be expressed in conic form:

| Problem Type | Cones Used | Example |
|-------------|-----------|---------|
| **LP** (Linear Program) | Nonneg cone | `min c^T x, s.t. Ax <= b` |
| **QP** (Quadratic Program) | Second-order cone | `min x^T P x + q^T x, s.t. Ax <= b` |
| **SOCP** (Second-Order Cone) | SOC | `min c^T x, s.t. ||Ax+b|| <= c^T x + d` |
| **SDP** (Semidefinite Program) | PSD cone | `min tr(CX), s.t. X >> 0` |
| **Exponential cone** | Exp cone | Problems with `cp.log`, `cp.exp`, `cp.entr` |

CVXPY automatically detects which cones are needed and selects a solver that supports them.

### Solver Backends

CVXPY ships with three open-source solvers:

| Solver | Type | Strengths | Cones |
|--------|------|-----------|-------|
| **CLARABEL** | Interior-point (Rust) | Fast, accurate, default since v1.5 | LP, QP, SOCP, SDP, Exp |
| **SCS** | ADMM (first-order) | Scales to large problems, all cones | LP, QP, SOCP, SDP, Exp |
| **OSQP** | ADMM | Fast for QPs specifically | LP, QP only |

Optional solvers (require separate install): **MOSEK** (commercial, state-of-the-art), **Gurobi** (LP/QP/SOCP), **ECOS** (legacy IPM for SOCP).

CVXPY automatically selects the best available solver based on the problem's cone types. You can override: `prob.solve(solver=cp.SCS)`.

### Dual Values and Sensitivity

After solving, every constraint object exposes `.dual_value` — the **shadow price** (Lagrange multiplier). For a constraint `g(x) <= 0`:

- `dual_value > 0`: constraint is active; relaxing it by delta improves the objective by approximately `dual_value * delta`
- `dual_value = 0`: constraint is not active (slack exists)

This generalizes LP shadow prices (from practice 032b/040a) to all convex problems.

### Parameters for Fast Re-Solving

`cp.Parameter()` creates a symbolic constant whose value can be changed without re-parsing the problem:

```python
lambd = cp.Parameter(nonneg=True)
prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + lambd * cp.norm1(x)))

for lam_val in [0.01, 0.1, 1.0, 10.0]:
    lambd.value = lam_val
    prob.solve()  # Re-uses cached problem structure
```

When the problem is DPP-compliant (Disciplined Parametrized Programming), CVXPY caches the canonicalization and only updates the numerical data. This can be 10-100x faster than constructing a new problem each time.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **DCP** | Disciplined Convex Programming — compositional rules that verify convexity of expressions |
| **Atom** | Built-in CVXPY function with known curvature and sign (e.g., `cp.norm`, `cp.quad_form`) |
| **Curvature** | Property of an expression: convex, concave, affine, or unknown |
| **Conic form** | Standard representation: `min c^T x s.t. Ax + s = b, s in K` (product of cones) |
| **SOCP** | Second-Order Cone Program — constraints of form `||Ax+b|| <= c^T x + d` |
| **SDP** | Semidefinite Program — constraints involving positive semidefinite matrices |
| **Dual value** | Shadow price / Lagrange multiplier of a constraint at the optimum |
| **Parameter** | `cp.Parameter()` — symbolic constant for fast re-solving without re-parsing |
| **DPP** | Disciplined Parametrized Programming — rules for parameter-dependent problems |
| **Canonicalization** | Transformation from user-level expression tree to standard conic form |
| **Warm start** | Using previous solution as initial guess for next solve |
| **Total variation** | `sum(|x[i+1] - x[i]|)` — regularizer that preserves edges while smoothing |
| **Robust optimization** | Optimizing against worst-case perturbations in problem data |
| **Efficient frontier** | Set of Pareto-optimal portfolios (max return for given risk) |
| **Ridge regression** | L2-regularized least squares — shrinks all coefficients |
| **LASSO** | L1-regularized least squares — produces sparse solutions (feature selection) |

### Where CVXPY Fits in the Ecosystem

```
Convex Optimization Libraries (verify convexity)
├── CVXPY ............. Python DSL, DCP rules, auto conic form ← this practice
├── Convex.jl ......... Julia equivalent of CVXPY
└── CVXR .............. R equivalent of CVXPY

General Algebraic Modeling (no convexity guarantee)
├── PuLP .............. LP/MIP only, simplest (practice 040a)
├── Pyomo ............. Full NLP/MIP/stochastic (practice 040b)
├── JuMP (Julia) ...... Fast, extensible
└── GAMS/AMPL ......... Commercial algebraic modeling

Solvers (called by CVXPY)
├── CLARABEL .......... Default open-source IPM (Rust)
├── SCS ............... First-order, all cones
├── OSQP .............. QP-specialized first-order
├── MOSEK ............. Commercial, state-of-the-art
└── Gurobi ............ Commercial LP/QP/SOCP
```

CVXPY's unique value: **DCP verification prevents modeling errors**. With PuLP/Pyomo, you can accidentally write non-convex constraints and get wrong answers or solver failures. With CVXPY, a non-convex formulation is rejected at construction time. The trade-off: CVXPY only handles convex problems. For MIP, use PuLP/Pyomo. For non-convex NLP, use Pyomo with IPOPT.

## Description

Model and solve convex optimization problems using CVXPY. Progress from regularized regression (QP) through portfolio optimization (QP with quadratic form) to signal denoising (SOCP) and robust optimization. Each phase introduces new atoms, DCP patterns, and solver features.

### What you'll build

1. **Regularized regression** — Ridge (L2) and LASSO (L1) regression. Compare sparsity patterns and bias-variance tradeoff across regularization strengths.
2. **Portfolio optimization** — Markowitz mean-variance model with CVXPY's `quad_form`. Trace the efficient frontier and analyze dual values.
3. **Signal denoising** — Total variation denoising on a 1D piecewise-constant signal. Demonstrate edge-preserving smoothing.
4. **Robust linear program** — Solve an LP with uncertain constraint data. Compare nominal vs robust solutions across uncertainty levels.

## Instructions

### Phase 1: Least-Squares and Regularized Regression — LP/QP (~25 min)

**File:** `src/phase1_regression.py`

This phase teaches the fundamental CVXPY workflow on a classic statistical problem. You'll implement ridge regression (L2 penalty) and LASSO regression (L1 penalty), both expressible as convex programs. The key learning: CVXPY atoms like `cp.sum_squares` and `cp.norm1` have known curvature, so CVXPY can verify that `minimize(convex + convex)` is indeed convex.

**What you implement:**
- `solve_ridge(A, b, lambd)` — Create a `cp.Variable`, form the ridge objective `||Ax-b||_2^2 + lambda*||x||_2^2` using `cp.sum_squares`, solve, return coefficients.
- `solve_lasso(A, b, lambd)` — Same structure but with L1 penalty `lambda*||x||_1` using `cp.norm1`. Observe that LASSO produces sparse solutions (some coefficients exactly zero).

**Why it matters:** These are the simplest non-trivial CVXPY problems. Ridge is a QP (quadratic objective, no constraints). LASSO involves the non-smooth L1 norm, which CVXPY handles by converting to a second-order cone program internally. Comparing the two teaches the fundamental L1 vs L2 tradeoff: sparsity vs shrinkage.

### Phase 2: Portfolio Optimization — QP (~25 min)

**File:** `src/phase2_portfolio.py`

This phase introduces `cp.quad_form(w, Sigma)` — the quadratic form atom that is convex when Sigma is positive semidefinite. You'll implement the Markowitz mean-variance portfolio model: minimize portfolio variance subject to a target return and weight constraints. Then sweep over target returns to trace the efficient frontier.

**What you implement:**
- `solve_markowitz(mu, Sigma, target_return)` — Create weight variables, form the risk objective `w^T Sigma w` using `cp.quad_form`, add return and budget constraints, solve. Access dual values to understand the "price of risk."

**Why it matters:** Portfolio optimization is the canonical QP in finance. The `quad_form` atom is one of CVXPY's most important atoms — it appears in any problem involving covariance matrices. The dual value of the return constraint gives the marginal cost of increasing target return, directly connecting to the efficient frontier slope.

### Phase 3: Signal Denoising with Total Variation — SOCP (~25 min)

**File:** `src/phase3_denoising.py`

This phase combines `cp.sum_squares` (data fidelity) with `cp.norm1(cp.diff(x))` (total variation penalty). The total variation of a signal is the sum of absolute first differences — it penalizes oscillations while allowing sharp jumps. This is an SOCP (the L1 norm on first differences requires second-order cone constraints internally).

**What you implement:**
- `total_variation_denoise(y, lambd)` — Form the TV denoising objective, solve, return the denoised signal. Experiment with different lambda values.

**Why it matters:** TV denoising is the prototypical application of convex optimization in signal/image processing. It demonstrates a non-trivial atom composition: `norm1(diff(x))` — and the DCP check that `convex(affine) = convex`. The lambda parameter controls the smoothness-fidelity tradeoff, analogous to the regularization parameter in Phase 1.

### Phase 4: Robust Linear Program (~25 min)

**File:** `src/phase4_robust.py`

This phase formulates an LP where the constraint matrix A is uncertain. Instead of solving for nominal data, you solve the **robust counterpart**: guarantee feasibility for all perturbations within an uncertainty set. The robust counterpart of a linear constraint with ellipsoidal uncertainty is a second-order cone constraint — CVXPY handles this naturally.

**What you implement:**
- `solve_robust_lp(c, A_nom, b, epsilon)` — For each constraint row, add the robust counterpart `a_i^T x + epsilon * ||x||_2 <= b_i`. Solve and compare with the nominal LP solution.

**Why it matters:** Robust optimization is one of the major applications of conic programming. Real-world data is never exact — robust formulations provide worst-case guarantees. The key insight: adding robustness turns an LP into an SOCP, and CVXPY handles the transformation transparently. The epsilon parameter controls conservatism: larger epsilon = more conservative (worse objective) but more robust to perturbations.

## Motivation

CVXPY is the de-facto standard for convex optimization in Python, used in finance (portfolio optimization), machine learning (regularized regression, SVM), signal processing (denoising, compressed sensing), control systems, and operations research. Its DCP verification prevents modeling errors that would silently produce wrong answers in general-purpose tools.

After implementing convex optimization algorithms from scratch (practices 033a/033b — gradient descent, proximal methods, Newton's method), this practice shows how those algorithms are invoked via a high-level modeling language. Understanding CVXPY's atoms, DCP rules, and solver selection is essential for any optimization practitioner — the modeling patterns transfer directly to any conic optimization tool.

## Commands

All commands are run from the `practice_041a_cvxpy/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (CVXPY, NumPy, SciPy) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_regression.py` | Phase 1: Ridge and LASSO regression — L2 vs L1 regularization comparison |
| `uv run python src/phase2_portfolio.py` | Phase 2: Markowitz portfolio — efficient frontier with `quad_form` |
| `uv run python src/phase3_denoising.py` | Phase 3: Total variation signal denoising — edge-preserving smoothing |
| `uv run python src/phase4_robust.py` | Phase 4: Robust linear program — nominal vs robust under uncertainty |

## References

- [CVXPY Documentation](https://www.cvxpy.org/) — Official documentation, tutorials, and API reference
- [CVXPY DCP Tutorial](https://www.cvxpy.org/tutorial/dcp/index.html) — Disciplined Convex Programming rules explained with examples
- [CVXPY Atoms Reference](https://www.cvxpy.org/tutorial/functions/index.html) — Complete list of atoms with curvature and sign
- [CVXPY Solver Features](https://www.cvxpy.org/tutorial/solvers/index.html) — Solver capabilities, selection, and tuning
- [CVXPY DPP Tutorial](https://www.cvxpy.org/tutorial/dpp/index.html) — Disciplined Parametrized Programming for fast re-solving
- [CLARABEL Solver](https://arxiv.org/html/2405.12762v1) — Paper describing the default solver's algorithm
- [CVXPY GitHub](https://github.com/cvxpy/cvxpy) — Source code and discussions
- [Boyd & Vandenberghe, Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/) — The textbook behind CVXPY's design
- [CVXPY TV In-painting Example](https://www.cvxpy.org/examples/applications/tv_inpainting.html) — Total variation example from CVXPY docs

## State

`not-started`
