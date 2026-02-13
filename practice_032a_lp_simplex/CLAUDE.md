# Practice 032a: LP — Simplex Method

## Technologies

- **C++17** — Modern C++ with structured bindings, constexpr, auto
- **Eigen** — Header-only linear algebra library (matrices, vectors, block operations), fetched via CMake FetchContent
- **CMake 3.16+** — Build system with FetchContent for dependency management

## Stack

- C++17
- Eigen 3.4.0 (header-only, fetched via FetchContent)

## Theoretical Context

### What Linear Programming Is

**Linear Programming (LP)** is the problem of optimizing (minimizing or maximizing) a linear function subject to linear constraints. It is the most fundamental optimization problem in operations research — every MIP solver, interior point method, and sensitivity analysis technique is built on LP.

LP models appear everywhere: resource allocation (how to distribute limited resources across competing activities), production planning (how much of each product to manufacture), blending problems (how to mix ingredients to meet specifications at minimum cost), transportation (how to ship goods from warehouses to customers at minimum cost), and portfolio optimization (how to allocate investments subject to risk constraints).

The defining characteristic is **linearity**: the objective function is a weighted sum of decision variables, and every constraint is a linear inequality or equality. This restriction enables powerful theoretical guarantees (global optimality, polynomial solvability via interior point methods) and extremely efficient practical algorithms (the simplex method solves problems with millions of variables routinely).

### Standard Form

Every LP can be written in **standard form**:

```
minimize    c^T x
subject to  A x = b
            x >= 0
```

Where:
- `x` is the vector of decision variables (what we control)
- `c` is the cost/objective vector (coefficients in the objective)
- `A` is the constraint matrix (m constraints, n variables)
- `b` is the right-hand side vector (resource limits)

**Converting to standard form:**
- **Maximization** → negate objective: `max c^T x` becomes `min (-c)^T x`
- **Inequality <=** → add slack variable: `a^T x <= b` becomes `a^T x + s = b, s >= 0`
- **Inequality >=** → add surplus variable: `a^T x >= b` becomes `a^T x - s = b, s >= 0`
- **Unrestricted variable** → split: `x = x+ - x-` where `x+, x- >= 0`
- **Negative RHS** → multiply row by -1 (flip inequality direction)

### Geometry: The Feasible Polytope

The feasible region of an LP (the set of all x satisfying Ax = b, x >= 0) is a **convex polytope** — a bounded convex shape with flat faces in n-dimensional space. Key geometric facts:

- The **vertices** (corner points) of this polytope correspond exactly to **basic feasible solutions (BFS)** — solutions where exactly m variables are nonzero (the "basic" variables) and the remaining n-m variables are zero (the "non-basic" variables).
- **The Fundamental Theorem of LP**: If an LP has an optimal solution, there exists an optimal solution at a vertex. This is why the simplex method only needs to examine vertices, not the entire feasible region.
- The number of vertices is at most C(n, m) = n! / (m!(n-m)!), which is finite but can be exponential in n.

### The Simplex Algorithm

The simplex method (George Dantzig, 1947) works by:

1. **Start** at a vertex (basic feasible solution).
2. **Look around**: examine the edges emanating from the current vertex. Each edge corresponds to swapping one basic variable with one non-basic variable.
3. **Move** along the edge that improves the objective the most (steepest descent). This is determined by the **reduced costs** — the rate of change of the objective per unit increase of each non-basic variable.
4. **Arrive** at the adjacent vertex (a new BFS).
5. **Repeat** until no improving edge exists (all reduced costs are non-negative for minimization). The current vertex is then optimal.

### Simplex Tableau

The simplex algorithm is implemented using a **tableau** — an augmented matrix that encodes the current BFS and all information needed for pivoting:

```
         x1    x2    ...  xn    | RHS
z  [  c_bar_1  c_bar_2  ...  c_bar_n | -z  ]    ← objective row (reduced costs)
s1 [  a_11     a_12     ...  a_1n    | b_1 ]    ← constraint rows
s2 [  a_21     a_22     ...  a_2n    | b_2 ]
...
```

Key elements:
- **Basis matrix B**: The m columns of A corresponding to the m basic variables. At any BFS, B is invertible and the basic variables have values `B^{-1} b`.
- **Non-basis N**: The remaining n-m columns.
- **Reduced costs** `c_bar_j = c_j - c_B^T B^{-1} a_j`: The rate of change of the objective when variable j enters the basis. Negative reduced cost → entering j improves the objective.
- **Pivoting**: A single step of Gauss-Jordan elimination that swaps one basic variable out and one non-basic variable in. The pivot element determines the row operations.

### Degeneracy and Cycling

A BFS is **degenerate** if one or more basic variables have value zero. Degeneracy is common in practice and can cause **cycling**: the simplex method pivots between bases without improving the objective, potentially looping forever.

**Bland's rule** prevents cycling: instead of choosing the most negative reduced cost, always choose the variable with the smallest index. This guarantees finite termination but may require more pivots. In practice, cycling is extremely rare and most implementations use the "most negative reduced cost" rule (Dantzig's rule) with a maximum iteration limit as a safeguard.

### Complexity

- **Worst case**: Exponential. The Klee-Minty cube (1972) is a family of LPs where the simplex method (with Dantzig's rule) visits all 2^n vertices before finding the optimum. The cube is constructed so that the steepest-descent rule is maximally misled.
- **Average case**: Polynomial. Smoothed analysis (Spielman & Teng, 2004) shows that under small random perturbations of the input, the simplex method runs in polynomial time. This explains why it is extremely fast in practice despite the exponential worst case.
- **Interior point methods** (Karmarkar, 1984) are polynomial worst-case but not always faster in practice. Most commercial solvers (Gurobi, CPLEX) offer both simplex and interior point and choose automatically.

### Big-M Method and Two-Phase Simplex

When the LP has no obvious initial BFS (e.g., ">=" or "=" constraints), two approaches exist:

**Big-M Method**: Add artificial variables with a very large cost M in the objective. The solver drives artificials to zero to avoid the huge penalty. Simple but numerically fragile — choosing M is tricky.

**Two-Phase Simplex** (preferred):
- **Phase I**: Introduce artificial variables and minimize their sum. If the minimum is 0, a BFS for the original problem exists (and Phase I found it). If > 0, the original LP is infeasible.
- **Phase II**: Starting from the Phase I BFS, optimize the original objective using the standard simplex method.

This cleanly separates feasibility from optimality and avoids the numerical issues of Big-M.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Decision variable** | A quantity we control (e.g., units to produce) |
| **Objective function** | Linear function to minimize/maximize: `c^T x` |
| **Constraint** | Linear inequality/equality: `a^T x <= b`, `a^T x >= b`, or `a^T x = b` |
| **Standard form** | `min c^T x` s.t. `Ax = b, x >= 0` (equalities only, non-negative vars) |
| **Slack variable** | Added to convert `<=` to `=`: absorbs the "slack" in the constraint |
| **Surplus variable** | Added to convert `>=` to `=`: measures how much the constraint is exceeded |
| **Basic feasible solution (BFS)** | A vertex of the feasible polytope; m variables are basic, rest are 0 |
| **Basis** | The set of m basic variables; their columns in A form an invertible matrix |
| **Reduced cost** | `c_bar_j = c_j - c_B^T B^{-1} a_j` — rate of objective change for variable j |
| **Pivot** | Gauss-Jordan elimination step that swaps one basic/non-basic variable |
| **Entering variable** | Non-basic variable with negative reduced cost that enters the basis |
| **Leaving variable** | Basic variable that hits zero first (minimum ratio test) and leaves the basis |
| **Degeneracy** | A BFS where a basic variable has value 0; can cause cycling |
| **Bland's rule** | Anti-cycling pivot rule: choose smallest-index variable with negative reduced cost |
| **Artificial variable** | Temporary variable added in Phase I to find an initial BFS |
| **Two-phase method** | Phase I finds feasibility, Phase II optimizes — cleanly separated |
| **Unbounded LP** | No finite optimum; objective can be made arbitrarily good |
| **Infeasible LP** | No solution satisfies all constraints simultaneously |

### Where LP Fits in the Optimization Hierarchy

```
LP (Linear Programming)
 └── Convex Optimization (includes QP, SOCP, SDP)
      └── Non-convex Optimization (general NLP, MIP)
           └── Combinatorial Optimization (discrete, NP-hard problems)
```

LP is the simplest and most tractable class. Every LP is convex (linear functions are both convex and concave), so any local optimum is a global optimum. LP is solvable in polynomial time (via interior point methods) and has strong duality (the primal and dual optimal values are equal). Mixed-Integer Programming (MIP) solvers work by solving a sequence of LP relaxations (Branch & Bound), so LP speed directly determines MIP performance.

## Description

Build a simplex solver from scratch using Eigen for matrix operations. Start by formulating LP problems in standard form, then implement the pivoting mechanics, assemble the full simplex loop, and finally handle arbitrary constraint types with the two-phase method. Each phase solves concrete problems (production planning, diet problem, resource allocation) to see the algorithm work step by step.

### What you'll build

1. **Standard form converter** — Transform inequality-based LP models into the canonical form the simplex requires
2. **Pivot engine** — Select entering/leaving variables and perform Gauss-Jordan elimination on the tableau
3. **Full simplex solver** — Iterate pivots until optimality or detect unboundedness
4. **Two-phase simplex** — Handle problems with ">=" and "=" constraints via auxiliary LP

## Instructions

### Phase 1: LP Formulation & Standard Form (~20 min)

**File:** `src/formulation.cpp`

This phase teaches the critical first step: translating a human-readable optimization problem into the machine-friendly standard form. Without this conversion, the simplex algorithm cannot operate. Understanding standard form is essential because every LP textbook, solver manual, and research paper assumes it.

**What you implement:**
- `to_standard_form()` — Convert inequalities to equalities by adding slack/surplus variables, handle max-to-min conversion, ensure non-negative RHS.
- `setup_initial_tableau()` — Construct the augmented matrix (tableau) from the standard-form LP, identify the initial basis from slack variables.

**Why it matters:** Formulation errors are the #1 source of incorrect LP results in practice. If you add a slack variable with the wrong sign or forget to negate the objective for maximization, the solver will produce wrong answers silently. This phase builds the mechanical precision needed to formulate correctly every time.

### Phase 2: Simplex Pivoting Mechanics (~20 min)

**File:** `src/pivoting.cpp`

This phase teaches the atomic operation of the simplex method: the pivot. You perform a single pivot step on a hardcoded tableau to see exactly how the algorithm moves from one vertex to another.

**What you implement:**
- `select_pivot_column()` — Dantzig's rule: find the most negative reduced cost (entering variable).
- `select_pivot_row()` — Minimum ratio test: find the tightest constraint (leaving variable).
- `pivot()` — Gauss-Jordan elimination to update the tableau.

**Why it matters:** Every iteration of the simplex method is one pivot. Understanding the geometry (moving along an edge of the polytope) and the algebra (elementary row operations on the tableau) is necessary to debug solver behavior, interpret sensitivity analysis, and understand why certain formulations solve faster than others.

### Phase 3: Full Simplex Solver (~25 min)

**File:** `src/simplex.cpp`

This phase assembles the pivot functions into a complete solver loop that runs from initial BFS to optimality. The pivot functions from Phase 2 are provided (already implemented) so you can focus on the loop logic, termination conditions, and solution extraction.

**What you implement:**
- `solve()` — The main simplex loop: repeatedly select pivot column, check for optimality, select pivot row, check for unboundedness, pivot, repeat. Extract the optimal solution from the final tableau.

**Why it matters:** The loop structure (with its three possible outcomes: optimal, unbounded, max-iterations) is the skeleton of every LP solver. Extracting the solution correctly from the tableau — reading basic variable values from the RHS column — is a skill you'll use when interpreting solver output in production.

### Phase 4: Two-Phase Simplex (~25 min)

**File:** `src/two_phase.cpp`

This phase handles the general case: LPs where the initial BFS is not obvious because constraints include ">=" or "=". The two-phase method first solves an auxiliary LP to find feasibility, then optimizes the original objective.

**What you implement:**
- `phase_one()` — Construct the auxiliary LP (minimize sum of artificials), solve it, check feasibility, and prepare the tableau for Phase II.
- `two_phase_simplex()` — The complete procedure: convert to standard form, decide if Phase I is needed, run Phase I if necessary, then run Phase II with the original objective.

**Why it matters:** Most real-world LPs have mixed constraint types. The two-phase method is how production solvers handle this internally. Understanding Phase I also teaches you about infeasibility detection — a critical capability when debugging models that "should" be feasible but aren't (common in supply chain and scheduling problems).

## Motivation

LP is the foundation of all operations research. Every MIP solver (Gurobi, CPLEX, HiGHS) uses LP relaxation via the simplex method or interior point methods at its core. Sensitivity analysis, reduced costs, and shadow prices — the tools for interpreting and improving optimization models — all come from LP duality theory. Understanding the simplex algorithm's internals is essential for:

- **Debugging solver output**: Why did the solver return "infeasible"? Why is this variable at zero? What does the reduced cost mean?
- **Formulating tight models**: Knowing how the simplex moves through vertices helps you write formulations that solve faster.
- **Understanding solver parameters**: Pivot rules, tolerances, preprocessing — all make sense once you've built a simplex solver.
- **Interview preparation**: LP and simplex are standard topics in OR/optimization interviews at trading firms, logistics companies, and tech companies with optimization teams.

This practice builds the mental model that makes every subsequent OR topic (duality, MIP, convex optimization) click.

## Commands

All commands are run from the `practice_032a_lp_simplex/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches Eigen via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all four phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_formulation 2>&1"` | Build Phase 1: LP formulation & standard form |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_pivoting 2>&1"` | Build Phase 2: Simplex pivoting mechanics |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_simplex 2>&1"` | Build Phase 3: Full simplex solver |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_two_phase 2>&1"` | Build Phase 4: Two-phase simplex |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\phase1_formulation.exe` | Run Phase 1: LP formulation & standard form |
| `build\Release\phase2_pivoting.exe` | Run Phase 2: Simplex pivoting mechanics |
| `build\Release\phase3_simplex.exe` | Run Phase 3: Full simplex solver |
| `build\Release\phase4_two_phase.exe` | Run Phase 4: Two-phase simplex |

## References

- [Dantzig, G.B. (1963). Linear Programming and Extensions](https://press.princeton.edu/books/paperback/9780691059136/linear-programming-and-extensions) — Original reference by the inventor of the simplex method
- [Bertsimas, D. & Tsitsiklis, J. (1997). Introduction to Linear Optimization](https://athenasc.com/linoptbook.html) — Standard graduate textbook covering simplex, duality, and interior point methods
- [Vanderbei, R. (2020). Linear Programming: Foundations and Extensions](https://vanderbei.princeton.edu/307/textbook/BasissAll.pdf) — Excellent exposition of simplex tableau operations
- [Eigen Documentation](https://eigen.tuxfamily.org/dox/) — Matrix/vector operations used throughout
- [Klee-Minty Cube](https://en.wikipedia.org/wiki/Klee%E2%80%93Minty_cube) — Worst-case example for the simplex method
- [Spielman & Teng (2004). Smoothed Analysis of Algorithms](https://arxiv.org/abs/cs/0111050) — Why simplex is fast in practice despite exponential worst case

## State

`not-started`
