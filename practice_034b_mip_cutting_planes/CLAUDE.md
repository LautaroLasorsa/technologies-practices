# Practice 034b: MIP --- Cutting Planes & Heuristics

## Technologies

- **Rust** --- Systems programming language with zero-cost abstractions and strong type safety
- **`good_lp`** (HiGHS backend) --- Mixed Integer Linear Programming modeler for Rust; HiGHS is a free, high-performance parallel MIP solver (MIT-licensed)
- **`ordered-float`** --- Wrapper types for f64 that implement Ord/Hash, useful for sorting and deduplication of floating-point values

## Stack

- Rust (edition 2021)
- `good_lp` 1.10+ with `highs` feature
- `ordered-float` 4

## Theoretical Context

### Cutting Planes Motivation

The LP relaxation of a MIP gives a bound on the optimal value (lower bound for minimization), but the gap between the LP relaxation optimum and the true MIP optimum can be large. The LP relaxation ignores integrality constraints, so its feasible region (a convex polytope) is a superset of the convex hull of integer-feasible points.

**Cutting planes** add valid inequalities that tighten the LP relaxation --- they shrink the LP polytope closer to the convex hull of integer solutions --- WITHOUT cutting off any integer-feasible point. The tighter the LP relaxation, the smaller the integrality gap, and the fewer branch-and-bound nodes needed to prove optimality.

This is the single most important idea in modern MIP solving. The combination of cutting planes with branch-and-bound (branch-and-cut) is responsible for the >10,000x speedup in MIP solvers over the past 25 years (Bixby, 2012).

### Valid Inequality

A constraint `a^T x <= b` is a **valid inequality** for a MIP if every integer-feasible solution satisfies it, but some fractional LP-feasible solutions may violate it. Adding a valid inequality:

- Does NOT remove any integer solution from the feasible set
- MAY remove fractional solutions, tightening the LP relaxation
- Improves the LP bound (moves it toward the MIP optimum)

The ideal set of valid inequalities would describe the convex hull of integer solutions exactly, making the LP relaxation solve the MIP directly. In practice, we add a subset of cuts that gives the most tightening.

### Gomory Fractional Cuts

**Gomory cuts** are the foundational cutting plane technique, derived directly from the simplex tableau. They are **general-purpose** --- applicable to any MIP, regardless of problem structure.

Given a simplex tableau row for basic variable x_i:

```
x_i + sum_j (a_bar_ij * x_j) = b_bar_i    (where x_j are non-basic)
```

If `b_bar_i` is fractional (not integer), define:
- `f_0 = frac(b_bar_i)` = fractional part of the RHS
- `f_j = frac(a_bar_ij)` = fractional part of each non-basic coefficient

The **Gomory fractional cut** is:

```
sum_j (f_j * x_j) >= f_0
```

**Why it's valid:** Any integer solution must make the LHS an integer minus an integer = integer, and that integer must be >= f_0 > 0, hence >= 1. But the current LP solution has all non-basic variables at 0, so LHS = 0 < f_0. The cut is violated by the current fractional LP optimum but satisfied by all integer solutions.

**Key subtlety:** The `frac()` function must always return a non-negative value. For negative numbers, `frac(-0.3) = 0.7`, not `-0.3`. The convention is `frac(a) = a - floor(a)`, where `floor` rounds toward negative infinity.

### Cover Inequalities (for Knapsack)

For a **knapsack constraint** `sum_j (a_j * x_j) <= b` with binary variables `x_j in {0, 1}`:

- A **cover** C is a subset of variables where `sum_{j in C} a_j > b`. If all variables in C are set to 1, the knapsack overflows.
- The **cover inequality** is: `sum_{j in C} x_j <= |C| - 1`. At least one variable in C must be 0.
- A cover is **minimal** if no proper subset is also a cover (removing any element makes the total weight <= b).

Cover inequalities are **structure-specific** --- they exploit the knapsack structure, unlike Gomory cuts which are general. They are among the most effective cuts for 0-1 knapsack substructures, which appear in many practical MIP models (scheduling, bin packing, capital budgeting).

**Finding violated covers:** Given an LP solution x*, we want a cover C where `sum_{j in C} x*_j > |C| - 1` (the cover inequality is violated by the current LP solution). A greedy heuristic: sort variables by LP value descending, greedily add to C until total weight exceeds capacity, then minimize by removing unnecessary elements.

### Cutting Plane Algorithm (Gomory's Algorithm)

The pure cutting plane algorithm:

1. Solve LP relaxation
2. If solution is integer-feasible, STOP --- optimal
3. Select a fractional basic variable
4. Generate a Gomory cut from that tableau row
5. Add the cut to the LP
6. Re-solve (dual simplex is efficient since we added a constraint)
7. Go to step 2

Gomory (1958) proved that this algorithm terminates finitely for pure integer programs (all variables integer). However, in practice:

- **Cuts accumulate**, making the LP larger and harder to solve
- **Numerical precision degrades** as fractional parts compound
- **Convergence can be very slow** --- many rounds may be needed

This is why pure cutting planes are rarely used alone. Modern solvers combine them with branch-and-bound.

### Branch-and-Cut

**Branch-and-cut** is the algorithm used by ALL modern MIP solvers (Gurobi, CPLEX, SCIP, HiGHS). It integrates cutting planes into branch-and-bound:

At each B&B node:
1. Solve the LP relaxation
2. Generate cuts to tighten the LP (Gomory cuts, cover cuts, flow covers, MIR cuts, etc.)
3. Re-solve the LP with cuts
4. If still fractional, branch as usual

The cuts tighten the LP bound at each node, leading to more pruning (more nodes can be fathomed by bound). This dramatically reduces the B&B tree size. The art of modern MIP solving is in cut selection --- which cuts to generate, how many, when to stop cutting and start branching.

### Rounding Heuristics

Given an LP relaxation solution with fractional values, **rounding heuristics** try to quickly find feasible integer solutions (incumbents) without solving the MIP to optimality. A good incumbent enables pruning in B&B.

**Simple rounding:** Round each fractional integer variable to the nearest integer. Fast but often produces infeasible solutions (violates constraints).

**Rounding down (for covering/packing):** For knapsack-like constraints, rounding down preserves feasibility at the cost of optimality.

**Feasibility pump** (Fischetti, Glover & Lodi, 2005): Alternate between:
- Rounding the LP solution to get an integer (possibly infeasible) point x_hat
- Projecting x_hat back to the LP feasible region by solving `min ||x - x_hat||` s.t. LP constraints

Iterate until the two points coincide (found a feasible integer solution) or a limit is reached.

**Diving heuristics:** Fix the most fractional variable to its nearest integer, re-solve LP, repeat. More sophisticated than simple rounding but still fast.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Valid inequality** | Constraint satisfied by all integer solutions but possibly violated by LP solutions |
| **Cutting plane** | A valid inequality added to tighten the LP relaxation |
| **Gomory cut** | Cut derived from simplex tableau row with fractional RHS |
| **Fractional part** | `frac(a) = a - floor(a)`, always in [0, 1) |
| **Cover (knapsack)** | Subset C where `sum_{j in C} a_j > b`; all-1 assignment overflows |
| **Minimal cover** | Cover where no proper subset is also a cover |
| **Cover inequality** | `sum_{j in C} x_j <= \|C\| - 1` |
| **Integrality gap** | Difference between LP relaxation bound and MIP optimum |
| **Branch-and-cut** | B&B + cutting planes at each node; the modern MIP algorithm |
| **Separation problem** | Given x*, find a valid inequality violated by x*, or prove none exists |
| **Cut pool** | Collection of generated cuts for reuse across B&B nodes |
| **Rounding heuristic** | Round fractional LP solution to integer; check feasibility |
| **Feasibility pump** | Alternate rounding + LP projection to find feasible integer point |
| **Incumbent** | Best known feasible integer solution (provides upper bound for min) |

### Where Cutting Planes Fit

```
LP Relaxation (weak bound)
  |
  +-- Add cutting planes --> Tighter LP (better bound)
  |                            |
  |                            +-- Still fractional? Branch (B&B)
  |                            |     At each node, add more cuts (Branch-and-Cut)
  |                            |
  |                            +-- Integer? Optimal!
  |
  +-- Rounding heuristic --> Feasible incumbent (upper bound for min)
```

## Description

Implement Gomory fractional cuts, cover inequalities for knapsack, a pure cutting plane algorithm, and a branch-and-cut skeleton in Rust. Each phase builds on cutting plane theory to show how modern MIP solvers achieve their performance. You will manually track simplex tableau information for small problems (since `good_lp` abstracts away solver internals) and use `good_lp`/HiGHS for LP solving.

### What you'll build

1. **Gomory cut generator** --- Derive Gomory fractional cuts from tableau data and observe LP bound tightening
2. **Cover inequality finder** --- Identify minimal covers in knapsack constraints, generate and add cover cuts
3. **Pure cutting plane solver** --- Iterate Gomory cut generation until integer optimality or iteration limit
4. **Branch-and-cut skeleton** --- Integrate cuts into B&B with a rounding heuristic for finding incumbents

## Instructions

### Phase 1: Gomory Cuts (~25 min)

**File:** `src/gomory.rs`

This phase teaches the most fundamental cutting plane: the Gomory fractional cut. You will derive a cut from a simplex tableau row and observe how it tightens the LP relaxation. The key insight is that the cut is mechanically derived from the tableau --- no problem-specific knowledge is needed.

**What you implement:**
- `compute_gomory_cut()` --- Given a tableau row (non-basic coefficients and fractional RHS), compute the Gomory cut coefficients and RHS using the fractional-part formula.
- `add_cut_and_resolve()` --- Add the generated cut as a new constraint to the LP and re-solve to observe bound tightening.

**Why it matters:** Gomory cuts are the theoretical foundation of all cutting plane methods. Every MIP textbook starts here. The mechanical derivation from the tableau is what makes them general-purpose --- they work on ANY MIP. Understanding the `frac()` function and why negative coefficients need careful handling prevents subtle bugs that plague implementations.

### Phase 2: Cover Inequalities (~25 min)

**File:** `src/covers.rs`

This phase teaches structure-specific cuts: cover inequalities for knapsack constraints. Unlike Gomory cuts (which come from the tableau), cover cuts exploit the combinatorial structure of binary knapsack constraints.

**What you implement:**
- `find_minimal_cover()` --- Given knapsack weights, capacity, and LP solution, find a minimal cover whose cover inequality is violated by the LP solution.
- `generate_cover_inequality()` --- Convert a cover set into a linear inequality `sum_{j in C} x_j <= |C| - 1`.

**Why it matters:** Most practical MIP models contain knapsack-like substructures (resource limits, capacity constraints, budget constraints). Cover cuts are among the most effective cuts for these structures. Learning to exploit problem structure is what separates competent MIP modelers from beginners.

### Phase 3: Cutting Plane Algorithm (~25 min)

**File:** `src/cutting_plane.rs`

This phase assembles the Gomory cut generator into a complete iterative algorithm. You will see how pure cutting planes converge (slowly) to the integer optimum, understand why cut accumulation causes numerical issues, and appreciate why branch-and-cut is preferred.

**What you implement:**
- `cutting_plane_algorithm()` --- The full loop: solve LP, check integrality, generate Gomory cut, add cut, re-solve, repeat. Track LP bound progression per round.

**Why it matters:** Running the pure cutting plane algorithm on a concrete example makes the convergence behavior visceral --- you see the bound tighten slowly, the LP grow larger, and the numerical precision degrade. This motivates the combination with branch-and-bound and explains why solver logs show "cuts added" followed by "branching."

### Phase 4: Branch-and-Cut + Rounding Heuristic (~25 min)

**File:** `src/branch_cut.rs`

This phase integrates everything: cutting planes inside branch-and-bound, plus a rounding heuristic to find incumbents quickly. This is the algorithm that powers every modern MIP solver.

**What you implement:**
- `rounding_heuristic()` --- Round fractional variables to nearest integer, check constraint feasibility, return incumbent if feasible.
- `branch_and_cut()` --- B&B loop with cut generation at each node: solve LP, try rounding, generate cuts, re-solve, branch if still fractional. Compare node counts with and without cuts.

**Why it matters:** Branch-and-cut is THE algorithm for MIP. Understanding how cuts reduce the B&B tree (better bounds = more pruning) and how heuristics provide incumbents (enabling pruning by bound) gives you the mental model to tune solver parameters, write tighter formulations, and interpret solver logs.

## Motivation

Cutting planes are responsible for the dramatic speedup in MIP solvers over the past 25 years. Bixby (2012) estimates that algorithmic improvements (primarily cutting planes and preprocessing) contribute a factor of >10,000x on top of hardware speedups. Understanding cutting planes:

- **Explains solver performance**: Why does adding one constraint make the model solve 100x faster? Because it tightens the LP relaxation.
- **Guides formulation**: Knowing which cuts the solver generates helps you write formulations that already include the strongest inequalities.
- **Enables solver tuning**: Solver parameters like "cut aggressiveness," "Gomory passes," and "cover cuts" all refer to concepts in this practice.
- **Interview preparation**: Cutting planes and branch-and-cut are standard topics in OR/optimization interviews at trading firms, logistics companies, and consulting firms.

This practice bridges the gap between "I know branch-and-bound" (034a) and "I understand how real MIP solvers work."

## Commands

All commands are run from the `practice_034b_mip_cutting_planes/` folder root.

### Build

| Command | Description |
|---------|-------------|
| `cargo build` | Build all binaries (debug mode) |
| `cargo build --release` | Build all binaries (optimized release mode) |
| `cargo build --bin phase1_gomory` | Build only Phase 1: Gomory cuts |
| `cargo build --bin phase2_covers` | Build only Phase 2: Cover inequalities |
| `cargo build --bin phase3_cutting_plane` | Build only Phase 3: Cutting plane algorithm |
| `cargo build --bin phase4_branch_cut` | Build only Phase 4: Branch-and-cut |

### Run

| Command | Description |
|---------|-------------|
| `cargo run --bin phase1_gomory` | Run Phase 1: Gomory cut derivation and LP tightening |
| `cargo run --bin phase2_covers` | Run Phase 2: Cover inequality generation for knapsack |
| `cargo run --bin phase3_cutting_plane` | Run Phase 3: Pure cutting plane algorithm iteration |
| `cargo run --bin phase4_branch_cut` | Run Phase 4: Branch-and-cut vs pure B&B comparison |

### Check

| Command | Description |
|---------|-------------|
| `cargo check` | Type-check all binaries without compiling |
| `cargo clippy` | Run linter for idiomatic Rust suggestions |

## References

- [Gomory, R.E. (1958). Outline of an algorithm for integer solutions to linear programs](https://doi.org/10.1090/S0002-9904-1958-10224-4) --- Original paper on Gomory fractional cuts
- [Wolsey, L.A. (1998). Integer Programming](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119606475) --- Standard textbook covering cutting planes, covers, and branch-and-cut
- [Nemhauser & Wolsey (1988). Integer and Combinatorial Optimization](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118627372) --- Comprehensive reference for polyhedral theory and valid inequalities
- [Bixby, R.E. (2012). A brief history of linear and mixed-integer programming computation](https://doi.org/10.1007/s10751-012-0624-2) --- Quantifies the 10,000x algorithmic speedup
- [Fischetti, Glover & Lodi (2005). The feasibility pump](https://doi.org/10.1007/s10107-004-0570-3) --- Seminal paper on rounding heuristics
- [A brief tutorial on Gomory fractional cuts](https://farkasdilemma.wordpress.com/2016/09/16/a-brief-tutorial-on-gomory-fractional-cuts/) --- Accessible walkthrough with examples
- [good_lp documentation](https://docs.rs/good_lp/latest/good_lp/) --- Rust LP/MIP modeling crate API

## State

`not-started`
