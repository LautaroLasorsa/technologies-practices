# Practice 034a: MIP — Branch & Bound

## Technologies

- **Rust** — Systems language with ownership model, zero-cost abstractions
- **good_lp 1.10** — Linear programming modeler with HiGHS backend for LP relaxation solving
- **ordered-float 4** — `OrderedFloat<f64>` wrapper enabling `Ord` on floats (needed for `BinaryHeap`)

## Stack

- Rust (cargo)

## Theoretical Context

### What Mixed-Integer Programming Is

**Mixed-Integer Programming (MIP)** is the problem of optimizing a linear objective function subject to linear constraints where **some or all variables must take integer values**. It generalizes Linear Programming by adding integrality constraints.

The standard MIP formulation:

```
maximize    c^T x
subject to  A x <= b
            x_i in Z   for i in I  (integer variables)
            x_j >= 0   for j not in I  (continuous variables)
```

Where `I` is the subset of variables required to be integer. When all variables are integer, it is a **pure integer program (IP)**. When all integer variables are restricted to {0, 1}, it is a **binary integer program (BIP)** or **0-1 program**.

MIP appears everywhere: production scheduling (make/don't-make binary decisions), facility location (open/close), vehicle routing (which routes to use), portfolio optimization with cardinality constraints, network design, crew scheduling, and virtually every real-world optimization problem that involves discrete choices combined with continuous allocation.

### Why MIP Is Hard

LP is solvable in polynomial time (interior point methods) and very efficiently in practice (simplex). Adding integrality constraints makes the problem **NP-hard**. The fundamental reason: the feasible region of an LP is a convex polytope (connected, smooth), but adding integrality constraints makes it a **discrete set of points** inside that polytope — a fundamentally harder geometric object to optimize over.

A naive approach — enumerate all possible integer assignments — has complexity O(k^n) where k is the range of each integer variable and n is the number of integer variables. For a problem with 50 binary variables, that is 2^50 ≈ 10^15 possibilities. Clearly infeasible.

### LP Relaxation

The **LP relaxation** of a MIP is obtained by dropping all integrality constraints — treating every variable as continuous. This is the key connection between LP and MIP:

- The LP relaxation is a **relaxation**: its feasible region contains every feasible MIP solution (and more).
- For **maximization**: LP_opt >= MIP_opt (LP is less constrained, so it can achieve at least as good an objective).
- For **minimization**: LP_opt <= MIP_opt.
- The LP relaxation optimal value provides a **bound** on the MIP optimal value.

If the LP relaxation solution happens to satisfy all integrality constraints, it is also optimal for the MIP. This rarely happens, but when it does, we are done immediately.

### Integrality Gap

The **integrality gap** is the difference (or ratio) between the LP relaxation optimum and the MIP optimum:

```
gap = |LP_opt - MIP_opt|
```

A small integrality gap means the LP relaxation is a tight approximation of the MIP — the LP bound is close to the true integer optimum. This is desirable because:

1. Branch & Bound can prune more aggressively with tight bounds.
2. Fewer nodes need exploration.
3. Solver time is typically proportional to the integrality gap.

**Formulation matters**: two mathematically equivalent MIP formulations can have vastly different integrality gaps. Tighter formulations (with smaller gaps) solve orders of magnitude faster. This is why OR practitioners spend significant effort on formulation quality.

### Branch and Bound Algorithm

Branch and Bound (B&B) is the fundamental algorithm for solving MIPs. It was introduced by Land & Doig (1960) and is the core engine inside every modern MIP solver (Gurobi, CPLEX, HiGHS, SCIP). The algorithm systematically explores the space of integer solutions by:

1. **Relaxing** — solving LP relaxations to obtain bounds.
2. **Branching** — splitting the problem into subproblems to eliminate fractional solutions.
3. **Bounding** — using LP bounds to prune subproblems that cannot improve the best known solution.

**The algorithm in detail:**

```
1. INITIALIZE
   - Create root node (the original MIP, no extra bounds)
   - Set incumbent = None (no integer solution found yet)
   - Set best_obj = -infinity (for maximization)
   - Add root node to the open node list

2. LOOP while open nodes exist:
   a. SELECT a node from the open list (node selection strategy)

   b. SOLVE the LP relaxation of this node
      If LP is infeasible:
        → PRUNE (fathom by infeasibility) — this subtree has no feasible solutions

   c. BOUND CHECK
      If LP_obj <= best_obj (for maximization):
        → PRUNE (fathom by bound) — even the relaxed optimum can't beat incumbent

   d. INTEGER FEASIBILITY CHECK
      If the LP solution satisfies all integrality constraints:
        → Update incumbent: best_obj = LP_obj, best_solution = LP_solution
        → PRUNE (fathom by optimality) — no need to branch further
        → Optionally: re-examine open nodes and prune any with bound <= new best_obj

   e. BRANCH
      Select a fractional variable x_i with value f (branching variable selection)
      Create LEFT child:  original node + constraint x_i <= floor(f)
      Create RIGHT child: original node + constraint x_i >= ceil(f)
      Add both children to the open node list

3. TERMINATE
   If incumbent exists → it is optimal (all other possibilities were pruned)
   If no incumbent → problem is infeasible (no integer-feasible solution exists)
```

**Why it works**: Every integer-feasible solution lies in exactly one leaf of the B&B tree. By solving LP relaxations at each node, we get bounds that tell us "the best possible integer solution in this subtree is at most X." If X is worse than an integer solution we already found, we skip the entire subtree. In practice, the vast majority of the tree is pruned — a problem with 2^50 potential solutions might only require exploring a few thousand nodes.

### Node Selection Strategies

The order in which we explore open nodes significantly impacts performance:

| Strategy | Data Structure | Pros | Cons |
|----------|---------------|------|------|
| **DFS (depth-first)** | Stack (LIFO) | Finds feasible solutions fast, low memory (O(depth)) | May explore suboptimal subtrees extensively |
| **BFS (breadth-first)** | Queue (FIFO) | Explores tree level by level | High memory, slow to find feasible solutions |
| **Best-first (best-bound)** | Priority queue (by LP bound) | Proves optimality fastest, best gap reduction | High memory (entire frontier), slow to find incumbents |
| **Hybrid (diving + best-bound)** | Combination | Production solvers use this — DFS to find incumbents, then best-bound to close gap | Complex implementation |

**Best-first** is theoretically optimal for minimizing nodes explored, but **DFS** is often better in practice because finding a good incumbent early enables aggressive pruning throughout the rest of the search.

### Variable Selection (Branching) Strategies

Which fractional variable to branch on affects tree size dramatically:

| Strategy | Rule | Complexity | Quality |
|----------|------|------------|---------|
| **First fractional** | Branch on the first integer variable with fractional value | O(n) | Baseline — simple, fast per node |
| **Most fractional** | Branch on the variable closest to 0.5 (max fractionality) | O(n) | Often better — creates most "balanced" split |
| **Strong branching** | For each candidate, solve both child LPs, pick the one that improves bounds most | O(n * LP_solve) | Best tree size — expensive per node |
| **Pseudocost branching** | Track historical branching effectiveness per variable, use as estimate | O(n) | Amortized strong branching — good after warmup |
| **Reliability branching** | Strong branch until pseudocosts become reliable, then switch | Adaptive | Production solvers use this (Gurobi, CPLEX) |

### Incumbent and Pruning

**Incumbent**: The best integer-feasible solution found so far during the search. It provides the **primal bound** — a concrete achievable objective value.

**LP bound**: The LP relaxation value at an open node provides the **dual bound** — the best possible objective value in that subtree.

**Pruning** is the key to B&B efficiency. Three pruning conditions:

1. **Fathom by infeasibility**: The LP relaxation of the node is infeasible → no integer solution exists in this subtree.
2. **Fathom by bound**: The LP bound at this node is no better than the incumbent → even the best possible solution in this subtree can't beat what we already have.
3. **Fathom by optimality**: The LP solution is integer-feasible → it is optimal for this subtree, update incumbent if better.

**Optimality gap**: `gap = |dual_bound - primal_bound| / |primal_bound|`. When gap = 0, the incumbent is provably optimal. Production solvers report this gap continuously and can terminate early at a user-specified gap tolerance (e.g., 1% gap is "good enough" for many applications).

### Complexity

- **Worst case**: Exponential. B&B may need to explore all 2^n leaves for n binary variables.
- **Practice**: The combination of LP bounds + pruning + good branching heuristics makes B&B practical for problems with thousands of integer variables. Modern solvers add cutting planes, presolve, heuristics, and symmetry breaking to further reduce the tree.
- **Key insight**: B&B efficiency depends critically on (1) tightness of LP relaxation (small integrality gap), (2) quality of incumbent (found early), and (3) branching strategy (small tree).

### Key Concepts

| Concept | Definition |
|---------|------------|
| **MIP** | Optimization with linear objective/constraints and integrality requirements on some variables |
| **LP relaxation** | Drop integrality constraints → continuous LP that provides a bound |
| **Integrality gap** | Difference between LP relaxation optimum and true MIP optimum |
| **Branch** | Split a node into two children by adding floor/ceil bounds on a fractional variable |
| **Bound** | LP relaxation value at a node — the best possible objective in that subtree |
| **Prune (fathom)** | Discard a subtree because it cannot contain a better solution than the incumbent |
| **Incumbent** | Best integer-feasible solution found so far |
| **Node** | A subproblem in the B&B tree with additional variable bound constraints |
| **DFS** | Depth-first search: explore children before siblings (uses stack) |
| **Best-first** | Explore the node with the best LP bound next (uses priority queue) |
| **First fractional** | Branch on the first integer variable with a fractional LP value |
| **Most fractional** | Branch on the integer variable whose value is closest to 0.5 |
| **Primal bound** | Objective value of the incumbent (achievable) |
| **Dual bound** | Best LP bound among all open nodes (theoretical limit) |
| **Optimality gap** | `|dual_bound - primal_bound| / |primal_bound|` — measures proof of optimality |

### Where B&B Fits in the Optimization Hierarchy

```
LP Relaxation (polynomial, provides bounds)
 └── Branch & Bound (uses LP relaxation at each node)
      ├── + Cutting Planes → Branch & Cut (tighten LP relaxation)
      ├── + Pricing → Branch & Price (column generation)
      └── + Heuristics → Modern MIP Solvers (Gurobi, CPLEX, HiGHS, SCIP)
```

B&B is the backbone. Every enhancement (cuts, heuristics, presolve, symmetry breaking) plugs into the B&B framework. Understanding B&B is prerequisite to understanding any modern MIP solver.

## Description

Build a Branch & Bound MIP solver in Rust. Use `good_lp` with the HiGHS backend for solving LP relaxations (continuous problems). Implement the B&B tree search, branching logic, bounding, and pruning from scratch. Progress through four phases: LP relaxation observation, single branching step, full B&B with DFS, and advanced strategies (best-first search, most-fractional branching).

### What you'll build

1. **LP relaxation solver** — Formulate a MIP, drop integrality, solve the relaxed LP, observe fractional solutions
2. **Branching mechanics** — Manually branch on one variable, solve both children, compare bounds
3. **Full B&B solver** — Complete branch-and-bound loop with DFS node selection and pruning
4. **Strategy comparison** — Best-first node selection and most-fractional variable selection, benchmarked against DFS

## Instructions

### Phase 1: LP Relaxation (~20 min)

**File:** `src/relaxation.rs`

This phase teaches the foundational concept: LP relaxation as a bound on the MIP optimum. You formulate a binary knapsack problem, relax integrality (treat 0-1 variables as continuous [0,1]), solve the resulting LP, and observe that the LP solution is fractional — motivating the need for Branch & Bound.

**What you implement:**
- `solve_lp_relaxation()` — Use `good_lp` to create continuous variables, set objective coefficients, add constraints, solve, and extract the solution vector. This is the core operation called at every B&B node.

**Why it matters:** LP relaxation is called at every single node of the B&B tree. Understanding what it computes (a bound, not the answer) and why it returns fractional values is essential. The quality of this bound (how close to the integer optimum) determines how fast B&B will converge.

### Phase 2: Single Branch (~20 min)

**File:** `src/single_branch.rs`

This phase teaches the branching operation in isolation. Given a fractional LP solution, you pick one variable, create two child subproblems (floor/ceil), solve both, and compare. This is the atomic step of B&B — everything else is repetition of this step with pruning.

**What you implement:**
- `branch_on_variable()` — Clone the problem, tighten variable bounds to create left (floor) and right (ceil) children. For binary variables, this means fixing x_i = 0 or x_i = 1.

**Why it matters:** Branching is how B&B eliminates fractional solutions. Each branch tightens the feasible region, and the child LP bounds can only get worse (or equal) compared to the parent. Understanding this monotonicity is key to understanding why B&B terminates and why pruning is sound.

### Phase 3: Full Branch & Bound (~30 min)

**File:** `src/branch_bound.rs`

This phase assembles everything into a complete B&B solver. The LP solver and branching functions are provided. You implement the search loop: select a node, solve its LP, check bounds, check integrality, prune or branch, repeat.

**What you implement:**
- `select_branching_variable_first_fractional()` — Find the first integer variable with a fractional LP value.
- `branch_and_bound()` — The complete DFS-based B&B loop with all three pruning conditions.

**Why it matters:** This is THE algorithm behind every MIP solver. Implementing it yourself builds intuition for: why some MIPs solve in seconds and others take hours (tree size), why formulation tightness matters (pruning effectiveness), and what solver logs mean (nodes explored, gap, incumbent updates).

### Phase 4: Branching Strategies (~25 min)

**File:** `src/strategies.rs`

This phase explores how strategy choices affect solver performance. You implement most-fractional variable selection and best-first node selection, then compare them against the Phase 3 DFS solver on the same problems.

**What you implement:**
- `select_branching_variable_most_fractional()` — Pick the variable whose fractional part is closest to 0.5.
- `branch_and_bound_best_first()` — Use a `BinaryHeap` (max-heap by LP bound) instead of a stack for node selection.

**Why it matters:** Strategy selection is the difference between a solver that takes 100 nodes and one that takes 100,000 on the same problem. Production solvers (Gurobi, CPLEX) use sophisticated hybrid strategies. Understanding the trade-offs (DFS = fast incumbents, best-first = fast proofs) helps you tune solvers for your specific problems.

## Motivation

Branch & Bound is THE algorithm behind every MIP solver — Gurobi, CPLEX, HiGHS, SCIP, CBC all use B&B as their core search framework. Understanding B&B is essential for:

- **Formulating tight models**: Knowing that LP relaxation quality drives pruning effectiveness motivates writing tight formulations (fewer fractional variables, smaller integrality gap).
- **Interpreting solver logs**: "Nodes explored: 45,231, Gap: 2.3%, Incumbent: 847.5" — these numbers are meaningless without understanding B&B.
- **Writing solver callbacks**: Production solvers let you inject custom branching, cutting planes, and heuristics via callbacks — all within the B&B framework.
- **Debugging slow solves**: When a MIP takes too long, understanding B&B tells you whether the bottleneck is weak bounds (need cuts/tighter formulation), lack of good incumbents (need heuristics), or inherent problem difficulty.
- **Career relevance**: B&B knowledge is expected in optimization roles at trading firms, logistics companies (Amazon, FedEx), scheduling companies (AutoScheduler.AI), and any team using MIP solvers.

## Commands

All commands are run from the `practice_034a_mip_branch_bound/` folder root.

### Build

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all binaries (fetches good_lp + HiGHS on first run) |
| `cargo build --release` | Compile with optimizations (faster LP solves for Phase 4 benchmarking) |
| `cargo check` | Fast type-check without producing binaries |

### Run — Phase 1: LP Relaxation

| Command | Description |
|---------|-------------|
| `cargo run --bin phase1_relaxation` | Run Phase 1: solve LP relaxation of binary knapsack, observe fractional solution |

### Run — Phase 2: Single Branch

| Command | Description |
|---------|-------------|
| `cargo run --bin phase2_single_branch` | Run Phase 2: branch on one variable, solve both children, compare bounds |

### Run — Phase 3: Full Branch & Bound

| Command | Description |
|---------|-------------|
| `cargo run --bin phase3_branch_bound` | Run Phase 3: complete B&B with DFS on knapsack and set cover problems |

### Run — Phase 4: Branching Strategies

| Command | Description |
|---------|-------------|
| `cargo run --bin phase4_strategies` | Run Phase 4: compare DFS vs best-first, first-fractional vs most-fractional |
| `cargo run --release --bin phase4_strategies` | Run Phase 4 with optimizations for accurate timing comparison |

## References

- [Land, A.H. & Doig, A.G. (1960). An automatic method of solving discrete programming problems](https://doi.org/10.2307/1910129) — Original Branch & Bound paper
- [Wolsey, L.A. (1998). Integer Programming](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119606475) — Standard graduate textbook on MIP and B&B
- [Nemhauser, G.L. & Wolsey, L.A. (1988). Integer and Combinatorial Optimization](https://onlinelibrary.wiley.com/doi/book/10.1002/9781118627372) — Comprehensive reference on integer programming theory
- [good_lp Documentation](https://docs.rs/good_lp/latest/good_lp/) — Rust LP modeler API reference
- [HiGHS Solver](https://highs.dev/) — Open-source LP/MIP solver used as backend
- [Achterberg, T. (2007). Constraint Integer Programming](https://opus4.kobv.de/opus4-zib/files/1088/Achterberg_Constraint_Integer_Programming.pdf) — Deep dive into modern MIP solver techniques (SCIP)
- [Linderoth, J.T. & Savelsbergh, M.W.P. (1999). A computational study of branch and bound search strategies](https://pubsonline.informs.org/doi/10.1287/ijoc.11.2.173) — Empirical comparison of node selection strategies

## State

`not-started`
