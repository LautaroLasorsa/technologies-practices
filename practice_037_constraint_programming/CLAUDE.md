# Practice 037: Constraint Programming — Propagation & Search

## Technologies

- **Rust** — Systems language with ownership model, zero-cost abstractions
- No external crates — pure algorithmic implementation using only `std`

## Stack

- Rust (cargo)

## Theoretical Context

### What Constraint Programming Is

**Constraint Programming (CP)** is a declarative paradigm for solving combinatorial problems. Instead of describing *how* to find a solution (imperative), you declare *what* the solution must satisfy:

1. **Variables** — the unknowns (e.g., which color for each node, which value in each Sudoku cell).
2. **Domains** — the set of possible values each variable can take.
3. **Constraints** — relationships between variables that must hold simultaneously.

The solver's job: find an assignment of values to all variables such that every constraint is satisfied. This is a **Constraint Satisfaction Problem (CSP)**. When there is also an objective function to optimize, it is a **Constraint Optimization Problem (COP)**.

CP originated in AI (constraint logic programming in Prolog, 1980s) and operations research independently. Today it is one of the two major paradigms for combinatorial optimization, alongside Mixed-Integer Programming (MIP).

### Variables and Domains

Each variable X_i has a finite domain D_i — the set of values it can take. For example:

- Sudoku cell (3,5): D = {1, 2, 3, 4, 5, 6, 7, 8, 9}
- Graph coloring vertex 0: D = {Red, Green, Blue} = {0, 1, 2}
- N-Queens row 3: D = {0, 1, 2, ..., n-1} (column positions)

The goal of CP solving is to **reduce domains** until either:
- Every domain has exactly **one** value → solution found.
- Some domain becomes **empty** → no solution exists (inconsistency detected).
- Domains are reduced but still have multiple values → need search (branching).

Domain reduction is the fundamental operation. Every technique in CP — propagation, arc consistency, search — is about making domains smaller efficiently.

### Constraints

A constraint restricts which combinations of values are allowed for a subset of variables. Constraints come in several forms:

**Unary constraints** — involve one variable: X > 3, X != 5, X is even.
These are trivially enforced by filtering the domain.

**Binary constraints** — involve two variables: X != Y, X + Y = 10, X < Y.
These are the core building block for arc consistency algorithms.

**Global constraints** — involve many variables with specialized propagation:
- **AllDifferent(X_1, ..., X_n)**: all variables must take distinct values. Decomposable into O(n^2) binary != constraints, but dedicated propagators are far more powerful (detect failures earlier).
- **Cumulative**: resource scheduling — activities with start times, durations, and resource demands must not exceed capacity at any time point.
- **Element**: array indexing — X[I] = V, where I is a variable.

For this practice, we decompose global constraints into binary constraints. Production solvers (OR-Tools CP-SAT, Gecode, Choco) use specialized propagators for global constraints.

### Constraint Propagation

**Propagation** is the process of inferring domain reductions from constraints without search. Example:

```
X + Y = 10,  X in {1..9},  Y in {1..9}
Now add: X >= 5
Propagate: X in {5..9}, so Y = 10 - X, Y in {1..5}
Now add: Y != 3
Propagate: Y in {1, 2, 4, 5}, so X = 10 - Y, X in {5, 6, 8, 9}
```

Each domain reduction can trigger further reductions on other variables through their shared constraints. This cascade is propagation. The key insight: **propagation is cheap** (polynomial) compared to search (exponential), so we want to propagate as much as possible before branching.

### Arc Consistency (AC-3)

**Arc consistency** is the most common form of local consistency for binary CSPs. A constraint arc (X_i, X_j) is arc-consistent if:

> For every value v in D_i, there exists at least one value w in D_j such that the constraint between X_i and X_j is satisfied by (v, w).

The value w is called a **support** for v. If v has no support, it can never participate in a solution involving this constraint, so it is safe to remove v from D_i.

**The AC-3 Algorithm** (Mackworth, 1977):

```
1. Initialize a queue with ALL constraint arcs (both directions).
   For constraint (i, j): enqueue both (i, j) and (j, i).

2. While queue is not empty:
     Dequeue (i, j)
     If REVISE(i, j) removes any value from D_i:
       If D_i is empty: return FAILURE (inconsistency)
       For each constraint (k, i) where k != j:
         Enqueue (k, i)   // k's domain might need re-checking

3. Return SUCCESS (all arcs consistent)
```

**REVISE(i, j)**: For each value v in D_i, check if any value w in D_j satisfies the constraint. If not, remove v from D_i. Return whether D_i changed.

**Complexity**: O(e * d^3) where e = number of constraint arcs, d = maximum domain size. Each arc can be enqueued O(d) times (each time a value is removed from a domain), and each REVISE call costs O(d^2) (check each value in D_i against each in D_j).

**What AC-3 achieves**: After AC-3, every remaining value in every domain has at least one support in every constraining neighbor. This doesn't guarantee a solution exists (arc consistency is a local property), but it often dramatically reduces domain sizes and can detect many inconsistencies early.

**AC-3 vs stronger consistencies**: AC-3 enforces arc consistency (pairwise). Stronger forms exist — path consistency (triples), k-consistency (k-tuples) — but are more expensive. In practice, AC-3 + search is the standard approach for binary CSPs.

### Backtracking Search

When propagation alone doesn't solve the CSP (some domains still have multiple values), we need **search**. Backtracking search is the standard:

```
function BACKTRACK(csp, assignment):
    if assignment is complete: return SUCCESS

    var = SELECT-UNASSIGNED-VARIABLE(csp)  // variable ordering
    for value in ORDER-DOMAIN-VALUES(var, csp):  // value ordering
        save domains
        assignment[var] = value
        reduce domain of var to {value}
        if PROPAGATE(csp):  // AC-3
            if BACKTRACK(csp, assignment): return SUCCESS
        restore domains  // undo propagation
        assignment[var] = None

    return FAILURE  // backtrack
```

The key optimization: **run AC-3 after each assignment** (called Maintaining Arc Consistency, MAC). Without propagation, backtracking only detects failures when it tries to assign a variable and no value works. With propagation, failures are detected much earlier — when propagation empties a domain — pruning huge subtrees.

### Variable Ordering: MRV (Fail-First)

**Minimum Remaining Values (MRV)**, also called the **fail-first** heuristic: always branch on the variable with the **smallest domain**.

Intuition: if a variable has only 2 values left, any mistake in choosing its value is detected quickly (only 2 options to try). If we instead branched on a variable with 100 values, we might explore 99 failing subtrees before finding the right value (or backtracking).

MRV is the single most effective generic variable ordering heuristic. It is used in virtually every CP solver.

### Value Ordering: Least Constraining Value

**Least Constraining Value (LCV)**: try the value that rules out the fewest choices for neighboring variables first.

Intuition: if we are looking for *any* solution (not all solutions), we want to pick the value most likely to lead to success — the one that leaves the most flexibility for other variables.

In practice, LCV is less impactful than MRV and more expensive to compute. For this practice, we use simple domain-order iteration.

### CP vs MIP

| Aspect | Constraint Programming | Mixed-Integer Programming |
|--------|----------------------|--------------------------|
| **Model** | Variables, domains, constraints (arbitrary) | Linear objective, linear constraints, integrality |
| **Relaxation** | Domain reduction (propagation) | LP relaxation (continuous) |
| **Search** | Backtracking + propagation | Branch & Bound + LP |
| **Bounds** | Weak (no LP relaxation) | Strong (LP provides tight dual bounds) |
| **Strengths** | Scheduling, timetabling, sequencing, AllDifferent, complex logical constraints | LP-relaxable problems, allocation, network flow, when LP relaxation is tight |
| **Weaknesses** | No good dual bounds, hard to prove optimality gap | Awkward for logical/combinatorial constraints (big-M formulations) |
| **Solvers** | OR-Tools CP-SAT, Gecode, Choco, MiniZinc | Gurobi, CPLEX, HiGHS, SCIP |

**Modern hybrid**: OR-Tools CP-SAT combines CP propagation with SAT (clause learning) and MIP techniques (LP relaxation, cutting planes). It is currently one of the fastest open-source solvers for scheduling and combinatorial problems.

### Global Constraints in Detail

**AllDifferent(X_1, ..., X_n)**: All variables must take distinct values.

- **Naive decomposition**: O(n^2) binary constraints X_i != X_j. This is what we implement in this practice.
- **Dedicated propagator**: Uses bipartite matching (Hall's theorem) to detect failures earlier. If k variables share a domain of size < k, there is no solution. The naive decomposition cannot detect this without search.
- **Applications**: Sudoku (rows, columns, boxes), graph coloring, scheduling (no two tasks at the same time slot).

**Why AllDifferent is important**: It is the most common global constraint in practice. Every Sudoku row/column/box is an AllDifferent constraint. Graph coloring is AllDifferent on adjacent vertices. Latin squares, tournament scheduling, register allocation — all use AllDifferent.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **CSP** | Constraint Satisfaction Problem: variables, domains, constraints — find assignment satisfying all |
| **Domain** | Set of possible values for a variable |
| **Arc consistency** | Every value in D_i has a support in D_j for each constraint (i, j) |
| **Support** | A value w in D_j that satisfies the constraint with v in D_i |
| **AC-3** | Algorithm to enforce arc consistency by iteratively revising arcs |
| **REVISE** | Remove unsupported values from D_i w.r.t. constraint (i, j) |
| **Propagation** | Inferring domain reductions from constraints without search |
| **Backtracking** | Depth-first search: assign, propagate, recurse or undo |
| **MAC** | Maintaining Arc Consistency — run AC-3 after each assignment during search |
| **MRV** | Minimum Remaining Values — branch on variable with smallest domain |
| **LCV** | Least Constraining Value — try value that eliminates fewest options |
| **AllDifferent** | Global constraint: all variables take distinct values |
| **Global constraint** | Constraint over many variables with specialized propagation |
| **Node consistency** | Every value in D_i satisfies all unary constraints on X_i |
| **Fail-first** | Synonym for MRV — detect failures as early as possible |
| **Chromatic number** | Minimum colors needed to properly color a graph |

## Description

Implement a CSP solver from scratch in pure Rust: domain representation, constraint modeling, the AC-3 propagation algorithm, and backtracking search with MRV variable ordering. Apply the solver to three classic problems: Sudoku, N-Queens, and graph coloring.

### What you'll build

1. **Domain & constraint types** — `Variable`, `BinaryConstraint`, `CSP` data structures, and the REVISE function (the building block of arc consistency)
2. **AC-3 propagation** — The full arc consistency algorithm: initialize arc queue, revise, re-enqueue affected arcs, detect inconsistency
3. **Backtracking search** — MAC (Maintaining Arc Consistency) with MRV variable ordering: select variable, try values, propagate, recurse or backtrack
4. **Applications** — Model and solve Sudoku (81 variables, AllDifferent on rows/cols/boxes), N-Queens (n variables, diagonal + column constraints), and graph coloring (Petersen graph, Australia map)

## Instructions

### Phase 1: Domain Representation & Revise (~25 min)

**File:** `src/domains.rs`

This phase teaches the fundamental data structures of CP (variables, domains, constraints) and the REVISE operation — the atomic building block of arc consistency. REVISE checks whether each value in one variable's domain has at least one "support" (compatible value) in another variable's domain, and removes unsupported values.

**What you implement:**
- `revise()` — For each value v in D_i, check if any value w in D_j satisfies the constraint. Remove v if no support exists. Return whether D_i changed. This is the core operation called hundreds of times during AC-3.
- `node_consistency()` — Apply unary-style filtering: remove values from domains that cannot satisfy the variable's constraints even in isolation. Simpler than binary arc consistency but demonstrates the same principle: use constraints to shrink domains.

**Why it matters:** REVISE is called at every step of AC-3, which is called at every node of backtracking search. Understanding what "support" means and why removing unsupported values is sound (can never eliminate a solution) is the conceptual foundation of all CP propagation.

### Phase 2: AC-3 Algorithm (~25 min)

**File:** `src/ac3.rs`

This phase assembles REVISE into the full AC-3 arc consistency algorithm. AC-3 maintains a queue of arcs to check, processes them one by one, and re-enqueues affected arcs when a domain changes. It is the workhorse propagation algorithm of CP.

**What you implement:**
- `ac3()` — The complete AC-3 loop: initialize queue with all arcs, dequeue and revise, re-enqueue neighbors on domain change, detect empty domains. This is the propagation engine used inside backtracking search.

**Why it matters:** AC-3 is the standard propagation algorithm in CP. It runs in O(e*d^3) and is the key to making backtracking efficient. Without propagation, backtracking is essentially brute force. With AC-3, domain reductions cascade through the constraint network, pruning huge portions of the search space.

### Phase 3: Backtracking Search with MAC + MRV (~30 min)

**File:** `src/backtracking.rs`

This phase combines AC-3 with backtracking search to create a complete CSP solver. After each variable assignment, AC-3 propagates the consequences, detecting failures early. MRV picks the most constrained variable to branch on, minimizing backtracking.

**What you implement:**
- `select_unassigned_variable()` — MRV heuristic: find the unassigned variable with the smallest domain. This is the fail-first strategy that makes backtracking practical.
- `backtrack()` — The recursive MAC search: pick variable (MRV), try each value, save domains, assign, propagate (AC-3), recurse, restore on failure. This is the complete CSP solver.

**Why it matters:** MAC (Maintaining Arc Consistency) is the standard algorithm in CP solvers. The combination of propagation (AC-3) and intelligent variable ordering (MRV) transforms backtracking from impractical brute force into an efficient solver capable of handling Sudoku, N-Queens, scheduling, and many other combinatorial problems.

### Phase 4: Applications (~30 min)

**File:** `src/applications.rs`

This phase applies the solver to three classic CSP benchmarks. Sudoku and N-Queens CSPs are provided (they are modeling exercises, not algorithmic). You model graph coloring as a CSP and solve various graph instances.

**What you implement:**
- `build_graph_coloring_csp()` — Create variables (one per vertex, domain = colors), constraints (adjacent vertices must differ). This is a straightforward modeling exercise that demonstrates how real problems map to the CSP framework.

**Why it matters:** Modeling is half the skill in CP. Knowing that graph coloring = "AllDifferent on each edge" and that Sudoku = "AllDifferent on rows + cols + boxes" lets you recognize CP-amenable problems in practice. Graph coloring specifically appears in register allocation (compilers), scheduling (exam timetabling), and frequency assignment (telecommunications).

## Motivation

Constraint Programming is the other major paradigm alongside MIP for solving combinatorial optimization problems. Specific reasons to learn it:

- **OR-Tools CP-SAT** is currently one of the fastest open-source solvers for scheduling and combinatorial problems — used extensively at Google and in industry.
- **Scheduling and timetabling** are natural CP problems — AllDifferent, Cumulative, NoOverlap constraints express scheduling requirements directly, while MIP formulations require awkward big-M linearizations.
- **Complementary to MIP knowledge** — understanding both paradigms lets you choose the right tool for each problem. Some problems are easy for CP and hard for MIP (and vice versa).
- **Foundation for CP-SAT** — OR-Tools CP-SAT combines CP propagation, SAT clause learning, and MIP relaxations. Understanding pure CP propagation is prerequisite to understanding how CP-SAT works.
- **Career relevance** — CP skills are valued at scheduling companies (AutoScheduler.AI), logistics optimization, manufacturing planning, and anywhere complex combinatorial constraints arise.

## Commands

All commands are run from the `practice_037_constraint_programming/` folder root.

### Build

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all four binaries |
| `cargo build --release` | Compile with optimizations (faster solving for Phase 4 Sudoku) |
| `cargo check` | Fast type-check without producing binaries |

### Run — Phase 1: Domains & Revise

| Command | Description |
|---------|-------------|
| `cargo run --bin phase1_domains` | Run Phase 1: domain reduction with REVISE on simple constraints |

### Run — Phase 2: AC-3

| Command | Description |
|---------|-------------|
| `cargo run --bin phase2_ac3` | Run Phase 2: AC-3 arc consistency on constraint examples |

### Run — Phase 3: Backtracking Search

| Command | Description |
|---------|-------------|
| `cargo run --bin phase3_backtracking` | Run Phase 3: MAC solver on 4-Queens and small graph coloring |

### Run — Phase 4: Applications

| Command | Description |
|---------|-------------|
| `cargo run --bin phase4_applications` | Run Phase 4: solve Sudoku, 8-Queens, and graph coloring |
| `cargo run --release --bin phase4_applications` | Run Phase 4 with optimizations for faster Sudoku solving |

## References

- [Mackworth, A.K. (1977). Consistency in Networks of Relations](https://doi.org/10.1016/0004-3702(77)90007-8) — Original AC-3 paper
- [Russell, S. & Norvig, P. (2020). Artificial Intelligence: A Modern Approach, Ch. 6](https://aima.cs.berkeley.edu/) — CSP chapter covering AC-3, backtracking, MRV
- [Rossi, F., van Beek, P., Walsh, T. (2006). Handbook of Constraint Programming](https://www.elsevier.com/books/handbook-of-constraint-programming/rossi/978-0-444-52726-4) — Comprehensive reference
- [OR-Tools CP-SAT Documentation](https://developers.google.com/optimization/cp/cp_solver) — Production CP solver combining CP, SAT, and MIP techniques
- [Lecoutre, C. (2009). Constraint Networks](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470611821) — Deep dive into arc consistency algorithms and their complexity

## State

`not-started`
