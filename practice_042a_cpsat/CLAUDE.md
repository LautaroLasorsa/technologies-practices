# Practice 042a: Constraint Programming — OR-Tools CP-SAT

## Technologies

- **OR-Tools CP-SAT** — Google's constraint programming solver combining SAT (satisfiability), constraint propagation, and MIP techniques. State-of-the-art open-source solver for scheduling, timetabling, and combinatorial optimization. Winner of the MiniZinc CP Challenge since 2018.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- ortools >= 9.10 (Google OR-Tools with CP-SAT solver)
- uv (package manager)

## Theoretical Context

### What CP-SAT Is and the Problem It Solves

**CP-SAT** is Google's flagship constraint programming solver within the OR-Tools optimization suite. It solves **combinatorial optimization problems** — problems where the goal is to find the best assignment of discrete variables subject to constraints. Examples include scheduling (which task goes where and when), assignment (who does what), timetabling, and puzzle solving.

CP-SAT stands for **Constraint Programming — Satisfiability**. It combines three paradigms:

1. **SAT solving** — CDCL (Conflict-Driven Clause Learning) solver that reasons over Boolean variables and clauses.
2. **Constraint Propagation** — domain reduction through global constraints (AllDifferent, NoOverlap, Cumulative) that prune infeasible values early.
3. **MIP techniques** — linear relaxations, cutting planes, and LP-based bounding used opportunistically alongside SAT search.

The key innovation is **Lazy Clause Generation (LCG)**: instead of pre-encoding all constraints as SAT clauses (intractable), CP-SAT generates explanatory clauses on-demand during search. When a constraint propagator detects a domain reduction or conflict, it produces a SAT clause explaining *why*, and the SAT solver uses this clause for backtracking and learning. This gives the solver both the propagation power of CP and the conflict-learning power of SAT.

**Reference:** [Stuckey, P. — Lazy Clause Generation: Combining SAT and CP](https://people.eng.unimelb.edu.au/pstuckey/papers/lazy.pdf)

### Integer-Only Variables

CP-SAT works exclusively with **integer variables**. There are no continuous (floating-point) variables. If your problem involves continuous quantities:

- **Scale and round**: multiply by 1000 (or 10^k) and work in fixed-point integer arithmetic.
- **Example**: a cost of $3.14 becomes 3140 in CP-SAT (with implicit scale factor 1000).

This differs from MIP solvers (HiGHS, Gurobi, CPLEX) which natively support continuous variables alongside integers. If your problem is predominantly continuous with linear constraints, an LP/MIP solver is more appropriate.

### Modeling API

The Python API revolves around three classes:

```python
from ortools.sat.python import cp_model

model = cp_model.CpModel()       # The model (variables + constraints + objective)
solver = cp_model.CpSolver()     # The solver engine
status = solver.solve(model)     # Returns OPTIMAL, FEASIBLE, INFEASIBLE, etc.
```

**Variable creation:**

| Method | Description |
|--------|-------------|
| `model.new_int_var(lb, ub, name)` | Integer variable with domain [lb, ub] |
| `model.new_bool_var(name)` | Boolean variable (0 or 1) |
| `model.new_interval_var(start, size, end, name)` | Interval variable: enforces start + size == end |
| `model.new_optional_interval_var(start, size, end, is_present, name)` | Optional interval: active only if is_present is true |
| `model.new_fixed_size_interval_var(start, size, name)` | Interval with constant size (cheapest to optimize) |

**Adding constraints:**

```python
model.add(x + y <= 10)                    # Linear constraint
model.add(x != y)                         # Disequality
model.add_all_different([x, y, z])         # Global: all different values
model.add_no_overlap([interval1, interval2])  # No temporal overlap
model.add_cumulative(intervals, demands, capacity)  # Resource capacity
model.add_implication(a, b)                # Boolean: a => b
model.add_exactly_one([a, b, c])           # Exactly one Boolean is true
model.add_at_most_one([a, b, c])           # At most one Boolean is true
```

**Objective:**

```python
model.minimize(makespan)    # Minimize
model.maximize(profit)      # Maximize
```

### Key Constraints

| Constraint | Use Case | Example |
|------------|----------|---------|
| **AllDifferent** | Every variable takes a unique value | Sudoku rows/columns, graph coloring, N-Queens |
| **NoOverlap** | Intervals cannot overlap in time | Single-machine scheduling, room booking |
| **NoOverlap2D** | Rectangles cannot overlap on a plane | 2D packing, chip floorplanning |
| **Cumulative** | Sum of concurrent demands <= capacity | Multi-machine scheduling, resource allocation |
| **Element** | Array indexing: `y == array[x]` | Lookup tables, routing |
| **Circuit** | Variables form a Hamiltonian circuit | TSP, vehicle routing |
| **AddImplication** | Boolean implication: `a => b` | Temporal rules (no night then morning shift) |
| **AddExactlyOne / AddAtMostOne** | Cardinality on Booleans | Assignment constraints (exactly one nurse per shift) |
| **AddMultiplicationEquality** | `target == var1 * var2` | Nonlinear constraints (linearized internally) |
| **AddMinEquality / AddMaxEquality** | `target == min/max(vars)` | Makespan = max of completion times |

### Interval Variables and Scheduling

Interval variables are the foundation of CP-SAT scheduling:

```python
start = model.new_int_var(0, horizon, "start")
duration = 5  # constant
end = model.new_int_var(0, horizon, "end")
interval = model.new_interval_var(start, duration, end, "task")
```

Internally, `interval` enforces `start + duration == end`. Interval variables are then used in:

- **NoOverlap(intervals)** — no two intervals overlap (disjunctive/single-machine constraint).
- **Cumulative(intervals, demands, capacity)** — at any time t, the sum of demands of active intervals <= capacity.

For **optional tasks** (tasks that may or may not be scheduled), use `new_optional_interval_var` with a Boolean presence literal.

### Search Strategies and Solution Enumeration

```python
solver = cp_model.CpSolver()

# Time limit
solver.parameters.max_time_in_seconds = 30.0

# Parallel search (use all cores)
solver.parameters.num_workers = 8

# Enumerate all solutions (satisfaction problems)
solver.parameters.enumerate_all_solutions = True
status = solver.solve(model, callback)  # callback called for each solution
```

**Solution callbacks** inherit from `CpSolverSolutionCallback`:

```python
class SolutionCollector(cp_model.CpSolverSolutionCallback):
    def __init__(self, variables):
        super().__init__()
        self.variables = variables
        self.solutions = []

    def on_solution_callback(self):
        solution = [self.value(v) for v in self.variables]
        self.solutions.append(solution)
```

### CP-SAT vs MIP: When to Use Which

| Criterion | CP-SAT | MIP (HiGHS/Gurobi) |
|-----------|--------|---------------------|
| **Variable types** | Integer/Boolean only | Continuous + Integer |
| **Strength** | Disjunctive/logical constraints | LP relaxation tight → fast bounds |
| **Scheduling** | Excellent (NoOverlap, Cumulative) | Requires big-M linearization (weaker) |
| **Logical rules** | Native (implications, reification) | Indicator constraints or big-M |
| **Continuous problems** | Must scale to integers | Native support |
| **Parallelism** | Built-in multi-worker | Solver-dependent |
| **License** | Apache 2.0 (free) | HiGHS: MIT; Gurobi: commercial |
| **Best for** | Scheduling, timetabling, puzzles, assignment with complex rules | Blending, network flow, problems with tight LP relaxations |

**Rule of thumb:** If your constraints are naturally "disjunctive" (either A or B happens, but not both overlapping) or involve logical implications, CP-SAT is likely superior. If your problem is mostly continuous with some integer variables and linear constraints, MIP is better.

**Reference:** [CP-SAT Primer — How Does It Work?](https://d-krupke.github.io/cpsat-primer/07_under_the_hood.html), [Google OR-Tools CP-SAT Documentation](https://developers.google.com/optimization/cp/cp_solver)

### Key Concepts Summary

| Concept | Definition |
|---------|------------|
| **CpModel** | Container for variables, constraints, and objective |
| **CpSolver** | Engine that searches for solutions |
| **IntVar** | Integer decision variable with finite domain [lb, ub] |
| **BoolVar** | Binary decision variable (0 or 1), subtype of IntVar |
| **IntervalVar** | Scheduling variable: (start, size, end) with start + size == end |
| **Global constraint** | Constraint over a set of variables with specialized propagation (AllDifferent, NoOverlap) |
| **Lazy Clause Generation** | On-demand SAT clause creation from constraint propagation conflicts |
| **Propagation** | Automatic domain reduction when variables are assigned |
| **CDCL** | Conflict-Driven Clause Learning — the SAT backbone of CP-SAT |
| **Makespan** | Completion time of the last task — common scheduling objective |
| **Horizon** | Upper bound on time: sum of all durations (safe but loose bound) |
| **Solution callback** | User-defined function called each time the solver finds a solution |

## Description

Model and solve four classic constraint satisfaction and optimization problems using OR-Tools CP-SAT:

1. **N-Queens** — Place N queens on an N×N board with no attacks. Use AllDifferent for columns and diagonals. Enumerate all solutions.
2. **Job-Shop Scheduling** — Schedule jobs with task precedences on machines, minimizing makespan. Use IntervalVar + NoOverlap.
3. **Nurse Scheduling** — Assign nurses to shifts over multiple days with coverage, fairness, and temporal constraints. Use BoolVar + implications.
4. **Sudoku Solver** — Model the classic puzzle as a CSP with AllDifferent constraints on rows, columns, and boxes.

## Instructions

### Phase 1: N-Queens with AllDifferent

**What it teaches:** Core CP-SAT workflow (model → variables → constraints → solve), the AllDifferent global constraint, and solution enumeration via callbacks.

**Why it matters:** N-Queens is the canonical CSP. The diagonal trick (queens[i]+i and queens[i]-i must all differ) demonstrates how clever variable design eliminates constraints. AllDifferent propagates far more effectively than pairwise != — this is the power of global constraints.

**Exercises:**
1. Implement `solve_n_queens(n, find_all=False)` — create variables, add AllDifferent constraints for columns, diagonals, and anti-diagonals.
2. Use `SolutionCollector` callback with `enumerate_all_solutions` to find all 92 solutions for n=8.
3. Verify solution counts: n=4→2, n=8→92, n=12→14200.

### Phase 2: Job-Shop Scheduling

**What it teaches:** IntervalVar, NoOverlap (the disjunctive constraint), precedence constraints, and makespan minimization.

**Why it matters:** Job-shop scheduling is the flagship problem for CP-SAT — it's where CP-SAT dramatically outperforms MIP. The NoOverlap constraint (a global constraint that reasons about time intervals on a single machine) is extremely hard to linearize for MIP solvers (requires big-M formulations with weak relaxations). CP-SAT handles it natively with specialized propagation.

**Exercises:**
1. Implement `solve_job_shop(jobs, n_machines)` — create interval variables for each task, add precedence constraints within each job, add NoOverlap on each machine, minimize makespan.
2. Solve the classic 3-job, 3-machine instance (optimal makespan = 11).
3. Examine solver statistics: `solver.wall_time`, `solver.num_branches`, `solver.num_conflicts`.

### Phase 3: Nurse Scheduling

**What it teaches:** BoolVar matrices, AddExactlyOne/AddAtMostOne cardinality constraints, AddImplication for temporal rules, and fairness objectives.

**Why it matters:** Nurse scheduling is a real-world problem with soft and hard constraints that mix logical rules (no night-then-morning) with coverage requirements (every shift staffed) and fairness (even workload). This demonstrates CP-SAT's strength with Boolean reasoning and implications — constraints that are natural in CP but awkward (big-M) in MIP.

**Exercises:**
1. Implement `solve_nurse_scheduling(n_nurses, n_days, shifts_per_day)` — create BoolVar matrix, add coverage (exactly one nurse per shift-day), add at-most-one-shift-per-day, add no-night-then-morning implication.
2. Add fairness: each nurse works between min_shifts and max_shifts.
3. Display the schedule as a formatted table.

### Phase 4: Sudoku Solver

**What it teaches:** Pure constraint satisfaction (no objective), AllDifferent on rows/columns/boxes, and the power of propagation on a well-known problem.

**Why it matters:** Sudoku is a pure CSP — no optimization, just feasibility. It demonstrates that CP-SAT with AllDifferent + propagation solves even "hard" Sudoku puzzles instantly (< 10ms), while naive backtracking would take much longer. This is the same AllDifferent you implemented from scratch in practice 037, but now with industrial-strength propagation.

**Exercises:**
1. Implement `solve_sudoku(grid)` — create 9×9 IntVar matrix, fix given clues, add AllDifferent on each row, column, and 3×3 box.
2. Solve the provided easy, medium, and hard puzzles.
3. Compare solve times across difficulty levels.

## Motivation

- **State-of-the-art:** CP-SAT is the best open-source solver for scheduling and combinatorial problems, winning the MiniZinc Challenge since 2018.
- **Industry adoption:** Google uses CP-SAT internally for scheduling, resource allocation, and planning. It powers Google Cloud's optimization tools.
- **Complementary to MIP:** Understanding when to use CP-SAT vs MIP (practices 040a/b) is essential for an optimization engineer. CP-SAT excels where MIP struggles (disjunctive scheduling, logical constraints).
- **Free and scalable:** Apache 2.0 license, built-in parallel search. Unlike Gurobi/CPLEX, no license restrictions.
- **Market demand:** Scheduling and constraint programming skills are in high demand in supply chain, manufacturing, healthcare, and logistics — domains where AutoScheduler.AI operates.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies |
| Phase 1 | `uv run python src/phase1_queens.py` | Run N-Queens solver |
| Phase 2 | `uv run python src/phase2_jobshop.py` | Run Job-Shop scheduling |
| Phase 3 | `uv run python src/phase3_nurse.py` | Run Nurse scheduling |
| Phase 4 | `uv run python src/phase4_sudoku.py` | Run Sudoku solver |

## State

`not-started`
