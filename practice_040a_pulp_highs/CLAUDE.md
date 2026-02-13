# Practice 040a: LP/MIP Modeling — PuLP & HiGHS

## Technologies

- **PuLP** — Python LP/MIP modeling library (COIN-OR). Provides algebraic modeling via operator overloading: variables, linear expressions, constraints, and objectives map directly to Python syntax.
- **HiGHS** — High-performance open-source LP/MIP/QP solver (MIT license). Default solver in SciPy (>=1.6), MathWorks Optimization Toolbox, and PuLP (>=2.8). Written in C++11 with Python bindings via `highspy`.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- PuLP >= 2.9 (modeling layer)
- highspy >= 1.7 (HiGHS solver Python bindings)
- uv (package manager)

## Theoretical Context

### What PuLP Is and the Problem It Solves

**PuLP** is a Python library for formulating Linear Programs (LP) and Mixed-Integer Programs (MIP) using an algebraic modeling interface. It translates a human-readable mathematical formulation into the data structures that solvers expect (MPS/LP file formats or direct API calls), then invokes a solver and parses the results back into Python objects.

The problem PuLP solves: solvers like HiGHS, CBC, CPLEX, and Gurobi accept problems in matrix form (`min c^T x, s.t. Ax <= b`), but writing raw matrices is error-prone and unreadable. PuLP lets you write:

```python
prob += x + 2*y <= 10       # a constraint
prob += 3*x + 5*y           # the objective
```

and it handles the translation to matrix form internally. This is called **algebraic modeling** — the same paradigm used by AMPL, GAMS, Pyomo, and JuMP.

### How PuLP Works Internally

1. **Expression trees**: When you write `2*x + 3*y`, PuLP builds an `LpAffineExpression` — a dictionary mapping variables to coefficients: `{x: 2, y: 3}`. The `<=`, `>=`, `==` operators create `LpConstraint` objects that store the expression and the sense/bound.

2. **Problem assembly**: `LpProblem` accumulates variables, constraints, and the objective. When you call `prob.solve()`, PuLP serializes the problem to MPS format (or calls the solver's Python API directly for supported solvers like HiGHS).

3. **Solver dispatch**: PuLP supports multiple solver backends. Each solver class (`HiGHS_CMD`, `PULP_CBC_CMD`, `GUROBI`, etc.) implements `actualSolve()` which handles invocation, parameter passing, and result parsing. After solving, variable values, constraint duals, and status are written back to the Python objects.

4. **Result extraction**: `variable.varValue` gives the optimal value, `constraint.pi` gives the shadow price (dual value), `constraint.slack` gives the slack, and `prob.objective.value()` gives the optimal objective.

### What HiGHS Is

**HiGHS** (High-performance Optimization Software) is an open-source solver for LP, MIP, and QP problems, developed at the University of Edinburgh. Key facts:

- **MIT license** — fully free for commercial use, unlike Gurobi/CPLEX.
- **Performance**: On the [Mittelmann benchmarks](https://plato.asu.edu/bench.html), HiGHS has the best LP, MIP, simplex, and barrier ratings among all open-source solvers. It is roughly 10-20x slower than the best commercial solvers (COPT, Gurobi) on average, but within 1-2 orders of magnitude — practical for most problems.
- **Algorithms**: Revised simplex method, interior point method (barrier), PDLP first-order method for LP. Branch-and-cut for MIP. Active set for QP.
- **Adoption**: Default LP solver in SciPy (>=1.6), default MIP solver in SciPy (>=1.9), default solver in MathWorks Optimization Toolbox, default solver in PuLP (>=2.8, replacing CBC).
- **Python interface**: `highspy` provides direct Python bindings. PuLP wraps it via `HiGHS_CMD()` (command-line) or `HiGHS()` (Python API).

### The Modeling Pattern

Every LP/MIP follows the same 6-step pattern in PuLP:

```
1. Define variables    →  LpVariable("x", lowBound=0)
2. Create problem      →  LpProblem("name", LpMinimize)
3. Set objective       →  prob += cost_expression
4. Add constraints     →  prob += expression <= bound, "name"
5. Solve               →  prob.solve(HiGHS_CMD(msg=0))
6. Extract solution    →  var.varValue, prob.status, constraint.pi
```

This pattern is universal across all algebraic modeling tools (Pyomo, CVXPY, JuMP). Learning it in PuLP transfers directly.

### Variable Types

PuLP supports three variable categories via the `cat` parameter:

| Category | `cat=` | Domain | Use case |
|----------|--------|--------|----------|
| **Continuous** | `'Continuous'` | Real numbers `[lb, ub]` | Quantities, amounts, fractions |
| **Integer** | `'Integer'` | Integers `{lb, ..., ub}` | Counts, discrete quantities |
| **Binary** | `'Binary'` | `{0, 1}` | Yes/no decisions, selection |

For indexed families of variables (e.g., `x[i]` for each item `i`), use:
```python
x = pulp.LpVariable.dicts("x", items, lowBound=0, cat='Continuous')
```
This creates a dictionary `{item: LpVariable}`.

### Solver Interface

```python
# Using HiGHS (default, recommended)
prob.solve(pulp.HiGHS_CMD(msg=0))        # Command-line interface, msg=0 suppresses output
prob.solve(pulp.HiGHS(msg=0))            # Python API interface (requires highspy)

# Using CBC (legacy default)
prob.solve(pulp.PULP_CBC_CMD(msg=0))

# Check solution status
print(pulp.LpStatus[prob.status])         # 'Optimal', 'Infeasible', 'Unbounded', etc.
print(pulp.value(prob.objective))          # Optimal objective value
```

Status codes: `1 = Optimal`, `0 = Not Solved`, `-1 = Infeasible`, `-2 = Unbounded`, `-3 = Undefined`.

### Solution Analysis: Shadow Prices, Reduced Costs, Sensitivity

After solving an LP, the solver provides dual information:

- **Shadow price** (`constraint.pi`): How much the objective improves per unit relaxation of a constraint's RHS. Also called the **dual variable** or **marginal value**. For `resource <= 100`, a shadow price of 5 means adding 1 unit of resource improves the objective by 5.

- **Reduced cost** (`variable.dj`): How much a variable's objective coefficient must improve before it enters the optimal basis. For a variable at zero, its reduced cost tells you "how far from being worthwhile" it is.

- **Slack** (`constraint.slack`): How much "room" remains in a constraint. Zero slack means the constraint is **binding** (tight). Non-zero slack means the constraint is not active at the optimum.

These values come directly from LP duality theory (practiced in 032b). They are critical for model interpretation in practice: "Which resources are bottlenecks?" (binding constraints with large shadow prices), "Which products are worth considering?" (small reduced costs).

### Common Modeling Patterns

| Pattern | Description | Example |
|---------|-------------|---------|
| **Big-M constraint** | `x <= M * y` links continuous `x` to binary `y` | "Can only produce if factory is open" |
| **Linking constraint** | `x[f][c] <= y[f]` | "Can only assign customer to open facility" |
| **Covering** | `sum(x[j]) >= 1` for each set | "Every customer must be covered" |
| **Partitioning** | `sum(x[j]) == 1` | "Exactly one option selected" |
| **Knapsack** | `sum(w[i]*x[i]) <= C`, `x` binary | "Select items within capacity" |
| **Flow balance** | `sum_in - sum_out == demand` | Network flow conservation |
| **Piecewise linear** | Breakpoints + SOS2 variables | Approximate nonlinear costs |

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Algebraic modeling** | Formulating optimization problems using mathematical syntax in code (`x + 2*y <= 10`) |
| **LpProblem** | PuLP's problem container: holds objective, constraints, variables, and solver results |
| **LpVariable** | A decision variable with name, bounds, and category (continuous/integer/binary) |
| **LpAffineExpression** | A linear combination of variables: `{var: coefficient}` pairs plus a constant |
| **LpConstraint** | An expression with a sense (`<=`, `>=`, `==`) and RHS bound |
| **Shadow price (pi)** | Dual variable: marginal value of relaxing a constraint by one unit |
| **Reduced cost (dj)** | How much a variable's coefficient must improve to enter the basis |
| **Slack** | Unused capacity in a constraint at the optimal solution |
| **LP relaxation** | Solving a MIP with integrality constraints removed (all variables continuous) |
| **Integrality gap** | Difference between LP relaxation bound and best integer solution |
| **Branch-and-bound** | Algorithm for MIP: recursively fix integer variables, solve LP relaxations |
| **Binding constraint** | A constraint with zero slack — active at the optimum |
| **Big-M** | A large constant used to "activate/deactivate" constraints via binary variables |
| **Linking constraint** | Constraint tying continuous variables to binary on/off decisions |

### Where PuLP + HiGHS Fit in the Ecosystem

```
Algebraic Modeling Libraries (high-level formulation)
├── PuLP .............. Simplest, LP/MIP only, great for learning
├── Pyomo ............. Full-featured, supports NLP, stochastic, DAE
├── CVXPY ............. Disciplined convex programming (LP, QP, SOCP, SDP)
├── Google OR-Tools ... CP-SAT, routing, LP/MIP
└── JuMP (Julia) ...... Fast, extensible, academic standard

Solvers (low-level computation)
├── HiGHS ............. Best open-source LP/MIP (MIT) ← used here
├── CBC ............... Legacy open-source MIP (EPL)
├── GLPK .............. Open-source LP/MIP (GPL)
├── Gurobi ............ Commercial, fastest MIP
├── CPLEX ............. Commercial (IBM)
└── COPT .............. Commercial, fastest LP
```

PuLP is the **entry point**: simplest API, minimal boilerplate, good for learning modeling patterns. Once you outgrow it (need nonlinear objectives, stochastic programming, or advanced decomposition), move to Pyomo or CVXPY. The modeling patterns (variables, constraints, objective, solve, extract) transfer directly.

## Description

Model and solve classic operations research problems using PuLP with the HiGHS solver. Progress from pure LP problems (continuous variables, duality analysis) to MIP problems (binary/integer variables, branch-and-bound). Each phase introduces a canonical OR problem that demonstrates specific modeling patterns.

### What you'll build

1. **Diet problem (LP)** — Minimize cost subject to nutritional requirements. Classic LP with shadow price analysis.
2. **Production planning (LP)** — Maximize profit with resource constraints. Analyze shadow prices, reduced costs, and binding constraints.
3. **Binary knapsack (MIP)** — Select items to maximize value within weight capacity. Compare MIP solution with LP relaxation.
4. **Facility location (MIP)** — Open facilities and assign customers. Demonstrates linking constraints, Big-M pattern, and the interplay between binary and continuous variables.

## Instructions

### Phase 1: Diet Problem — LP (~20 min)

**File:** `src/phase1_diet.py`

This phase teaches the fundamental PuLP modeling pattern on the simplest possible LP. The diet problem (Stigler, 1945) is historically the first LP ever formulated: choose food quantities to minimize cost while meeting nutritional requirements. It has only continuous variables and inequality constraints, making it ideal for learning the `define → constrain → solve → extract` workflow.

**What you implement:**
- `solve_diet_problem()` — Create LP variables for food quantities, set the cost-minimization objective, add nutritional requirement constraints, solve with HiGHS, extract solution and shadow prices.

**Why it matters:** This is the "hello world" of LP modeling. Every modeling pattern you'll use later (production planning, facility location, scheduling) follows exactly the same structure. Shadow prices from the nutritional constraints tell you which nutrients are "expensive" — the marginal cost of requiring one more unit. This connects directly to LP duality from practice 032b.

### Phase 2: Production Planning — LP (~25 min)

**File:** `src/phase2_production.py`

This phase introduces the other direction: maximization with resource constraints. You'll solve a production planning LP and then analyze the dual information (shadow prices, reduced costs) to answer managerial questions: "Which resources are bottlenecks?", "How much would we pay for more machine time?", "Why aren't we producing product X?"

**What you implement:**
- `solve_production_planning()` — Create variables for production quantities, maximize profit, add resource capacity constraints, solve, and extract full dual analysis (shadow prices, reduced costs, slack).

**Why it matters:** In practice, the solution itself is only half the value of an LP. The dual analysis tells you where to invest (resources with high shadow prices), what to ignore (non-binding constraints with zero shadow prices), and what's not competitive (products with positive reduced costs). This is the information that drives real business decisions.

### Phase 3: Binary Knapsack — MIP (~20 min)

**File:** `src/phase3_knapsack.py`

This phase introduces integer programming: the knapsack problem requires binary variables (`cat='Binary'`), which makes it NP-hard and requires branch-and-bound internally. You'll also solve the LP relaxation and compare: the gap between the LP bound and the MIP optimum is the **integrality gap**, a key concept for understanding solver difficulty.

**What you implement:**
- `solve_knapsack()` — Create binary variables for item selection, maximize total value, add the weight capacity constraint, solve the MIP, then solve the LP relaxation separately and compare bounds.

**Why it matters:** Binary variables are everywhere in real-world optimization: open/close decisions, scheduling assignments, yes/no selections. Understanding the LP relaxation bound tells you how hard the problem is for the solver (small gap = easy, large gap = hard) and provides a quality guarantee for the integer solution.

### Phase 4: Facility Location — MIP (~25 min)

**File:** `src/phase4_facility.py`

This phase combines binary and continuous variables in a single model: the Capacitated Facility Location Problem (CFLP). Binary variables decide which facilities to open, continuous variables assign customers to facilities. The **linking constraint** `x[f][c] <= y[f]` says "can't use a facility unless it's open" — this is the most important MIP modeling pattern.

**What you implement:**
- `solve_facility_location()` — Create binary open/close variables and continuous assignment variables, minimize total cost (fixed + transport), add demand satisfaction, linking, and capacity constraints, solve, and display which facilities are open and how customers are assigned.

**Why it matters:** Facility location is the canonical mixed-integer problem: it demonstrates how binary "design" decisions interact with continuous "operational" decisions. The linking constraint pattern (`continuous <= binary`) appears in facility location, network design, scheduling, and supply chain models. Understanding this pattern is essential for modeling any problem with on/off decisions.

## Motivation

PuLP is the simplest Python optimization modeling tool — the natural entry point for anyone who wants to use solvers rather than build them. After implementing simplex (032a), duality (032b), and branch-and-bound (034a) from scratch, this practice shows how those algorithms are invoked in production via a high-level API.

The modeling patterns learned here (algebraic formulation, dual analysis, linking constraints, LP relaxation comparison) transfer directly to:
- **Pyomo** (040b) — more powerful but same paradigm
- **CVXPY** (041a) — convex optimization with disciplined modeling
- **OR-Tools** (042a/b) — constraint programming and routing
- **Commercial solvers** (Gurobi, CPLEX) — same modeling patterns, faster solvers

Understanding PuLP + HiGHS is also practical for production work: HiGHS is the default solver in SciPy and can handle problems with millions of variables. For many real-world LP/MIP problems, PuLP + HiGHS is sufficient without commercial licenses.

## Commands

All commands are run from the `practice_040a_pulp_highs/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (PuLP, highspy) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_diet.py` | Phase 1: Diet problem — LP with shadow price analysis |
| `uv run python src/phase2_production.py` | Phase 2: Production planning — LP with full dual analysis |
| `uv run python src/phase3_knapsack.py` | Phase 3: Binary knapsack — MIP vs LP relaxation comparison |
| `uv run python src/phase4_facility.py` | Phase 4: Facility location — MIP with linking constraints |

## References

- [PuLP Documentation](https://coin-or.github.io/pulp/) — Official PuLP documentation with examples and API reference
- [PuLP GitHub (COIN-OR)](https://github.com/coin-or/pulp) — Source code and issue tracker
- [HiGHS Solver](https://highs.dev/) — Official HiGHS website with benchmarks and documentation
- [HiGHS GitHub](https://github.com/ERGO-Code/HiGHS) — Source code and discussion forum
- [Mittelmann Benchmarks](https://plato.asu.edu/bench.html) — Industry-standard LP/MIP solver benchmarks
- [Stigler Diet Problem (1945)](https://en.wikipedia.org/wiki/Stigler_diet) — Historical context for the diet problem
- [PuLP Solver Interface Docs](https://coin-or.github.io/pulp/technical/solvers.html) — How PuLP interfaces with different solvers

## State

`not-started`
