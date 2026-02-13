# Practice 040b: Advanced Modeling -- Pyomo

## Technologies

- **Pyomo** -- Algebraic modeling language for optimization in Python (supports LP, MIP, NLP, MINLP, stochastic programming, DAE)
- **HiGHS** -- High-performance open-source solver for LP, MIP, and QP (via `highspy`)
- **Ipopt** -- Interior-point solver for large-scale nonlinear optimization (via `cyipopt`, optional)

## Stack

- Python 3.12+, uv

## Theoretical Context

### Pyomo vs PuLP

PuLP is a lightweight LP/MIP modeling library -- it generates `.lp` or `.mps` files and calls an external solver. PuLP is limited to **linear** objectives and constraints with continuous or integer variables. It has no concept of abstract models, no nonlinear support, and no advanced modeling constructs.

Pyomo is a full **algebraic modeling language** (AML) embedded in Python -- the Python equivalent of AMPL/GAMS. Beyond LP/MIP, Pyomo supports:

- **Abstract models** -- separate model structure from data; reuse one model across many instances
- **Nonlinear programming (NLP)** -- smooth nonlinear objectives and constraints (solved by Ipopt, KNITRO)
- **Mixed-integer nonlinear programming (MINLP)** -- binary/integer variables + nonlinear expressions (Bonmin, Couenne)
- **Stochastic programming** -- `pyomo.pysp` for two-stage and multi-stage stochastic optimization
- **Differential-algebraic equations (DAE)** -- `pyomo.dae` for dynamic optimization
- **Generalized disjunctive programming (GDP)** -- logical constraints with `pyomo.gdp`
- **Piecewise linear approximations** -- built-in `Piecewise` component
- **Bilevel programming** -- `pyomo.bilevel`

PuLP is the right choice for simple LP/MIP models where you want minimal setup. Pyomo is required when you need any of the above capabilities, or when you want data-driven model construction via abstract models.

**Sources:** [Pyomo Documentation](https://pyomo.readthedocs.io/en/stable/), [PuLP Documentation](https://coin-or.github.io/pulp/), [Pyomo vs PuLP comparison (ResearchGate)](https://www.researchgate.net/post/Optimization-with-Pulp-vs-Pyomo-any-experience)

### Concrete vs Abstract Models

**ConcreteModel** -- data is supplied inline at model construction time. Each component (Set, Param, Var, Constraint) is immediately populated when defined. This is the Pythonic approach: build the model procedurally, populate data directly.

```python
model = pyo.ConcreteModel()
model.x = pyo.Var([1,2,3], within=pyo.NonNegativeReals)
model.obj = pyo.Objective(expr=sum(model.x[i] for i in [1,2,3]))
```

**AbstractModel** -- model *structure* is defined without data. Sets, Parameters, Variables, Constraints, and Objectives are declared symbolically. Data is supplied later via `model.create_instance(data)`, producing a concrete instance. This enables:

- **Model reuse** -- one model definition, many data sets (different customers, scenarios, time periods)
- **Data validation** -- Pyomo validates that supplied data matches declared Sets and Parameters
- **Separation of concerns** -- modelers define structure, analysts provide data

```python
model = pyo.AbstractModel()
model.I = pyo.Set()
model.c = pyo.Param(model.I)
model.x = pyo.Var(model.I, within=pyo.NonNegativeReals)
# ... define rules ...
instance = model.create_instance({None: {'I': {None: [1,2,3]}, 'c': {1: 10, 2: 20, 3: 30}}})
```

The data dictionary format uses `{None: {component_name: data_dict}}`. For Sets, the value is `{None: list_of_elements}`. For Params, the value is `{index: value}`.

**Sources:** [Abstract vs Concrete Models (Pyomo docs)](https://pyomo.readthedocs.io/en/stable/explanation/philosophy/abstract_modeling.html), [Pyomo Overview](https://pyomo.readthedocs.io/en/6.8.0/pyomo_overview/abstract_concrete.html)

### Pyomo Modeling Components

| Component | Purpose | Example |
|-----------|---------|---------|
| **Set** | Index space for variables/constraints | `model.I = Set(initialize=[1,2,3])` |
| **Param** | Fixed data (coefficients, bounds, RHS) | `model.cost = Param(model.I, initialize={1:10, 2:20})` |
| **Var** | Decision variables | `model.x = Var(model.I, within=NonNegativeReals)` |
| **Constraint** | Feasibility restrictions | `model.con = Constraint(model.I, rule=con_rule)` |
| **Objective** | Function to minimize/maximize | `model.obj = Objective(rule=obj_rule, sense=minimize)` |
| **RangeSet** | Contiguous integer set | `model.T = RangeSet(1, 12)` -- periods 1..12 |
| **Expression** | Named sub-expressions for reuse | `model.total = Expression(rule=total_rule)` |
| **Piecewise** | Piecewise-linear approximation | `model.pw = Piecewise(model.I, model.y, model.x, ...)` |
| **Suffix** | Solver communication (duals, etc.) | `model.dual = Suffix(direction=Suffix.IMPORT)` |

### Indexed Components

Pyomo's power comes from multi-dimensional indexing. Variables, Constraints, and Parameters can be indexed over one or more Sets:

```python
model.I = Set(initialize=['A','B'])      # suppliers
model.J = Set(initialize=[1, 2, 3])      # customers
model.x = Var(model.I, model.J, within=NonNegativeReals)   # x['A',1], x['A',2], ...

def supply_rule(m, i):
    return sum(m.x[i,j] for j in m.J) <= m.supply[i]
model.supply_con = Constraint(model.I, rule=supply_rule)   # one constraint per supplier
```

Constraint rules receive the model `m` plus the index values as arguments. The rule is called once per element of the indexing set(s).

### Nonlinear Programming (NLP)

Pyomo expresses nonlinear objectives and constraints naturally in Python:

```python
model.obj = Objective(expr=sum(cov[i,j] * model.w[i] * model.w[j] for i in I for j in I))
```

Pyomo builds an expression tree and communicates it to nonlinear solvers via the NL file format or direct Python interface. Key NLP solvers:

- **Ipopt** -- open-source interior-point method for large-scale NLP. Handles convex and non-convex smooth problems. Available via `cyipopt` Python package or standalone binary.
- **HiGHS** -- primarily LP/MIP but also handles convex QP (quadratic programming). Can solve Markowitz portfolio problems directly.
- **Bonmin** -- Branch-and-Bound for MINLP (mixes integer variables with nonlinear constraints). Open-source via COIN-OR.
- **Couenne** -- global MINLP solver (finds global optimum, not just local). Slower but more reliable for non-convex MINLP.

### Solver Interface

```python
solver = pyo.SolverFactory('highs')                 # or 'ipopt', 'bonmin', 'glpk'
solver.options['time_limit'] = 60                    # solver-specific options
result = solver.solve(model, tee=True)               # tee=True prints solver log

# Check termination
from pyomo.opt import TerminationCondition
assert result.solver.termination_condition == TerminationCondition.optimal

# Warm-starting (provide initial values for faster re-solve)
model.x[1].value = 5.0
solver.solve(model, warmstart=True)
```

### Results Analysis

After solving, extract sensitivity information:

```python
# Declare suffixes BEFORE solving
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.rc = pyo.Suffix(direction=pyo.Suffix.IMPORT)       # reduced costs
model.slack = pyo.Suffix(direction=pyo.Suffix.IMPORT)

solver.solve(model)

# Shadow prices (dual values) -- marginal value of relaxing a constraint by 1 unit
for c in model.supply_con:
    print(f"Supply {c}: dual = {model.dual[model.supply_con[c]]:.4f}")

# Reduced costs -- how much the objective coefficient must improve before variable enters basis
for v in model.x:
    print(f"x{v}: rc = {model.rc[model.x[v]]:.4f}")
```

**Shadow prices** tell you: "if I had one more unit of supply at source i, how much would my objective improve?" This is critical for resource allocation decisions.

**Note:** Not all solvers export duals. HiGHS and Gurobi support dual export; CBC has limited support. Ipopt provides duals for NLP problems.

### Modeling Patterns

**Big-M constraints** -- activate/deactivate constraints using binary variables:
```python
# x <= capacity * y  (if y=0, x forced to 0; if y=1, x <= capacity)
model.link = Constraint(rule=lambda m: m.x <= m.capacity * m.y)
```

**Indicator constraints** -- Pyomo GDP provides cleaner logical constraints:
```python
from pyomo.gdp import Disjunct, Disjunction
# Either facility is open (with capacity constraint) or closed (x=0)
```

**Piecewise linear** -- approximate nonlinear functions:
```python
model.pw = Piecewise(model.I, model.y, model.x,
                     pw_pts=breakpoints, f_rule=values,
                     pw_constr_type='EQ')
```
Pyomo automatically detects convexity/concavity and chooses the tightest formulation (SOS2, big-M binary, or disaggregated convex combination).

**Sources:** [Pyomo Piecewise docs](https://pyomo.readthedocs.io/en/stable/api/pyomo.core.base.piecewise.Piecewise.html), [Pyomo Solver Recipes](https://pyomo.readthedocs.io/en/stable/howto/solver_recipes.html), [Solving Pyomo Models](https://pyomo.readthedocs.io/en/6.8.0/solving_pyomo_models.html)

### Key Concepts Summary

| Concept | Definition |
|---------|------------|
| **AbstractModel** | Model structure without data; instantiated via `create_instance(data)` for reuse across datasets |
| **ConcreteModel** | Model with data supplied inline at construction time |
| **Set** | Index space defining valid indices for Params, Vars, and Constraints |
| **Param** | Immutable data attached to the model (costs, capacities, demands) |
| **Var domain** | `NonNegativeReals`, `Binary`, `Integers`, `Reals`, `UnitInterval` -- restricts variable values |
| **Constraint rule** | Python function `rule(model, *indices)` returning an expression or `Constraint.Skip` |
| **Suffix** | Mechanism to import/export solver information (duals, reduced costs, slack) |
| **NL file format** | Standard format for communicating nonlinear models to solvers (Ipopt, Bonmin) |
| **Warm-starting** | Providing initial variable values to speed up re-solves (useful for parameter sweeps) |
| **Piecewise** | Built-in component for piecewise-linear approximation of nonlinear functions |
| **Big-M** | Modeling pattern linking binary activation variables to continuous constraints |
| **QP** | Quadratic programming -- linear constraints, quadratic objective (convex if Q is PSD) |
| **MINLP** | Mixed-integer nonlinear programming -- hardest class; combines integer and nonlinear |

## Description

Model advanced OR problems with Pyomo: an abstract transportation model (reusable across data sets), multi-period production planning with inventory balance constraints, nonlinear Markowitz portfolio optimization (QP), and a mixed-integer nonlinear facility location problem with congestion costs.

## Instructions

### Phase 1: Abstract Transportation Model (~25 min)

The classic transportation problem minimizes shipping cost from suppliers to customers. This phase teaches **abstract models** -- the key differentiator between Pyomo and PuLP.

1. Read `src/phase1_transportation.py` -- understand the two data instances provided
2. **User implements** `build_transportation_model()`:
   - Create an `AbstractModel` with Sets (suppliers, customers), Params (supply, demand, cost), Vars (shipment quantities)
   - Define objective rule (minimize total shipping cost) and constraint rules (supply limits, demand satisfaction)
   - Return the abstract model (not an instance)
3. The `main()` function creates instances from two different datasets and solves each
4. **Key learning:** One model definition, two data sets -- this is impossible with PuLP's concrete-only approach. Abstract models enable data-driven optimization where analysts change `.dat` files without touching model code.

### Phase 2: Multi-Period Production Planning (~25 min)

A manufacturing plant must decide how much of each product to produce in each time period, balancing production costs against inventory holding costs while respecting capacity constraints.

1. Read `src/phase2_production.py` -- understand the data structure (products, periods, costs, capacity)
2. **User implements** `solve_multi_period_production(data)`:
   - Create a `ConcreteModel` with Sets (products, periods), indexed Vars (production, inventory)
   - The **inventory balance constraint** links consecutive periods: `inv[p,t] = inv[p,t-1] + produce[p,t] - demand[p,t]`
   - Capacity constraint limits total production per period
3. **Key learning:** Multi-dimensional indexing (`Var(products, periods)`) and inter-period linking constraints. This pattern appears everywhere in supply chain planning, scheduling, and resource allocation.

### Phase 3: Portfolio Optimization -- NLP (~25 min)

The Markowitz mean-variance model minimizes portfolio risk (variance) subject to achieving a target expected return. The objective is **quadratic** (nonlinear).

1. Read `src/phase3_portfolio.py` -- understand the stock data (expected returns, covariance matrix)
2. **User implements** `solve_portfolio(returns, covariance, target_return)`:
   - Create a ConcreteModel with continuous variables `w[i]` (portfolio weights in [0,1])
   - **Nonlinear objective:** minimize `sum(cov[i,j] * w[i] * w[j])`
   - Constraints: budget (weights sum to 1), target return, no short selling
3. The `main()` function sweeps target returns to trace the **efficient frontier**
4. **Key learning:** Pyomo handles nonlinear expressions naturally. HiGHS can solve convex QP directly. This is impossible in PuLP.

### Phase 4: Facility Location with Congestion -- MINLP (~25 min)

Extend the classic facility location problem: opening costs are fixed, but service cost at each facility increases **nonlinearly** with load (congestion). This combines binary decisions (open/close) with nonlinear costs.

1. Read `src/phase4_minlp.py` -- understand the facility and customer data
2. **User implements** `solve_facility_congestion(facilities, customers)`:
   - Binary variables `y[f]` (open facility), continuous variables `x[f,c]` (assignment fraction)
   - **Nonlinear congestion term:** `congestion_cost * load^2` where `load = sum(demand * x)`
   - This is MINLP -- the hardest optimization class
3. For this practice, solving the continuous relaxation (y in [0,1]) is sufficient to understand the modeling
4. **Key learning:** MINLP modeling in Pyomo. Understanding why MINLP is hard (combines combinatorial explosion with non-convexity). Relaxation as a practical approach when exact MINLP solvers are unavailable.

## Motivation

- **Most flexible open-source optimization modeling:** Pyomo is the Python equivalent of AMPL/GAMS. It handles LP, MIP, NLP, MINLP, stochastic, and dynamic optimization -- all in one framework.
- **Industry standard:** Used extensively in energy systems, supply chain, manufacturing, and financial optimization. AutoScheduler.AI's domain (warehouse optimization) involves multi-period planning that benefits from Pyomo's indexed component system.
- **Prerequisite for advanced topics:** Stochastic programming (practice 045), robust optimization (practice 046), and integrated supply chain models (practice 048) all build on Pyomo.
- **Abstract models enable production deployment:** Separate model code from data -- analysts change parameters without touching optimization logic. Critical for building optimization-as-a-service products.
- **Nonlinear modeling:** Real-world problems are rarely purely linear. Congestion, economies of scale, risk (variance), and diminishing returns all require nonlinear terms.

## Commands

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install all dependencies (Pyomo, HiGHS) |
| `uv add cyipopt` | (Optional) Install Ipopt solver for NLP -- requires system libraries, may not work on all platforms |

### Run Phases

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_transportation.py` | Phase 1: Abstract transportation model -- solves two instances with same model |
| `uv run python src/phase2_production.py` | Phase 2: Multi-period production planning with inventory balance |
| `uv run python src/phase3_portfolio.py` | Phase 3: Markowitz portfolio optimization (NLP/QP) -- traces efficient frontier |
| `uv run python src/phase4_minlp.py` | Phase 4: Facility location with nonlinear congestion (MINLP relaxation) |

### Debugging

| Command | Description |
|---------|-------------|
| `uv run python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('highs').available())"` | Verify HiGHS solver is available |
| `uv run python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('ipopt').available())"` | Verify Ipopt solver is available (optional) |
| `uv run python -c "import pyomo.environ as pyo; m = pyo.ConcreteModel(); m.x = pyo.Var(); print('Pyomo OK')"` | Quick Pyomo smoke test |

## References

- [Pyomo Documentation (stable)](https://pyomo.readthedocs.io/en/stable/)
- [Pyomo -- Optimization Modeling in Python (Springer book)](https://link.springer.com/book/10.1007/978-3-030-68928-5)
- [Abstract vs Concrete Models](https://pyomo.readthedocs.io/en/stable/explanation/philosophy/abstract_modeling.html)
- [Pyomo Piecewise API](https://pyomo.readthedocs.io/en/stable/api/pyomo.core.base.piecewise.Piecewise.html)
- [Pyomo Solver Recipes](https://pyomo.readthedocs.io/en/stable/howto/solver_recipes.html)
- [HiGHS Solver](https://highs.dev/)
- [Ipopt Documentation](https://coin-or.github.io/Ipopt/)
- [Markowitz Portfolio Theory](https://en.wikipedia.org/wiki/Modern_portfolio_theory)

## State

`not-started`
