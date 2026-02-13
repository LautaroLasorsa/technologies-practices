# Practice 045: Stochastic Programming

## Technologies

- **Pyomo** -- Algebraic modeling language for optimization in Python. Used here to formulate two-stage and multi-stage stochastic programs as deterministic equivalents (extensive forms) using indexed Blocks for scenario representation.
- **HiGHS** -- High-performance open-source LP/MIP solver (via `highspy`). Solves the extensive form LP/MIP models generated from stochastic programs.
- **NumPy** -- Scenario generation, sampling, distance computations, and statistical analysis (confidence intervals, convergence).
- **Matplotlib** -- Visualization of scenario trees, SAA convergence plots, solution comparisons, and decision analysis.
- **Python 3.12+** -- Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- Pyomo >= 6.7, highspy >= 1.7
- NumPy >= 1.26, Matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### Why Stochastic Programming? The Limits of Deterministic Models

Deterministic optimization assumes all parameters are known with certainty. In reality, demand, supply, prices, yields, and processing times are **uncertain**. A deterministic model using expected values can produce solutions that are:

- **Infeasible** under some realizations (e.g., planned capacity cannot meet high demand).
- **Suboptimal** because they miss hedging opportunities (e.g., diversifying supply sources).

Stochastic programming explicitly models uncertainty by optimizing over a set of **scenarios** -- discrete realizations of the uncertain parameters, each with an associated probability. The goal is to find decisions that perform well **on average across all scenarios**, while allowing adaptive recourse actions after uncertainty is revealed.

**Source:** [Birge & Louveaux, "Introduction to Stochastic Programming" (Springer, 2011)](https://link.springer.com/book/10.1007/978-1-4614-0237-4)

### Two-Stage Stochastic Programming

The most common formulation. Decisions are split into two stages:

**Stage 1 (here-and-now):** Decisions `x` made **before** uncertainty is revealed. These must be the same regardless of which scenario occurs (non-anticipativity). Example: how much land to allocate to each crop before knowing the weather.

**Stage 2 (wait-and-see / recourse):** Decisions `y(ξ)` made **after** observing the random outcome `ξ`. These can adapt to each scenario. Example: how much crop to buy/sell after observing the yield.

**Mathematical formulation:**

```
min   c^T x + E_ξ [Q(x, ξ)]
s.t.  Ax = b
      x >= 0

where Q(x, ξ) = min  q(ξ)^T y
                 s.t. W(ξ) y = h(ξ) - T(ξ) x    (linking first and second stage)
                      y >= 0
```

- `c^T x`: first-stage cost (known with certainty)
- `Q(x, ξ)`: recourse function -- optimal second-stage cost given first-stage decision x and realization ξ
- `E_ξ[Q(x,ξ)]`: expected second-stage cost across all scenarios
- `T(ξ)`: technology matrix linking stages (how first-stage decisions affect second-stage feasibility)
- `W(ξ)`: recourse matrix (often fixed, called "fixed recourse")

### The Extensive Form (Deterministic Equivalent)

When uncertainty is represented by a finite set of scenarios `ξ_1, ..., ξ_S` with probabilities `p_1, ..., p_S`, the stochastic program can be written as a single large deterministic problem:

```
min   c^T x + Σ_s p_s * q_s^T y_s
s.t.  Ax = b
      T_s x + W_s y_s = h_s,    for each scenario s = 1,...,S
      x >= 0, y_s >= 0
```

This is called the **extensive form** or **deterministic equivalent**. First-stage variables `x` appear once (shared across scenarios). Second-stage variables `y_s` are duplicated for each scenario. The model size grows linearly with the number of scenarios.

In Pyomo, this is implemented using **indexed Blocks** -- one Block per scenario containing that scenario's second-stage variables and constraints, all linked to the shared first-stage variables.

### Value Metrics: VSS and EVPI

Two key metrics quantify the value of modeling uncertainty:

**RP (Recourse Problem):** The optimal objective of the stochastic program (the extensive form). This is the best achievable expected cost when we account for uncertainty.

**EEV (Expected result of using the Expected Value solution):** Solve the deterministic problem using expected values of uncertain parameters (the "mean scenario"). Then evaluate this solution's actual expected cost across all scenarios. Often worse than RP because the mean-value solution doesn't hedge.

**WS (Wait-and-See):** The expected cost if we had perfect information -- solve each scenario independently as if we knew it would occur, then take the probability-weighted average. This is a lower bound (we can never do better than perfect information).

```
VSS = EEV - RP >= 0
```
**Value of Stochastic Solution:** How much we gain by solving the stochastic model instead of the deterministic mean-value model. If VSS is large, uncertainty matters and stochastic programming pays off.

```
EVPI = RP - WS >= 0
```
**Expected Value of Perfect Information:** How much we would pay for a perfect forecast. If EVPI is small, our stochastic solution is already close to what we'd achieve with perfect information.

**Relationship:** `WS <= RP <= EEV`, so `EVPI >= 0` and `VSS >= 0`.

**Source:** [Birge & Louveaux (2011), Chapter 4: The Value of Information](https://link.springer.com/book/10.1007/978-1-4614-0237-4)

### Sample Average Approximation (SAA)

When the number of possible scenarios is very large (or continuous), the extensive form becomes intractable. SAA replaces the true expectation with an empirical average over N sampled scenarios:

```
min   c^T x + (1/N) Σ_{i=1}^{N} Q(x, ξ_i)
s.t.  x in X

where ξ_1, ..., ξ_N are i.i.d. samples from the distribution of ξ
```

**SAA procedure:**
1. Draw M independent batches, each of N scenarios.
2. Solve the SAA problem for each batch, obtaining candidate solutions x_1*, ..., x_M*.
3. For each candidate, evaluate the objective on a large independent sample (N' >> N) to get an unbiased estimate.
4. Select the best candidate and compute confidence intervals.

**Convergence:** As N -> infinity, the SAA optimal value converges to the true optimal value (with probability 1). The rate is O(1/sqrt(N)), so quadrupling the sample size halves the error.

**Statistical output:**
- Lower bound: average SAA objective across M batches (biased low because of optimization bias).
- Upper bound: evaluation of the best candidate on a large independent sample.
- Optimality gap estimate: upper bound - lower bound, with confidence intervals.

**Source:** [Shapiro, Dentcheva & Ruszczynski, "Lectures on Stochastic Programming" (SIAM, 2021)](https://doi.org/10.1137/1.9781611976595), [Kleywegt, Shapiro & Homem-de-Mello (2002)](https://doi.org/10.1137/S1052623499363220)

### Scenario Generation Methods

**Direct sampling:** Draw scenarios from known distributions (normal, uniform, discrete). Simple but may require many scenarios for convergence.

**Moment matching:** Generate scenarios whose first few moments (mean, variance, skewness) match the target distribution. Produces compact scenario sets but limited to matching specified moments.

**Latin Hypercube Sampling (LHS):** Stratified sampling that ensures coverage of the entire distribution range. Better coverage than pure Monte Carlo for the same number of samples.

### Scenario Reduction

When you have too many scenarios (from simulation or historical data), **scenario reduction** selects a representative subset that approximates the original distribution:

**Kantorovich (Wasserstein) distance:** Measures the distance between two discrete probability distributions by the minimum-cost transportation of probability mass. For scenarios ξ_1,...,ξ_S with probabilities p_1,...,p_S and a reduced set J ⊂ {1,...,S}:

```
D_K = Σ_{i not in J} p_i * min_{j in J} ||ξ_i - ξ_j||
```

**Fast forward selection (Heitsch & Romisch, 2003):**
1. Start with empty selected set J = {}.
2. At each step, add the scenario that **reduces the Kantorovich distance the most**.
3. The greedy metric: for each candidate j, compute D_j = Σ_{i not in J∪{j}} p_i * min_{k in J∪{j}} ||ξ_i - ξ_k||. Pick j that minimizes D_j.
4. Redistribute probabilities: each deleted scenario's probability goes to its nearest retained scenario.
5. Repeat until |J| = target size.

**Source:** [Heitsch & Romisch, "Scenario Reduction Algorithms in Stochastic Programming" (2003)](https://doi.org/10.1023/A:1021805924152), [Dupacova, Growe-Kuska & Romisch (2003)](https://doi.org/10.1007/s101070100270)

### Multi-Stage Stochastic Programming

Extends two-stage to T stages, where decisions at each stage depend on information revealed up to that point. The scenario structure is represented as a **scenario tree**:

```
         Stage 1          Stage 2          Stage 3
                          ┌── ξ_H ──── ξ_HH
           ┌── ξ_H ──────┤
           │              └── ξ_H ──── ξ_HL
    x_1 ───┤
           │              ┌── ξ_L ──── ξ_LH
           └── ξ_L ──────┤
                          └── ξ_L ──── ξ_LL
```

**Non-anticipativity constraints:** At each node in the scenario tree, scenarios that share the same history (are indistinguishable at that point) must make the **same decision**. Formally, if scenarios s and s' are in the same partition at stage t, then `x_t^s = x_t^{s'}`.

In the extensive form, non-anticipativity is enforced by:
- Using a single variable for each decision node (implicit), or
- Creating per-scenario variables and adding explicit equality constraints: `x_t^s = x_t^{s'}` for all (s, s') in the same node.

**Multi-stage formulation:**
```
min  c_1^T x_1 + E[c_2^T x_2(ξ_2) + ... + c_T^T x_T(ξ_2,...,ξ_T)]
s.t. A_1 x_1 = b_1
     T_{t-1} x_{t-1} + W_t x_t = h_t(ξ_t),  t = 2,...,T
     x_t measurable w.r.t. information at stage t (non-anticipativity)
     x_t >= 0
```

### Relationship to Robust Optimization

Stochastic programming and robust optimization are two paradigms for optimization under uncertainty:

| Aspect | Stochastic Programming | Robust Optimization |
|--------|----------------------|---------------------|
| **Uncertainty model** | Probability distribution (scenarios with probabilities) | Uncertainty set (worst-case, no probabilities) |
| **Objective** | Minimize expected cost | Minimize worst-case cost |
| **Philosophy** | Risk-neutral on average | Risk-averse (worst-case protection) |
| **Data requirement** | Scenario probabilities (or distribution) | Uncertainty set bounds |
| **Computational** | Grows with number of scenarios | Often tractable reformulations |
| **When to use** | Good probability estimates available | Distribution unknown, need guarantees |

Practice 046 (Robust Optimization) covers the complementary paradigm.

**Source:** [Ben-Tal, El Ghaoui & Nemirovski, "Robust Optimization" (Princeton, 2009)](https://press.princeton.edu/books/hardcover/9780691143682/robust-optimization)

### Pyomo Implementation Pattern for Stochastic Programs

This practice uses **plain Pyomo ConcreteModel with indexed Blocks** to build extensive forms directly, without requiring PySP or mpi-sppy. This approach:

1. Creates a ConcreteModel with first-stage variables at the top level.
2. Creates a `Block` for each scenario containing second-stage variables and constraints.
3. Links stages via constraints that reference both top-level and block-level variables.
4. Weights the objective by scenario probabilities.

```python
model = pyo.ConcreteModel()
# First-stage variables (shared across all scenarios)
model.x = pyo.Var(CROPS, within=pyo.NonNegativeReals)

# Scenario blocks (one per scenario)
def scenario_block_rule(block, s):
    # Second-stage variables specific to scenario s
    block.y_buy = pyo.Var(CROPS, within=pyo.NonNegativeReals)
    block.y_sell = pyo.Var(CROPS, within=pyo.NonNegativeReals)
    # Linking constraints (reference model.x from parent)
    def balance_rule(b, c):
        return yield_data[s,c] * model.x[c] + b.y_buy[c] - b.y_sell[c] >= requirement[c]
    block.balance = pyo.Constraint(CROPS, rule=balance_rule)

model.scenarios = pyo.Block(SCENARIO_SET, rule=scenario_block_rule)

# Objective: first-stage cost + expected second-stage cost
model.obj = pyo.Objective(expr=
    first_stage_cost(model) +
    sum(prob[s] * second_stage_cost(model.scenarios[s]) for s in SCENARIO_SET)
)
```

This is the standard pattern used throughout all four phases.

**Source:** [Pyomo Documentation -- Blocks](https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Blocks.html), [mpi-sppy farmer example](https://mpi-sppy.readthedocs.io/en/latest/examples.html)

## Description

Model optimization problems under uncertainty using stochastic programming with Pyomo. Progress from the classic two-stage farmer's problem through Sample Average Approximation, scenario generation and reduction techniques, to a full multi-stage inventory management problem.

### What you'll build

1. **Two-stage farmer's problem** -- The classic Birge & Louveaux example: allocate land before knowing crop yields, then buy/sell crops as recourse. Compute VSS and EVPI to quantify the value of stochastic modeling.
2. **Sample Average Approximation** -- When scenarios are too numerous for the extensive form, sample batches and solve repeatedly. Compute confidence intervals and observe convergence.
3. **Scenario generation and reduction** -- Generate scenarios from distributions, then reduce them to a compact representative set using the fast forward selection algorithm based on Kantorovich distance.
4. **Multi-stage inventory management** -- A 3-period inventory problem with uncertain demand at each stage. Build the scenario tree, enforce non-anticipativity, and compare with a rolling horizon approach.

## Instructions

### Phase 1: Two-Stage Stochastic LP -- The Farmer's Problem (~25 min)

**File:** `src/phase1_two_stage.py`

The farmer's problem (Birge & Louveaux, Section 1.1) is the "hello world" of stochastic programming. A farmer has 500 acres and must decide how to allocate land among wheat, corn, and sugar beets **before** knowing the weather. After observing yields, the farmer can buy or sell crops to meet requirements and maximize profit. Three weather scenarios (good, average, bad) are considered with equal probability.

**What you implement:**

- `build_stochastic_farmer_model(scenarios, probabilities)` -- Build the full extensive form Pyomo model. First-stage: land allocation variables. Scenario blocks: purchase/sell recourse variables with yield-dependent constraints. Probability-weighted objective.
- `build_deterministic_farmer_model(yields)` -- Build and solve the deterministic model using a single yield vector (the expected/average yields). This serves as the baseline for computing VSS.
- `compute_vss_and_evpi(scenarios, probabilities)` -- Compute the three key values (RP, EEV, WS) and derive VSS = EEV - RP and EVPI = RP - WS. This requires solving the stochastic model, the mean-value model, and each scenario independently.

**Why it matters:** This is the foundational stochastic programming workflow. The extensive form pattern (shared first-stage variables + scenario blocks) is used everywhere. VSS and EVPI are the standard metrics to justify stochastic modeling to stakeholders -- "here's how much money we save by accounting for uncertainty."

### Phase 2: Sample Average Approximation (~25 min)

**File:** `src/phase2_saa.py`

When the number of scenarios is large or continuous, the extensive form is intractable. SAA uses Monte Carlo sampling: draw N scenarios, solve the resulting extensive form, repeat M times, and use statistics to bound the true optimal value. This phase applies SAA to a variant of the farmer's problem with continuously distributed yields.

**What you implement:**

- `sample_scenarios(n_scenarios, rng)` -- Generate a batch of N random yield scenarios by sampling from a distribution (normal around expected yields, clipped to reasonable bounds).
- `saa_single_replication(n_scenarios, rng)` -- One SAA replication: sample N scenarios with equal probability 1/N, build and solve the extensive form, return the optimal objective and first-stage solution.
- `run_saa(n_scenarios, n_replications, seed)` -- The full SAA loop: run M replications, collect objective values, compute the lower bound (mean of SAA objectives), evaluate the best candidate on a large independent sample for the upper bound, and compute confidence intervals.

**Why it matters:** SAA is the practical workhorse for stochastic programming when you cannot enumerate all scenarios. Understanding the sampling-solving-evaluation loop and the statistical guarantees (confidence intervals, convergence rate O(1/sqrt(N))) is essential for real applications where distributions are estimated from data.

### Phase 3: Scenario Generation & Reduction (~25 min)

**File:** `src/phase3_scenarios.py`

In practice, you need to create scenario sets from distributions and then reduce them to a manageable size while preserving solution quality. This phase covers generation from multivariate distributions and reduction using the fast forward selection algorithm.

**What you implement:**

- `generate_scenarios_from_distribution(n_scenarios, mean, cov, rng)` -- Generate scenarios from a multivariate normal distribution. Each scenario is a yield vector for all crops. Assign equal probabilities.
- `kantorovich_distance(scenarios_full, probs_full, selected_indices)` -- Compute the Kantorovich distance between the full scenario set and the selected subset. Each non-selected scenario contributes its probability times the distance to its nearest selected scenario.
- `fast_forward_selection(scenarios, probabilities, n_target)` -- The greedy forward selection algorithm: iteratively add the scenario that reduces the Kantorovich distance most. Redistribute deleted scenarios' probabilities to their nearest retained scenario.
- `compare_solutions(scenarios_full, probs_full, scenarios_reduced, probs_reduced)` -- Solve the farmer's problem with both the full and reduced scenario sets. Compare first-stage decisions and objective values to assess reduction quality.

**Why it matters:** Real-world stochastic programs may have thousands of scenarios from simulation. Scenario reduction is how you make them tractable without discarding information randomly. The Kantorovich distance provides a principled measure of approximation quality.

### Phase 4: Multi-Stage Stochastic Program (~25 min)

**File:** `src/phase4_multi_stage.py`

Extend from two-stage to three-stage: an inventory management problem where demand is uncertain at each period. The decision-maker orders inventory at the start of each period, observes demand, and carries excess to the next period (or incurs backlog cost). The scenario tree branches at each stage.

**What you implement:**

- `build_scenario_tree(demand_outcomes, probabilities_per_stage)` -- Construct a scenario tree by taking the Cartesian product of per-stage demand outcomes. Each leaf-to-root path is a scenario. Compute path probabilities.
- `build_multi_stage_model(scenario_tree, params)` -- Build the extensive form with explicit non-anticipativity constraints. Variables: order quantity at each stage for each scenario. Constraints: inventory balance, non-negativity, and non-anticipativity (scenarios sharing the same history must order the same amount).
- `rolling_horizon(demand_scenarios, probabilities, params)` -- The myopic alternative: at each stage, solve a two-stage problem using the remaining scenarios, implement the first-stage decision, observe the outcome, and repeat. Compare with the full multi-stage solution.

**Why it matters:** Multi-stage models are the general case -- most real decisions unfold over time (production planning, portfolio rebalancing, supply chain management). Non-anticipativity constraints are the critical concept: they encode the information structure (what you know when you decide). The rolling horizon comparison shows the cost of myopia.

## Motivation

- **Handles real-world uncertainty:** Deterministic models are idealizations. Stochastic programming is the standard framework for supply chain planning, energy dispatch, portfolio management, and capacity expansion under uncertainty. AutoScheduler.AI's warehouse optimization faces demand, processing time, and resource uncertainty daily.
- **Builds on Pyomo mastery:** Practice 040b introduced Pyomo modeling. This practice uses its advanced features (indexed Blocks, complex linking constraints) in a new domain.
- **Practical methodology:** SAA and scenario reduction are not academic curiosities -- they are how stochastic programs are solved in practice when you have data or simulation models instead of neat 3-scenario examples.
- **Gateway to robust optimization:** Practice 046 covers the complementary paradigm (worst-case instead of expected-value). Understanding stochastic programming first makes the comparison meaningful.
- **Industry demand:** Stochastic programming skills are sought in energy trading, supply chain optimization, financial risk management, and operations research consulting.

## Commands

All commands are run from the `practice_045_stochastic_programming/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (Pyomo, HiGHS, NumPy, Matplotlib) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_two_stage.py` | Phase 1: Two-stage farmer's problem, extensive form, VSS & EVPI computation |
| `uv run python src/phase2_saa.py` | Phase 2: Sample Average Approximation with confidence intervals and convergence |
| `uv run python src/phase3_scenarios.py` | Phase 3: Scenario generation from distributions, fast forward reduction, comparison |
| `uv run python src/phase4_multi_stage.py` | Phase 4: Multi-stage inventory problem, non-anticipativity, rolling horizon comparison |

### Debugging

| Command | Description |
|---------|-------------|
| `uv run python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('highs').available())"` | Verify HiGHS solver is available |
| `uv run python -c "import pyomo.environ as pyo; m = pyo.ConcreteModel(); m.x = pyo.Var(); print('Pyomo OK')"` | Quick Pyomo smoke test |
| `uv run python -c "import numpy; print(numpy.__version__)"` | Verify NumPy version |

## References

- [Birge & Louveaux, "Introduction to Stochastic Programming" (Springer, 2011)](https://link.springer.com/book/10.1007/978-1-4614-0237-4) -- The standard textbook; farmer's problem from Chapter 1
- [Shapiro, Dentcheva & Ruszczynski, "Lectures on Stochastic Programming" (SIAM, 2021)](https://doi.org/10.1137/1.9781611976595) -- SAA theory and convergence
- [Heitsch & Romisch (2003), "Scenario Reduction Algorithms in Stochastic Programming"](https://doi.org/10.1023/A:1021805924152) -- Fast forward/backward selection
- [Dupacova, Growe-Kuska & Romisch (2003), "Scenario Reduction in Stochastic Programming"](https://doi.org/10.1007/s101070100270) -- Kantorovich distance foundations
- [Pyomo Documentation -- Blocks](https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Blocks.html) -- Indexed blocks for scenario representation
- [mpi-sppy Documentation & Examples](https://mpi-sppy.readthedocs.io/en/latest/examples.html) -- Farmer problem reference implementation
- [PySP Documentation](https://pysp.readthedocs.io/en/latest/pysp.html) -- Original Pyomo stochastic programming extension
- [Pyomo Documentation (stable)](https://pyomo.readthedocs.io/en/stable/) -- General Pyomo reference
- [HiGHS Solver](https://highs.dev/) -- LP/MIP solver used throughout

## State

`not-started`
