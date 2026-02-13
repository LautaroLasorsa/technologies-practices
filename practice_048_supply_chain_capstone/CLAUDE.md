# Practice 048: Integrated OR -- Supply Chain Optimization (Capstone)

## Technologies

- **Pyomo** -- Algebraic modeling language for optimization in Python. Used here to formulate facility location MIPs, lot-sizing MIPs, and two-stage stochastic programs as extensive forms using indexed Blocks for scenario representation.
- **HiGHS** -- High-performance open-source LP/MIP solver (via `highspy`). Solves all formulations: CFLP, lot-sizing, and the integrated stochastic model.
- **OR-Tools** -- Google's routing solver (`ortools.constraint_solver`) for solving multi-depot CVRP. Uses RoutingIndexManager with per-vehicle start/end depots, capacity dimensions, and GUIDED_LOCAL_SEARCH metaheuristic.
- **NumPy** -- Instance generation, distance computation, demand scenario sampling, and numerical analysis.
- **Matplotlib** -- Visualization of production schedules, inventory profiles, and cost breakdowns.
- **Python 3.12+** -- Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- Pyomo >= 6.7, highspy >= 1.7
- ortools >= 9.9
- NumPy >= 1.26, Matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### Supply Chain Optimization Hierarchy

Supply chain decisions are organized into three temporal levels, each with its own optimization problem class and planning horizon:

| Level | Horizon | Decision | OR Model | Example |
|-------|---------|----------|----------|---------|
| **Strategic** | 3-10 years | WHERE to locate facilities, capacity sizing | Facility Location (MIP) | Amazon builds a new fulfillment center |
| **Tactical** | 1-12 months | WHEN and HOW MUCH to produce/stock | Lot-sizing, production planning (MIP) | P&G's monthly production schedule |
| **Operational** | Daily/weekly | HOW to route deliveries, schedule shifts | VRP, scheduling (CP/heuristics) | FedEx daily route optimization |

Each level's decisions constrain the next: you can only produce at open facilities, and you can only deliver from facilities with inventory. This hierarchical structure means that poor strategic decisions (wrong facility locations) cannot be fully compensated by excellent operational execution. Conversely, optimal facility locations are wasted if routing is inefficient.

**Source:** [Simchi-Levi, Kaminsky & Simchi-Levi, "Designing and Managing the Supply Chain" (McGraw-Hill, 2008)](https://www.mhprofessional.com/designing-and-managing-the-supply-chain-9780073341521-usa), [Shapiro, "Modeling the Supply Chain" (Cengage, 2006)](https://www.cengage.com/c/modeling-the-supply-chain-2e-shapiro/)

### Capacitated Facility Location Problem (CFLP)

The CFLP is the fundamental strategic supply chain problem: given a set of potential facility sites and customer demand zones, decide which facilities to open and how to allocate customer demand to minimize total cost (fixed opening costs + transportation costs) subject to facility capacity limits.

**Mathematical formulation:**

```
min  SUM_j f[j]*y[j] + SUM_i SUM_j c[i,j]*x[i,j]
s.t. SUM_j x[i,j] = d[i]          for all customers i    (demand satisfaction)
     SUM_i x[i,j] <= s[j]*y[j]     for all facilities j   (capacity + linking)
     x[i,j] >= 0                   for all i, j
     y[j] in {0, 1}                for all j
```

Where:
- `y[j]` = 1 if facility j is opened (binary, strategic decision)
- `x[i,j]` = units shipped from facility j to customer i (continuous, operational)
- `f[j]` = fixed annual cost of operating facility j
- `c[i,j]` = per-unit transportation cost from j to i
- `d[i]` = demand of customer i
- `s[j]` = capacity of facility j

**Complexity:** CFLP is NP-hard (reduction from Set Cover). The LP relaxation provides a lower bound but may have a significant integrality gap due to fractional facility openings (y[j] = 0.4 means "40% open" -- physically meaningless).

**Lagrangian relaxation** is a classical approach: relax the demand constraints into the objective with Lagrange multipliers. The resulting subproblem decomposes by facility (each facility independently decides whether to open based on its contribution to the Lagrangian). The multipliers provide a dual bound tighter than the LP relaxation. In practice, modern MIP solvers (HiGHS, Gurobi) solve medium-scale CFLP instances (hundreds of facilities, thousands of customers) directly.

**Source:** [Cornuejols, Nemhauser & Wolsey, "The Uncapacitated Facility Location Problem" (1990)](https://doi.org/10.1016/B978-0-444-87408-7.50007-1), [Scipbook -- Facility Location Problems](https://scipbook.readthedocs.io/en/latest/flp.html)

### Lot-Sizing Models: Wagner-Whitin and Extensions

The **Dynamic Lot-Sizing Problem** (Wagner & Whitin, 1958) answers: given time-varying demand, when and how much should we produce to minimize the total of setup costs and holding costs?

**Single-item uncapacitated lot-sizing (ULSM):**
```
min  SUM_t [ K*y[t] + h*I[t] ]
s.t. I[t] = I[t-1] + q[t] - d[t]    for all t    (inventory balance)
     q[t] <= M*y[t]                   for all t    (setup linking)
     I[t] >= 0, q[t] >= 0, y[t] in {0,1}
```

**Key property (Wagner-Whitin):** There exists an optimal solution where `I[t-1] * q[t] = 0` for all t -- either entering inventory is zero (produce fresh) or production is zero (use existing stock). This property enables O(n log n) dynamic programming.

**Capacitated lot-sizing (CLSP):** Adds `q[t] <= C*y[t]` where C is production capacity per period. This destroys the Wagner-Whitin property and makes the problem NP-hard. It also creates the key trade-off: capacity forces advance production (building inventory before high-demand periods), but holding costs penalize early production.

**Multi-item extensions** add products competing for shared capacity and setup times, leading to rich scheduling/planning interactions.

The **inventory balance constraint** is a flow conservation equation: `inflow (production) - outflow (demand) = change in stock`. This is the same structure as network flow conservation (practice 036a), applied across time instead of across space.

**Source:** [Wagner & Whitin, "Dynamic Version of the Economic Lot Size Model" (1958)](https://doi.org/10.1287/mnsc.5.1.89), [Pochet & Wolsey, "Production Planning by Mixed Integer Programming" (Springer, 2006)](https://link.springer.com/book/10.1007/0-387-33477-7)

### Multi-Depot Vehicle Routing Problem (MDVRP)

The **MDVRP** extends the classic VRP by allowing multiple depots, each with its own fleet of vehicles. The problem simultaneously decides:
1. **Customer-to-depot assignment:** Which depot serves which customer?
2. **Route construction:** In what order does each vehicle visit its assigned customers?

This is harder than single-depot VRP because the assignment and routing decisions interact: assigning a customer to a nearby depot reduces travel but may create capacity imbalance.

**Relationship to single-depot:** A single-depot CVRP with depot at the centroid of customers is a natural baseline. Multi-depot reduces "stem distance" (depot-to-first-customer travel) by placing depots closer to customer clusters. The difference in total distance quantifies the value of distributed warehousing.

**OR-Tools implementation:** Multi-depot is handled by passing per-vehicle start/end node indices to RoutingIndexManager. Vehicle v starts at `starts[v]` and ends at `ends[v]`. Vehicles from the same depot share the same start/end node but are distinct vehicles in the model.

**Source:** [Toth & Vigo, "Vehicle Routing: Problems, Methods, Applications" (SIAM, 2014)](https://epubs.siam.org/doi/book/10.1137/1.9781611973594), [Google OR-Tools VRP Documentation](https://developers.google.com/optimization/routing/vrp)

### Integration Challenges

Real supply chains don't solve these three problems independently. Decisions at each level interact:

- **Facility location affects routing:** If you open a warehouse in the wrong place, no amount of route optimization can compensate for the long travel distances.
- **Production planning affects inventory:** If you under-produce in advance of a demand spike, you face backlog costs; over-producing creates holding costs.
- **Routing feedback:** If routing from a particular warehouse is consistently expensive, the tactical model should reduce production there and increase at a closer facility.

**Integrated models** combine these decisions but face a combinatorial explosion: binary facility decisions x binary setup decisions x continuous flow/routing = massive MIP. In practice, companies use hierarchical decomposition (solve strategic first, then tactical, then operational) or two-stage stochastic models that combine strategic + tactical under uncertainty.

### Stochastic Supply Chain: Two-Stage Formulation

The **two-stage stochastic supply chain model** combines facility location (Stage 1) with production/allocation (Stage 2) under demand uncertainty:

```
Stage 1 (before demand known): y[j] in {0,1}  -- open facility j?
Stage 2 (after demand scenario s realized): x[s,i,j], slack[s,i], excess[s,j]

min  SUM_j f[j]*y[j] + E_s[ second_stage_cost(y, demand_s) ]
```

**Key insight:** The stochastic model typically opens MORE or DIFFERENT facilities than the deterministic model, because it must hedge against high-demand scenarios. A deterministic model using average demand may under-provision capacity, leading to expensive shortages when demand is high. The stochastic model "pays" for extra capacity in the first stage to reduce expected shortage costs in the second stage.

**Value of Stochastic Solution (VSS):**
```
VSS = EEV - RP
```
Where RP = stochastic optimal cost, and EEV = cost when using the deterministic solution in a stochastic world. VSS quantifies the dollar value of accounting for uncertainty in facility decisions. Studies show VSS of 3-10% in supply chain applications -- representing millions of dollars for large companies.

**Source:** [Birge & Louveaux, "Introduction to Stochastic Programming" (Springer, 2011)](https://link.springer.com/book/10.1007/978-1-4614-0237-4), [Santoso et al., "A Stochastic Programming Approach for Supply Chain Network Design Under Uncertainty" (2005)](https://doi.org/10.1016/j.ejor.2004.01.036)

### Real-World Applications

| Company | Supply Chain OR Application |
|---------|---------------------------|
| **Amazon** | Fulfillment center location, multi-echelon inventory, last-mile routing |
| **Walmart** | Distribution network design, cross-docking optimization, truck routing |
| **FedEx/UPS** | Hub-and-spoke network design, package sorting, vehicle routing with time windows |
| **Procter & Gamble** | Production planning across 130+ plants, inventory optimization |
| **Zara/Inditex** | Fast-fashion supply chain: production timing, distribution from Spain |

These companies use the exact same model families we implement here: CFLP for network design, lot-sizing for production planning, CVRP for distribution, and stochastic models for demand hedging.

**Source:** [Simchi-Levi et al., "Designing and Managing the Supply Chain" (2008)](https://www.mhprofessional.com/designing-and-managing-the-supply-chain-9780073341521-usa)

## Description

Integrate multiple OR techniques from the entire curriculum into a realistic supply chain optimization scenario. Progress through the supply chain hierarchy: strategic facility location, tactical production planning, operational vehicle routing, and an integrated model under demand uncertainty.

### What you'll build

1. **Capacitated Facility Location (CFLP)** -- Decide which warehouses to open from candidate sites, allocate customer demand to facilities respecting capacity, analyze sensitivity to fixed costs and demand changes.
2. **Multi-period Lot-Sizing** -- Plan production quantities across 12 periods at multiple facilities, balancing setup costs, production costs, inventory holding, and backlog penalties. Visualize the production schedule and compare with LP relaxation.
3. **Multi-Depot CVRP** -- Route delivery vehicles from open warehouses to customers using OR-Tools, compare multi-depot vs single-depot routing to quantify the value of distributed warehousing.
4. **Integrated Stochastic Model** -- Combine facility location and production/allocation in a two-stage stochastic program with demand uncertainty. Compute VSS to quantify the value of modeling uncertainty.

## Instructions

### Phase 1: Facility Location & Network Design (~25 min)

**File:** `src/phase1_facility_location.py`

The Capacitated Facility Location Problem is the strategic foundation of supply chain design. You decide which warehouses to open from a set of candidate sites, and how to allocate customer demand to those warehouses, minimizing total fixed + transportation cost while respecting capacity limits. This is an NP-hard MIP combining binary open/close decisions with continuous flow variables.

**What you implement:**

- `build_cflp_model(data)` -- Build the CFLP as a Pyomo MIP. Define binary y[j] variables for facility decisions, continuous x[i,j] for flow allocation, demand satisfaction constraints (equality), and capacity-linking constraints (the Big-M bound `sum_i x[i,j] <= s[j]*y[j]`). Minimize total cost. Solve with HiGHS.
- `sensitivity_analysis(base_data)` -- Scale fixed costs and demands independently, re-solve for each factor, and observe how the optimal network design changes. See the fundamental trade-off: more facilities = higher fixed cost but lower transport.

**Why it matters:** Every supply chain begins with facility location. Getting this wrong means years of suboptimal operations. The binary-continuous structure (open/close + allocate) is the same MIP pattern used in hundreds of real applications. Sensitivity analysis is how you present trade-offs to decision-makers.

### Phase 2: Production Planning & Inventory (~25 min)

**File:** `src/phase2_production_planning.py`

Given open facilities, plan production across 12 time periods. The lot-sizing problem decides when to produce (incurring setup costs) and how much (limited by capacity), while maintaining inventory to meet seasonal demand. Binary setup variables linked to continuous production quantities create the same MIP structure as facility location.

**What you implement:**

- `build_lot_sizing_model(data)` -- Build the multi-facility multi-period MIP. Key elements: inventory balance constraints (flow conservation across time), setup-linking Big-M constraints (`q[j,t] <= C[j]*y[j,t]`), and the four-way cost trade-off (setup vs. production vs. holding vs. backlog).
- `compute_lp_gap(data)` -- Compare MIP optimal with LP relaxation. The LP uses fractional setups (y=0.3 means "30% of a setup"), creating a weak bound. This motivates cutting planes in practice.

**Why it matters:** Production planning is the tactical level of supply chain optimization. The inventory balance equation (flow conservation across time) is a fundamental modeling pattern. The LP relaxation gap illustrates why MIP solving requires more than just LP relaxation. The comparison with lot-for-lot shows optimization savings.

### Phase 3: Vehicle Routing for Distribution (~25 min)

**File:** `src/phase3_distribution_routing.py`

The operational level: route delivery vehicles from warehouses to customers. This phase uses OR-Tools' routing solver for multi-depot CVRP, where each warehouse (depot) has its own fleet. The key OR-Tools feature is RoutingIndexManager with per-vehicle start/end indices.

**What you implement:**

- `solve_multi_depot_cvrp(data)` -- Set up multi-depot CVRP with OR-Tools. Create RoutingIndexManager with per-vehicle start/end depot indices, register distance and demand callbacks, add capacity dimension, configure GUIDED_LOCAL_SEARCH, solve and extract routes.
- `solve_single_depot_cvrp(data)` -- Single-depot baseline (centroid depot). Compare total distance with multi-depot to quantify the value of distributed warehousing.

**Why it matters:** Routing is where supply chain meets the real world -- physical trucks on real roads. The multi-depot vs single-depot comparison directly connects to Phase 1: better facility locations yield better routing. This is the operational payoff of good strategic decisions.

### Phase 4: Integrated Model Under Uncertainty (~25 min)

**File:** `src/phase4_integrated_model.py`

The capstone integration: combine facility location (strategic) with production/allocation (tactical) in a two-stage stochastic program. Stage 1 decides which facilities to open (before demand is known). Stage 2 decides production and allocation for each demand scenario (after demand is revealed).

**What you implement:**

- `build_stochastic_supply_chain(data)` -- Build the two-stage stochastic MIP using Pyomo Blocks (one per scenario). First-stage: binary y[j] at top level. Scenario blocks: continuous flow, shortage, excess variables with capacity-linking constraints referencing y[j]. Probability-weighted objective.
- `build_deterministic_supply_chain(data)` -- Deterministic baseline using expected demand. Single flat model (no scenarios).
- `compute_vss(data)` -- Value of Stochastic Solution: solve stochastic (RP), solve deterministic and evaluate in stochastic world (EEV), compute VSS = EEV - RP. This is the dollar value of modeling uncertainty.

**Why it matters:** This phase synthesizes everything: Pyomo MIP modeling (practice 040b), facility location (Phase 1), stochastic programming (practice 045), and value metrics (VSS). The result is a complete supply chain planning framework that accounts for demand uncertainty. VSS is how you justify the investment in stochastic modeling to stakeholders.

## Motivation

- **Capstone integration:** This practice pulls together facility location (MIP), production planning (MIP), vehicle routing (OR-Tools), and stochastic programming (Pyomo Blocks) into a single coherent supply chain scenario. It demonstrates mastery of the entire OR curriculum.
- **Industry relevance:** Supply chain optimization is the single largest commercial application of operations research. Amazon, Walmart, FedEx, and P&G solve these exact problem families. AutoScheduler.AI's warehouse optimization is a direct application.
- **Hierarchical thinking:** Understanding how strategic, tactical, and operational decisions interact -- and how uncertainty propagates through all levels -- is the mark of a senior OR practitioner.
- **Stochastic value:** Computing VSS is how you make the business case for sophisticated optimization. "Our stochastic model saves $X million compared to planning with average demand" is a concrete, defensible ROI.
- **Career differentiator:** End-to-end supply chain modeling (from network design to vehicle routing) is a rare and valuable skill in both consulting and industry.

## Commands

All commands are run from the `practice_048_supply_chain_capstone/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (Pyomo, HiGHS, OR-Tools, NumPy, Matplotlib) |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_facility_location.py` | Phase 1: Capacitated Facility Location MIP, sensitivity analysis |
| `uv run python src/phase2_production_planning.py` | Phase 2: Multi-period lot-sizing MIP, LP gap, visualization |
| `uv run python src/phase3_distribution_routing.py` | Phase 3: Multi-depot CVRP with OR-Tools, single-depot comparison |
| `uv run python src/phase4_integrated_model.py` | Phase 4: Integrated two-stage stochastic model, VSS computation |

### Debugging

| Command | Description |
|---------|-------------|
| `uv run python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('highs').available())"` | Verify HiGHS solver is available |
| `uv run python -c "from ortools.constraint_solver import pywrapcp; print('OR-Tools OK')"` | Verify OR-Tools routing solver |
| `uv run python -c "import numpy; print(numpy.__version__)"` | Verify NumPy version |

## References

- [Simchi-Levi, Kaminsky & Simchi-Levi, "Designing and Managing the Supply Chain" (McGraw-Hill, 2008)](https://www.mhprofessional.com/designing-and-managing-the-supply-chain-9780073341521-usa) -- The standard supply chain textbook covering all three decision levels
- [Birge & Louveaux, "Introduction to Stochastic Programming" (Springer, 2011)](https://link.springer.com/book/10.1007/978-1-4614-0237-4) -- Two-stage stochastic programming, VSS, EVPI
- [Pochet & Wolsey, "Production Planning by Mixed Integer Programming" (Springer, 2006)](https://link.springer.com/book/10.1007/0-387-33477-7) -- Lot-sizing formulations, Wagner-Whitin, cutting planes
- [Toth & Vigo, "Vehicle Routing: Problems, Methods, Applications" (SIAM, 2014)](https://epubs.siam.org/doi/book/10.1137/1.9781611973594) -- Multi-depot VRP theory and algorithms
- [Scipbook -- Facility Location Problems](https://scipbook.readthedocs.io/en/latest/flp.html) -- CFLP formulation and Python examples
- [Santoso et al., "A Stochastic Programming Approach for Supply Chain Network Design Under Uncertainty" (2005)](https://doi.org/10.1016/j.ejor.2004.01.036) -- Integrated stochastic facility location
- [Wagner & Whitin, "Dynamic Version of the Economic Lot Size Model" (1958)](https://doi.org/10.1287/mnsc.5.1.89) -- Original lot-sizing dynamic programming
- [Google OR-Tools VRP Documentation](https://developers.google.com/optimization/routing/vrp) -- Multi-depot VRP with OR-Tools
- [Pyomo Documentation -- Blocks](https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Blocks.html) -- Indexed blocks for stochastic programming
- [HiGHS Solver](https://highs.dev/) -- LP/MIP solver used throughout

## State

`not-started`
