"""Phase 4: Integrated Model Under Uncertainty.

The capstone integration: combine facility location decisions (Phase 1) with
production planning (Phase 2) under demand uncertainty using two-stage
stochastic programming (from practice 045).

Stage 1 (here-and-now): Decide which facilities to open and their capacities.
  These are strategic, irreversible decisions made BEFORE demand is known.

Stage 2 (recourse): Given the realized demand scenario, decide production
  quantities, inventory levels, and emergency procurement at each open facility.
  These are tactical/operational decisions that adapt to each scenario.

The model structure is the extensive form (deterministic equivalent):
  - First-stage variables appear once (shared across scenarios)
  - Second-stage variables are duplicated per scenario in Pyomo Blocks
  - Objective = first-stage cost + E[second-stage cost]

We compute the Value of Stochastic Solution (VSS) to quantify how much we
gain by modeling uncertainty vs. using deterministic expected demand.

Pyomo patterns used:
  - ConcreteModel with first-stage Vars at top level
  - Block(SCENARIOS, rule=...) for per-scenario second-stage variables
  - Linking constraints between stages (first-stage y in second-stage blocks)
  - Probability-weighted objective
  - Comparison with deterministic equivalent
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np
from typing import NamedTuple


# ============================================================================
# Data structures
# ============================================================================

class IntegratedData(NamedTuple):
    """Problem data for integrated stochastic supply chain model."""
    num_facilities: int         # J: potential facility sites
    num_customers: int          # I: customer demand zones
    num_scenarios: int          # S: demand scenarios
    fixed_costs: np.ndarray     # f[j]: fixed opening cost (J,)
    capacities: np.ndarray      # cap[j]: facility capacity (J,)
    production_costs: np.ndarray  # p[j]: per-unit production cost (J,)
    transport_costs: np.ndarray   # c[i,j]: per-unit transport cost (I x J)
    holding_cost: float         # h: per-unit inventory holding cost
    shortage_cost: float        # s: per-unit shortage penalty
    demand_scenarios: np.ndarray  # d[s,i]: demand of customer i in scenario s (S x I)
    scenario_probs: np.ndarray    # prob[s]: probability of scenario s (S,)


# ============================================================================
# Instance generation
# ============================================================================

def generate_integrated_instance(
    num_facilities: int = 5,
    num_customers: int = 10,
    num_scenarios: int = 5,
    seed: int = 42,
) -> IntegratedData:
    """Generate an integrated stochastic supply chain instance.

    Creates facility location + production data with multiple demand scenarios.
    Scenarios represent different demand realizations (low, below-avg, average,
    above-avg, high) with assigned probabilities.

    Args:
        num_facilities: Potential facility sites.
        num_customers: Customer demand zones.
        num_scenarios: Number of demand scenarios.
        seed: Random seed.

    Returns:
        IntegratedData with all parameters.
    """
    rng = np.random.default_rng(seed)

    # Facility locations and costs
    facility_locs = rng.uniform(0, 100, size=(num_facilities, 2))
    customer_locs = rng.uniform(0, 100, size=(num_customers, 2))

    # Base demand for each customer
    base_demand = rng.uniform(50, 200, size=num_customers).round()

    # Demand scenarios: scale base demand by different factors
    # Probabilities follow a roughly bell-shaped distribution
    demand_factors = np.array([0.6, 0.8, 1.0, 1.2, 1.5])[:num_scenarios]
    scenario_probs = np.array([0.10, 0.25, 0.35, 0.20, 0.10])[:num_scenarios]
    scenario_probs = scenario_probs / scenario_probs.sum()  # normalize

    demand_scenarios = np.zeros((num_scenarios, num_customers))
    for s in range(num_scenarios):
        noise = rng.normal(0, 10, size=num_customers)
        demand_scenarios[s] = np.maximum(
            base_demand * demand_factors[s] + noise, 10
        ).round()

    # Costs
    fixed_costs = rng.uniform(8000, 25000, size=num_facilities).round()
    capacities = rng.uniform(300, 800, size=num_facilities).round()
    production_costs = rng.uniform(5, 20, size=num_facilities).round()

    distances = np.linalg.norm(
        customer_locs[:, np.newaxis, :] - facility_locs[np.newaxis, :, :],
        axis=2,
    )
    transport_costs = (distances * 0.5).round(2)

    holding_cost = 2.0
    shortage_cost = 50.0  # High penalty for unmet demand

    # Compute expected demand
    expected_demand = sum(
        scenario_probs[s] * demand_scenarios[s] for s in range(num_scenarios)
    )

    print(f"  Integrated instance: {num_facilities} facilities, "
          f"{num_customers} customers, {num_scenarios} scenarios")
    print(f"  Expected total demand: {expected_demand.sum():.0f}")
    print(f"  Total capacity (all open): {capacities.sum():.0f}")
    print(f"  Scenario probabilities: {scenario_probs}")

    return IntegratedData(
        num_facilities=num_facilities,
        num_customers=num_customers,
        num_scenarios=num_scenarios,
        fixed_costs=fixed_costs,
        capacities=capacities,
        production_costs=production_costs,
        transport_costs=transport_costs,
        holding_cost=holding_cost,
        shortage_cost=shortage_cost,
        demand_scenarios=demand_scenarios,
        scenario_probs=scenario_probs,
    )


# ============================================================================
# TODO(human): Build the integrated two-stage stochastic model
# ============================================================================

def build_stochastic_supply_chain(data: IntegratedData) -> pyo.ConcreteModel:
    """Build the two-stage stochastic supply chain model (extensive form).

    Stage 1 (here-and-now):
      y[j] in {0,1}  -- open facility j or not
      These decisions are made BEFORE demand is revealed.

    Stage 2 (recourse, per scenario s):
      x[s,i,j] >= 0  -- units shipped from facility j to customer i
      slack[s,i] >= 0 -- unmet demand (shortage) for customer i
      excess[s,j] >= 0 -- unused capacity at facility j

    Objective:
      min  SUM_j f[j]*y[j]   (first-stage: fixed facility costs)
         + SUM_s prob[s] * [
             SUM_j SUM_i (p[j] + c[i,j]) * x[s,i,j]   (production + transport)
           + SUM_j h * excess[s,j]                       (holding)
           + SUM_i shortage_cost * slack[s,i]             (shortage penalty)
           ]

    Subject to (for each scenario s):
      Demand:    SUM_j x[s,i,j] + slack[s,i] >= d[s,i]   for all i
      Capacity:  SUM_i x[s,i,j] <= cap[j] * y[j]         for all j
      Excess:    excess[s,j] = cap[j]*y[j] - SUM_i x[s,i,j]  for all j

    Args:
        data: IntegratedData instance.

    Returns:
        A solved Pyomo ConcreteModel.
    """
    # TODO(human): Build the two-stage stochastic supply chain MIP
    #
    # This is the culmination of the entire OR curriculum. You are combining:
    # - Facility location (Phase 1 / practice 040b) for first-stage binary decisions
    # - Production/allocation for second-stage continuous decisions
    # - Stochastic programming (practice 045) for the scenario-based structure
    #
    # The extensive form creates ONE large MIP with:
    # - First-stage variables y[j] at the top level (shared across all scenarios)
    # - One Block per scenario containing that scenario's second-stage variables
    # - Linking constraints: second-stage capacity depends on first-stage y[j]
    # - Probability-weighted objective
    #
    # STEP 1: Create model and sets
    #   model = pyo.ConcreteModel("StochasticSupplyChain")
    #   model.J = pyo.RangeSet(0, data.num_facilities - 1)
    #   model.I = pyo.RangeSet(0, data.num_customers - 1)
    #   model.S = pyo.RangeSet(0, data.num_scenarios - 1)
    #
    # STEP 2: First-stage variables (BEFORE uncertainty is revealed)
    #   model.y = pyo.Var(model.J, within=pyo.Binary)
    #   These represent the STRATEGIC decisions: which facilities to build.
    #   They must be the same regardless of which demand scenario occurs.
    #   This is the "non-anticipativity" principle from stochastic programming.
    #
    # STEP 3: First-stage cost expression
    #   model.first_stage_cost = sum(data.fixed_costs[j] * model.y[j] for j in model.J)
    #
    # STEP 4: Scenario blocks (one per scenario)
    #   def scenario_rule(block, s):
    #       # Second-stage variables for scenario s
    #       block.x = pyo.Var(model.I, model.J, within=pyo.NonNegativeReals)
    #           # x[i,j] = units shipped from facility j to customer i in scenario s
    #       block.slack = pyo.Var(model.I, within=pyo.NonNegativeReals)
    #           # slack[i] = unmet demand for customer i (shortage)
    #       block.excess = pyo.Var(model.J, within=pyo.NonNegativeReals)
    #           # excess[j] = unused capacity at facility j
    #
    #       # Demand satisfaction: shipped + shortage >= demand
    #       def demand_rule(b, i):
    #           return sum(b.x[i, j] for j in model.J) + b.slack[i] >= data.demand_scenarios[s, i]
    #       block.demand_con = pyo.Constraint(model.I, rule=demand_rule)
    #
    #       # Capacity linking (THIS IS THE KEY CROSS-STAGE CONSTRAINT):
    #       # Total shipment from facility j in this scenario <= capacity * y[j]
    #       # Note: model.y[j] is the FIRST-STAGE variable, referenced inside the block
    #       def capacity_rule(b, j):
    #           return sum(b.x[i, j] for i in model.I) <= data.capacities[j] * model.y[j]
    #       block.capacity_con = pyo.Constraint(model.J, rule=capacity_rule)
    #
    #       # Excess capacity computation
    #       def excess_rule(b, j):
    #           return b.excess[j] == data.capacities[j] * model.y[j] - sum(b.x[i, j] for i in model.I)
    #       block.excess_con = pyo.Constraint(model.J, rule=excess_rule)
    #
    #       # Second-stage cost for this scenario
    #       block.cost = pyo.Expression(expr=
    #           sum((data.production_costs[j] + data.transport_costs[i, j]) * b.x[i, j]
    #               for i in model.I for j in model.J)
    #           + sum(data.holding_cost * b.excess[j] for j in model.J)
    #           + sum(data.shortage_cost * b.slack[i] for i in model.I)
    #       )
    #
    #   model.scenarios = pyo.Block(model.S, rule=scenario_rule)
    #
    # STEP 5: Objective = first-stage cost + E[second-stage cost]
    #   model.obj = pyo.Objective(
    #       expr=model.first_stage_cost
    #            + sum(data.scenario_probs[s] * model.scenarios[s].cost
    #                  for s in model.S),
    #       sense=pyo.minimize,
    #   )
    #
    # STEP 6: Solve
    #   solver = pyo.SolverFactory("highs")
    #   result = solver.solve(model, tee=False)
    #   assert result.solver.termination_condition == TerminationCondition.optimal
    #
    # Return the solved model.
    raise NotImplementedError("TODO(human): build two-stage stochastic supply chain MIP")


# ============================================================================
# TODO(human): Build the deterministic model and compute VSS
# ============================================================================

def build_deterministic_supply_chain(data: IntegratedData) -> pyo.ConcreteModel:
    """Build and solve the deterministic supply chain model using expected demand.

    Uses E[demand] = SUM_s prob[s] * demand[s] as if it were certain.
    This is the "EV problem" -- the model that ignores uncertainty.

    Args:
        data: IntegratedData instance.

    Returns:
        A solved Pyomo ConcreteModel with first-stage y and single-scenario allocation.
    """
    # TODO(human): Build deterministic model using expected demand
    #
    # The deterministic model is the supply chain model with a SINGLE scenario
    # whose demand equals the expected demand E[d] = SUM_s prob_s * d_s.
    #
    # This is structurally the same as Phase 1's CFLP but with production costs:
    #   min  SUM_j f[j]*y[j] + SUM_i SUM_j (p[j]+c[i,j])*x[i,j]
    #        + SUM_j h*excess[j] + SUM_i shortage_cost*slack[i]
    #   s.t. SUM_j x[i,j] + slack[i] >= E[d[i]]       for all i
    #        SUM_i x[i,j] <= cap[j]*y[j]               for all j
    #        excess[j] = cap[j]*y[j] - SUM_i x[i,j]    for all j
    #
    # No scenario Blocks needed -- just a single flat model.
    #
    # IMPLEMENTATION:
    # 1. Compute expected demand: expected_d = SUM_s prob[s] * demand_scenarios[s]
    # 2. Build model with y[j] binary, x[i,j] continuous, slack[i], excess[j]
    # 3. Add demand, capacity, and excess constraints with expected demand
    # 4. Minimize total cost
    # 5. Solve and return
    raise NotImplementedError("TODO(human): deterministic supply chain with expected demand")


def compute_vss(data: IntegratedData) -> dict:
    """Compute the Value of Stochastic Solution (VSS).

    VSS = EEV - RP

    RP:  Solve the stochastic model -> optimal expected cost
    EEV: Solve deterministic model -> get y_det -> fix y=y_det in stochastic
         model -> evaluate expected cost with fixed first-stage decisions

    VSS measures: how much we save by using the stochastic model instead of
    the deterministic "expected value" approach.

    Args:
        data: IntegratedData instance.

    Returns:
        Dict with 'RP', 'EEV', 'VSS', 'stoch_facilities', 'det_facilities'.
    """
    # TODO(human): Compute VSS for the supply chain model
    #
    # This is the ultimate value proposition of stochastic programming.
    # In a supply chain context, VSS answers: "How much money do we save by
    # considering demand uncertainty when deciding where to build warehouses?"
    #
    # STEP 1: Solve the stochastic model (RP)
    #   stoch_model = build_stochastic_supply_chain(data)
    #   RP = pyo.value(stoch_model.obj)
    #   stoch_facs = [j for j in stoch_model.J if pyo.value(stoch_model.y[j]) > 0.5]
    #
    # STEP 2: Solve the deterministic model (EV problem)
    #   det_model = build_deterministic_supply_chain(data)
    #   det_facs = [j for j in det_model.J if pyo.value(det_model.y[j]) > 0.5]
    #
    # STEP 3: Compute EEV (Expected result of EV solution)
    #   Fix the stochastic model's y variables to the deterministic solution:
    #     eval_model = build_stochastic_supply_chain(data)  # fresh copy
    #     -- but before solving, fix y:
    #     Actually, build a new stochastic model, then fix y[j] to det values:
    #       - Build model with same structure as build_stochastic_supply_chain
    #       - Before solving, fix first-stage: model.y[j].fix(det_y_value)
    #       - Solve the resulting model (now it's just an LP since y is fixed)
    #       - EEV = objective value
    #
    #   To fix variables in Pyomo: model.y[j].fix(1) or model.y[j].fix(0)
    #   A fixed variable acts as a constant -- the solver cannot change it.
    #
    # STEP 4: Compute VSS = EEV - RP
    #   VSS should be >= 0. If it's large, the stochastic model adds significant
    #   value over the deterministic approach. In supply chain terms:
    #   "Building warehouses based on average demand costs us $VSS more than
    #    building warehouses that hedge against demand uncertainty."
    #
    # INSIGHT: The stochastic solution typically opens MORE facilities and/or
    # different facilities than the deterministic solution, because it needs
    # to hedge against high-demand scenarios. The deterministic model, seeing
    # only average demand, may under-provision capacity, leading to costly
    # shortages in high-demand scenarios.
    #
    # Return {"RP": ..., "EEV": ..., "VSS": ...,
    #         "stoch_facilities": [...], "det_facilities": [...]}
    raise NotImplementedError("TODO(human): compute VSS for supply chain model")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_stochastic_solution(model: pyo.ConcreteModel, data: IntegratedData) -> None:
    """Print the stochastic supply chain solution."""
    print(f"\n{'=' * 70}")
    print(f"  Stochastic Supply Chain Solution")
    print(f"{'=' * 70}")
    print(f"  Total expected cost: {pyo.value(model.obj):>12,.2f}")

    # First-stage costs
    first_cost = sum(data.fixed_costs[j] * pyo.value(model.y[j]) for j in model.J)
    second_cost = pyo.value(model.obj) - first_cost
    print(f"  First-stage (facility) cost: {first_cost:>12,.2f}")
    print(f"  Expected second-stage cost:  {second_cost:>12,.2f}")

    # Open facilities
    open_facs = [j for j in model.J if pyo.value(model.y[j]) > 0.5]
    print(f"\n  Open facilities: {open_facs} ({len(open_facs)}/{data.num_facilities})")
    for j in open_facs:
        print(f"    Facility {j}: capacity={data.capacities[j]:.0f}, "
              f"fixed cost={data.fixed_costs[j]:.0f}, "
              f"prod cost={data.production_costs[j]:.0f}")

    # Per-scenario summary
    print(f"\n  Per-scenario costs:")
    print(f"  {'Scenario':>8} | {'Prob':>6} | {'Cost':>12} | {'Shortage':>10} | {'Excess Cap':>10}")
    print(f"  {'-'*8}-+-{'-'*6}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")
    for s in model.S:
        blk = model.scenarios[s]
        cost = pyo.value(blk.cost)
        total_shortage = sum(pyo.value(blk.slack[i]) for i in model.I)
        total_excess = sum(pyo.value(blk.excess[j]) for j in model.J)
        print(f"  {s:>8d} | {data.scenario_probs[s]:>6.3f} | {cost:>12,.2f} | "
              f"{total_shortage:>10.1f} | {total_excess:>10.1f}")


def print_vss_results(results: dict) -> None:
    """Print VSS computation results."""
    print(f"\n{'=' * 70}")
    print(f"  Value of Stochastic Solution (VSS)")
    print(f"{'=' * 70}")
    print(f"  RP  (stochastic optimal):     {results['RP']:>12,.2f}")
    print(f"  EEV (deterministic evaluated): {results['EEV']:>12,.2f}")
    print(f"  VSS = EEV - RP:               {results['VSS']:>12,.2f}")
    print(f"\n  Stochastic solution opens: {results['stoch_facilities']}")
    print(f"  Deterministic solution opens:  {results['det_facilities']}")

    if results["VSS"] > 0:
        pct = results["VSS"] / results["EEV"] * 100
        print(f"\n  The stochastic model saves ${results['VSS']:,.2f} ({pct:.1f}%) "
              f"over the deterministic approach.")
        print(f"  This represents the cost of IGNORING demand uncertainty")
        print(f"  when making facility location decisions.")
    else:
        print(f"\n  VSS ~ 0: uncertainty doesn't significantly affect facility decisions.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 4: Integrated Model Under Uncertainty")
    print("=" * 70)

    # --- Generate instance ---
    print("\n--- Generating integrated instance ---")
    data = generate_integrated_instance(
        num_facilities=5,
        num_customers=10,
        num_scenarios=5,
        seed=42,
    )

    # --- Solve stochastic model ---
    print("\n--- Solving stochastic supply chain model ---")
    stoch_model = build_stochastic_supply_chain(data)
    print_stochastic_solution(stoch_model, data)

    # --- Solve deterministic model ---
    print("\n--- Solving deterministic model (expected demand) ---")
    det_model = build_deterministic_supply_chain(data)
    det_facs = [j for j in det_model.J if pyo.value(det_model.y[j]) > 0.5]
    print(f"  Deterministic cost: {pyo.value(det_model.obj):>12,.2f}")
    print(f"  Open facilities: {det_facs}")

    # --- Compute VSS ---
    print("\n--- Computing Value of Stochastic Solution ---")
    vss_results = compute_vss(data)
    print_vss_results(vss_results)

    # --- Experiment: more scenarios ---
    print("\n--- Experiment: 10 scenarios ---")
    data10 = generate_integrated_instance(
        num_facilities=5,
        num_customers=10,
        num_scenarios=5,  # limited by our factor array but shows the pattern
        seed=123,
    )
    vss10 = compute_vss(data10)
    print(f"  VSS with different seed: ${vss10['VSS']:,.2f}")
    print(f"  Stochastic facilities: {vss10['stoch_facilities']}")
    print(f"  Deterministic facilities: {vss10['det_facilities']}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  CAPSTONE SUMMARY: Supply Chain Optimization Hierarchy")
    print(f"{'=' * 70}")
    print(f"  STRATEGIC  (Phase 1): WHERE to open warehouses       -> CFLP MIP")
    print(f"  TACTICAL   (Phase 2): WHEN and HOW MUCH to produce   -> Lot-sizing MIP")
    print(f"  OPERATIONAL(Phase 3): HOW to route deliveries        -> Multi-depot CVRP")
    print(f"  INTEGRATED (Phase 4): ALL levels under UNCERTAINTY   -> Two-stage stochastic MIP")
    print(f"\n  Each level's decisions constrain the next:")
    print(f"    Facility locations -> Production possibilities -> Routing options")
    print(f"    Uncertainty in demand propagates through all levels.")
    print(f"  The VSS quantifies the cost of ignoring this uncertainty.")

    print("\n[Phase 4 complete -- Capstone finished!]")


if __name__ == "__main__":
    main()
