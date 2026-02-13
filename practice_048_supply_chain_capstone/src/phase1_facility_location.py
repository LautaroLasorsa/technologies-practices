"""Phase 1: Facility Location & Network Design.

Strategic supply chain decision: where to open warehouses to minimize total
cost (fixed opening costs + variable transportation costs) while meeting
customer demand and respecting facility capacity limits.

This is the Capacitated Facility Location Problem (CFLP) -- an NP-hard MIP
that forms the backbone of supply chain network design. Companies like Amazon,
Walmart, and FedEx solve variants of this problem to decide where to build
fulfillment centers.

Pyomo patterns used:
  - ConcreteModel with Sets, Params, Vars, Constraints, Objective
  - Binary variables (within=pyo.Binary) for open/close decisions
  - Continuous variables for flow allocation
  - Big-M capacity linking constraints
  - SolverFactory('highs') for MIP solving
  - Sensitivity analysis via parameter perturbation
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np
from typing import NamedTuple


# ============================================================================
# Data structures
# ============================================================================

class FacilityLocationData(NamedTuple):
    """Problem data for Capacitated Facility Location."""
    num_facilities: int           # Number of potential warehouse sites
    num_customers: int            # Number of customer demand zones
    fixed_costs: np.ndarray       # f[j]: fixed cost of opening facility j (shape: J)
    capacities: np.ndarray        # s[j]: max capacity of facility j (shape: J)
    demands: np.ndarray           # d[i]: demand of customer i (shape: I)
    transport_costs: np.ndarray   # c[i,j]: cost to serve customer i from facility j (shape: I x J)


# ============================================================================
# Instance generation
# ============================================================================

def generate_cflp_instance(
    num_facilities: int = 8,
    num_customers: int = 20,
    seed: int = 42,
) -> FacilityLocationData:
    """Generate a random CFLP instance with geographic structure.

    Facilities and customers are placed randomly in a 100x100 grid.
    Transport costs are proportional to Euclidean distance times demand.
    Fixed costs and capacities vary to create interesting trade-offs.

    Args:
        num_facilities: Number of potential facility sites (J).
        num_customers: Number of customer zones (I).
        seed: Random seed for reproducibility.

    Returns:
        FacilityLocationData with all problem parameters.
    """
    rng = np.random.default_rng(seed)

    # Random locations in [0, 100] x [0, 100]
    facility_locations = rng.uniform(0, 100, size=(num_facilities, 2))
    customer_locations = rng.uniform(0, 100, size=(num_customers, 2))

    # Customer demands: between 50 and 200 units
    demands = rng.integers(50, 201, size=num_customers).astype(float)

    # Facility capacities: 2x-4x average demand per facility (some slack)
    total_demand = demands.sum()
    avg_capacity = total_demand / num_facilities
    capacities = rng.uniform(2.0 * avg_capacity, 4.0 * avg_capacity, size=num_facilities)
    capacities = np.round(capacities).astype(float)

    # Fixed costs: correlated with capacity (larger facility = higher fixed cost)
    # Range roughly 5000-25000
    fixed_costs = 5000 + 150 * capacities + rng.normal(0, 1000, size=num_facilities)
    fixed_costs = np.maximum(fixed_costs, 1000).round()

    # Transport costs: distance * cost_rate (per unit shipped)
    # c[i,j] = distance(i,j) * demand[i] * rate
    cost_rate = 0.5  # $/unit/km
    distances = np.linalg.norm(
        customer_locations[:, np.newaxis, :] - facility_locations[np.newaxis, :, :],
        axis=2,
    )
    transport_costs = distances * cost_rate

    print(f"  CFLP instance: {num_facilities} facilities, {num_customers} customers")
    print(f"  Total demand: {total_demand:.0f}")
    print(f"  Total capacity (all open): {capacities.sum():.0f}")
    print(f"  Fixed costs range: [{fixed_costs.min():.0f}, {fixed_costs.max():.0f}]")

    return FacilityLocationData(
        num_facilities=num_facilities,
        num_customers=num_customers,
        fixed_costs=fixed_costs,
        capacities=capacities,
        demands=demands,
        transport_costs=transport_costs,
    )


# ============================================================================
# TODO(human): Build and solve the CFLP MIP
# ============================================================================

def build_cflp_model(data: FacilityLocationData) -> pyo.ConcreteModel:
    """Build the Capacitated Facility Location Problem as a Pyomo MIP.

    The CFLP is:
        min  SUM_j f[j]*y[j] + SUM_i SUM_j c[i,j]*x[i,j]
        s.t. SUM_j x[i,j] = d[i]          for all i  (demand satisfaction)
             SUM_i x[i,j] <= s[j]*y[j]     for all j  (capacity + linking)
             x[i,j] >= 0                   for all i,j
             y[j] in {0, 1}                for all j

    Where:
      y[j] = 1 if facility j is opened, 0 otherwise (Binary)
      x[i,j] = amount shipped from facility j to customer i (Continuous)
      f[j] = fixed cost of opening facility j
      c[i,j] = per-unit transport cost from j to i
      d[i] = demand of customer i
      s[j] = capacity of facility j

    Args:
        data: FacilityLocationData instance.

    Returns:
        A solved Pyomo ConcreteModel.
    """
    # TODO(human): Build and solve the CFLP MIP model
    #
    # This is the canonical facility location formulation, one of the most
    # important models in supply chain optimization. Follow these steps:
    #
    # STEP 1: Create model and index sets
    #   model = pyo.ConcreteModel("CFLP")
    #   model.I = pyo.RangeSet(0, data.num_customers - 1)    # customer indices
    #   model.J = pyo.RangeSet(0, data.num_facilities - 1)   # facility indices
    #
    # STEP 2: Create parameters from the data arrays
    #   Use pyo.Param with initialize= that reads from data.fixed_costs,
    #   data.capacities, data.demands, data.transport_costs. For example:
    #     model.f = pyo.Param(model.J, initialize=lambda m, j: float(data.fixed_costs[j]))
    #     model.s = pyo.Param(model.J, initialize=lambda m, j: float(data.capacities[j]))
    #     model.d = pyo.Param(model.I, initialize=lambda m, i: float(data.demands[i]))
    #     model.c = pyo.Param(model.I, model.J, initialize=lambda m, i, j: float(data.transport_costs[i, j]))
    #
    # STEP 3: Decision variables
    #   model.y = pyo.Var(model.J, within=pyo.Binary)
    #     y[j] = 1 means open facility j (strategic, binary decision)
    #   model.x = pyo.Var(model.I, model.J, within=pyo.NonNegativeReals)
    #     x[i,j] = units shipped from facility j to customer i (continuous flow)
    #
    #   The combination of binary y and continuous x makes this a MIP.
    #   The binary variables represent the strategic "where to build" decisions.
    #   The continuous variables represent the operational "how to allocate" flows.
    #
    # STEP 4: Objective -- minimize total cost
    #   Total cost = sum of fixed opening costs + sum of transport costs
    #   model.obj = pyo.Objective(
    #       expr=sum(model.f[j] * model.y[j] for j in model.J)
    #            + sum(model.c[i,j] * model.x[i,j] for i in model.I for j in model.J),
    #       sense=pyo.minimize
    #   )
    #
    # STEP 5: Demand satisfaction constraints (one per customer)
    #   For each customer i, total flow from all facilities must equal demand:
    #   def demand_rule(m, i):
    #       return sum(m.x[i, j] for j in m.J) == m.d[i]
    #   model.demand_con = pyo.Constraint(model.I, rule=demand_rule)
    #
    #   Note: we use equality (==) not inequality (>=) because over-supplying
    #   a customer adds cost without benefit (transport costs are positive).
    #
    # STEP 6: Capacity-linking constraints (one per facility)
    #   If facility j is closed (y[j]=0), no flow can pass through it.
    #   If open (y[j]=1), total flow is limited by capacity s[j]:
    #   def capacity_rule(m, j):
    #       return sum(m.x[i, j] for i in m.I) <= m.s[j] * m.y[j]
    #   model.capacity_con = pyo.Constraint(model.J, rule=capacity_rule)
    #
    #   This is the KEY linking constraint: s[j]*y[j] is a big-M style
    #   bound. When y[j]=0, the RHS is 0, forcing all x[i,j]=0 for that j.
    #   When y[j]=1, the RHS is s[j], allowing flow up to capacity.
    #
    # STEP 7: Solve with HiGHS
    #   solver = pyo.SolverFactory("highs")
    #   result = solver.solve(model, tee=False)
    #   assert result.solver.termination_condition == TerminationCondition.optimal
    #
    # Return the solved model.
    raise NotImplementedError("TODO(human): build CFLP MIP with binary facility decisions and continuous flow")


# ============================================================================
# TODO(human): Sensitivity analysis
# ============================================================================

def sensitivity_analysis(base_data: FacilityLocationData) -> dict:
    """Analyze how the optimal network design changes with parameters.

    Perform two sensitivity experiments:
    1. Scale all fixed costs by factors [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
       and observe how the number of open facilities changes.
    2. Scale all demands by factors [0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
       and observe how cost and facility selection change.

    Args:
        base_data: The base CFLP instance.

    Returns:
        Dictionary with keys 'fixed_cost_results' and 'demand_results',
        each a list of dicts with 'factor', 'num_open', 'total_cost', 'open_facilities'.
    """
    # TODO(human): Run sensitivity analysis on fixed costs and demands
    #
    # This exercise teaches you about the economics of facility location:
    # - When fixed costs are LOW: it's cheap to open many small, nearby facilities
    #   -> more facilities open, lower transport costs, higher total fixed costs
    # - When fixed costs are HIGH: consolidate into fewer, larger facilities
    #   -> fewer facilities open, higher transport costs, lower total fixed costs
    # - When demand INCREASES: existing facilities may become capacity-constrained
    #   -> need to open additional facilities to handle overflow
    # - When demand DECREASES: some facilities become unnecessary
    #   -> close the least cost-effective ones
    #
    # IMPLEMENTATION:
    # For each sensitivity factor in the given lists:
    #   1. Create a modified copy of base_data with scaled fixed_costs (or demands)
    #      Use base_data._replace(fixed_costs=base_data.fixed_costs * factor)
    #   2. Build and solve the CFLP with build_cflp_model(modified_data)
    #   3. Extract: number of open facilities, which ones are open, total cost
    #      Open facilities: [j for j in model.J if pyo.value(model.y[j]) > 0.5]
    #      Total cost: pyo.value(model.obj)
    #   4. Store results
    #
    # INSIGHT: The trade-off curve between fixed costs and transport costs is
    # the fundamental economic insight of facility location. More facilities =
    # higher fixed cost but lower transport cost. The optimizer finds the sweet spot.
    #
    # Return results dict with both experiment results.
    raise NotImplementedError("TODO(human): sensitivity analysis on fixed costs and demands")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_cflp_solution(model: pyo.ConcreteModel, data: FacilityLocationData) -> None:
    """Print the CFLP solution details."""
    print(f"\n{'=' * 70}")
    print(f"  CFLP Solution")
    print(f"{'=' * 70}")
    print(f"  Total cost: {pyo.value(model.obj):>12,.2f}")

    # Separate fixed and transport costs
    fixed_cost = sum(
        pyo.value(model.f[j]) * pyo.value(model.y[j])
        for j in model.J
    )
    transport_cost = sum(
        pyo.value(model.c[i, j]) * pyo.value(model.x[i, j])
        for i in model.I for j in model.J
    )
    print(f"  Fixed cost:     {fixed_cost:>12,.2f}  ({fixed_cost / pyo.value(model.obj) * 100:.1f}%)")
    print(f"  Transport cost: {transport_cost:>12,.2f}  ({transport_cost / pyo.value(model.obj) * 100:.1f}%)")

    # Open facilities
    open_facilities = [j for j in model.J if pyo.value(model.y[j]) > 0.5]
    print(f"\n  Open facilities: {len(open_facilities)} / {data.num_facilities}")
    for j in open_facilities:
        used_capacity = sum(pyo.value(model.x[i, j]) for i in model.I)
        cap = pyo.value(model.s[j])
        utilization = used_capacity / cap * 100 if cap > 0 else 0
        print(
            f"    Facility {j:2d}: capacity {cap:>7.0f}, "
            f"used {used_capacity:>7.0f} ({utilization:>5.1f}%), "
            f"fixed cost {pyo.value(model.f[j]):>8.0f}"
        )

    # Customer assignments summary
    print(f"\n  Customer assignments (multi-sourced customers):")
    multi_sourced = 0
    for i in model.I:
        sources = [
            (j, pyo.value(model.x[i, j]))
            for j in model.J
            if pyo.value(model.x[i, j]) > 1e-6
        ]
        if len(sources) > 1:
            multi_sourced += 1
            src_str = ", ".join(f"F{j}:{amt:.0f}" for j, amt in sources)
            print(f"    Customer {i:2d} (demand {pyo.value(model.d[i]):.0f}): {src_str}")
    print(f"  Multi-sourced customers: {multi_sourced} / {data.num_customers}")


def print_sensitivity_results(results: dict) -> None:
    """Print sensitivity analysis results."""
    print(f"\n{'=' * 70}")
    print(f"  Sensitivity Analysis: Fixed Costs")
    print(f"{'=' * 70}")
    print(f"  {'Factor':>8} | {'# Open':>7} | {'Total Cost':>12} | Open Facilities")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*12}-+-{'-'*30}")
    for r in results["fixed_cost_results"]:
        facs = ", ".join(str(f) for f in r["open_facilities"])
        print(f"  {r['factor']:>8.2f} | {r['num_open']:>7d} | {r['total_cost']:>12,.2f} | {facs}")

    print(f"\n{'=' * 70}")
    print(f"  Sensitivity Analysis: Demand Scaling")
    print(f"{'=' * 70}")
    print(f"  {'Factor':>8} | {'# Open':>7} | {'Total Cost':>12} | Open Facilities")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*12}-+-{'-'*30}")
    for r in results["demand_results"]:
        facs = ", ".join(str(f) for f in r["open_facilities"])
        print(f"  {r['factor']:>8.2f} | {r['num_open']:>7d} | {r['total_cost']:>12,.2f} | {facs}")


def get_open_facilities(model: pyo.ConcreteModel) -> list[int]:
    """Extract list of open facility indices from a solved CFLP model."""
    return [j for j in model.J if pyo.value(model.y[j]) > 0.5]


def get_facility_flows(model: pyo.ConcreteModel) -> dict[int, float]:
    """Extract total flow through each open facility."""
    flows = {}
    for j in model.J:
        if pyo.value(model.y[j]) > 0.5:
            flows[j] = sum(pyo.value(model.x[i, j]) for i in model.I)
    return flows


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 1: Facility Location & Network Design (CFLP)")
    print("=" * 70)

    # --- Generate and solve the base instance ---
    print("\n--- Generating CFLP instance ---")
    data = generate_cflp_instance(num_facilities=8, num_customers=20, seed=42)

    print("\n--- Solving CFLP ---")
    model = build_cflp_model(data)
    print_cflp_solution(model, data)

    # --- LP relaxation comparison ---
    print(f"\n--- LP Relaxation Bound ---")
    model_lp = build_cflp_model(data)
    # Relax binary variables to continuous [0, 1]
    for j in model_lp.J:
        model_lp.y[j].domain = pyo.UnitInterval
    solver = pyo.SolverFactory("highs")
    solver.solve(model_lp, tee=False)
    lp_bound = pyo.value(model_lp.obj)
    mip_obj = pyo.value(model.obj)
    gap = (mip_obj - lp_bound) / mip_obj * 100
    print(f"  LP relaxation bound: {lp_bound:>12,.2f}")
    print(f"  MIP optimal:         {mip_obj:>12,.2f}")
    print(f"  Integrality gap:     {gap:>11.2f}%")

    # --- Sensitivity analysis ---
    print("\n--- Sensitivity Analysis ---")
    results = sensitivity_analysis(data)
    print_sensitivity_results(results)

    # --- Export open facilities for Phase 3 (multi-depot VRP) ---
    open_facs = get_open_facilities(model)
    print(f"\n  Open facilities for downstream phases: {open_facs}")

    print("\n[Phase 1 complete]")


if __name__ == "__main__":
    main()
