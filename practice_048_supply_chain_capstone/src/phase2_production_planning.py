"""Phase 2: Production Planning & Inventory (Lot-Sizing).

Tactical supply chain decision: when and how much to produce at each facility
across multiple time periods, balancing setup costs, production costs,
inventory holding costs, and backlog penalties.

This is the Multi-Item Capacitated Lot-Sizing Problem (MCLSP) -- a classic
production planning formulation first studied by Wagner & Whitin (1958) for
the single-item case. The MIP formulation uses binary "setup" variables
(y[j,t] = 1 if facility j produces in period t) linked to continuous
production quantities, creating the same binary-continuous structure as CFLP.

Real-world applications: manufacturing scheduling (Toyota, P&G), semiconductor
fab planning, food/beverage production with perishable inventory.

Pyomo patterns used:
  - Multi-indexed sets (facilities x products x time periods)
  - Inventory balance constraints (flow conservation across time)
  - Big-M setup linking (production only when setup is active)
  - Binary + continuous variable interaction
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np
from typing import NamedTuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================================
# Data structures
# ============================================================================

class LotSizingData(NamedTuple):
    """Problem data for multi-period lot-sizing."""
    num_facilities: int         # Number of production facilities (J)
    num_periods: int            # Number of time periods (T)
    demands: np.ndarray         # d[j,t]: demand at facility j in period t (J x T)
    setup_costs: np.ndarray     # K[j]: fixed setup cost per production run at facility j (J,)
    production_costs: np.ndarray  # p[j]: variable cost per unit produced at facility j (J,)
    holding_costs: np.ndarray   # h[j]: inventory holding cost per unit per period at j (J,)
    backlog_costs: np.ndarray   # b[j]: backlog penalty per unit per period at j (J,)
    capacities: np.ndarray      # C[j]: max production capacity per period at j (J,)
    initial_inventory: np.ndarray  # I0[j]: starting inventory at facility j (J,)


# ============================================================================
# Instance generation
# ============================================================================

def generate_lot_sizing_instance(
    num_facilities: int = 3,
    num_periods: int = 12,
    seed: int = 42,
) -> LotSizingData:
    """Generate a multi-facility lot-sizing instance.

    Demand follows a seasonal pattern with random noise to make the problem
    interesting: some periods have high demand requiring advance production
    and inventory buildup.

    Args:
        num_facilities: Number of production facilities.
        num_periods: Number of planning periods (e.g., months).
        seed: Random seed.

    Returns:
        LotSizingData with all problem parameters.
    """
    rng = np.random.default_rng(seed)

    # Seasonal demand pattern: base + seasonal + noise
    t_arr = np.arange(num_periods)
    base_demand = 100.0
    seasonal = 40.0 * np.sin(2 * np.pi * t_arr / num_periods)  # one full cycle

    demands = np.zeros((num_facilities, num_periods))
    for j in range(num_facilities):
        noise = rng.normal(0, 15, size=num_periods)
        demands[j] = np.maximum(base_demand + seasonal + noise, 20.0).round()

    # Costs vary by facility (different plants have different economics)
    setup_costs = rng.uniform(500, 1500, size=num_facilities).round()
    production_costs = rng.uniform(5, 15, size=num_facilities).round()
    holding_costs = rng.uniform(1, 4, size=num_facilities).round()
    backlog_costs = holding_costs * 5  # backlog is 5x holding cost (incentivizes meeting demand)

    # Capacity: 1.3x average demand (some slack but not enough to avoid planning)
    avg_demand = demands.mean(axis=1)
    capacities = (1.3 * avg_demand).round()

    initial_inventory = np.zeros(num_facilities)

    print(f"  Lot-sizing instance: {num_facilities} facilities, {num_periods} periods")
    print(f"  Demand range: [{demands.min():.0f}, {demands.max():.0f}]")
    print(f"  Setup costs: {setup_costs}")
    print(f"  Capacities: {capacities}")

    return LotSizingData(
        num_facilities=num_facilities,
        num_periods=num_periods,
        demands=demands,
        setup_costs=setup_costs,
        production_costs=production_costs,
        holding_costs=holding_costs,
        backlog_costs=backlog_costs,
        capacities=capacities,
        initial_inventory=initial_inventory,
    )


# ============================================================================
# TODO(human): Build and solve the lot-sizing MIP
# ============================================================================

def build_lot_sizing_model(data: LotSizingData) -> pyo.ConcreteModel:
    """Build a multi-facility multi-period lot-sizing MIP.

    The formulation is:
        min  SUM_j SUM_t [ K[j]*y[j,t] + p[j]*q[j,t] + h[j]*I[j,t] + b[j]*B[j,t] ]
        s.t.
          Inventory balance (flow conservation):
            I[j,t] - B[j,t] = I[j,t-1] - B[j,t-1] + q[j,t] - d[j,t]   for all j, t
          Setup-production linking:
            q[j,t] <= C[j] * y[j,t]                                      for all j, t
          Production capacity:
            q[j,t] <= C[j]                                                for all j, t
          Variable domains:
            y[j,t] in {0,1},  q[j,t] >= 0,  I[j,t] >= 0,  B[j,t] >= 0

    Where:
      y[j,t] = 1 if facility j produces in period t (setup/binary)
      q[j,t] = quantity produced at facility j in period t (continuous)
      I[j,t] = inventory at facility j at end of period t (continuous)
      B[j,t] = backlog at facility j at end of period t (continuous)

    Args:
        data: LotSizingData instance.

    Returns:
        A solved Pyomo ConcreteModel.
    """
    # TODO(human): Build the lot-sizing MIP
    #
    # This is the capacitated lot-sizing problem (CLSP), a generalization of
    # the classic Wagner-Whitin model. The key insight is the INVENTORY BALANCE
    # equation, which is a flow conservation constraint across time:
    #
    #   (ending inventory) - (ending backlog) =
    #       (previous inventory) - (previous backlog) + (production) - (demand)
    #
    # Or equivalently: I[j,t] - B[j,t] = I[j,t-1] - B[j,t-1] + q[j,t] - d[j,t]
    #
    # This is analogous to "flow in = flow out" in network optimization (practice 036a).
    # Time periods are nodes, production is inflow, demand is outflow, and inventory
    # is the arc connecting consecutive periods.
    #
    # STEP 1: Create model and index sets
    #   model = pyo.ConcreteModel("LotSizing")
    #   model.J = pyo.RangeSet(0, data.num_facilities - 1)   # facilities
    #   model.T = pyo.RangeSet(0, data.num_periods - 1)      # time periods
    #
    # STEP 2: Parameters
    #   model.K = pyo.Param(model.J, initialize=...)   # setup costs
    #   model.p = pyo.Param(model.J, initialize=...)   # production costs
    #   model.h = pyo.Param(model.J, initialize=...)   # holding costs
    #   model.b = pyo.Param(model.J, initialize=...)   # backlog costs
    #   model.C = pyo.Param(model.J, initialize=...)   # capacities
    #   model.d = pyo.Param(model.J, model.T, initialize=lambda m, j, t: float(data.demands[j, t]))
    #   model.I0 = pyo.Param(model.J, initialize=...)  # initial inventory
    #
    # STEP 3: Decision variables
    #   model.y = pyo.Var(model.J, model.T, within=pyo.Binary)
    #     y[j,t] = 1 means "set up production at facility j in period t"
    #     This incurs the fixed setup cost K[j] (e.g., machine changeover, cleaning)
    #
    #   model.q = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    #     q[j,t] = production quantity. Can only be > 0 if y[j,t] = 1.
    #
    #   model.Inv = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    #     I[j,t] = inventory at end of period t (positive stock)
    #
    #   model.Back = pyo.Var(model.J, model.T, within=pyo.NonNegativeReals)
    #     B[j,t] = backlog at end of period t (unmet demand carried forward)
    #
    # STEP 4: Inventory balance constraints
    #   For t = 0 (first period), use initial inventory:
    #     model.Inv[j,0] - model.Back[j,0] = model.I0[j] + model.q[j,0] - model.d[j,0]
    #   For t >= 1:
    #     model.Inv[j,t] - model.Back[j,t] = model.Inv[j,t-1] - model.Back[j,t-1] + model.q[j,t] - model.d[j,t]
    #
    #   These are EQUALITY constraints (not inequalities) because inventory and
    #   backlog variables are separate non-negative variables. The difference
    #   I[j,t] - B[j,t] is the "net inventory position" -- positive means surplus,
    #   negative means backlog. But since both I and B are >= 0, the optimizer
    #   will naturally set at most one of them to be positive (both positive
    #   would waste cost).
    #
    # STEP 5: Setup-linking constraints (Big-M)
    #   q[j,t] <= C[j] * y[j,t]    for all j, t
    #
    #   When y[j,t] = 0 (no setup), production q[j,t] is forced to 0.
    #   When y[j,t] = 1 (setup active), production is bounded by capacity C[j].
    #   This is exactly the same Big-M pattern as CFLP's capacity-linking.
    #
    # STEP 6: Objective -- minimize total cost across all facilities and periods
    #   min SUM_j SUM_t [ K[j]*y[j,t] + p[j]*q[j,t] + h[j]*Inv[j,t] + b[j]*Back[j,t] ]
    #
    #   The four cost components create a rich trade-off:
    #   - Setup cost pushes toward fewer, larger production batches
    #   - Holding cost pushes toward producing just-in-time
    #   - Backlog cost pushes toward meeting demand on time
    #   - Production cost is proportional to volume (less interesting for decisions)
    #
    # STEP 7: Solve with HiGHS
    #   solver = pyo.SolverFactory("highs")
    #   result = solver.solve(model, tee=False)
    #   assert result.solver.termination_condition == TerminationCondition.optimal
    #
    # Return the solved model.
    raise NotImplementedError("TODO(human): build lot-sizing MIP with setup, inventory, backlog")


# ============================================================================
# TODO(human): LP relaxation gap analysis
# ============================================================================

def compute_lp_gap(data: LotSizingData) -> dict:
    """Compare MIP optimal with LP relaxation bound.

    The LP relaxation of lot-sizing problems often has a large integrality
    gap because the LP can use fractional setup variables (e.g., y=0.3 means
    "30% of a setup"). This gap motivates cutting plane methods that
    tighten the formulation.

    Args:
        data: LotSizingData instance.

    Returns:
        Dict with 'mip_obj', 'lp_obj', 'gap_percent'.
    """
    # TODO(human): Compute and compare MIP vs LP relaxation
    #
    # The LP relaxation gap for lot-sizing is notoriously large because:
    # - The LP solution uses fractional setups: y[j,t] = 0.3 means "produce
    #   at 30% capacity with 30% of the setup cost" -- physically meaningless
    #   but mathematically valid in the relaxation.
    # - In reality, you either set up the machine (y=1) or you don't (y=0).
    #   Fractional values have no physical interpretation.
    #
    # IMPLEMENTATION:
    # 1. Solve the MIP normally: build_lot_sizing_model(data) -> mip_obj
    # 2. Build the model again, then relax binary variables:
    #      for j in model.J:
    #          for t in model.T:
    #              model.y[j, t].domain = pyo.UnitInterval
    #    Solve this LP relaxation -> lp_obj
    # 3. Compute gap = (mip_obj - lp_obj) / mip_obj * 100
    #
    # The gap tells you how much the "easy" LP bound underestimates the true
    # cost. A large gap means: (a) the LP relaxation is weak, and (b) the
    # MIP solver has to work harder (more branching) to close the gap.
    #
    # In production, cutting planes (like (l,S) inequalities for lot-sizing)
    # tighten the LP relaxation, shrinking the gap and speeding up solving.
    #
    # Return: {"mip_obj": ..., "lp_obj": ..., "gap_percent": ...}
    raise NotImplementedError("TODO(human): compute LP relaxation gap")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_lot_sizing_solution(model: pyo.ConcreteModel, data: LotSizingData) -> None:
    """Print lot-sizing solution: production schedule, inventory, costs."""
    print(f"\n{'=' * 70}")
    print(f"  Lot-Sizing Solution")
    print(f"{'=' * 70}")
    print(f"  Total cost: {pyo.value(model.obj):>12,.2f}")

    # Cost breakdown
    setup_total = sum(
        pyo.value(model.K[j]) * pyo.value(model.y[j, t])
        for j in model.J for t in model.T
    )
    prod_total = sum(
        pyo.value(model.p[j]) * pyo.value(model.q[j, t])
        for j in model.J for t in model.T
    )
    hold_total = sum(
        pyo.value(model.h[j]) * pyo.value(model.Inv[j, t])
        for j in model.J for t in model.T
    )
    back_total = sum(
        pyo.value(model.b[j]) * pyo.value(model.Back[j, t])
        for j in model.J for t in model.T
    )
    total = pyo.value(model.obj)
    print(f"  Setup cost:      {setup_total:>10,.2f}  ({setup_total/total*100:.1f}%)")
    print(f"  Production cost: {prod_total:>10,.2f}  ({prod_total/total*100:.1f}%)")
    print(f"  Holding cost:    {hold_total:>10,.2f}  ({hold_total/total*100:.1f}%)")
    print(f"  Backlog cost:    {back_total:>10,.2f}  ({back_total/total*100:.1f}%)")

    for j in model.J:
        print(f"\n  Facility {j} (setup cost={pyo.value(model.K[j]):.0f}, "
              f"capacity={pyo.value(model.C[j]):.0f}):")
        print(f"    {'Period':>6} | {'Demand':>8} | {'Produce':>8} | {'Inv':>8} | {'Backlog':>8} | {'Setup':>6}")
        print(f"    {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
        for t in model.T:
            demand = pyo.value(model.d[j, t])
            prod = pyo.value(model.q[j, t])
            inv = pyo.value(model.Inv[j, t])
            back = pyo.value(model.Back[j, t])
            setup = "YES" if pyo.value(model.y[j, t]) > 0.5 else "no"
            print(f"    {t:>6d} | {demand:>8.0f} | {prod:>8.0f} | {inv:>8.0f} | {back:>8.0f} | {setup:>6}")
        num_setups = sum(1 for t in model.T if pyo.value(model.y[j, t]) > 0.5)
        print(f"    Total setups: {num_setups} / {data.num_periods}")


def visualize_production_schedule(model: pyo.ConcreteModel, data: LotSizingData) -> None:
    """Create a production schedule visualization and save to file."""
    fig, axes = plt.subplots(data.num_facilities, 1, figsize=(12, 3 * data.num_facilities),
                             sharex=True)
    if data.num_facilities == 1:
        axes = [axes]

    periods = list(model.T)

    for j, ax in enumerate(axes):
        demands = [pyo.value(model.d[j, t]) for t in periods]
        production = [pyo.value(model.q[j, t]) for t in periods]
        inventory = [pyo.value(model.Inv[j, t]) for t in periods]
        backlog = [pyo.value(model.Back[j, t]) for t in periods]

        x = np.arange(len(periods))
        width = 0.35

        ax.bar(x - width / 2, demands, width, label="Demand", alpha=0.7, color="steelblue")
        ax.bar(x + width / 2, production, width, label="Production", alpha=0.7, color="coral")
        ax.plot(x, inventory, "g-o", label="Inventory", markersize=4)
        if any(b > 0.1 for b in backlog):
            ax.plot(x, backlog, "r--x", label="Backlog", markersize=4)

        # Mark setup periods
        for t_idx, t in enumerate(periods):
            if pyo.value(model.y[j, t]) > 0.5:
                ax.axvline(x=t_idx, color="gray", linestyle=":", alpha=0.3)

        ax.set_ylabel(f"Facility {j}")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"Facility {j}: setup cost={pyo.value(model.K[j]):.0f}, "
                     f"capacity={pyo.value(model.C[j]):.0f}")

    axes[-1].set_xlabel("Period")
    axes[-1].set_xticks(np.arange(len(periods)))
    axes[-1].set_xticklabels([str(t) for t in periods])
    plt.tight_layout()
    plt.savefig("production_schedule.png", dpi=120)
    print("  Saved production_schedule.png")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 2: Production Planning & Inventory (Lot-Sizing)")
    print("=" * 70)

    # --- Generate and solve ---
    print("\n--- Generating lot-sizing instance ---")
    data = generate_lot_sizing_instance(num_facilities=3, num_periods=12, seed=42)

    print("\n--- Solving lot-sizing MIP ---")
    model = build_lot_sizing_model(data)
    print_lot_sizing_solution(model, data)

    # --- Visualize ---
    print("\n--- Visualizing production schedule ---")
    visualize_production_schedule(model, data)

    # --- LP relaxation gap ---
    print("\n--- LP Relaxation Gap ---")
    gap_results = compute_lp_gap(data)
    print(f"  MIP optimal:     {gap_results['mip_obj']:>12,.2f}")
    print(f"  LP relaxation:   {gap_results['lp_obj']:>12,.2f}")
    print(f"  Integrality gap: {gap_results['gap_percent']:>11.2f}%")

    # --- Compare: lot-for-lot vs optimized ---
    print(f"\n--- Comparison: Lot-for-Lot (produce exactly demand) ---")
    lfl_cost = compute_lot_for_lot_cost(data)
    mip_cost = gap_results["mip_obj"]
    savings = (lfl_cost - mip_cost) / lfl_cost * 100
    print(f"  Lot-for-lot cost: {lfl_cost:>12,.2f}")
    print(f"  Optimized cost:   {mip_cost:>12,.2f}")
    print(f"  Savings:          {savings:>11.2f}%")

    print("\n[Phase 2 complete]")


def compute_lot_for_lot_cost(data: LotSizingData) -> float:
    """Compute cost of naive lot-for-lot policy: produce exactly demand each period.

    This is the simplest feasible policy -- no inventory buildup, no batching.
    Every period with positive demand incurs a setup. This serves as a baseline
    to show how much the MIP optimization saves.
    """
    total_cost = 0.0
    for j in range(data.num_facilities):
        for t in range(data.num_periods):
            d = data.demands[j, t]
            if d > 0:
                total_cost += data.setup_costs[j]  # setup every period
                total_cost += data.production_costs[j] * d
    return total_cost


if __name__ == "__main__":
    main()
