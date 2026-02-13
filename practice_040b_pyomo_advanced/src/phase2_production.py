"""Phase 2: Multi-Period Production Planning with Pyomo.

A manufacturing plant produces multiple products over several time periods.
Decisions: how much to produce and how much inventory to carry forward.
Objective: minimize total production + holding cost subject to capacity.
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ── Problem data ───────────────────────────────────────────────────────────

PRODUCTION_DATA = {
    "products": ["Widget", "Gadget", "Gizmo"],
    "n_periods": 6,
    # Demand per product per period (1-indexed)
    "demand": {
        ("Widget", 1): 100, ("Widget", 2): 120, ("Widget", 3): 150,
        ("Widget", 4): 130, ("Widget", 5): 110, ("Widget", 6): 140,
        ("Gadget", 1): 80,  ("Gadget", 2): 90,  ("Gadget", 3): 110,
        ("Gadget", 4): 100, ("Gadget", 5): 95,  ("Gadget", 6): 85,
        ("Gizmo", 1): 60,   ("Gizmo", 2): 70,   ("Gizmo", 3): 65,
        ("Gizmo", 4): 80,   ("Gizmo", 5): 75,   ("Gizmo", 6): 90,
    },
    # Production cost per unit
    "prod_cost": {"Widget": 10.0, "Gadget": 15.0, "Gizmo": 12.0},
    # Inventory holding cost per unit per period
    "hold_cost": {"Widget": 1.5, "Gadget": 2.0, "Gizmo": 1.8},
    # Resource usage per unit of production
    "resource_usage": {"Widget": 1.0, "Gadget": 1.5, "Gizmo": 1.2},
    # Production capacity per period (in resource units)
    "capacity": {1: 300, 2: 280, 3: 320, 4: 310, 5: 290, 6: 300},
    # Initial inventory
    "initial_inventory": {"Widget": 20, "Gadget": 15, "Gizmo": 10},
}


# ── Solve multi-period production planning ─────────────────────────────────


def solve_multi_period_production(data: dict) -> pyo.ConcreteModel | None:
    # TODO(human): Multi-Period Production Planning with Pyomo
    #
    # Variables:
    #   x[p, t] = units of product p produced in period t
    #   inv[p, t] = inventory of product p at end of period t
    #
    # Objective: minimize total production cost + holding cost
    #   sum(prod_cost[p] * x[p,t]) + sum(hold_cost[p] * inv[p,t])
    #
    # Constraints:
    #   Inventory balance: inv[p,t] = inv[p,t-1] + x[p,t] - demand[p,t]
    #     For t=1: inv[p,1] = initial_inventory[p] + x[p,1] - demand[p,1]
    #   Capacity: sum_p resource_usage[p] * x[p,t] <= capacity[t]
    #   Non-negativity: x >= 0, inv >= 0
    #
    # Key modeling pattern: indexed variables over SETS (products x periods).
    # The inventory balance constraint LINKS consecutive periods.
    # This is a multi-period LP -- common in supply chain planning.
    #
    # Use pyo.ConcreteModel() with:
    #   model.T = pyo.RangeSet(1, n_periods)
    #   model.P = pyo.Set(initialize=product_names)
    #
    # For the inventory balance, handle t=1 separately (uses initial_inventory)
    # vs t>1 (uses inv[p, t-1]).
    #
    # Hint: pyo.Constraint(model.P, model.T, rule=balance_rule) creates one
    # constraint per (product, period) pair. The rule receives (m, p, t).
    raise NotImplementedError


# ── Display helpers ────────────────────────────────────────────────────────


def display_production_plan(model: pyo.ConcreteModel) -> None:
    """Print the optimal production plan in a readable table."""
    print(f"\n{'='*70}")
    print(f"  Multi-Period Production Plan")
    print(f"{'='*70}")
    print(f"  Total cost: {pyo.value(model.obj):.2f}\n")

    for p in model.P:
        print(f"  Product: {p}")
        print(f"  {'Period':>8} {'Demand':>8} {'Produce':>10} {'Inventory':>10}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
        for t in model.T:
            demand = model.demand_data[p, t]
            produce = pyo.value(model.x[p, t])
            inv = pyo.value(model.inv[p, t])
            print(f"  {t:>8} {demand:>8.0f} {produce:>10.1f} {inv:>10.1f}")
        print()

    print(f"  Capacity utilization:")
    print(f"  {'Period':>8} {'Used':>10} {'Capacity':>10} {'Util%':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*8}")
    for t in model.T:
        used = sum(
            PRODUCTION_DATA["resource_usage"][p] * pyo.value(model.x[p, t])
            for p in model.P
        )
        cap = PRODUCTION_DATA["capacity"][t]
        print(f"  {t:>8} {used:>10.1f} {cap:>10.0f} {100*used/cap:>7.1f}%")


def display_cost_breakdown(model: pyo.ConcreteModel) -> None:
    """Print cost breakdown by product and type."""
    print(f"\n  Cost breakdown:")
    print(f"  {'Product':<10} {'Prod Cost':>12} {'Hold Cost':>12} {'Total':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    total_prod = 0.0
    total_hold = 0.0
    for p in model.P:
        prod_cost = sum(
            PRODUCTION_DATA["prod_cost"][p] * pyo.value(model.x[p, t])
            for t in model.T
        )
        hold_cost = sum(
            PRODUCTION_DATA["hold_cost"][p] * pyo.value(model.inv[p, t])
            for t in model.T
        )
        total_prod += prod_cost
        total_hold += hold_cost
        print(f"  {p:<10} {prod_cost:>12.2f} {hold_cost:>12.2f} {prod_cost + hold_cost:>12.2f}")

    print(f"  {'TOTAL':<10} {total_prod:>12.2f} {total_hold:>12.2f} {total_prod + total_hold:>12.2f}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    model = solve_multi_period_production(PRODUCTION_DATA)
    if model is None:
        print("ERROR: Solver failed or not available.")
        return

    display_production_plan(model)
    display_cost_breakdown(model)

    print(f"\n{'='*70}")
    print("  Key insight: the model builds inventory BEFORE high-demand periods")
    print("  to avoid infeasibility when capacity is tight. This look-ahead")
    print("  behavior is why optimization beats greedy period-by-period planning.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
