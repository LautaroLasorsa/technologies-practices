"""Phase 1: Abstract Transportation Model with Pyomo.

The transportation problem: ship goods from suppliers to customers
minimizing total shipping cost. We define the model ONCE as an
AbstractModel and solve it with TWO different data sets.
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition


# ── Data Instance 1: Small network (3 suppliers, 4 customers) ──────────────

DATA_INSTANCE_1 = {None: {
    "I": {None: ["Seattle", "Denver", "Chicago"]},
    "J": {None: ["NYC", "LA", "Atlanta", "Houston"]},
    "supply": {
        "Seattle": 350,
        "Denver": 600,
        "Chicago": 500,
    },
    "demand": {
        "NYC": 325,
        "LA": 300,
        "Atlanta": 275,
        "Houston": 250,
    },
    "cost": {
        ("Seattle", "NYC"): 2.5,
        ("Seattle", "LA"): 1.7,
        ("Seattle", "Atlanta"): 1.8,
        ("Seattle", "Houston"): 2.2,
        ("Denver", "NYC"): 3.5,
        ("Denver", "LA"): 1.4,
        ("Denver", "Atlanta"): 2.8,
        ("Denver", "Houston"): 1.6,
        ("Chicago", "NYC"): 2.0,
        ("Chicago", "LA"): 3.2,
        ("Chicago", "Atlanta"): 1.5,
        ("Chicago", "Houston"): 2.1,
    },
}}


# ── Data Instance 2: Different network (2 suppliers, 3 customers) ──────────

DATA_INSTANCE_2 = {None: {
    "I": {None: ["FactoryA", "FactoryB"]},
    "J": {None: ["Store1", "Store2", "Store3"]},
    "supply": {
        "FactoryA": 1000,
        "FactoryB": 800,
    },
    "demand": {
        "Store1": 500,
        "Store2": 600,
        "Store3": 400,
    },
    "cost": {
        ("FactoryA", "Store1"): 4.0,
        ("FactoryA", "Store2"): 6.0,
        ("FactoryA", "Store3"): 5.0,
        ("FactoryB", "Store1"): 7.0,
        ("FactoryB", "Store2"): 3.0,
        ("FactoryB", "Store3"): 4.5,
    },
}}


# ── Build the abstract model ──────────────────────────────────────────────


def build_transportation_model() -> pyo.AbstractModel:
    # TODO(human): Build an Abstract Transportation Model with Pyomo
    #
    # The transportation problem: ship goods from suppliers to customers
    # minimizing total shipping cost.
    #
    # Abstract model (data-independent):
    #   model = pyo.AbstractModel()
    #
    #   model.I = pyo.Set()          # suppliers
    #   model.J = pyo.Set()          # customers
    #   model.supply = pyo.Param(model.I)    # supply at each supplier
    #   model.demand = pyo.Param(model.J)    # demand at each customer
    #   model.cost = pyo.Param(model.I, model.J)  # shipping cost per unit
    #
    #   model.x = pyo.Var(model.I, model.J, within=pyo.NonNegativeReals)
    #
    #   def obj_rule(m):
    #       return sum(m.cost[i,j] * m.x[i,j] for i in m.I for j in m.J)
    #   model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    #
    #   def supply_rule(m, i):
    #       return sum(m.x[i,j] for j in m.J) <= m.supply[i]
    #   model.supply_con = pyo.Constraint(model.I, rule=supply_rule)
    #
    #   def demand_rule(m, j):
    #       return sum(m.x[i,j] for i in m.I) >= m.demand[j]
    #   model.demand_con = pyo.Constraint(model.J, rule=demand_rule)
    #
    # Then create instances with different data:
    #   instance = model.create_instance(data_dict)
    #   solver.solve(instance)
    #
    # The power of abstract models: same model, different data sets.
    raise NotImplementedError


# ── Display helpers ────────────────────────────────────────────────────────


def display_solution(instance: pyo.ConcreteModel, label: str) -> None:
    """Print the transportation solution in a readable format."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Optimal cost: {pyo.value(instance.obj):.2f}\n")

    print(f"  {'Route':<30} {'Shipment':>10}")
    print(f"  {'-'*30} {'-'*10}")
    for i in instance.I:
        for j in instance.J:
            val = pyo.value(instance.x[i, j])
            if val > 1e-6:
                print(f"  {i} -> {j:<20} {val:>10.1f}")

    print(f"\n  Supply usage:")
    for i in instance.I:
        shipped = sum(pyo.value(instance.x[i, j]) for j in instance.J)
        supply = pyo.value(instance.supply[i])
        print(f"    {i}: {shipped:.0f} / {supply:.0f}")

    print(f"\n  Demand satisfaction:")
    for j in instance.J:
        received = sum(pyo.value(instance.x[i, j]) for i in instance.I)
        demand = pyo.value(instance.demand[j])
        print(f"    {j}: {received:.0f} / {demand:.0f}")


def display_duals(instance: pyo.ConcreteModel) -> None:
    """Print shadow prices (dual values) for supply and demand constraints."""
    if not hasattr(instance, "dual"):
        print("\n  (No dual information available)")
        return

    print(f"\n  Shadow prices (supply constraints):")
    for i in instance.I:
        dual_val = instance.dual.get(instance.supply_con[i], 0.0)
        print(f"    {i}: {dual_val:.4f}")

    print(f"\n  Shadow prices (demand constraints):")
    for j in instance.J:
        dual_val = instance.dual.get(instance.demand_con[j], 0.0)
        print(f"    {j}: {dual_val:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    # Build the abstract model once
    model = build_transportation_model()

    solver = pyo.SolverFactory("highs")
    if not solver.available():
        print("ERROR: HiGHS solver not available. Run: uv sync")
        return

    # Solve instance 1
    instance1 = model.create_instance(DATA_INSTANCE_1)
    instance1.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    result1 = solver.solve(instance1, tee=False)

    if result1.solver.termination_condition == TerminationCondition.optimal:
        display_solution(instance1, "Instance 1: 3 Suppliers -> 4 Customers")
        display_duals(instance1)
    else:
        print(f"Instance 1 failed: {result1.solver.termination_condition}")

    # Solve instance 2 -- same model, different data
    instance2 = model.create_instance(DATA_INSTANCE_2)
    instance2.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    result2 = solver.solve(instance2, tee=False)

    if result2.solver.termination_condition == TerminationCondition.optimal:
        display_solution(instance2, "Instance 2: 2 Factories -> 3 Stores")
        display_duals(instance2)
    else:
        print(f"Instance 2 failed: {result2.solver.termination_condition}")

    print(f"\n{'='*60}")
    print("  Same model definition, two different data sets.")
    print("  This is the power of Pyomo's AbstractModel.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
