"""Phase 4: Facility Location with Congestion Costs (MINLP).

Extends the classic facility location problem: service cost at each
facility increases nonlinearly with load (congestion). This is a
Mixed-Integer Nonlinear Program (MINLP).

Since exact MINLP solvers (Bonmin, Couenne) may not be available,
we solve the continuous relaxation (y in [0,1]) as a convex NLP.
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import math


# ── Problem data ───────────────────────────────────────────────────────────

FACILITIES = {
    "F1": {"fixed_cost": 500, "capacity": 400, "congestion_cost": 0.005,
            "location": (0.0, 0.0)},
    "F2": {"fixed_cost": 600, "capacity": 350, "congestion_cost": 0.008,
            "location": (10.0, 0.0)},
    "F3": {"fixed_cost": 450, "capacity": 300, "congestion_cost": 0.006,
            "location": (5.0, 8.0)},
    "F4": {"fixed_cost": 550, "capacity": 450, "congestion_cost": 0.004,
            "location": (3.0, 5.0)},
}

CUSTOMERS = {
    "C1": {"demand": 80, "location": (1.0, 2.0)},
    "C2": {"demand": 120, "location": (8.0, 3.0)},
    "C3": {"demand": 100, "location": (4.0, 7.0)},
    "C4": {"demand": 90, "location": (6.0, 1.0)},
    "C5": {"demand": 110, "location": (2.0, 6.0)},
    "C6": {"demand": 70, "location": (9.0, 5.0)},
}


def compute_transport_cost(
    facilities: dict, customers: dict
) -> dict[tuple[str, str], float]:
    """Compute per-unit transport cost as Euclidean distance between locations."""
    costs = {}
    for f, fdata in facilities.items():
        fx, fy = fdata["location"]
        for c, cdata in customers.items():
            cx, cy = cdata["location"]
            costs[f, c] = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
    return costs


# ── Solve facility location with congestion ────────────────────────────────


def solve_facility_congestion(
    facilities: dict, customers: dict
) -> pyo.ConcreteModel | None:
    # TODO(human): Facility Location with Congestion (MINLP)
    #
    # Extension of the facility location problem (040a Phase 4):
    # now, the service cost at each facility increases nonlinearly with load
    # (congestion effect: more customers -> longer waiting times).
    #
    # Variables:
    #   y[f] = Binary (or continuous [0,1] for relaxation): open facility f
    #   x[f,c] = Continuous [0,1]: fraction of customer c served by facility f
    #
    # Objective: minimize
    #   sum(fixed_cost[f] * y[f])                           # fixed costs
    #   + sum(transport_cost[f,c] * demand[c] * x[f,c])     # transport
    #   + sum(congestion_cost[f] * load[f]^2)                # NONLINEAR congestion
    #   where load[f] = sum(demand[c] * x[f,c] for c in customers)
    #
    # Constraints:
    #   1. Each customer fully served: sum(x[f,c] for f) == 1 for each c
    #   2. Capacity: load[f] <= capacity[f] * y[f] for each f
    #      (if facility closed, y[f]=0 forces load=0)
    #   3. Linking: x[f,c] <= y[f] for each f,c
    #      (can't assign to closed facility)
    #   4. Bounds: 0 <= x[f,c] <= 1, 0 <= y[f] <= 1
    #
    # This is MINLP: binary variables (y) + nonlinear objective (load^2).
    # Much harder than MIP or NLP alone.
    #
    # Solve approach: relax y from Binary to continuous [0,1] for a convex
    # NLP relaxation. Use SolverFactory('highs') for the QP, or 'ipopt'
    # for the general NLP formulation.
    #
    # Hint: define load[f] as an Expression or auxiliary variable:
    #   model.load = pyo.Var(model.F, within=pyo.NonNegativeReals)
    #   Then add constraint: load[f] == sum(demand[c] * x[f,c] for c)
    #   Then use load[f]**2 in the objective.
    #
    # Note: for this practice, solving the relaxation is sufficient to
    # understand MINLP modeling. Exact MINLP solving requires specialized solvers.
    raise NotImplementedError


# ── Display helpers ────────────────────────────────────────────────────────


def display_facility_solution(model: pyo.ConcreteModel) -> None:
    """Print the facility location solution."""
    print(f"\n{'='*65}")
    print(f"  Facility Location with Congestion (Relaxation)")
    print(f"{'='*65}")
    print(f"  Total cost: {pyo.value(model.obj):.2f}\n")

    # Facility status
    print(f"  {'Facility':<10} {'Open':>6} {'Load':>8} {'Capacity':>10} {'Congestion':>12}")
    print(f"  {'-'*10} {'-'*6} {'-'*8} {'-'*10} {'-'*12}")
    for f in model.F:
        y_val = pyo.value(model.y[f])
        load_val = pyo.value(model.load[f])
        cap = FACILITIES[f]["capacity"]
        cong_cost = FACILITIES[f]["congestion_cost"] * load_val ** 2
        status = f"{y_val:.2f}"
        print(f"  {f:<10} {status:>6} {load_val:>8.1f} {cap:>10} {cong_cost:>12.2f}")

    # Assignment matrix
    print(f"\n  Assignment fractions (x[f,c]):")
    header = f"  {'':>6}" + "".join(f"{c:>8}" for c in model.C)
    print(header)
    for f in model.F:
        row = f"  {f:>6}"
        for c in model.C:
            val = pyo.value(model.x[f, c])
            row += f"{val:>8.2f}" if val > 1e-4 else f"{'---':>8}"
        print(row)

    # Cost breakdown
    print(f"\n  Cost breakdown:")
    fixed_total = sum(FACILITIES[f]["fixed_cost"] * pyo.value(model.y[f]) for f in model.F)
    transport_total = sum(
        compute_transport_cost(FACILITIES, CUSTOMERS)[f, c]
        * CUSTOMERS[c]["demand"]
        * pyo.value(model.x[f, c])
        for f in model.F for c in model.C
    )
    congestion_total = sum(
        FACILITIES[f]["congestion_cost"] * pyo.value(model.load[f]) ** 2
        for f in model.F
    )
    print(f"    Fixed costs:      {fixed_total:>10.2f}")
    print(f"    Transport costs:  {transport_total:>10.2f}")
    print(f"    Congestion costs: {congestion_total:>10.2f}")
    print(f"    Total:            {fixed_total + transport_total + congestion_total:>10.2f}")


def compare_with_without_congestion(model_with: pyo.ConcreteModel) -> None:
    """Explain the impact of congestion costs on facility selection."""
    print(f"\n  Impact of congestion:")
    print(f"  Without congestion, the optimal solution concentrates demand")
    print(f"  at the cheapest facilities. With congestion (load^2 cost),")
    print(f"  the optimizer SPREADS load across facilities to avoid the")
    print(f"  superlinear penalty -- a classic load-balancing effect.")
    print(f"  This is why MINLP matters: real costs are rarely linear.")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"{'='*65}")
    print(f"  MINLP: Facility Location with Congestion")
    print(f"{'='*65}")

    print(f"\n  Facilities: {', '.join(FACILITIES.keys())}")
    print(f"  Customers:  {', '.join(CUSTOMERS.keys())}")
    total_demand = sum(c["demand"] for c in CUSTOMERS.values())
    total_capacity = sum(f["capacity"] for f in FACILITIES.values())
    print(f"  Total demand: {total_demand}, Total capacity: {total_capacity}")

    model = solve_facility_congestion(FACILITIES, CUSTOMERS)
    if model is None:
        print("\nERROR: Solver failed or not available.")
        return

    display_facility_solution(model)
    compare_with_without_congestion(model)

    print(f"\n{'='*65}")
    print("  This is a MINLP relaxation (y continuous). In practice,")
    print("  y values near 0 or 1 can be rounded. For exact solutions,")
    print("  use Bonmin (branch-and-bound for MINLP) or Couenne (global).")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
