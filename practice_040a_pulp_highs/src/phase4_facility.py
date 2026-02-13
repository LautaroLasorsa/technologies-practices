"""Phase 4: Capacitated Facility Location — MIP with Linking Constraints.

Decide which facilities to open (binary) and how to assign customers to
facilities (continuous). The linking constraint x[f][c] <= y[f] says
"can't serve from a closed facility" — this is the most important MIP
modeling pattern, appearing in network design, scheduling, and supply chain.

This model mixes binary "design" decisions (open/close) with continuous
"operational" decisions (assignment fractions), showing how they interact.
"""

import pulp


# ============================================================================
# Data: Facilities, customers, costs
# ============================================================================

# Facilities with fixed opening cost and capacity
FACILITIES = {
    "warehouse_A": {"fixed_cost": 1000, "capacity": 120},
    "warehouse_B": {"fixed_cost": 1500, "capacity": 160},
    "warehouse_C": {"fixed_cost": 800,  "capacity": 90},
    "warehouse_D": {"fixed_cost": 1200, "capacity": 130},
    "warehouse_E": {"fixed_cost": 2000, "capacity": 200},
}

# Customers with demand
CUSTOMERS = {
    "store_1": {"demand": 40},
    "store_2": {"demand": 55},
    "store_3": {"demand": 30},
    "store_4": {"demand": 45},
    "store_5": {"demand": 60},
    "store_6": {"demand": 35},
    "store_7": {"demand": 50},
    "store_8": {"demand": 25},
}

# Transportation cost per unit demand: transport_cost[facility][customer]
# (Think of this as distance * per-unit-shipping-cost)
TRANSPORT_COSTS = {
    "warehouse_A": {"store_1": 4, "store_2": 8, "store_3": 6, "store_4": 10, "store_5": 12, "store_6": 7, "store_7": 14, "store_8": 9},
    "warehouse_B": {"store_1": 9, "store_2": 3, "store_3": 7, "store_4": 5,  "store_5": 6,  "store_6": 11, "store_7": 4,  "store_8": 8},
    "warehouse_C": {"store_1": 5, "store_2": 10, "store_3": 3, "store_4": 9, "store_5": 11, "store_6": 4,  "store_7": 12, "store_8": 6},
    "warehouse_D": {"store_1": 11, "store_2": 6, "store_3": 9, "store_4": 3, "store_5": 5,  "store_6": 8,  "store_7": 7,  "store_8": 10},
    "warehouse_E": {"store_1": 7, "store_2": 5, "store_3": 8, "store_4": 6,  "store_5": 3,  "store_6": 9,  "store_7": 5,  "store_8": 4},
}


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def solve_facility_location(
    facilities: dict[str, dict[str, float]],
    customers: dict[str, dict[str, float]],
    transport_costs: dict[str, dict[str, float]],
) -> tuple[dict[str, int], dict[str, dict[str, float]], float]:
    """Solve the capacitated facility location problem.

    Args:
        facilities: {facility: {fixed_cost, capacity}}
        customers: {customer: {demand}}
        transport_costs: {facility: {customer: cost_per_unit}}

    Returns:
        Tuple of:
        - open_facilities: {facility: 0 or 1}
        - assignments: {facility: {customer: fraction_served}} (only non-zero)
        - total_cost: optimal total cost (fixed + transportation)
    """
    # TODO(human): Capacitated Facility Location Problem (CFLP)
    #
    # Decide which facilities to open and how to assign customers.
    #
    # Variables:
    #   y[f] = Binary: 1 if facility f is open
    #   x[f][c] = Continuous [0,1]: fraction of customer c served by facility f
    #
    # Objective: minimize
    #   sum(fixed_cost[f] * y[f] for f in facilities)
    #   + sum(transport_cost[f][c] * demand[c] * x[f][c] for f, c)
    #
    # Constraints:
    #   1. Each customer fully served:
    #      sum(x[f][c] for f in facilities) == 1  for all c
    #   2. Only open facilities serve (linking constraint):
    #      x[f][c] <= y[f]  for all f, c
    #   3. Capacity: total demand served by f <= capacity * y[f]
    #      sum(demand[c] * x[f][c] for c in customers) <= capacity[f] * y[f]  for all f
    #
    # This is the classic MIP linking binary (open/close) with continuous (assignment).
    # The "x <= y" constraint is a linking constraint — it says "can't use facility
    # unless it's open." This pattern appears in many real-world models:
    #   - Network design: can't route flow through an unbuilt link
    #   - Scheduling: can't assign tasks to an unavailable machine
    #   - Supply chain: can't ship from a warehouse that isn't built
    #
    # The LP relaxation will likely partially open facilities (y[f] = 0.5).
    # The MIP solver uses branch-and-bound to find integer y values.
    #
    # Return (open_dict, assignments_dict, total_cost)
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_facility_solution(
    facilities: dict[str, dict[str, float]],
    customers: dict[str, dict[str, float]],
    transport_costs: dict[str, dict[str, float]],
    open_facilities: dict[str, int],
    assignments: dict[str, dict[str, float]],
    total_cost: float,
) -> None:
    """Display which facilities are open and how customers are assigned."""
    print("=" * 70)
    print("CAPACITATED FACILITY LOCATION — Optimal Solution")
    print("=" * 70)

    # Cost breakdown
    fixed_total = sum(
        facilities[f]["fixed_cost"]
        for f, is_open in open_facilities.items()
        if is_open > 0.5
    )
    transport_total = total_cost - fixed_total

    print(f"\nTotal cost:          ${total_cost:.2f}")
    print(f"  Fixed costs:       ${fixed_total:.2f}")
    print(f"  Transportation:    ${transport_total:.2f}")

    # Facility status
    print("\nFacility decisions:")
    for f in sorted(facilities.keys()):
        is_open = open_facilities.get(f, 0) > 0.5
        status = "OPEN" if is_open else "CLOSED"
        cap = facilities[f]["capacity"]
        cost = facilities[f]["fixed_cost"]

        if is_open and f in assignments:
            used = sum(
                customers[c]["demand"] * frac
                for c, frac in assignments[f].items()
            )
            util = used / cap * 100
            print(f"  {f:<14s}  [{status}]  cost=${cost:5.0f}  "
                  f"used={used:5.1f}/{cap}  ({util:.0f}% utilized)")
        else:
            print(f"  {f:<14s}  [{status}]  cost=${cost:5.0f}")

    # Customer assignments
    print("\nCustomer assignments:")
    for c in sorted(customers.keys()):
        demand = customers[c]["demand"]
        print(f"  {c:<10s} (demand={demand:3.0f}):")
        for f in sorted(facilities.keys()):
            frac = assignments.get(f, {}).get(c, 0.0)
            if frac > 1e-6:
                units = demand * frac
                tc = transport_costs[f][c] * units
                print(f"    <- {f:<14s}  {frac*100:5.1f}%  ({units:.1f} units, ${tc:.2f} transport)")

    # Summary
    n_open = sum(1 for v in open_facilities.values() if v > 0.5)
    n_total = len(facilities)
    print(f"\nSummary: {n_open}/{n_total} facilities opened, "
          f"{len(customers)} customers served")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Solving capacitated facility location with PuLP + HiGHS...\n")

    open_facilities, assignments, total_cost = solve_facility_location(
        FACILITIES, CUSTOMERS, TRANSPORT_COSTS
    )

    print_facility_solution(
        FACILITIES, CUSTOMERS, TRANSPORT_COSTS,
        open_facilities, assignments, total_cost
    )


if __name__ == "__main__":
    main()
