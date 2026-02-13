"""Phase 3: Binary Knapsack — MIP with LP Relaxation Comparison.

The 0-1 knapsack problem: select items to maximize total value without exceeding
weight capacity. Each item is either fully taken (1) or left (0) — no fractions.

This is the simplest MIP: one binary variable per item, one constraint. Yet it
is NP-hard, so HiGHS uses branch-and-bound to solve it. Comparing the MIP
solution with the LP relaxation (fractional items allowed) shows the integrality
gap — a key concept for understanding solver difficulty.
"""

import pulp


# ============================================================================
# Data: Items with value and weight
# ============================================================================

ITEMS = {
    "laptop":      {"value": 600, "weight": 2.5},
    "camera":      {"value": 500, "weight": 1.5},
    "headphones":  {"value": 150, "weight": 0.3},
    "book_set":    {"value": 200, "weight": 3.0},
    "tablet":      {"value": 450, "weight": 0.8},
    "jacket":      {"value": 100, "weight": 1.2},
    "toolkit":     {"value": 300, "weight": 4.0},
    "speaker":     {"value": 250, "weight": 2.0},
    "watch":       {"value": 350, "weight": 0.2},
    "binoculars":  {"value": 200, "weight": 1.8},
    "drone":       {"value": 700, "weight": 3.5},
    "charger_kit": {"value": 120, "weight": 0.5},
}

CAPACITY = 10.0  # kg


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def solve_knapsack(
    items: dict[str, dict[str, float]],
    capacity: float,
) -> tuple[dict[str, int], float, dict[str, float], float]:
    """Solve the binary knapsack problem and compare with LP relaxation.

    Args:
        items: {item_name: {value: v, weight: w}}
        capacity: maximum total weight

    Returns:
        Tuple of:
        - mip_solution: {item_name: 0 or 1} selected items
        - mip_value: optimal integer objective value
        - lp_solution: {item_name: float in [0,1]} LP relaxation solution
        - lp_value: LP relaxation objective value (upper bound on MIP)
    """
    # TODO(human): Binary Knapsack with PuLP
    #
    # Select items to maximize total value within weight capacity.
    # Variables: x[item] = pulp.LpVariable(item, cat='Binary')
    #   Binary = {0, 1}. 1 means item is selected.
    #
    # Objective: maximize pulp.lpSum(items[i]["value"] * x[i] for i in items)
    # Constraint: pulp.lpSum(items[i]["weight"] * x[i] for i in items) <= capacity
    #
    # This is a MIP (binary integer program). HiGHS uses branch-and-bound
    # (which you implemented in practice 034a!) to solve it.
    #
    # After solving the MIP, also solve the LP relaxation separately:
    #   - Create a NEW problem with cat='Continuous' and 0 <= x <= 1 bounds
    #   - The LP relaxation allows fractional items (e.g., take 0.7 of a laptop)
    #   - The LP objective is always >= MIP objective (for maximization)
    #   - The gap between them is the integrality gap:
    #       gap = (lp_value - mip_value) / lp_value * 100%
    #   - Small gap means the LP relaxation is tight (good formulation)
    #   - Large gap means the solver has to work harder (more branching)
    #
    # Return (mip_solution, mip_value, lp_solution, lp_value)
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_knapsack_results(
    items: dict[str, dict[str, float]],
    mip_solution: dict[str, int],
    mip_value: float,
    lp_solution: dict[str, float],
    lp_value: float,
    capacity: float,
) -> None:
    """Display MIP solution, LP relaxation, and integrality gap analysis."""
    print("=" * 65)
    print("BINARY KNAPSACK — MIP Solution")
    print("=" * 65)

    selected = {k: v for k, v in mip_solution.items() if v > 0.5}
    total_weight = sum(items[i]["weight"] for i in selected)

    print(f"\nOptimal value: ${mip_value:.0f}")
    print(f"Total weight:  {total_weight:.1f} / {capacity:.1f} kg\n")

    print("Selected items:")
    for item in sorted(selected.keys()):
        v, w = items[item]["value"], items[item]["weight"]
        ratio = v / w
        print(f"  {item:<14s}  value=${v:4.0f}  weight={w:.1f}kg  (ratio={ratio:.0f}$/kg)")

    not_selected = {k for k, v in mip_solution.items() if v < 0.5}
    if not_selected:
        print("\nNot selected:")
        for item in sorted(not_selected):
            v, w = items[item]["value"], items[item]["weight"]
            print(f"  {item:<14s}  value=${v:4.0f}  weight={w:.1f}kg")

    print("\n" + "-" * 65)
    print("LP RELAXATION — Upper Bound")
    print("-" * 65)

    print(f"\nLP relaxation value: ${lp_value:.2f}")

    print("\nLP solution (fractional items allowed):")
    for item, frac in sorted(lp_solution.items(), key=lambda x: -x[1]):
        if frac > 1e-6:
            status = "FULL" if frac > 0.999 else f"FRAC ({frac:.3f})"
            print(f"  {item:<14s}  x={frac:.4f}  [{status}]")

    gap = (lp_value - mip_value) / lp_value * 100 if lp_value > 0 else 0
    print(f"\nIntegrality gap: {gap:.2f}%")
    print(f"  LP bound:   ${lp_value:.2f}")
    print(f"  MIP optimal: ${mip_value:.0f}")
    if gap < 1:
        print("  -> Tight relaxation! LP bound is very close to MIP optimum.")
    elif gap < 5:
        print("  -> Moderate gap. Branch-and-bound closes it quickly.")
    else:
        print("  -> Large gap. Solver needs significant branching effort.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Solving binary knapsack with PuLP + HiGHS...\n")

    mip_solution, mip_value, lp_solution, lp_value = solve_knapsack(
        ITEMS, CAPACITY
    )

    print_knapsack_results(
        ITEMS, mip_solution, mip_value, lp_solution, lp_value, CAPACITY
    )


if __name__ == "__main__":
    main()
