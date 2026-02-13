"""Phase 2: Production Planning — LP with Full Dual Analysis.

Maximize total profit from producing a mix of products, subject to limited
resources (labor hours, machine hours, raw materials). After solving, analyze
shadow prices (which resources are bottlenecks?), reduced costs (why aren't
we producing certain products?), and slack (which resources are underutilized?).

The dual analysis is often more valuable than the primal solution itself — it
answers "what should we invest in?" rather than just "what should we produce?"
"""

import pulp


# ============================================================================
# Data: Products, profits, resource usage
# ============================================================================

# Products with profit per unit
PRODUCTS = {
    "desk":      {"profit": 60},
    "table":     {"profit": 90},
    "bookshelf": {"profit": 80},
    "chair":     {"profit": 40},
    "cabinet":   {"profit": 100},
}

# Resource usage per unit of each product
# resource_usage[product][resource] = units of resource consumed per product unit
RESOURCE_USAGE = {
    "desk":      {"labor_hrs": 4,  "machine_hrs": 2,  "wood_kg": 8,  "metal_kg": 3},
    "table":     {"labor_hrs": 6,  "machine_hrs": 3,  "wood_kg": 12, "metal_kg": 2},
    "bookshelf": {"labor_hrs": 5,  "machine_hrs": 4,  "wood_kg": 15, "metal_kg": 1},
    "chair":     {"labor_hrs": 2,  "machine_hrs": 1,  "wood_kg": 5,  "metal_kg": 2},
    "cabinet":   {"labor_hrs": 8,  "machine_hrs": 5,  "wood_kg": 20, "metal_kg": 4},
}

# Available resources per week
RESOURCE_LIMITS = {
    "labor_hrs":   120,   # hours of labor available
    "machine_hrs":  80,   # hours of machine time
    "wood_kg":     300,   # kg of wood
    "metal_kg":     60,   # kg of metal
}

# Maximum demand per product per week (market constraint)
MAX_DEMAND = {
    "desk":      15,
    "table":     10,
    "bookshelf":  8,
    "chair":     20,
    "cabinet":    5,
}


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def solve_production_planning(
    products: dict[str, dict[str, float]],
    resource_usage: dict[str, dict[str, float]],
    resource_limits: dict[str, float],
    max_demand: dict[str, int],
) -> tuple[dict[str, float], float, dict[str, float], dict[str, float], dict[str, float]]:
    """Solve the production planning LP and extract full dual analysis.

    Args:
        products: {product_name: {profit: value}}
        resource_usage: {product_name: {resource: usage_per_unit}}
        resource_limits: {resource: max_available}
        max_demand: {product_name: max_units_demanded}

    Returns:
        Tuple of:
        - production: {product: quantity_to_produce}
        - total_profit: optimal weekly profit
        - shadow_prices: {resource: shadow_price} for resource constraints
        - reduced_costs: {product: reduced_cost}
        - slacks: {resource: slack_value}
    """
    # TODO(human): Production Planning LP
    #
    # Maximize total profit subject to resource limits and demand caps.
    # Variables: x[product] = quantity to produce (continuous, >= 0)
    # Objective: maximize sum(profit[p] * x[p] for p in products)
    # Constraints:
    #   1. Resource limits: for each resource r,
    #      sum(usage[p][r] * x[p] for p in products) <= limit[r]
    #   2. Demand caps: x[p] <= max_demand[p] for each product p
    #
    # After solving, extract the full dual analysis:
    #   1. Shadow prices: prob.constraints[name].pi for resource constraints
    #      - Which resources are fully used (binding constraints)?
    #      - What are the shadow prices? (marginal value of each resource)
    #      - A shadow price of $5/hr for labor means 1 extra labor hour
    #        would increase profit by $5.
    #   2. Reduced costs: x[p].dj for each product variable
    #      - How much would a product's profit need to improve before
    #        we'd want to produce it?
    #      - For products already being produced, reduced cost is 0.
    #      - For products NOT produced, reduced cost tells you the gap.
    #   3. Slack: prob.constraints[name].slack for resource constraints
    #      - How much of each resource is unused?
    #      - Zero slack = binding = bottleneck resource
    #
    # These correspond to LP duality from practice 032b!
    #   shadow_price(resource) = dual variable of that constraint
    #   reduced_cost(product) = dual variable of the non-negativity constraint
    #
    # Return (production_dict, total_profit, shadow_prices, reduced_costs, slacks)
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_production_plan(
    production: dict[str, float],
    total_profit: float,
    products: dict[str, dict[str, float]],
) -> None:
    """Display what to produce and how much."""
    print("=" * 65)
    print("PRODUCTION PLANNING — Optimal Plan")
    print("=" * 65)
    print(f"\nTotal weekly profit: ${total_profit:.2f}\n")

    print("Production quantities:")
    for product, qty in sorted(production.items(), key=lambda x: -x[1]):
        profit = products[product]["profit"] * qty
        if qty > 1e-6:
            print(f"  {product:<12s}  {qty:6.2f} units  (${profit:.2f} profit)")
        else:
            print(f"  {product:<12s}  {qty:6.2f} units  (not produced)")


def print_dual_analysis(
    shadow_prices: dict[str, float],
    reduced_costs: dict[str, float],
    slacks: dict[str, float],
    resource_limits: dict[str, float],
) -> None:
    """Display shadow prices, reduced costs, and slack analysis."""
    print("\n" + "-" * 65)
    print("DUAL ANALYSIS — What the LP Tells Us Beyond the Solution")
    print("-" * 65)

    print("\nResource analysis (shadow prices + slack):")
    print(f"  {'Resource':<14s} {'Limit':>8s} {'Slack':>8s} {'Shadow Price':>14s}  Interpretation")
    print(f"  {'-'*14} {'-'*8} {'-'*8} {'-'*14}  {'-'*30}")
    for resource in sorted(resource_limits.keys()):
        limit = resource_limits[resource]
        slack = slacks.get(resource, 0.0)
        sp = shadow_prices.get(resource, 0.0)
        if abs(slack) < 1e-6:
            interp = f"BOTTLENECK: +1 unit → +${sp:.2f} profit"
        else:
            interp = f"Surplus: {slack:.1f} unused"
        print(f"  {resource:<14s} {limit:8.1f} {slack:8.1f} {sp:14.4f}  {interp}")

    print("\nReduced costs (why aren't we producing more?):")
    print(f"  {'Product':<12s} {'Reduced Cost':>14s}  Interpretation")
    print(f"  {'-'*12} {'-'*14}  {'-'*40}")
    for product, rc in sorted(reduced_costs.items()):
        if abs(rc) < 1e-6:
            interp = "In the optimal mix"
        else:
            interp = f"Profit must increase by ${abs(rc):.2f} to be worth producing"
        print(f"  {product:<12s} {rc:14.4f}  {interp}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Solving production planning LP with PuLP + HiGHS...\n")

    production, total_profit, shadow_prices, reduced_costs, slacks = (
        solve_production_planning(
            PRODUCTS, RESOURCE_USAGE, RESOURCE_LIMITS, MAX_DEMAND
        )
    )

    print_production_plan(production, total_profit, PRODUCTS)
    print_dual_analysis(shadow_prices, reduced_costs, slacks, RESOURCE_LIMITS)


if __name__ == "__main__":
    main()
