"""Phase 1: Diet Problem — Linear Programming with PuLP & HiGHS.

The diet problem (Stigler, 1945) is historically the first LP ever formulated.
Choose food quantities to minimize total cost while meeting minimum nutritional
requirements. All variables are continuous (you can buy fractional servings).

This is the "hello world" of LP modeling — every concept here (variables,
objective, constraints, solve, extract) is reused in every subsequent phase.
"""

import pulp


# ============================================================================
# Data: Foods, costs, and nutritional content per serving
# ============================================================================

# Each food: name -> {cost, calories, protein_g, fat_g, carbs_g, fiber_g}
FOODS = {
    "oatmeal":       {"cost": 0.30, "calories": 150, "protein": 5,  "fat": 3,  "carbs": 27, "fiber": 4},
    "chicken_breast": {"cost": 2.50, "calories": 165, "protein": 31, "fat": 3.6, "carbs": 0,  "fiber": 0},
    "eggs":          {"cost": 0.80, "calories": 155, "protein": 13, "fat": 11, "carbs": 1,  "fiber": 0},
    "whole_milk":    {"cost": 0.60, "calories": 149, "protein": 8,  "fat": 8,  "carbs": 12, "fiber": 0},
    "rice":          {"cost": 0.20, "calories": 206, "protein": 4,  "fat": 0.4, "carbs": 45, "fiber": 0.6},
    "broccoli":      {"cost": 1.00, "calories": 55,  "protein": 3.7, "fat": 0.6, "carbs": 11, "fiber": 5.1},
    "peanut_butter": {"cost": 0.50, "calories": 188, "protein": 7,  "fat": 16, "carbs": 6,  "fiber": 2},
    "banana":        {"cost": 0.25, "calories": 105, "protein": 1.3, "fat": 0.4, "carbs": 27, "fiber": 3.1},
}

# Minimum daily nutritional requirements
REQUIREMENTS = {
    "calories": 2000,   # kcal
    "protein":  50,     # grams
    "fat":      20,     # grams (minimum healthy fat)
    "carbs":    130,    # grams
    "fiber":    25,     # grams
}

# Maximum servings per food (to keep the diet reasonable)
MAX_SERVINGS = 10.0


# ============================================================================
# Solver function — TODO(human)
# ============================================================================

def solve_diet_problem(
    foods: dict[str, dict[str, float]],
    requirements: dict[str, float],
    max_servings: float,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Solve the diet problem: minimize cost while meeting nutritional requirements.

    Args:
        foods: {food_name: {cost, calories, protein, fat, carbs, fiber}}
        requirements: {nutrient: minimum_daily_amount}
        max_servings: maximum servings of any single food

    Returns:
        Tuple of:
        - solution: {food_name: servings} (only foods with servings > 0)
        - total_cost: optimal daily cost
        - shadow_prices: {nutrient: shadow_price} for each requirement constraint
    """
    # TODO(human): Solve the Diet Problem using PuLP
    #
    # The diet problem: choose quantities of foods to minimize total cost
    # while meeting minimum nutritional requirements.
    #
    # Steps:
    #   1. Create the problem: prob = pulp.LpProblem("Diet", pulp.LpMinimize)
    #   2. Create variables: x[food] = pulp.LpVariable(food, lowBound=0, upBound=max_servings)
    #      (continuous, non-negative — can buy fractional servings)
    #   3. Set objective: prob += pulp.lpSum(foods[f]["cost"] * x[f] for f in foods)
    #   4. Add constraints: for each nutrient n in requirements,
    #      prob += pulp.lpSum(foods[f][n] * x[f] for f in foods) >= requirements[n], n
    #      The second argument is the constraint name — needed for accessing shadow prices later.
    #   5. Solve: prob.solve(pulp.HiGHS_CMD(msg=0))
    #   6. Check status: assert prob.status == pulp.constants.LpStatusOptimal
    #   7. Extract solution: {f: x[f].varValue for f in foods if x[f].varValue > 1e-6}
    #   8. Extract shadow prices from the constraints:
    #      for name, constraint in prob.constraints.items():
    #          shadow_prices[name] = constraint.pi
    #      Shadow price = how much would the objective (cost) INCREASE if we raised
    #      this nutrient requirement by one unit? High shadow price = expensive nutrient.
    #
    # Return (solution_dict, total_cost, shadow_prices_dict)
    raise NotImplementedError


# ============================================================================
# Display helpers
# ============================================================================

def print_solution(
    solution: dict[str, float],
    total_cost: float,
    shadow_prices: dict[str, float],
    foods: dict[str, dict[str, float]],
    requirements: dict[str, float],
) -> None:
    """Display the diet solution with nutritional breakdown and dual analysis."""
    print("=" * 65)
    print("DIET PROBLEM — Optimal Solution")
    print("=" * 65)

    print(f"\nTotal daily cost: ${total_cost:.2f}\n")

    print("Food servings:")
    for food, servings in sorted(solution.items(), key=lambda x: -x[1]):
        cost = foods[food]["cost"] * servings
        print(f"  {food:<18s} {servings:6.2f} servings  (${cost:.2f})")

    print("\nNutritional intake vs requirements:")
    for nutrient, req in requirements.items():
        intake = sum(
            foods[f][nutrient] * solution.get(f, 0.0)
            for f in foods
        )
        status = "BINDING" if abs(intake - req) < 1e-4 else "slack"
        print(f"  {nutrient:<10s}  intake={intake:8.1f}  req={req:8.1f}  [{status}]")

    print("\nShadow prices (dual variables):")
    print("  (How much daily cost increases per unit increase in requirement)")
    for nutrient, price in shadow_prices.items():
        interpretation = ""
        if abs(price) < 1e-6:
            interpretation = " (non-binding, free to relax)"
        else:
            interpretation = f" (raising req by 1 costs +${price:.4f}/day)"
        print(f"  {nutrient:<10s}  pi = {price:8.4f}{interpretation}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("Solving the diet problem with PuLP + HiGHS...\n")

    solution, total_cost, shadow_prices = solve_diet_problem(
        FOODS, REQUIREMENTS, MAX_SERVINGS
    )

    print_solution(solution, total_cost, shadow_prices, FOODS, REQUIREMENTS)


if __name__ == "__main__":
    main()
