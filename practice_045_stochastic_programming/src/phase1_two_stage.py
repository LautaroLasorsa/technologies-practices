"""Phase 1: Two-Stage Stochastic LP -- The Farmer's Problem.

The classic Birge & Louveaux farmer problem:
  - 500 acres of land to allocate among wheat, corn, sugar beets
  - Yields depend on weather (3 scenarios: good, average, bad)
  - First-stage: land allocation (before knowing weather)
  - Second-stage: buy/sell crops to meet requirements (after observing yields)

We build the extensive form (deterministic equivalent) using Pyomo indexed
Blocks, one per scenario. Then compute VSS and EVPI to quantify the value
of modeling uncertainty.

Pyomo patterns used:
  - ConcreteModel with top-level first-stage Vars
  - Block(SCENARIOS, rule=...) for second-stage per-scenario variables
  - Linking constraints referencing parent model variables inside blocks
  - Probability-weighted Objective
  - SolverFactory('highs') to solve the extensive form LP
"""

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
import numpy as np


# ============================================================================
# Problem data -- Birge & Louveaux farmer's problem
# ============================================================================

CROPS = ["wheat", "corn", "sugar_beets"]
TOTAL_LAND = 500.0  # acres

# Planting costs ($/acre)
PLANTING_COST = {"wheat": 150.0, "corn": 230.0, "sugar_beets": 260.0}

# Selling prices ($/ton)
SELLING_PRICE = {"wheat": 170.0, "corn": 150.0, "sugar_beets_favorable": 36.0, "sugar_beets_excess": 10.0}

# Purchase prices ($/ton) -- buying is more expensive than selling
PURCHASE_PRICE = {"wheat": 238.0, "corn": 210.0}

# Minimum requirements (tons) -- must meet or buy from market
REQUIREMENTS = {"wheat": 200.0, "corn": 240.0}

# Sugar beet quota (tons) -- can sell up to this at favorable price, excess at lower price
SUGAR_BEET_QUOTA = 6000.0

# Yield data per scenario (tons/acre)
# Three weather scenarios: good (above average), average, bad (below average)
SCENARIO_DATA = {
    "good": {
        "probability": 1.0 / 3.0,
        "yields": {"wheat": 3.0, "corn": 3.6, "sugar_beets": 24.0},
    },
    "average": {
        "probability": 1.0 / 3.0,
        "yields": {"wheat": 2.5, "corn": 3.0, "sugar_beets": 20.0},
    },
    "bad": {
        "probability": 1.0 / 3.0,
        "yields": {"wheat": 2.0, "corn": 2.4, "sugar_beets": 16.0},
    },
}

# Expected (average) yields -- used for the deterministic model
EXPECTED_YIELDS = {
    crop: sum(SCENARIO_DATA[s]["yields"][crop] for s in SCENARIO_DATA) / len(SCENARIO_DATA)
    for crop in CROPS
}


# ============================================================================
# TODO(human): Build the two-stage stochastic farmer model (extensive form)
# ============================================================================

def build_stochastic_farmer_model(scenario_data: dict) -> pyo.ConcreteModel:
    """Build the extensive form of the stochastic farmer's problem.

    The extensive form creates ONE large LP with:
      - First-stage variables (land allocation) shared across all scenarios
      - Second-stage variables (buy/sell) duplicated per scenario in Blocks
      - Objective = first-stage cost + probability-weighted second-stage costs

    Args:
        scenario_data: Dictionary with scenario names as keys, each containing
                       'probability' and 'yields' (dict of crop -> tons/acre).

    Returns:
        Solved Pyomo ConcreteModel.
    """
    # TODO(human): Build and solve the extensive form Pyomo model
    #
    # This is the core of two-stage stochastic programming. You need to:
    #
    # STEP 1: Create model and first-stage variables
    #   model = pyo.ConcreteModel("Farmer_Stochastic")
    #   model.x = pyo.Var(CROPS, within=pyo.NonNegativeReals)
    #   These are the "here-and-now" decisions: acres allocated to each crop.
    #   They must be the same regardless of which scenario occurs.
    #
    # STEP 2: Land constraint (first-stage)
    #   Total land allocated cannot exceed TOTAL_LAND (500 acres):
    #   sum(model.x[c] for c in CROPS) <= TOTAL_LAND
    #
    # STEP 3: Define scenario blocks using Block(scenario_names, rule=...)
    #   For each scenario s, the block should contain:
    #     block.y_sell_wheat   -- tons of wheat sold (NonNegativeReals)
    #     block.y_sell_corn    -- tons of corn sold (NonNegativeReals)
    #     block.y_sell_beets_fav  -- sugar beets sold at favorable price (NonNegativeReals)
    #     block.y_sell_beets_exc  -- sugar beets sold at excess price (NonNegativeReals)
    #     block.y_buy_wheat    -- tons of wheat purchased (NonNegativeReals)
    #     block.y_buy_corn     -- tons of corn purchased (NonNegativeReals)
    #
    #   And the following LINKING constraints that reference model.x:
    #     Wheat balance:  yield[s,"wheat"] * model.x["wheat"] + y_buy_wheat - y_sell_wheat >= REQUIREMENTS["wheat"]
    #     Corn balance:   yield[s,"corn"] * model.x["corn"] + y_buy_corn - y_sell_corn >= REQUIREMENTS["corn"]
    #     Sugar beet sell: y_sell_beets_fav + y_sell_beets_exc <= yield[s,"sugar_beets"] * model.x["sugar_beets"]
    #     Sugar beet quota: y_sell_beets_fav <= SUGAR_BEET_QUOTA
    #
    #   The key insight: model.x (first-stage) appears in the block constraints.
    #   This is the LINKING between stages -- first-stage decisions determine
    #   what's available in the second stage.
    #
    # STEP 4: Define the probability-weighted objective
    #   first_stage_cost = sum(PLANTING_COST[c] * model.x[c] for c in CROPS)
    #
    #   For each scenario s with probability p_s:
    #     second_stage_profit_s =
    #       SELLING_PRICE["wheat"] * y_sell_wheat
    #       + SELLING_PRICE["corn"] * y_sell_corn
    #       + SELLING_PRICE["sugar_beets_favorable"] * y_sell_beets_fav
    #       + SELLING_PRICE["sugar_beets_excess"] * y_sell_beets_exc
    #       - PURCHASE_PRICE["wheat"] * y_buy_wheat
    #       - PURCHASE_PRICE["corn"] * y_buy_corn
    #
    #   Objective (minimize COST = planting cost - expected profit):
    #     model.obj = Objective(expr=first_stage_cost - sum(p_s * profit_s for s in scenarios))
    #     (equivalently: minimize cost, which is planting cost minus revenue)
    #
    # STEP 5: Solve with HiGHS
    #   solver = pyo.SolverFactory('highs')
    #   result = solver.solve(model)
    #   assert result.solver.termination_condition == TerminationCondition.optimal
    #
    # Return the solved model.
    raise NotImplementedError("TODO(human): build extensive form with Blocks, link stages, solve")


# ============================================================================
# TODO(human): Build the deterministic model (using expected/fixed yields)
# ============================================================================

def build_deterministic_farmer_model(yields: dict) -> pyo.ConcreteModel:
    """Build and solve a deterministic farmer model with known yields.

    This is a single-scenario version: no uncertainty, no blocks.
    Used for two purposes:
      1. The "EV problem" (expected value): solve with mean yields to get x_EV
      2. The "WS problem" (wait-and-see): solve separately for each scenario

    Args:
        yields: Dictionary of crop -> tons/acre (a single yield scenario).

    Returns:
        Solved Pyomo ConcreteModel.
    """
    # TODO(human): Build and solve the deterministic (single-scenario) model
    #
    # This is the same farmer problem but WITHOUT scenarios -- just one
    # set of yields. It's a standard LP, much simpler than the stochastic version.
    #
    # Variables:
    #   x[c] -- acres allocated to crop c (NonNegativeReals)
    #   y_sell_wheat, y_sell_corn, y_sell_beets_fav, y_sell_beets_exc (NonNegativeReals)
    #   y_buy_wheat, y_buy_corn (NonNegativeReals)
    #
    # Constraints:
    #   Land: sum(x[c]) <= TOTAL_LAND
    #   Wheat balance: yields["wheat"] * x["wheat"] + y_buy_wheat - y_sell_wheat >= REQUIREMENTS["wheat"]
    #   Corn balance:  yields["corn"] * x["corn"] + y_buy_corn - y_sell_corn >= REQUIREMENTS["corn"]
    #   Sugar beet sell: y_sell_beets_fav + y_sell_beets_exc <= yields["sugar_beets"] * x["sugar_beets"]
    #   Quota: y_sell_beets_fav <= SUGAR_BEET_QUOTA
    #
    # Objective (minimize cost = planting - revenue):
    #   min  sum(PLANTING_COST[c] * x[c]) - selling_revenue + purchase_cost
    #
    # Solve with HiGHS, return the model.
    #
    # WHY this function exists: We need it twice:
    #   - With EXPECTED_YIELDS to get the "EV solution" for VSS
    #   - With each scenario's yields separately for the "WS" (wait-and-see) bound
    raise NotImplementedError("TODO(human): build single-scenario deterministic farmer LP")


# ============================================================================
# TODO(human): Compute VSS and EVPI
# ============================================================================

def compute_vss_and_evpi(scenario_data: dict) -> dict:
    """Compute Value of Stochastic Solution (VSS) and EVPI.

    Three problems to solve:
      RP  = Recourse Problem (stochastic model, the extensive form)
      EEV = Expected result of EV solution (fix x from mean model, evaluate across scenarios)
      WS  = Wait-and-See (solve each scenario independently, take expected optimal)

    Then:
      VSS  = EEV - RP  (value of accounting for uncertainty)
      EVPI = RP - WS   (value of perfect information)

    Args:
        scenario_data: The SCENARIO_DATA dictionary.

    Returns:
        Dictionary with keys 'RP', 'EEV', 'WS', 'VSS', 'EVPI' and their values.
    """
    # TODO(human): Compute all five values
    #
    # STEP 1: Solve the Recourse Problem (RP)
    #   Use build_stochastic_farmer_model(scenario_data).
    #   RP = the optimal objective value (pyo.value(model.obj)).
    #   This is the best expected cost when we model uncertainty.
    #
    # STEP 2: Compute EEV (Expected result of Expected Value solution)
    #   a) Solve the deterministic model with EXPECTED_YIELDS -> get x_EV (land allocations)
    #   b) For EACH scenario s:
    #      - Fix x = x_EV (the mean-value solution) and solve only the second stage
    #      - One way: build a deterministic model for scenario s but fix x to x_EV
    #        by adding constraints x[c] == x_EV[c], or by setting x[c].fix(x_EV_value)
    #      - Record the objective value for this scenario
    #   c) EEV = sum(p_s * objective_s for all scenarios)
    #   EEV represents: "what happens if I use the deterministic solution in a stochastic world?"
    #
    #   Hint: To fix a Pyomo variable, use model.x[c].fix(value).
    #   A fixed variable is treated as a parameter -- the solver cannot change it.
    #
    # STEP 3: Compute WS (Wait-and-See)
    #   For each scenario s independently:
    #     - Solve build_deterministic_farmer_model(scenario_data[s]["yields"])
    #     - Record the optimal objective
    #   WS = sum(p_s * optimal_obj_s for all scenarios)
    #   WS represents: "what if I had a perfect forecast for each scenario?"
    #   This is a LOWER BOUND -- we can never do better than perfect information.
    #
    # STEP 4: Compute the value metrics
    #   VSS = EEV - RP    (should be >= 0; larger means uncertainty matters more)
    #   EVPI = RP - WS    (should be >= 0; smaller means our stochastic solution is good)
    #
    # Return {"RP": ..., "EEV": ..., "WS": ..., "VSS": ..., "EVPI": ...}
    #
    # INTERPRETATION:
    #   - VSS answers: "How much do I save by using the stochastic model vs the naive
    #     deterministic approach?" If VSS = $0, uncertainty doesn't matter for decisions.
    #   - EVPI answers: "How much would I pay for a perfect weather forecast?" If EVPI is
    #     small, our stochastic solution already captures most of the achievable value.
    raise NotImplementedError("TODO(human): compute RP, EEV, WS, VSS, EVPI")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_stochastic_solution(model: pyo.ConcreteModel, scenario_data: dict) -> None:
    """Print the solution of a stochastic farmer model."""
    print(f"\n{'=' * 65}")
    print(f"  Stochastic Farmer Model -- Solution")
    print(f"{'=' * 65}")
    print(f"  Objective (min cost): {pyo.value(model.obj):>12,.2f}")
    print(f"\n  First-stage decisions (land allocation):")
    for c in CROPS:
        print(f"    {c:>12}: {pyo.value(model.x[c]):>8.1f} acres")
    total_planted = sum(pyo.value(model.x[c]) for c in CROPS)
    print(f"    {'TOTAL':>12}: {total_planted:>8.1f} / {TOTAL_LAND:.0f} acres")

    for s in scenario_data:
        blk = model.scenarios[s]
        print(f"\n  Scenario '{s}' (p={scenario_data[s]['probability']:.3f}):")
        print(f"    Sell wheat:      {pyo.value(blk.y_sell_wheat):>8.1f} tons")
        print(f"    Sell corn:       {pyo.value(blk.y_sell_corn):>8.1f} tons")
        print(f"    Sell beets (fav):{pyo.value(blk.y_sell_beets_fav):>8.1f} tons")
        print(f"    Sell beets (exc):{pyo.value(blk.y_sell_beets_exc):>8.1f} tons")
        print(f"    Buy wheat:       {pyo.value(blk.y_buy_wheat):>8.1f} tons")
        print(f"    Buy corn:        {pyo.value(blk.y_buy_corn):>8.1f} tons")


def print_deterministic_solution(model: pyo.ConcreteModel, label: str) -> None:
    """Print the solution of a deterministic farmer model."""
    print(f"\n  {label}:")
    print(f"    Objective (min cost): {pyo.value(model.obj):>12,.2f}")
    for c in CROPS:
        print(f"    {c:>12}: {pyo.value(model.x[c]):>8.1f} acres")


def print_value_metrics(metrics: dict) -> None:
    """Print VSS and EVPI analysis."""
    print(f"\n{'=' * 65}")
    print(f"  Value Metrics")
    print(f"{'=' * 65}")
    print(f"  WS  (Wait-and-See, perfect info):   {metrics['WS']:>12,.2f}")
    print(f"  RP  (Recourse Problem, stochastic):  {metrics['RP']:>12,.2f}")
    print(f"  EEV (Expected value of EV solution): {metrics['EEV']:>12,.2f}")
    print(f"  ---")
    print(f"  EVPI = RP - WS  = {metrics['EVPI']:>12,.2f}  (value of perfect forecast)")
    print(f"  VSS  = EEV - RP = {metrics['VSS']:>12,.2f}  (value of stochastic model)")
    print(f"\n  Interpretation:")
    if metrics["VSS"] > 0:
        print(f"    The stochastic model saves ${metrics['VSS']:,.2f} over the deterministic approach.")
    else:
        print(f"    The deterministic solution is equally good -- uncertainty doesn't matter here.")
    if metrics["EVPI"] < metrics["VSS"]:
        print(f"    EVPI < VSS: our stochastic solution captures most of the achievable improvement.")
    else:
        print(f"    There is still ${metrics['EVPI']:,.2f} of value in better forecasting.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 65)
    print("Phase 1: Two-Stage Stochastic LP -- The Farmer's Problem")
    print("=" * 65)

    # --- Solve the stochastic model ---
    print("\n--- Stochastic Model (Extensive Form) ---")
    stoch_model = build_stochastic_farmer_model(SCENARIO_DATA)
    print_stochastic_solution(stoch_model, SCENARIO_DATA)

    # --- Solve the deterministic model (average yields) ---
    print("\n--- Deterministic Model (Expected Yields) ---")
    det_model = build_deterministic_farmer_model(EXPECTED_YIELDS)
    print_deterministic_solution(det_model, "Deterministic (mean yields)")

    # --- VSS and EVPI ---
    print("\n--- Value of Stochastic Solution & Perfect Information ---")
    metrics = compute_vss_and_evpi(SCENARIO_DATA)
    print_value_metrics(metrics)

    # --- Compare first-stage decisions ---
    print(f"\n{'=' * 65}")
    print(f"  First-Stage Decision Comparison")
    print(f"{'=' * 65}")
    print(f"  {'Crop':>12} | {'Stochastic':>12} | {'Deterministic':>14}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*14}")
    for c in CROPS:
        x_stoch = pyo.value(stoch_model.x[c])
        x_det = pyo.value(det_model.x[c])
        print(f"  {c:>12} | {x_stoch:>12.1f} | {x_det:>14.1f}")

    print("\n[Phase 1 complete]")


if __name__ == "__main__":
    main()
