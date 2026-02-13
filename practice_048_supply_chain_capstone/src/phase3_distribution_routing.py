"""Phase 3: Vehicle Routing for Distribution (Multi-Depot CVRP).

Operational supply chain decision: how to route vehicles from warehouses
(open facilities from Phase 1) to customers to deliver goods.

This is the Multi-Depot Capacitated Vehicle Routing Problem (MDCVRP) --
each depot (warehouse) has its own fleet of vehicles, and each customer
must be served by exactly one vehicle from one depot. The problem combines
facility-customer assignment (which depot serves which customer) with
route construction (in what order does each vehicle visit its customers).

OR-Tools routing solver handles multi-depot VRP by specifying different
start/end depots for different vehicles.

Real-world applications: last-mile delivery from distribution centers (Amazon),
grocery delivery (Walmart, Instacart), field service routing.

OR-Tools patterns used:
  - RoutingIndexManager with per-vehicle start/end indices
  - Distance callback via RegisterTransitCallback
  - Demand callback via RegisterUnaryTransitCallback
  - AddDimensionWithVehicleCapacity for load tracking
  - GUIDED_LOCAL_SEARCH metaheuristic
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp
import numpy as np
from typing import NamedTuple


# ============================================================================
# Data structures
# ============================================================================

class RoutingData(NamedTuple):
    """Data for multi-depot CVRP."""
    num_depots: int                # Number of depot locations
    num_customers: int             # Number of customer locations
    depot_locations: np.ndarray    # (num_depots, 2) coordinates
    customer_locations: np.ndarray # (num_customers, 2) coordinates
    customer_demands: np.ndarray   # (num_customers,) demand at each customer
    vehicles_per_depot: int        # Number of vehicles at each depot
    vehicle_capacity: int          # Capacity of each vehicle


# ============================================================================
# Instance generation
# ============================================================================

def generate_routing_instance(
    num_depots: int = 3,
    num_customers: int = 30,
    vehicles_per_depot: int = 3,
    vehicle_capacity: int = 100,
    seed: int = 42,
) -> RoutingData:
    """Generate a multi-depot CVRP instance.

    Depots are placed in a pattern that spreads them across the region.
    Customers are scattered uniformly. Demands are random.

    Args:
        num_depots: Number of depots (warehouses).
        num_customers: Number of customer locations.
        vehicles_per_depot: Vehicles available at each depot.
        vehicle_capacity: Max load per vehicle.
        seed: Random seed.

    Returns:
        RoutingData with all problem parameters.
    """
    rng = np.random.default_rng(seed)

    # Spread depots across the region
    depot_locations = np.zeros((num_depots, 2))
    for k in range(num_depots):
        angle = 2 * np.pi * k / num_depots
        depot_locations[k] = [50 + 30 * np.cos(angle), 50 + 30 * np.sin(angle)]

    # Customers scattered uniformly
    customer_locations = rng.uniform(5, 95, size=(num_customers, 2))

    # Customer demands: 10-40 units
    customer_demands = rng.integers(10, 41, size=num_customers)

    total_demand = customer_demands.sum()
    total_capacity = num_depots * vehicles_per_depot * vehicle_capacity

    print(f"  Routing instance: {num_depots} depots, {num_customers} customers")
    print(f"  Vehicles: {num_depots * vehicles_per_depot} total "
          f"({vehicles_per_depot}/depot, capacity={vehicle_capacity})")
    print(f"  Total demand: {total_demand}, total capacity: {total_capacity}")

    return RoutingData(
        num_depots=num_depots,
        num_customers=num_customers,
        depot_locations=depot_locations,
        customer_locations=customer_locations,
        customer_demands=customer_demands,
        vehicles_per_depot=vehicles_per_depot,
        vehicle_capacity=vehicle_capacity,
    )


def build_distance_matrix(data: RoutingData) -> np.ndarray:
    """Build a full distance matrix for all locations (depots + customers).

    Locations are ordered: [depot_0, depot_1, ..., depot_K, cust_0, cust_1, ...].

    Args:
        data: RoutingData instance.

    Returns:
        Integer distance matrix of shape (N, N) where N = num_depots + num_customers.
    """
    all_locations = np.vstack([data.depot_locations, data.customer_locations])
    n = len(all_locations)
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dist[i, j] = int(np.round(np.linalg.norm(all_locations[i] - all_locations[j])))
    return dist


# ============================================================================
# TODO(human): Set up and solve multi-depot CVRP with OR-Tools
# ============================================================================

def solve_multi_depot_cvrp(data: RoutingData) -> dict | None:
    """Solve the multi-depot CVRP using OR-Tools routing solver.

    Key difference from single-depot: each vehicle has its own start and end
    depot. OR-Tools handles this via the RoutingIndexManager constructor that
    takes lists of start/end node indices for each vehicle.

    Args:
        data: RoutingData instance.

    Returns:
        Dictionary with 'total_distance', 'routes' (list of route dicts),
        and 'unserved' (list of unserved customer indices), or None if infeasible.
    """
    # TODO(human): Implement multi-depot CVRP with OR-Tools
    #
    # Multi-depot VRP is the operational counterpart to facility location (Phase 1).
    # Once you know WHICH warehouses are open, you need to decide HOW to route
    # delivery vehicles from each warehouse to its assigned customers.
    #
    # The key OR-Tools feature for multi-depot is the RoutingIndexManager
    # constructor that accepts per-vehicle start/end indices:
    #   RoutingIndexManager(num_locations, num_vehicles, starts, ends)
    # where starts[v] and ends[v] are the depot node indices for vehicle v.
    #
    # STEP 1: Build the distance matrix
    #   dist_matrix = build_distance_matrix(data)
    #   Node ordering: [depot_0, depot_1, ..., depot_K, cust_0, cust_1, ...]
    #   So depot k is at node index k, customer i is at node index (num_depots + i).
    #
    # STEP 2: Set up vehicle start/end indices
    #   Each depot has vehicles_per_depot vehicles. Vehicle v at depot k
    #   starts and ends at node index k.
    #   num_vehicles = data.num_depots * data.vehicles_per_depot
    #   starts = []
    #   ends = []
    #   for k in range(data.num_depots):
    #       for _ in range(data.vehicles_per_depot):
    #           starts.append(k)
    #           ends.append(k)
    #
    # STEP 3: Create RoutingIndexManager with per-vehicle depots
    #   manager = pywrapcp.RoutingIndexManager(
    #       len(dist_matrix),    # total locations (depots + customers)
    #       num_vehicles,
    #       starts,              # start depot for each vehicle
    #       ends,                # end depot for each vehicle
    #   )
    #   This tells OR-Tools that vehicle 0 starts/ends at node starts[0],
    #   vehicle 1 at starts[1], etc. The solver handles the multi-depot
    #   structure internally.
    #
    # STEP 4: Create RoutingModel and register distance callback
    #   routing = pywrapcp.RoutingModel(manager)
    #   def distance_callback(from_index, to_index):
    #       from_node = manager.IndexToNode(from_index)
    #       to_node = manager.IndexToNode(to_index)
    #       return dist_matrix[from_node][to_node]
    #   transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    #   routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
    #
    # STEP 5: Register demand callback and add capacity dimension
    #   Demands: depot nodes have 0 demand, customer nodes have their demand.
    #   demands = [0] * data.num_depots + list(data.customer_demands)
    #   def demand_callback(from_index):
    #       from_node = manager.IndexToNode(from_index)
    #       return demands[from_node]
    #   demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    #   routing.AddDimensionWithVehicleCapacity(
    #       demand_cb_idx,
    #       0,                                          # zero slack
    #       [data.vehicle_capacity] * num_vehicles,     # vehicle capacities
    #       True,                                       # start cumul at 0
    #       "Capacity",
    #   )
    #
    # STEP 6: Allow dropping customers (with high penalty) in case infeasible
    #   for cust_idx in range(data.num_customers):
    #       node = data.num_depots + cust_idx
    #       routing.AddDisjunction(
    #           [manager.NodeToIndex(node)],
    #           100000,  # penalty for not visiting
    #       )
    #
    # STEP 7: Search parameters
    #   search_params = pywrapcp.DefaultRoutingSearchParameters()
    #   search_params.first_solution_strategy = (
    #       routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    #   )
    #   search_params.local_search_metaheuristic = (
    #       routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    #   )
    #   search_params.time_limit.FromSeconds(5)
    #
    # STEP 8: Solve and extract routes
    #   solution = routing.SolveWithParameters(search_params)
    #   if not solution:
    #       return None
    #
    #   Extract routes: for each vehicle, follow NextVar from start to end:
    #   routes = []
    #   total_distance = 0
    #   for v in range(num_vehicles):
    #       index = routing.Start(v)
    #       route_nodes = []
    #       route_distance = 0
    #       route_load = 0
    #       while not routing.IsEnd(index):
    #           node = manager.IndexToNode(index)
    #           route_nodes.append(node)
    #           route_load += demands[node]
    #           prev = index
    #           index = solution.Value(routing.NextVar(index))
    #           route_distance += routing.GetArcCostForVehicle(prev, index, v)
    #       route_nodes.append(manager.IndexToNode(index))  # end depot
    #       if len(route_nodes) > 2:  # non-empty route (not just depot->depot)
    #           routes.append({
    #               "vehicle": v,
    #               "depot": starts[v],
    #               "nodes": route_nodes,
    #               "distance": route_distance,
    #               "load": route_load,
    #           })
    #           total_distance += route_distance
    #
    #   Return {"total_distance": total_distance, "routes": routes, "unserved": []}
    raise NotImplementedError("TODO(human): multi-depot CVRP with OR-Tools")


# ============================================================================
# TODO(human): Single-depot comparison
# ============================================================================

def solve_single_depot_cvrp(data: RoutingData) -> dict | None:
    """Solve single-depot CVRP (all vehicles from a central depot).

    Uses the centroid of all customer locations as the single depot.
    This serves as a baseline to show the benefit of multiple depots.

    Args:
        data: RoutingData instance.

    Returns:
        Same format as solve_multi_depot_cvrp, or None if infeasible.
    """
    # TODO(human): Solve single-depot CVRP for comparison
    #
    # The single-depot case demonstrates WHY multi-depot matters:
    # - With one central depot, all vehicles must travel out and back from
    #   the same point. Customers far from the depot incur high travel costs.
    # - With multiple depots, vehicles can start closer to their assigned
    #   customers, reducing "stem distance" (depot-to-first-customer).
    #
    # IMPLEMENTATION:
    # 1. Create a single depot at the centroid of customer locations:
    #      centroid = data.customer_locations.mean(axis=0)
    # 2. Build a modified RoutingData with 1 depot, same customers,
    #    total vehicles = data.num_depots * data.vehicles_per_depot
    # 3. Build distance matrix: [centroid] + customer_locations
    # 4. Set up standard single-depot RoutingIndexManager:
    #      manager = pywrapcp.RoutingIndexManager(N, num_vehicles, depot=0)
    # 5. Register callbacks, add capacity dimension, solve (same as multi-depot
    #    but simpler -- all vehicles start/end at node 0)
    # 6. Extract and return routes in the same format
    #
    # EXPECTED RESULT: Single-depot total distance should be HIGHER than
    # multi-depot. The difference quantifies the value of distributed warehousing
    # -- a direct connection to Phase 1's facility location decisions.
    raise NotImplementedError("TODO(human): single-depot CVRP baseline")


# ============================================================================
# Helpers (provided)
# ============================================================================

def print_routing_solution(result: dict | None, label: str) -> None:
    """Print routing solution details."""
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")

    if result is None:
        print("  No feasible solution found!")
        return

    print(f"  Total distance: {result['total_distance']}")
    print(f"  Active routes: {len(result['routes'])}")

    for route in result["routes"]:
        nodes_str = " -> ".join(str(n) for n in route["nodes"])
        print(f"\n  Vehicle {route['vehicle']} (depot {route['depot']}):")
        print(f"    Route: {nodes_str}")
        print(f"    Distance: {route['distance']}, Load: {route['load']}")

    if result.get("unserved"):
        print(f"\n  WARNING: Unserved customers: {result['unserved']}")


def compare_routing_solutions(multi: dict | None, single: dict | None) -> None:
    """Compare multi-depot vs single-depot routing costs."""
    print(f"\n{'=' * 70}")
    print(f"  Multi-Depot vs Single-Depot Comparison")
    print(f"{'=' * 70}")

    if multi is None or single is None:
        print("  Cannot compare -- one or both solutions infeasible.")
        return

    md_dist = multi["total_distance"]
    sd_dist = single["total_distance"]
    savings = (sd_dist - md_dist) / sd_dist * 100

    print(f"  Multi-depot distance:  {md_dist:>8}")
    print(f"  Single-depot distance: {sd_dist:>8}")
    print(f"  Distance savings:      {savings:>7.1f}%")
    print(f"  Active routes (multi): {len(multi['routes'])}")
    print(f"  Active routes (single):{len(single['routes'])}")

    if savings > 0:
        print(f"\n  Multi-depot saves {savings:.1f}% in routing distance.")
        print(f"  This quantifies the OPERATIONAL benefit of distributed warehousing.")
    else:
        print(f"\n  Single-depot is better (or equal) -- the depot configuration")
        print(f"  does not help for this customer distribution.")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("Phase 3: Vehicle Routing for Distribution (Multi-Depot CVRP)")
    print("=" * 70)

    # --- Generate routing instance ---
    print("\n--- Generating routing instance ---")
    data = generate_routing_instance(
        num_depots=3,
        num_customers=30,
        vehicles_per_depot=3,
        vehicle_capacity=100,
        seed=42,
    )

    # --- Multi-depot CVRP ---
    print("\n--- Solving Multi-Depot CVRP ---")
    multi_result = solve_multi_depot_cvrp(data)
    print_routing_solution(multi_result, "Multi-Depot CVRP Solution")

    # --- Single-depot baseline ---
    print("\n--- Solving Single-Depot CVRP (baseline) ---")
    single_result = solve_single_depot_cvrp(data)
    print_routing_solution(single_result, "Single-Depot CVRP Solution")

    # --- Comparison ---
    compare_routing_solutions(multi_result, single_result)

    # --- Experiment: more depots ---
    print("\n--- Experiment: 5 depots ---")
    data5 = generate_routing_instance(
        num_depots=5,
        num_customers=30,
        vehicles_per_depot=2,
        vehicle_capacity=100,
        seed=42,
    )
    result5 = solve_multi_depot_cvrp(data5)
    if result5:
        print(f"  5-depot total distance: {result5['total_distance']}")
        if multi_result:
            pct = (multi_result["total_distance"] - result5["total_distance"]) / multi_result["total_distance"] * 100
            print(f"  vs 3-depot: {pct:+.1f}% change")

    print("\n[Phase 3 complete]")


if __name__ == "__main__":
    main()
