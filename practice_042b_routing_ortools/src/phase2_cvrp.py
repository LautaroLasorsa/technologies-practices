"""Phase 2: Capacitated Vehicle Routing Problem (CVRP) with OR-Tools.

Multiple vehicles with capacity limits serve customers with demands from a central depot.
Demonstrates: dimensions, demand callbacks, AddDimensionWithVehicleCapacity, multi-vehicle routing.
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_cvrp_data() -> dict:
    """Create a CVRP instance: 16 customers + depot, 4 vehicles.

    Based on the classic OR-Tools CVRP example with modifications.
    The depot is node 0. Customers are nodes 1-16.
    """
    data = {}

    # Distance matrix: 17 locations (depot=0, customers=1..16)
    # Distances are symmetric and satisfy triangle inequality.
    data["distance_matrix"] = [
        [0, 548, 776, 696, 582, 274, 502, 194, 308, 194, 536, 502, 388, 354, 468, 776, 662],
        [548, 0, 684, 308, 194, 502, 730, 354, 696, 742, 1084, 594, 480, 674, 1016, 868, 1210],
        [776, 684, 0, 992, 878, 502, 274, 810, 468, 742, 400, 1278, 1164, 1130, 788, 1552, 754],
        [696, 308, 992, 0, 114, 650, 878, 502, 844, 890, 1232, 514, 628, 822, 1164, 560, 1358],
        [582, 194, 878, 114, 0, 536, 764, 388, 730, 776, 1118, 400, 514, 708, 1050, 674, 1244],
        [274, 502, 502, 650, 536, 0, 228, 308, 194, 240, 582, 776, 662, 628, 514, 1050, 708],
        [502, 730, 274, 878, 764, 228, 0, 536, 194, 468, 354, 1004, 890, 856, 514, 1278, 480],
        [194, 354, 810, 502, 388, 308, 536, 0, 342, 388, 730, 468, 354, 320, 662, 742, 856],
        [308, 696, 468, 844, 730, 194, 194, 342, 0, 274, 388, 810, 696, 662, 320, 1084, 514],
        [194, 742, 742, 890, 776, 240, 468, 388, 274, 0, 342, 536, 422, 388, 274, 810, 468],
        [536, 1084, 400, 1232, 1118, 582, 354, 730, 388, 342, 0, 878, 764, 730, 388, 1152, 354],
        [502, 594, 1278, 514, 400, 776, 1004, 468, 810, 536, 878, 0, 114, 308, 650, 274, 844],
        [388, 480, 1164, 628, 514, 662, 890, 354, 696, 422, 764, 114, 0, 194, 536, 388, 730],
        [354, 674, 1130, 822, 708, 628, 856, 320, 662, 388, 730, 308, 194, 0, 342, 422, 536],
        [468, 1016, 788, 1164, 1050, 514, 514, 662, 320, 274, 388, 650, 536, 342, 0, 764, 194],
        [776, 868, 1552, 560, 674, 1050, 1278, 742, 1084, 810, 1152, 274, 388, 422, 764, 0, 798],
        [662, 1210, 754, 1358, 1244, 708, 480, 856, 514, 468, 354, 844, 730, 536, 194, 798, 0],
    ]

    # Demand at each location (depot has 0 demand)
    data["demands"] = [0, 1, 1, 2, 4, 2, 4, 8, 8, 1, 2, 1, 2, 4, 4, 8, 8]

    # Vehicle capacities (4 vehicles, each with capacity 15)
    data["vehicle_capacities"] = [15, 15, 15, 15]

    data["num_vehicles"] = 4
    data["depot"] = 0

    return data


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_cvrp_solution(
    data: dict,
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
) -> None:
    """Print CVRP routes with load at each stop.

    Args:
        data: Problem data dict.
        manager: The routing index manager.
        routing: The routing model.
        solution: The assignment returned by the solver.
    """
    total_distance = 0
    total_load = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_loads = []
        route_distance = 0
        cumul_load = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            cumul_load += data["demands"][node]
            route_nodes.append(node)
            route_loads.append(cumul_load)
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        # Add final depot
        node = manager.IndexToNode(index)
        route_nodes.append(node)
        route_loads.append(cumul_load)

        # Display
        route_str = " -> ".join(
            f"{n}(load={l})" for n, l in zip(route_nodes, route_loads)
        )
        cap = data["vehicle_capacities"][vehicle_id]
        print(f"  Vehicle {vehicle_id}: {route_str}")
        print(f"    Distance: {route_distance}, Load: {cumul_load}/{cap}")

        total_distance += route_distance
        total_load += cumul_load

    print(f"\n  Total distance: {total_distance}")
    print(f"  Total load delivered: {total_load}")


# ---------------------------------------------------------------------------
# TODO(human): Implement demand callback and solver
# ---------------------------------------------------------------------------

def create_demand_callback(
    manager: pywrapcp.RoutingIndexManager,
    demands: list[int],
):
    """Create a unary transit callback that returns the demand at a location.

    Args:
        manager: The routing index manager.
        demands: List where demands[i] = demand at node i.

    Returns:
        A callable(from_index) -> int suitable for RegisterUnaryTransitCallback.
    """
    # TODO(human): Demand callback for CVRP
    #
    # Unlike the distance callback (which takes TWO indices: from and to), the
    # demand callback takes only ONE index: the location being visited. This is
    # because demand is a property of the location, not of the arc between two
    # locations.
    #
    # Pattern:
    #   def demand_callback(from_index):
    #       from_node = manager.IndexToNode(from_index)
    #       return demands[from_node]
    #
    # This is registered with RegisterUnaryTransitCallback (not RegisterTransitCallback).
    # The "unary" indicates it depends on a single index, not a pair.
    #
    # The demand value is what gets ACCUMULATED in the Capacity dimension.
    # At each stop, the vehicle's cumulative load increases by demands[node].
    # The solver ensures the cumul never exceeds the vehicle's capacity.
    #
    # Important: depot (node 0) should have demand 0. If it doesn't, the
    # vehicle starts with a non-zero load, eating into its capacity.
    #
    # Return: a closure or function that takes (from_index) -> int
    raise NotImplementedError("TODO(human): create_demand_callback")


def create_distance_callback(
    manager: pywrapcp.RoutingIndexManager,
    distance_matrix: list[list[int]],
):
    """Create a transit callback for distances (same as Phase 1).

    Args:
        manager: The routing index manager.
        distance_matrix: NxN distance matrix.

    Returns:
        A callable(from_index, to_index) -> int.
    """
    # TODO(human): Distance callback (same pattern as Phase 1)
    #
    # This is identical to Phase 1's create_distance_callback.
    # Convert from_index and to_index to node indices via manager.IndexToNode(),
    # then look up distance_matrix[from_node][to_node].
    #
    # In CVRP you still need this callback for arc costs (minimizing total
    # distance traveled by all vehicles). The demand callback is SEPARATE —
    # it feeds the Capacity dimension, not the arc cost.
    #
    # Return: a closure that takes (from_index, to_index) -> int
    raise NotImplementedError("TODO(human): create_distance_callback")


def solve_cvrp(data: dict) -> None:
    """Solve a CVRP instance using OR-Tools routing solver.

    Args:
        data: Problem data dict with keys: distance_matrix, demands,
              vehicle_capacities, num_vehicles, depot.
    """
    # TODO(human): Set up and solve the CVRP
    #
    # Steps:
    #
    # 1. Create RoutingIndexManager:
    #      manager = pywrapcp.RoutingIndexManager(
    #          len(data["distance_matrix"]),   # num locations
    #          data["num_vehicles"],            # num vehicles
    #          data["depot"]                    # depot node
    #      )
    #
    # 2. Create RoutingModel:
    #      routing = pywrapcp.RoutingModel(manager)
    #
    # 3. Register and set the DISTANCE callback as arc cost (same as Phase 1):
    #      distance_cb = create_distance_callback(manager, data["distance_matrix"])
    #      transit_cb_index = routing.RegisterTransitCallback(distance_cb)
    #      routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)
    #
    # 4. Register the DEMAND callback and add the Capacity dimension:
    #      demand_cb = create_demand_callback(manager, data["demands"])
    #      demand_cb_index = routing.RegisterUnaryTransitCallback(demand_cb)
    #
    #      routing.AddDimensionWithVehicleCapacity(
    #          demand_cb_index,               # callback index
    #          0,                             # slack = 0 (no slack for capacity)
    #          data["vehicle_capacities"],    # max cumul per vehicle
    #          True,                          # fix_start_cumul_to_zero (vehicles start empty)
    #          "Capacity"                     # dimension name
    #      )
    #
    #    What this does: creates a "Capacity" dimension that tracks the cumulative
    #    demand picked up by each vehicle. At each location visited, the cumul
    #    increases by demands[node]. The constraint is: cumul <= vehicle_capacity
    #    at every point along the route. Slack=0 means no "buffer" — the vehicle
    #    can't magically shed load between stops.
    #
    # 5. Configure search parameters:
    #      search_params = pywrapcp.DefaultRoutingSearchParameters()
    #      search_params.first_solution_strategy = (
    #          routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    #      )
    #      search_params.local_search_metaheuristic = (
    #          routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    #      )
    #      search_params.time_limit.FromSeconds(3)
    #
    # 6. Solve and print:
    #      solution = routing.SolveWithParameters(search_params)
    #      if solution:
    #          print_cvrp_solution(data, manager, routing, solution)
    raise NotImplementedError("TODO(human): solve_cvrp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Capacitated VRP (CVRP) ===\n")

    data = create_cvrp_data()
    print(f"Locations: {len(data['distance_matrix'])} (depot + {len(data['distance_matrix'])-1} customers)")
    print(f"Vehicles: {data['num_vehicles']} (capacities: {data['vehicle_capacities']})")
    print(f"Total demand: {sum(data['demands'])}")
    print()

    solve_cvrp(data)

    # --- Experiment: what if we use fewer vehicles? ---
    print("\n--- Experiment: 3 vehicles with capacity 20 ---\n")
    data2 = create_cvrp_data()
    data2["num_vehicles"] = 3
    data2["vehicle_capacities"] = [20, 20, 20]
    solve_cvrp(data2)

    # --- Experiment: heterogeneous fleet ---
    print("\n--- Experiment: heterogeneous fleet (capacities: 10, 15, 20, 25) ---\n")
    data3 = create_cvrp_data()
    data3["vehicle_capacities"] = [10, 15, 20, 25]
    solve_cvrp(data3)


if __name__ == "__main__":
    main()
