"""Phase 3: VRP with Time Windows (VRPTW) with OR-Tools.

Each customer must be visited within a specific time window [earliest, latest].
Vehicles travel between locations with known travel times.
Demonstrates: time dimension, slack variables, CumulVar.SetRange, time window constraints.
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_vrptw_data() -> dict:
    """Create a VRPTW instance: 16 customers + depot, 4 vehicles.

    Based on the classic OR-Tools VRPTW example.
    Time windows define when each location can be serviced.
    """
    data = {}

    # Time matrix (travel time between locations in minutes)
    # Symmetric; depot = node 0.
    data["time_matrix"] = [
        [0, 6, 9, 8, 7, 3, 6, 2, 3, 2, 6, 6, 4, 4, 5, 9, 7],
        [6, 0, 8, 3, 2, 6, 8, 4, 8, 8, 13, 7, 5, 8, 12, 10, 14],
        [9, 8, 0, 11, 10, 6, 3, 9, 5, 8, 4, 15, 14, 13, 9, 18, 9],
        [8, 3, 11, 0, 1, 7, 10, 6, 10, 10, 14, 6, 7, 9, 14, 6, 16],
        [7, 2, 10, 1, 0, 6, 9, 4, 8, 9, 13, 4, 6, 8, 12, 8, 14],
        [3, 6, 6, 7, 6, 0, 2, 3, 2, 2, 7, 9, 7, 7, 6, 12, 8],
        [6, 8, 3, 10, 9, 2, 0, 6, 2, 5, 4, 12, 10, 10, 6, 15, 5],
        [2, 4, 9, 6, 4, 3, 6, 0, 4, 4, 8, 5, 4, 3, 7, 8, 10],
        [3, 8, 5, 10, 8, 2, 2, 4, 0, 3, 4, 9, 8, 7, 3, 13, 6],
        [2, 8, 8, 10, 9, 2, 5, 4, 3, 0, 4, 6, 5, 4, 3, 9, 5],
        [6, 13, 4, 14, 13, 7, 4, 8, 4, 4, 0, 10, 9, 8, 4, 13, 4],
        [6, 7, 15, 6, 4, 9, 12, 5, 9, 6, 10, 0, 1, 3, 7, 3, 10],
        [4, 5, 14, 7, 6, 7, 10, 4, 8, 5, 9, 1, 0, 2, 6, 4, 8],
        [4, 8, 13, 9, 8, 7, 10, 3, 7, 4, 8, 3, 2, 0, 4, 5, 6],
        [5, 12, 9, 14, 12, 6, 6, 7, 3, 3, 4, 7, 6, 4, 0, 9, 2],
        [9, 10, 18, 6, 8, 12, 15, 8, 13, 9, 13, 3, 4, 5, 9, 0, 9],
        [7, 14, 9, 16, 14, 8, 5, 10, 6, 5, 4, 10, 8, 6, 2, 9, 0],
    ]

    # Time windows: [earliest_arrival, latest_arrival] for each location.
    # The vehicle must arrive at location i between time_windows[i][0] and time_windows[i][1].
    # Depot (node 0) has window [0, 50] — all vehicles must return by time 50.
    data["time_windows"] = [
        (0, 50),    # 0: depot
        (7, 12),    # 1
        (10, 15),   # 2
        (16, 18),   # 3
        (10, 13),   # 4
        (0, 5),     # 5
        (5, 10),    # 6
        (0, 4),     # 7
        (5, 10),    # 8
        (0, 3),     # 9
        (10, 16),   # 10
        (10, 15),   # 11
        (0, 5),     # 12
        (5, 10),    # 13
        (7, 8),     # 14
        (10, 15),   # 15
        (11, 15),   # 16
    ]

    data["num_vehicles"] = 4
    data["depot"] = 0

    return data


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_vrptw_solution(
    data: dict,
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
) -> None:
    """Print VRPTW routes with arrival times and time windows.

    Args:
        data: Problem data dict.
        manager: The routing index manager.
        routing: The routing model.
        solution: The assignment returned by the solver.
    """
    time_dimension = routing.GetDimensionOrDie("Time")
    total_time = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_info = []

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            time_var = time_dimension.CumulVar(index)
            arrival = solution.Min(time_var)
            window = data["time_windows"][node]
            route_info.append((node, arrival, window))
            index = solution.Value(routing.NextVar(index))

        # Final depot return
        node = manager.IndexToNode(index)
        time_var = time_dimension.CumulVar(index)
        arrival = solution.Min(time_var)
        window = data["time_windows"][node]
        route_info.append((node, arrival, window))

        route_time = route_info[-1][1] - route_info[0][1]
        total_time += route_time

        print(f"  Vehicle {vehicle_id}:")
        for node, arrival, (earliest, latest) in route_info:
            status = "ok" if earliest <= arrival <= latest else "LATE!"
            print(f"    Node {node:>2} | arrive={arrival:>3} | window=[{earliest:>2}, {latest:>2}] | {status}")
        print(f"    Route time: {route_time}")
        print()

    print(f"  Total time across all vehicles: {total_time}")


# ---------------------------------------------------------------------------
# TODO(human): Implement time callback and solver
# ---------------------------------------------------------------------------

def create_time_callback(
    manager: pywrapcp.RoutingIndexManager,
    time_matrix: list[list[int]],
):
    """Create a transit callback that returns travel time between two locations.

    Args:
        manager: The routing index manager.
        time_matrix: NxN matrix where time_matrix[i][j] = travel time from i to j.

    Returns:
        A callable(from_index, to_index) -> int suitable for RegisterTransitCallback.
    """
    # TODO(human): Time callback for VRPTW
    #
    # This follows the same pattern as the distance callback from Phase 1:
    #   1. Convert from_index -> from_node via manager.IndexToNode()
    #   2. Convert to_index -> to_node via manager.IndexToNode()
    #   3. Return time_matrix[from_node][to_node]
    #
    # The key difference from CVRP's demand callback: this is a BINARY callback
    # (depends on both from and to), registered with RegisterTransitCallback.
    # Travel time depends on WHICH two locations you're traveling between.
    #
    # This callback will be used for BOTH:
    #   a) The arc cost evaluator (minimizing total travel time)
    #   b) The Time dimension (tracking cumulative time along the route)
    #
    # Same callback, two purposes. The arc cost measures "how expensive is this
    # route?" and the Time dimension tracks "what time does the vehicle arrive
    # at each stop?" — enabling time window constraints.
    #
    # Return: a closure that takes (from_index, to_index) -> int
    raise NotImplementedError("TODO(human): create_time_callback")


def solve_vrptw(data: dict) -> None:
    """Solve a VRPTW instance using OR-Tools routing solver.

    Args:
        data: Problem data dict with keys: time_matrix, time_windows,
              num_vehicles, depot.
    """
    # TODO(human): Set up and solve the VRPTW
    #
    # Steps:
    #
    # 1. Create RoutingIndexManager and RoutingModel (same pattern as Phase 2).
    #
    # 2. Register the TIME callback and set as arc cost:
    #      time_cb = create_time_callback(manager, data["time_matrix"])
    #      transit_cb_index = routing.RegisterTransitCallback(time_cb)
    #      routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_index)
    #
    # 3. Add the Time dimension — THIS IS THE KEY NEW CONCEPT:
    #
    #      routing.AddDimension(
    #          transit_cb_index,   # callback index (travel time between locations)
    #          30,                 # slack_max: maximum WAITING time at each location
    #          50,                 # max cumulative time (horizon — vehicle must finish by this time)
    #          False,              # fix_start_cumul_to_zero = False
    #          "Time"              # dimension name
    #      )
    #
    #    IMPORTANT differences from the Capacity dimension in Phase 2:
    #
    #    a) slack_max = 30 (not 0): Slack represents WAITING TIME. If a vehicle
    #       arrives at a location before the time window opens, it must WAIT.
    #       The slack variable captures this waiting. Example: vehicle arrives at
    #       node 5 at time 3, but the window is [7, 12]. Slack = 4 (wait until 7).
    #       Setting slack_max=0 would mean "no waiting allowed" — the vehicle must
    #       arrive exactly within the window, which is usually infeasible.
    #
    #    b) fix_start_cumul_to_zero = False: In CVRP, vehicles start with 0 load
    #       (True). But for time, vehicles may DEPART at different times. Setting
    #       False lets the solver choose each vehicle's departure time. If True,
    #       all vehicles depart at time 0, which may be overly restrictive.
    #
    #    c) max cumul = 50: This is the horizon — the latest any vehicle can
    #       complete its route. Match this to the depot's time window upper bound.
    #
    # 4. Set time windows on each location:
    #
    #      time_dimension = routing.GetDimensionOrDie("Time")
    #
    #      for location_idx, (earliest, latest) in enumerate(data["time_windows"]):
    #          if location_idx == data["depot"]:
    #              continue  # handle depot separately
    #          index = manager.NodeToIndex(location_idx)
    #          time_dimension.CumulVar(index).SetRange(earliest, latest)
    #
    #    CumulVar(index).SetRange(a, b) constrains the cumulative time at that
    #    location to be in [a, b]. If the vehicle arrives before 'a', it waits
    #    (slack absorbs the difference). If it would arrive after 'b', the route
    #    is infeasible — the solver must find a different route.
    #
    # 5. Set depot time windows (for vehicle start and end):
    #
    #      depot_idx = data["depot"]
    #      for vehicle_id in range(data["num_vehicles"]):
    #          start_index = routing.Start(vehicle_id)
    #          end_index = routing.End(vehicle_id)
    #          time_dimension.CumulVar(start_index).SetRange(
    #              data["time_windows"][depot_idx][0],
    #              data["time_windows"][depot_idx][1]
    #          )
    #          time_dimension.CumulVar(end_index).SetRange(
    #              data["time_windows"][depot_idx][0],
    #              data["time_windows"][depot_idx][1]
    #          )
    #
    # 6. Configure search parameters with GUIDED_LOCAL_SEARCH + time limit.
    #
    # 7. Solve and call print_vrptw_solution().
    raise NotImplementedError("TODO(human): solve_vrptw")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== VRP with Time Windows (VRPTW) ===\n")

    data = create_vrptw_data()
    n_locations = len(data["time_matrix"])
    print(f"Locations: {n_locations} (depot + {n_locations - 1} customers)")
    print(f"Vehicles: {data['num_vehicles']}")
    print(f"Depot time window: {data['time_windows'][data['depot']]}")
    print()

    # Print time windows for all customers
    print("Customer time windows:")
    for i, (earliest, latest) in enumerate(data["time_windows"]):
        if i == data["depot"]:
            continue
        print(f"  Node {i:>2}: [{earliest:>2}, {latest:>2}]")
    print()

    solve_vrptw(data)


if __name__ == "__main__":
    main()
