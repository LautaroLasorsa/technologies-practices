"""Phase 1: Traveling Salesman Problem with OR-Tools Routing Solver.

Solve TSP for 15 cities using RoutingIndexManager, RoutingModel, and callbacks.
Demonstrates: core routing workflow, arc cost evaluators, first-solution strategies,
local search metaheuristics.
"""

import numpy as np
from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_distance_matrix() -> list[list[int]]:
    """Create a 15-city distance matrix (symmetric, metric).

    This is a randomly generated but fixed instance so results are reproducible.
    Distances are in arbitrary units (think km or miles).
    """
    # 15-city instance — hand-crafted to have interesting structure
    # City 0 is the depot (start/end point for the salesman).
    rng = np.random.RandomState(42)
    n = 15
    # Generate random 2D coordinates, then compute Euclidean distances
    coords = rng.randint(0, 100, size=(n, 2))
    dist = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            dx = int(coords[i][0]) - int(coords[j][0])
            dy = int(coords[i][1]) - int(coords[j][1])
            dist[i][j] = int(np.sqrt(dx * dx + dy * dy) + 0.5)
    return dist.tolist()


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_tsp_solution(
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
) -> tuple[list[int], int]:
    """Extract and print the TSP tour from a solution.

    Args:
        manager: The routing index manager.
        routing: The routing model.
        solution: The assignment returned by the solver.

    Returns:
        Tuple of (tour as list of node indices, total distance).
    """
    index = routing.Start(0)  # single vehicle (vehicle 0)
    tour = []
    total_distance = 0
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        tour.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        total_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    tour.append(manager.IndexToNode(index))  # return to depot

    route_str = " -> ".join(str(n) for n in tour)
    print(f"  Tour: {route_str}")
    print(f"  Distance: {total_distance}")
    return tour, total_distance


# ---------------------------------------------------------------------------
# Available strategies for comparison
# ---------------------------------------------------------------------------

FIRST_SOLUTION_STRATEGIES = {
    "PATH_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC,
    "SAVINGS": routing_enums_pb2.FirstSolutionStrategy.SAVINGS,
    "CHRISTOFIDES": routing_enums_pb2.FirstSolutionStrategy.CHRISTOFIDES,
    "PARALLEL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION,
    "LOCAL_CHEAPEST_INSERTION": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_INSERTION,
    "GLOBAL_CHEAPEST_ARC": routing_enums_pb2.FirstSolutionStrategy.GLOBAL_CHEAPEST_ARC,
    "FIRST_UNBOUND_MIN_VALUE": routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE,
}

METAHEURISTICS = {
    "GREEDY_DESCENT": routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT,
    "GUIDED_LOCAL_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
    "SIMULATED_ANNEALING": routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    "TABU_SEARCH": routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
}


# ---------------------------------------------------------------------------
# TODO(human): Implement the distance callback and solver
# ---------------------------------------------------------------------------

def create_distance_callback(
    manager: pywrapcp.RoutingIndexManager,
    distance_matrix: list[list[int]],
):
    """Create a transit callback that returns the distance between two nodes.

    Args:
        manager: The routing index manager (for index-to-node conversion).
        distance_matrix: NxN matrix where distance_matrix[i][j] = distance from i to j.

    Returns:
        A callable(from_index, to_index) -> int suitable for RegisterTransitCallback.
    """
    # TODO(human): Distance callback for TSP
    #
    # The OR-Tools routing solver uses INTERNAL INDICES (not your node IDs) when
    # calling callbacks. You must convert from internal index to node index using
    # the manager before looking up the distance matrix.
    #
    # Pattern:
    #   def distance_callback(from_index, to_index):
    #       from_node = manager.IndexToNode(from_index)
    #       to_node = manager.IndexToNode(to_index)
    #       return distance_matrix[from_node][to_node]
    #
    # Why the indirection? The solver may have multiple internal variables for the
    # same node (e.g., depot appears as both start and end of each vehicle's route).
    # The RoutingIndexManager handles this many-to-one mapping. Your callback must
    # always go through IndexToNode() — never use the raw index as a matrix index.
    #
    # Return type must be int (or castable to int). OR-Tools routing works with
    # integer costs only (similar to CP-SAT's integer-only variables).
    #
    # Return: a closure or function that takes (from_index, to_index) -> int
    raise NotImplementedError("TODO(human): create_distance_callback")


def solve_tsp(
    distance_matrix: list[list[int]],
    strategy_name: str = "PATH_CHEAPEST_ARC",
    metaheuristic_name: str | None = None,
    time_limit_seconds: int = 5,
) -> tuple[list[int], int] | None:
    """Solve a TSP instance using OR-Tools routing solver.

    Args:
        distance_matrix: NxN symmetric distance matrix.
        strategy_name: Key into FIRST_SOLUTION_STRATEGIES dict.
        metaheuristic_name: Key into METAHEURISTICS dict. None = no metaheuristic.
        time_limit_seconds: Time limit for metaheuristic search.

    Returns:
        Tuple of (tour, total_distance) or None if no solution found.
    """
    # TODO(human): Set up and solve the TSP
    #
    # Steps:
    #
    # 1. Create the RoutingIndexManager:
    #      manager = pywrapcp.RoutingIndexManager(
    #          len(distance_matrix),  # number of nodes
    #          1,                     # number of vehicles (TSP = 1 vehicle)
    #          0                      # depot index (start/end node)
    #      )
    #
    # 2. Create the RoutingModel:
    #      routing = pywrapcp.RoutingModel(manager)
    #
    # 3. Register the distance callback:
    #      callback = create_distance_callback(manager, distance_matrix)
    #      transit_callback_index = routing.RegisterTransitCallback(callback)
    #
    # 4. Set the arc cost evaluator — this tells the solver "the cost of going
    #    from node A to node B is given by this callback":
    #      routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    #
    # 5. Configure search parameters:
    #      search_params = pywrapcp.DefaultRoutingSearchParameters()
    #      search_params.first_solution_strategy = FIRST_SOLUTION_STRATEGIES[strategy_name]
    #
    #    If metaheuristic_name is not None, also set:
    #      search_params.local_search_metaheuristic = METAHEURISTICS[metaheuristic_name]
    #      search_params.time_limit.FromSeconds(time_limit_seconds)
    #
    #    IMPORTANT: When using a metaheuristic, you MUST set a time limit.
    #    Metaheuristics never prove optimality — they improve indefinitely.
    #    Without a time limit, the solver will run forever.
    #
    # 6. Solve:
    #      solution = routing.SolveWithParameters(search_params)
    #
    # 7. If solution exists, call print_tsp_solution() and return (tour, distance).
    #    If solution is None, print a message and return None.
    #
    # The entire pipeline: Manager -> Model -> Callback -> ArcCost -> Params -> Solve
    # This is the canonical OR-Tools routing pattern you'll reuse in every phase.
    raise NotImplementedError("TODO(human): solve_tsp")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    distance_matrix = create_distance_matrix()
    n = len(distance_matrix)
    print(f"=== TSP with {n} cities ===\n")

    # --- Compare first-solution strategies (no metaheuristic) ---
    print("--- Comparing First-Solution Strategies (no metaheuristic) ---\n")
    results: dict[str, int] = {}
    for name in ["PATH_CHEAPEST_ARC", "SAVINGS", "CHRISTOFIDES",
                  "PARALLEL_CHEAPEST_INSERTION", "GLOBAL_CHEAPEST_ARC"]:
        print(f"Strategy: {name}")
        result = solve_tsp(distance_matrix, strategy_name=name)
        if result:
            _, dist = result
            results[name] = dist
        print()

    if results:
        best_name = min(results, key=results.get)
        print(f"Best initial strategy: {best_name} (distance={results[best_name]})")
        print()

    # --- Now apply metaheuristics on top of PATH_CHEAPEST_ARC ---
    print("--- Comparing Metaheuristics (initial: PATH_CHEAPEST_ARC, 3s limit) ---\n")
    meta_results: dict[str, int] = {}
    for meta_name in ["GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH",
                       "SIMULATED_ANNEALING", "TABU_SEARCH"]:
        print(f"Metaheuristic: {meta_name}")
        result = solve_tsp(
            distance_matrix,
            strategy_name="PATH_CHEAPEST_ARC",
            metaheuristic_name=meta_name,
            time_limit_seconds=3,
        )
        if result:
            _, dist = result
            meta_results[meta_name] = dist
        print()

    if meta_results:
        best_meta = min(meta_results, key=meta_results.get)
        print(f"Best metaheuristic: {best_meta} (distance={meta_results[best_meta]})")


if __name__ == "__main__":
    main()
