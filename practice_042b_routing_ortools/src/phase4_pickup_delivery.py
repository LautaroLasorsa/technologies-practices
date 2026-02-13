"""Phase 4: Pickup and Delivery Problem (PDP) with OR-Tools.

Each request has a pickup location and a delivery location. The same vehicle must
handle both, pickup before delivery. Combines routing + capacity + ordering constraints.
Demonstrates: AddPickupAndDelivery, VehicleVar, precedence constraints, capacity + PD.
"""

from ortools.constraint_solver import routing_enums_pb2, pywrapcp


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def create_pdp_data() -> dict:
    """Create a Pickup-Delivery instance: 8 requests (16 P/D nodes) + depot.

    Depot is node 0.
    Pickup-delivery pairs: each request (pickup_node, delivery_node).
    Each request carries 1 unit of load: pickup adds +1, delivery removes -1.
    """
    data = {}

    # Distance matrix: 17 locations
    # Node 0: depot
    # Nodes 1-16: pickup/delivery locations
    data["distance_matrix"] = [
        # fmt: off
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
        # fmt: on
    ]

    # Pickup-Delivery pairs: (pickup_node, delivery_node)
    # 8 requests. Each pickup picks up 1 unit; each delivery drops off 1 unit.
    data["pickups_deliveries"] = [
        (1, 6),    # Request 0: pick up at node 1, deliver to node 6
        (2, 10),   # Request 1: pick up at node 2, deliver to node 10
        (4, 3),    # Request 2: pick up at node 4, deliver to node 3
        (5, 9),    # Request 3: pick up at node 5, deliver to node 9
        (7, 8),    # Request 4: pick up at node 7, deliver to node 8
        (15, 11),  # Request 5: pick up at node 15, deliver to node 11
        (13, 12),  # Request 6: pick up at node 13, deliver to node 12
        (16, 14),  # Request 7: pick up at node 16, deliver to node 14
    ]

    # Demands: +1 at pickup, -1 at delivery, 0 at depot.
    # This models load carried: picking up adds weight, delivering removes it.
    demands = [0] * len(data["distance_matrix"])
    for pickup, delivery in data["pickups_deliveries"]:
        demands[pickup] = 1
        demands[delivery] = -1
    data["demands"] = demands

    # Each vehicle can carry at most 3 packages simultaneously.
    data["vehicle_capacities"] = [3, 3, 3, 3]
    data["num_vehicles"] = 4
    data["depot"] = 0

    return data


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_pdp_solution(
    data: dict,
    manager: pywrapcp.RoutingIndexManager,
    routing: pywrapcp.RoutingModel,
    solution: pywrapcp.Assignment,
) -> None:
    """Print PDP routes showing pickup/delivery pairs and vehicle loads.

    Args:
        data: Problem data dict.
        manager: The routing index manager.
        routing: The routing model.
        solution: The assignment returned by the solver.
    """
    # Build lookup: node -> role (P=pickup, D=delivery)
    node_role: dict[int, str] = {}
    for req_id, (pickup, delivery) in enumerate(data["pickups_deliveries"]):
        node_role[pickup] = f"P{req_id}"
        node_role[delivery] = f"D{req_id}"

    total_distance = 0

    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        route_nodes = []
        route_distance = 0
        current_load = 0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            current_load += data["demands"][node]
            role = node_role.get(node, "depot")
            route_nodes.append((node, role, current_load))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)

        # Final depot
        node = manager.IndexToNode(index)
        current_load += data["demands"][node]
        route_nodes.append((node, "depot", current_load))

        total_distance += route_distance

        print(f"  Vehicle {vehicle_id} (distance={route_distance}):")
        for node, role, load in route_nodes:
            cap = data["vehicle_capacities"][vehicle_id]
            print(f"    Node {node:>2} [{role:>5}] | load={load}/{cap}")
        print()

    print(f"  Total distance: {total_distance}")

    # Verify pickup-before-delivery
    print("\n  Verification — pickup before delivery:")
    for vehicle_id in range(data["num_vehicles"]):
        index = routing.Start(vehicle_id)
        visited_order = []
        while not routing.IsEnd(index):
            visited_order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))

        for req_id, (pickup, delivery) in enumerate(data["pickups_deliveries"]):
            if pickup in visited_order and delivery in visited_order:
                p_pos = visited_order.index(pickup)
                d_pos = visited_order.index(delivery)
                status = "ok" if p_pos < d_pos else "VIOLATION!"
                print(f"    Vehicle {vehicle_id}, Request {req_id}: "
                      f"pickup(node {pickup}) at pos {p_pos}, "
                      f"delivery(node {delivery}) at pos {d_pos} -> {status}")


# ---------------------------------------------------------------------------
# TODO(human): Implement the solver
# ---------------------------------------------------------------------------

def solve_pickup_delivery(data: dict) -> None:
    """Solve a Pickup-Delivery Problem using OR-Tools routing solver.

    Args:
        data: Problem data dict with keys: distance_matrix, pickups_deliveries,
              demands, vehicle_capacities, num_vehicles, depot.
    """
    # TODO(human): Set up and solve the Pickup-Delivery Problem
    #
    # This phase combines everything: distance callback, demand callback,
    # capacity dimension, AND pickup-delivery constraints. Follow these steps:
    #
    # 1. Create RoutingIndexManager and RoutingModel (standard pattern).
    #
    # 2. Register DISTANCE callback and set as arc cost (same as Phase 1/2).
    #
    # 3. Add a DISTANCE dimension — needed for precedence constraints:
    #
    #      routing.AddDimension(
    #          transit_cb_index,  # distance callback
    #          0,                 # no slack
    #          3000,              # max cumulative distance per route
    #          True,              # start cumul at zero
    #          "Distance"
    #      )
    #      distance_dimension = routing.GetDimensionOrDie("Distance")
    #
    #    Why a distance DIMENSION in addition to arc cost? Because we need
    #    CumulVar for precedence constraints: pickup cumul < delivery cumul.
    #    Arc cost alone doesn't give us cumulative values at each node.
    #
    # 4. Register DEMAND callback and add Capacity dimension:
    #
    #      Note: demands are +1 for pickups, -1 for deliveries, 0 for depot.
    #      The cumulative load goes UP at pickups and DOWN at deliveries.
    #
    #      demand_cb = ... (same pattern as Phase 2)
    #      demand_cb_index = routing.RegisterUnaryTransitCallback(demand_cb)
    #      routing.AddDimensionWithVehicleCapacity(
    #          demand_cb_index,
    #          0,                             # no slack
    #          data["vehicle_capacities"],    # max load per vehicle
    #          True,                          # start empty
    #          "Capacity"
    #      )
    #
    # 5. Add PICKUP-DELIVERY constraints — the core of this phase:
    #
    #      for pickup_node, delivery_node in data["pickups_deliveries"]:
    #          pickup_index = manager.NodeToIndex(pickup_node)
    #          delivery_index = manager.NodeToIndex(delivery_node)
    #
    #          # Tell the solver these two nodes form a pickup-delivery pair:
    #          routing.AddPickupAndDelivery(pickup_index, delivery_index)
    #
    #          # SAME VEHICLE constraint: the pickup and delivery must be on
    #          # the same route. VehicleVar(index) returns the vehicle assigned
    #          # to that node. Constraining them equal forces same vehicle.
    #          routing.solver().Add(
    #              routing.VehicleVar(pickup_index) ==
    #              routing.VehicleVar(delivery_index)
    #          )
    #
    #          # PRECEDENCE constraint: pickup must come BEFORE delivery on
    #          # the route. We use the Distance dimension's CumulVar: the
    #          # cumulative distance at pickup must be <= cumulative at delivery.
    #          # This enforces ordering because distance is monotonically
    #          # increasing along a route.
    #          routing.solver().Add(
    #              distance_dimension.CumulVar(pickup_index) <=
    #              distance_dimension.CumulVar(delivery_index)
    #          )
    #
    #    Three constraints per pair:
    #      a) AddPickupAndDelivery — registers the pair with the solver
    #      b) VehicleVar equality — same vehicle
    #      c) CumulVar ordering — pickup before delivery
    #
    #    Without (a), the solver doesn't know the nodes are related.
    #    Without (b), pickup might go on vehicle 1 and delivery on vehicle 3.
    #    Without (c), delivery could come first — the vehicle arrives at the
    #    delivery location before it has picked up the package!
    #
    # 6. Configure search parameters:
    #      Use PATH_CHEAPEST_ARC + GUIDED_LOCAL_SEARCH with 5s time limit.
    #      PDP is harder to solve than plain CVRP — give it more time.
    #
    # 7. Solve and call print_pdp_solution().
    raise NotImplementedError("TODO(human): solve_pickup_delivery")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Pickup and Delivery Problem (PDP) ===\n")

    data = create_pdp_data()
    n_locations = len(data["distance_matrix"])
    print(f"Locations: {n_locations} (depot + {n_locations - 1} P/D nodes)")
    print(f"Vehicles: {data['num_vehicles']} (capacity: {data['vehicle_capacities']})")
    print(f"Requests: {len(data['pickups_deliveries'])}")
    print()

    print("Pickup-Delivery pairs:")
    for i, (p, d) in enumerate(data["pickups_deliveries"]):
        print(f"  Request {i}: pickup at node {p:>2}, deliver to node {d:>2}")
    print()

    print("Node demands (positive = pickup, negative = delivery):")
    for i, dem in enumerate(data["demands"]):
        if dem != 0:
            role = "pickup" if dem > 0 else "delivery"
            print(f"  Node {i:>2}: {dem:>+2} ({role})")
    print()

    solve_pickup_delivery(data)


if __name__ == "__main__":
    main()
