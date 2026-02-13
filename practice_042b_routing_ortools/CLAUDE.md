# Practice 042b: Vehicle Routing — OR-Tools Routing Solver

## Technologies

- **OR-Tools Routing Solver** — Google's constraint-based routing library (`ortools.constraint_solver`) for solving TSP, VRP, and their rich variants (capacity, time windows, pickup-delivery). Uses first-solution heuristics + local search metaheuristics. Production-grade solver used in logistics, fleet management, and last-mile delivery worldwide.
- **Python 3.12+** — Runtime with `uv` for dependency management.
- **NumPy** — Distance matrix generation and data manipulation.

## Stack

- Python 3.12+
- ortools >= 9.9 (Google OR-Tools routing solver)
- numpy >= 1.26
- uv (package manager)

## Theoretical Context

### The Vehicle Routing Problem Family

The **Vehicle Routing Problem (VRP)** is a combinatorial optimization problem that asks: *given a fleet of vehicles at a depot and a set of customers with demands, what is the optimal set of routes for the vehicles to serve all customers?* It generalizes the Traveling Salesman Problem (TSP) from one vehicle to many, and adds constraints like vehicle capacity, time windows, and pickup-delivery pairs.

The VRP family is one of the most studied and commercially important problems in operations research. Real-world applications include:

- **Last-mile delivery** (Amazon, FedEx, UPS route optimization)
- **Fleet management** (waste collection, field service)
- **Ride-sharing** (Uber/Lyft pooling = pickup-delivery VRP with time windows)
- **Supply chain** (warehouse-to-store distribution)

**Key variants:**

| Variant | Abbreviation | Extra Constraints |
|---------|-------------|-------------------|
| Traveling Salesman Problem | TSP | 1 vehicle, visit all cities, minimize distance |
| Capacitated VRP | CVRP | Vehicles have max load capacity |
| VRP with Time Windows | VRPTW | Each customer available only during [earliest, latest] |
| Pickup and Delivery | PDP/PDPTW | Each request has pickup + delivery, same vehicle |
| VRP with Multiple Depots | MDVRP | Vehicles start/end at different depots |
| Open VRP | OVRP | Vehicles don't return to depot |

All are NP-hard. Exact methods work for ~20-30 nodes; real instances (hundreds/thousands of nodes) require heuristics and metaheuristics.

**References:** [Toth & Vigo — Vehicle Routing: Problems, Methods, Applications (2014)](https://epubs.siam.org/doi/book/10.1137/1.9781611973594), [Google OR-Tools VRP Documentation](https://developers.google.com/optimization/routing/vrp)

### OR-Tools Routing Solver Architecture

The OR-Tools routing solver lives in `ortools.constraint_solver` (NOT `ortools.sat` — that is CP-SAT). It is a specialized solver purpose-built for routing problems. The architecture has three core components:

#### 1. RoutingIndexManager

The manager that maps between **node indices** (your problem's location IDs: 0, 1, ..., N-1) and **internal variable indices** (the solver's internal representation). This mapping is necessary because the solver may create multiple internal variables per node (e.g., for multiple vehicles starting/ending at the depot).

```python
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# num_locations, num_vehicles, depot_index
manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)

# Convert between node and internal index:
manager.NodeToIndex(node)    # node ID -> solver index
manager.IndexToNode(index)   # solver index -> node ID
```

#### 2. RoutingModel

The model that holds the optimization problem: arc costs, dimensions, constraints, and search parameters. Created from a manager.

```python
routing = pywrapcp.RoutingModel(manager)
```

Key methods:

| Method | Purpose |
|--------|---------|
| `RegisterTransitCallback(fn)` | Register a function that returns the cost/time/distance between two indices |
| `RegisterUnaryTransitCallback(fn)` | Register a function that depends on a single location (e.g., demand) |
| `SetArcCostEvaluatorOfAllVehicles(cb_idx)` | Set the cost function for arc traversal |
| `AddDimension(cb, slack, max_cumul, fix_start, name)` | Add a dimension tracking cumulative quantities |
| `AddDimensionWithVehicleCapacity(cb, slack, caps, fix_start, name)` | Dimension with per-vehicle capacity |
| `AddPickupAndDelivery(pickup, delivery)` | Link a pickup-delivery pair |
| `SolveWithParameters(params)` | Solve and return assignment |

#### 3. Callbacks (Transit Functions)

Callbacks are user-defined functions that the solver calls to get costs/quantities. They receive **solver indices** (not node indices), so you must convert using the manager:

```python
def distance_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return distance_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
```

For quantities that depend only on the current location (e.g., demand at a customer), use `RegisterUnaryTransitCallback`.

### Dimensions: Tracking Cumulative Quantities

**Dimensions** are the most important concept in OR-Tools routing. A dimension tracks a quantity that **accumulates along a route** — distance traveled, time elapsed, load carried, etc. Without dimensions, the solver only minimizes arc cost. With dimensions, you can enforce constraints on cumulative values.

A dimension has:

- **Transit variable** — how much the quantity changes at each arc (e.g., travel time between locations).
- **Cumulative variable (`CumulVar`)** — the total accumulated quantity at each location (e.g., arrival time).
- **Slack variable (`SlackVar`)** — allowed waiting/idle time at a location (relevant for time windows).

```python
# AddDimension(callback_index, slack_max, cumul_max, fix_start_cumul_to_zero, name)
routing.AddDimension(
    time_callback_index,
    30,      # max slack (waiting time) at each location
    1800,    # max cumulative value (e.g., max route time = 1800 min)
    False,   # don't fix start cumul to zero (vehicles may depart at different times)
    "Time"
)
```

**AddDimensionWithVehicleCapacity** is similar but allows different max cumulative values per vehicle:

```python
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index,
    0,                   # no slack for capacity
    vehicle_capacities,  # list: [cap_v0, cap_v1, ...]
    True,                # start cumul at zero (empty vehicle)
    "Capacity"
)
```

To set time windows on the cumulative variable:

```python
time_dimension = routing.GetDimensionOrDie("Time")
index = manager.NodeToIndex(location)
time_dimension.CumulVar(index).SetRange(earliest, latest)
```

### First Solution Strategies

The solver needs an initial feasible solution before it can improve it with local search. OR-Tools offers several construction heuristics:

| Strategy | Description | Best For |
|----------|-------------|----------|
| `PATH_CHEAPEST_ARC` | Extend partial route by cheapest next arc | General purpose, fast |
| `PATH_MOST_CONSTRAINED_ARC` | Extend by most constrained arc | Highly constrained problems |
| `SAVINGS` | Clarke-Wright savings algorithm: merge routes that save the most distance | CVRP, good initial quality |
| `CHRISTOFIDES` | 3/2-approximation for metric TSP | Symmetric TSP |
| `PARALLEL_CHEAPEST_INSERTION` | Insert cheapest unserved node into any route | Multi-vehicle, fast |
| `LOCAL_CHEAPEST_INSERTION` | Insert cheapest unserved node into current route | Single-route focus |
| `FIRST_UNBOUND_MIN_VALUE` | Assign minimum value to first unbound variable | Fast but low quality |
| `GLOBAL_CHEAPEST_ARC` | Choose globally cheapest arc | Dense small instances |

```python
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
```

**Reference:** [Google OR-Tools Routing Options](https://developers.google.com/optimization/routing/routing_options)

### Local Search Metaheuristics

After finding an initial solution, the solver improves it using local search. Available metaheuristics:

| Metaheuristic | Description |
|---------------|-------------|
| `GREEDY_DESCENT` | Accept only improving moves (hill-climbing). Fast but gets stuck in local optima. |
| `GUIDED_LOCAL_SEARCH` (GLS) | Penalizes features of local optima to escape them. **Most commonly used for VRP.** Augments the objective with penalties on frequently-used arcs, then resolves. |
| `SIMULATED_ANNEALING` | Accept worsening moves with probability decreasing over time (temperature schedule). Good exploration. |
| `TABU_SEARCH` | Maintain a "tabu list" of recent moves; forbid reversing them. Prevents cycling. |
| `GENERIC_TABU_SEARCH` | Tabu on variable-value pairs instead of moves. |

```python
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.FromSeconds(5)  # MUST set time limit with metaheuristics
```

**Important:** Metaheuristics don't know when they've found the optimal solution, so you **must** set a time limit. Without it, the solver runs indefinitely.

**Reference:** [Voudouris & Tsang — Guided Local Search (1999)](https://link.springer.com/chapter/10.1007/978-1-4615-5775-3_12)

### Disjunctions and Penalties

Sometimes not all locations can be visited (e.g., insufficient vehicles, conflicting time windows). OR-Tools supports **disjunctions** — groups of nodes where visiting is optional, with a penalty for skipping:

```python
penalty = 1000  # cost of not visiting this location
routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
```

If the penalty is large enough, the solver will always visit the node. If it's finite, the solver trades off the penalty vs the cost of visiting.

### Pickup and Delivery Constraints

For pickup-delivery problems, you must ensure:

1. **Same vehicle:** The pickup and delivery of a request are served by the same vehicle.
2. **Precedence:** Pickup occurs before delivery on the route.

```python
for pickup_node, delivery_node in pickup_delivery_pairs:
    pickup_index = manager.NodeToIndex(pickup_node)
    delivery_index = manager.NodeToIndex(delivery_node)
    routing.AddPickupAndDelivery(pickup_index, delivery_index)
    # Same vehicle constraint
    routing.solver().Add(
        routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
    )
    # Precedence: pickup cumul <= delivery cumul
    routing.solver().Add(
        distance_dimension.CumulVar(pickup_index)
        <= distance_dimension.CumulVar(delivery_index)
    )
```

### Key Concepts Summary

| Concept | Definition |
|---------|------------|
| **RoutingIndexManager** | Maps between node IDs and solver's internal variable indices |
| **RoutingModel** | The optimization model: costs, dimensions, constraints |
| **Transit callback** | User function returning cost/distance/time between two locations |
| **Unary transit callback** | User function returning a quantity at a single location (demand) |
| **Dimension** | Tracks a cumulative quantity along a route (time, distance, load) |
| **CumulVar** | The cumulative value of a dimension at a specific location |
| **SlackVar** | Allowed waiting/idle time at a location (used with time windows) |
| **Disjunction** | A set of nodes that may optionally be skipped (with penalty) |
| **First solution strategy** | Heuristic for constructing an initial feasible solution |
| **Local search metaheuristic** | Algorithm for improving the initial solution (GLS, SA, tabu) |
| **Arc cost evaluator** | The callback that defines the cost of traveling between nodes |
| **VehicleVar** | The variable indicating which vehicle visits a given node |

## Description

Model and solve four classic vehicle routing problems using OR-Tools' routing solver:

1. **TSP** — Solve a Traveling Salesman Problem for 15 cities. Define the distance callback, set arc costs, and compare multiple first-solution strategies and metaheuristics.
2. **CVRP** — Capacitated Vehicle Routing Problem with 4 vehicles and 16 customers. Add capacity constraints using dimensions. Show routes with cumulative loads.
3. **VRPTW** — VRP with Time Windows. Each customer has an [earliest, latest] service window. Use a time dimension with slack for waiting. Display arrival times at each stop.
4. **Pickup and Delivery** — Each request has a pickup location and a delivery location. Same vehicle must handle both in order. Add pickup-delivery constraints and capacity.

## Instructions

### Phase 1: TSP with OR-Tools (~25 min)

**What it teaches:** The core OR-Tools routing workflow (RoutingIndexManager, RoutingModel, callbacks, search parameters), arc cost evaluators, and how different heuristic strategies affect solution quality and speed.

**Why it matters:** TSP is the simplest routing problem and the foundation for all VRP variants. Mastering the manager/model/callback pattern here makes all subsequent phases straightforward. Comparing strategies builds intuition for which heuristics work on which problem structures.

**Exercises:**
1. Implement `create_distance_callback(manager, distance_matrix)` — define the transit callback that converts solver indices to node indices and returns the distance.
2. Implement `solve_tsp(distance_matrix, strategy, metaheuristic, time_limit)` — set up the routing model, register the callback, set the arc cost evaluator, configure search parameters, and solve.
3. Run with multiple strategies (PATH_CHEAPEST_ARC, SAVINGS, CHRISTOFIDES) and compare solution distances.
4. Enable GUIDED_LOCAL_SEARCH and observe improvement over the initial solution.

### Phase 2: Capacitated VRP (~25 min)

**What it teaches:** The dimension concept (the key abstraction in OR-Tools routing), unary transit callbacks for demand, AddDimensionWithVehicleCapacity, and multi-vehicle routing.

**Why it matters:** CVRP is the workhorse of logistics optimization. The dimension concept is what separates simple TSP from real-world routing. Understanding how cumulative variables track load along a route is essential for all constrained routing.

**Exercises:**
1. Implement `create_demand_callback(manager, demands)` — unary callback returning demand at each location.
2. Implement `solve_cvrp(distance_matrix, demands, vehicle_capacities, depot)` — register both distance and demand callbacks, add capacity dimension, solve and extract per-vehicle routes with loads.
3. Verify that no vehicle exceeds its capacity.
4. Experiment with different numbers of vehicles and capacities.

### Phase 3: VRPTW — VRP with Time Windows (~25 min)

**What it teaches:** Time dimensions with slack variables, CumulVar.SetRange for time windows, and the interaction between travel time, waiting time, and service windows.

**Why it matters:** Time windows are the most common real-world constraint in delivery and service routing. Understanding slack (waiting at a location until the window opens) is crucial. VRPTW is the standard benchmark problem in academic routing research.

**Exercises:**
1. Implement `create_time_callback(manager, time_matrix)` — transit callback for travel time between locations.
2. Implement `solve_vrptw(time_matrix, time_windows, vehicle_count, depot)` — add time dimension with slack, set time window ranges on each location's CumulVar, set depot departure window, solve.
3. Display routes with arrival time at each stop.
4. Observe how slack allows vehicles to wait for a window to open.

### Phase 4: Pickup and Delivery (~25 min)

**What it teaches:** AddPickupAndDelivery, VehicleVar constraints (same vehicle for pickup and delivery), precedence constraints via CumulVar, and combining capacity + routing constraints.

**Why it matters:** Pickup-delivery is the model for ride-sharing, courier services, and LTL (less-than-truckload) freight. The constraint that the same vehicle handles both pickup and delivery, in order, is a structural constraint that goes beyond simple capacity.

**Exercises:**
1. Implement `solve_pickup_delivery(distance_matrix, pickups_deliveries, demands, vehicle_capacities, depot)` — add distance dimension, add pickup-delivery pairs with same-vehicle and precedence constraints, add capacity dimension, solve.
2. Verify each route respects pickup-before-delivery ordering.
3. Verify capacity is never exceeded (pickups add load, deliveries remove it).
4. Display routes showing pickup/delivery pairs clearly.

## Motivation

- **Industry relevance:** Vehicle routing is the most commercially impactful application of combinatorial optimization. Companies spend billions on logistics; even 1% improvement in routing saves enormous costs.
- **OR-Tools adoption:** Google's routing solver is used by thousands of companies for production routing. It's the go-to open-source solution for VRP.
- **Complementary to CP-SAT:** Practice 042a covered CP-SAT (exact solver for scheduling/constraints). The routing solver uses a fundamentally different approach (heuristics + metaheuristics) designed specifically for routing topology. Knowing both is essential.
- **Career relevance:** Logistics, supply chain, and fleet management are core domains for AutoScheduler.AI. Understanding VRP modeling with OR-Tools directly applies to real optimization engineering work.
- **Builds on foundations:** Practices 036a-b implemented TSP/VRP heuristics from scratch (nearest neighbor, 2-opt, Clarke-Wright). Now you use industrial-grade implementations of these same ideas.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `uv sync` | Install dependencies (ortools, numpy) |
| Phase 1 | `uv run python src/phase1_tsp.py` | Run TSP solver with strategy comparison |
| Phase 2 | `uv run python src/phase2_cvrp.py` | Run Capacitated VRP |
| Phase 3 | `uv run python src/phase3_vrptw.py` | Run VRP with Time Windows |
| Phase 4 | `uv run python src/phase4_pickup_delivery.py` | Run Pickup and Delivery problem |

## State

`not-started`
