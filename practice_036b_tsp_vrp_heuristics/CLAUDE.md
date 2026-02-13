# Practice 036b: Network Optimization — TSP & VRP Heuristics

**Technologies:** C++17 (STL only — no external libraries)
**Stack:** C++17, CMake 3.16+
**State:** `not-started`

---

## Theoretical Context

### The Traveling Salesman Problem (TSP)

**Problem:** Given n cities with pairwise distances, find the shortest tour that visits every city exactly once and returns to the starting city. Formally, find a Hamiltonian cycle of minimum total weight in a complete weighted graph.

**Complexity:** TSP is NP-hard. The decision version ("is there a tour of length <= k?") is NP-complete. No known polynomial-time algorithm finds an optimal solution. Brute force is O(n!), dynamic programming (Held-Karp) is O(n^2 * 2^n) — feasible for n <= 25. For practical instances (hundreds to millions of cities), heuristics are essential.

**Metric TSP:** When distances satisfy the triangle inequality d(a,c) <= d(a,b) + d(b,c), Christofides' algorithm guarantees a 3/2-approximation. Euclidean TSP (cities as 2D points) is a special case of metric TSP.

The practical approach is: **(1) construct** an initial tour with a greedy heuristic, then **(2) improve** it with local search.

### TSP Construction Heuristics

**Nearest Neighbor (NN):**
1. Start at a given city
2. At each step, go to the nearest unvisited city
3. Return to start when all cities visited

- Complexity: O(n^2)
- Quality: typically 20-25% above optimal for random Euclidean instances
- Very fast, good starting point for improvement
- Weakness: makes locally greedy decisions — the last few edges are often very long because the unvisited cities are scattered

**Cheapest Insertion:**
1. Start with a "seed" subtour (e.g., triangle of 3 mutually farthest cities)
2. Among all cities not yet in the tour, find the one whose insertion causes the least increase in tour length
3. Insert it at the best position in the current tour
4. Repeat until all cities are in the tour

- Insertion cost for city k between consecutive tour cities i and j: d(i,k) + d(k,j) - d(i,j)
- Complexity: O(n^2) with efficient bookkeeping
- Quality: typically 10-15% above optimal — better than NN because it builds the tour shape incrementally

**Farthest Insertion:** Same as cheapest insertion, but always insert the city farthest from the current tour. This produces better initial tour shape by establishing the "convex hull" early. Quality is comparable to cheapest insertion.

### TSP Local Search / Improvement Heuristics

After construction, **local search** iteratively improves the tour by making small modifications:

**2-opt:**
- Remove two non-adjacent edges from the tour, creating two segments
- Reconnect them the only other possible way (reversing one segment)
- For edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]):
  - Remove these two edges
  - Reverse the segment tour[i+1..j]
  - New edges become (tour[i], tour[j]) and (tour[i+1], tour[j+1])
- Accept the move if it reduces total distance:
  - delta = d(tour[i], tour[j]) + d(tour[i+1], tour[j+1]) - d(tour[i], tour[i+1]) - d(tour[j], tour[j+1])
  - If delta < 0: the move improves the tour
- Try all O(n^2) pairs per pass; repeat until no improving move exists (**2-opt local optimum**)
- A tour is "2-optimal" when no 2-opt move can improve it
- Typically improves NN tours by 15-25%, reaching within 5-8% of optimal

**3-opt:** Remove three edges, try all reconnections (there are 8 ways to reconnect 3 segments). O(n^3) per pass. Better results but much slower. In practice, Or-opt (a restricted 3-opt) is preferred.

**Or-opt:** Move a chain of 1, 2, or 3 consecutive cities to another position in the tour. Faster than 3-opt but captures many of its improvements.

### Lin-Kernighan Heuristic

The **Lin-Kernighan (LK)** heuristic is a variable-depth search: instead of fixed k-opt, it performs a sequence of improving exchanges of variable length. At each step, it considers extending the current sequence of edge swaps by one more, stopping when no further extension is improving. This makes it adaptive — it can perform the equivalent of 4-opt, 5-opt, or deeper moves when beneficial.

**LKH (Lin-Kernighan-Helighen):** The state-of-the-art TSP solver by Keld Helsgott. Uses 5-opt moves with backtracking, candidate lists to prune the search, and sensitivity analysis. Finds near-optimal solutions for instances with millions of cities. The LKH-3 solver regularly wins DIMACS and TSPLIB competitions.

### The Vehicle Routing Problem (VRP)

**Problem:** A fleet of vehicles, each with capacity Q, must serve a set of customers with known demands from a central depot. Each customer must be served by exactly one vehicle. Minimize total distance traveled by all vehicles.

**VRP = TSP generalization:** TSP is VRP with one vehicle of infinite capacity. VRP adds:
- Multiple vehicles (routes), each starting and ending at the depot
- Vehicle capacity constraint: sum of demands on each route <= Q
- Often: maximum route duration or distance constraints

**CVRP (Capacitated VRP):** The basic variant — capacity is the only constraint beyond routing. NP-hard (contains TSP as special case).

**VRPTW (VRP with Time Windows):** Each customer has a time window [e_i, l_i] — service must begin within this interval. Even more constrained and harder.

**Practical scale:** Amazon delivers ~20 million packages/day. UPS's ORION system (OR + ML) saves ~100 million miles/year. These systems use variants of the algorithms in this practice.

### VRP Construction: Clarke-Wright Savings Algorithm

The most influential VRP construction heuristic (Clarke & Wright, 1964):

1. **Initial solution:** Create n routes, each serving exactly one customer: depot -> customer_i -> depot. Total distance = 2 * sum of d(depot, i).

2. **Compute savings:** For each pair (i, j) of customers:
   s(i, j) = d(0, i) + d(0, j) - d(i, j)
   This is the distance saved by merging the routes serving i and j into a single route depot -> ... -> i -> j -> ... -> depot, instead of two separate round-trips.

3. **Sort savings** in descending order (biggest savings first).

4. **Merge routes greedily:** For each saving (i, j) in order:
   - Check: i and j are in different routes
   - Check: i is at an endpoint (first or last customer) of its route
   - Check: j is at an endpoint of its route
   - Check: combined demand of merged route <= vehicle capacity Q
   - If all checks pass: merge the two routes by connecting i to j

The endpoint check ensures we only concatenate routes, not splice into the middle. The algorithm runs in O(n^2 log n) due to sorting.

### VRP Local Search

After construction, local search improves the solution with **inter-route** and **intra-route** moves:

**Intra-route (within a single route):**
- 2-opt: same as TSP 2-opt, applied to each route individually
- Or-opt: relocate a chain of 1-3 customers within the same route

**Inter-route (between two routes):**
- **Relocate:** Move a customer from one route to another. For customer c in route r1: remove c from r1, insert at best position in r2 (if capacity allows). Accept if total distance decreases.
- **Swap:** Exchange a customer from route r1 with a customer from route r2. Accept if total distance decreases and both routes remain feasible.
- **2-opt* (inter-route 2-opt):** For routes r1 and r2, try swapping their tails at every pair of positions. Route r1 = [depot, a1, ..., ai, ai+1, ..., depot] and r2 = [depot, b1, ..., bj, bj+1, ..., depot]. After 2-opt*: r1 = [depot, a1, ..., ai, bj+1, ..., depot] and r2 = [depot, b1, ..., bj, ai+1, ..., depot]. Check capacity feasibility of both new routes.

**Improvement loop:** Alternate between inter-route and intra-route moves until no improving move is found (local optimum).

### Key Concepts

| Concept | Definition |
|---------|------------|
| TSP | Visit all cities once, return to start, minimize total distance |
| NP-hard | No known polynomial optimal algorithm; heuristics are essential |
| Nearest neighbor | Greedy construction: always go to nearest unvisited city |
| Cheapest insertion | Insert city that causes least tour length increase |
| 2-opt | Remove 2 edges, reconnect — accept if tour improves |
| 2-opt local optimum | Tour where no 2-opt move improves it |
| Lin-Kernighan | Variable-depth search — best known TSP heuristic |
| VRP | Multiple capacitated vehicles serve customers from a depot |
| CVRP | VRP with vehicle capacity as the only side constraint |
| Clarke-Wright savings | Merge routes greedily by distance saved: s(i,j) = d(0,i) + d(0,j) - d(i,j) |
| Relocate | Move a customer from one route to another |
| 2-opt* | Inter-route 2-opt: swap tails between two routes |
| Construction + improvement | Standard heuristic pattern: greedy build, then local search |

---

## Description

Implement TSP construction heuristics (nearest neighbor, cheapest insertion), TSP 2-opt local search improvement, VRP Clarke-Wright savings construction, and VRP inter-route local search (relocate + 2-opt*). All from scratch using only C++ STL. Benchmark on random Euclidean instances and observe the quality-speed tradeoffs.

---

## Instructions

### Phase 1: TSP Construction — Nearest Neighbor & Cheapest Insertion

**File:** `src/tsp_construction.cpp`

The `Point` struct, `distance()`, `tour_length()`, `generate_random_points()`, `print_tour()`, sample instances, and `main()` are provided. You implement:

1. **`nearest_neighbor()`** — The simplest TSP construction heuristic. This teaches greedy construction: at each step, commit to the locally best choice (nearest unvisited city). The O(n^2) loop structure is straightforward. Observe how the last few edges are typically long — the greedy strategy "paints itself into a corner." Compare with the competitive ratio: NN can be O(log n) times optimal in the worst case, but averages 20-25% above optimal on random instances.

2. **`cheapest_insertion()`** — A smarter construction that builds the tour shape incrementally. Starting from a triangle of 3 farthest cities, repeatedly find the city whose best insertion costs least. This teaches the insertion paradigm: the tour is always a valid (partial) tour, and each insertion preserves feasibility. The initial triangle matters — starting with farthest cities establishes the tour's "skeleton." Typical quality is 10-15% above optimal.

### Phase 2: TSP Improvement — 2-opt Local Search

**File:** `src/tsp_improvement.cpp`

The point infrastructure and `nearest_neighbor()` (fully implemented as a starting tour generator) are provided. You implement:

3. **`two_opt_pass()`** — A single pass trying all O(n^2) pairs of edges for 2-opt improvement. This teaches the 2-opt move: for a pair (i, j), compute the delta (change in tour length) from removing edges (tour[i], tour[i+1]) and (tour[j], tour[j+1]) and adding edges (tour[i], tour[j]) and (tour[i+1], tour[j+1]). If delta < 0, reverse the segment and accept. The reversal is the key geometric operation: reversing tour[i+1..j] reconnects the tour correctly.

4. **`two_opt()`** — Iterate 2-opt passes until no improvement is found (convergence to a 2-opt local optimum). This teaches the local search loop pattern used across all combinatorial optimization: repeat improvement moves until stuck. Observe how many passes are needed and the diminishing returns per pass.

### Phase 3: VRP Construction — Clarke-Wright Savings

**File:** `src/vrp_construction.cpp`

The `Customer` struct, `VRPSolution` struct, distance matrix computation, route helpers, sample instances, and `main()` are provided. You implement:

5. **`clarke_wright_savings()`** — The most famous VRP construction heuristic. This teaches the savings concept: merging routes serving customers i and j saves s(i,j) = d(0,i) + d(0,j) - d(i,j) distance. The greedy strategy is to merge in order of decreasing savings, subject to capacity and endpoint constraints. You must track which route each customer belongs to and whether they are at a route endpoint. This teaches the bookkeeping challenges of route-based algorithms.

### Phase 4: VRP Improvement — Relocate & 2-opt*

**File:** `src/vrp_improvement.cpp`

The customer infrastructure, distance helpers, and `clarke_wright_savings()` (fully implemented) are provided. You implement:

6. **`relocate_pass()`** — Try moving each customer from its current route to the best position in every other route. This teaches inter-route improvement: for each customer c in route r1, compute the removal savings (removing c from r1) and the insertion cost (inserting c at the best position in r2). If the net delta is negative and capacity is feasible, perform the move. This is the simplest inter-route operator and often finds significant improvements.

7. **`two_opt_star_pass()`** — Inter-route 2-opt: for each pair of routes (r1, r2), try swapping their tails at every pair of cut points. This teaches the combinatorial structure of inter-route moves: you're exploring O(|r1| * |r2|) moves per route pair, and O(R^2) route pairs. The capacity check after tail-swap is essential — unlike intra-route 2-opt, the new routes may violate capacity.

---

## Motivation

TSP and VRP are the most studied combinatorial optimization problems in operations research. Every logistics company — Amazon, FedEx, UPS, Uber — uses variants of these algorithms at massive scale daily. The construction + local search paradigm (greedy build, then iterative improvement) is the universal pattern for heuristic optimization: it applies not just to routing but to scheduling, packing, assignment, and virtually every NP-hard problem. Understanding 2-opt and Clarke-Wright savings is foundational for any OR practitioner. This practice builds on the network flow foundations from 036a, moving from polynomial-time exact algorithms to heuristic approaches for NP-hard problems — the practical reality of most real-world optimization.

---

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `cmake -S . -B build` | Configure CMake build (from practice root) |
| Setup | `cmake --build build --config Release` | Build all four executables |
| Phase 1 | `cmake --build build --config Release --target phase1_tsp_construction` | Build TSP construction only |
| Phase 1 | `./build/Release/phase1_tsp_construction.exe` | Run nearest-neighbor and cheapest insertion |
| Phase 2 | `cmake --build build --config Release --target phase2_tsp_improvement` | Build TSP improvement only |
| Phase 2 | `./build/Release/phase2_tsp_improvement.exe` | Run 2-opt improvement on NN tours |
| Phase 3 | `cmake --build build --config Release --target phase3_vrp_construction` | Build VRP construction only |
| Phase 3 | `./build/Release/phase3_vrp_construction.exe` | Run Clarke-Wright savings algorithm |
| Phase 4 | `cmake --build build --config Release --target phase4_vrp_improvement` | Build VRP improvement only |
| Phase 4 | `./build/Release/phase4_vrp_improvement.exe` | Run relocate + 2-opt* improvement |
| All | `cmake --build build --config Release --target all_phases` | Build all four phases |

**Note:** On Windows with MSVC, executables are in `build\Release\`. On Linux/Mac with Make, they are in `build/`.
