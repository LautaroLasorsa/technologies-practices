# Practice 036a: Network Optimization — Flows & Assignment

**Technologies:** C++17 (STL only — no external libraries)
**Stack:** C++17, CMake 3.16+
**State:** `not-started`

---

## Theoretical Context

### Network Flow Fundamentals

A **flow network** is a directed graph G = (V, E) where each edge (u, v) has a non-negative **capacity** c(u, v) >= 0. Two distinguished vertices exist: a **source** s (produces flow) and a **sink** t (absorbs flow). A **flow** f assigns a value f(u, v) to each edge satisfying:

1. **Capacity constraint:** 0 <= f(u, v) <= c(u, v) for all edges
2. **Flow conservation:** For every vertex v except s and t, flow in = flow out:
   sum of f(u, v) for all u = sum of f(v, w) for all w

The **value** of a flow |f| = total flow leaving source = total flow entering sink.

### Max Flow / Min Cut

**Goal:** Find a flow of maximum value from s to t.

**Ford-Fulkerson method** (general framework):
1. Start with zero flow
2. While there exists an **augmenting path** from s to t in the **residual graph**, augment flow along it
3. The **residual graph** G_f has edge (u, v) with residual capacity c_f(u, v) = c(u, v) - f(u, v) for forward edges, and c_f(v, u) = f(u, v) for reverse edges (allowing flow "undoing")

**Edmonds-Karp algorithm:** Ford-Fulkerson with BFS (shortest augmenting path). Complexity: O(V * E^2). BFS guarantees that augmenting path lengths are non-decreasing, bounding the number of augmentations to O(V * E).

**Max-flow min-cut theorem** (Ford & Fulkerson, 1956): The maximum flow from s to t equals the minimum capacity of an **s-t cut** (a partition of V into S and T where s in S, t in T). The min cut consists of edges (u, v) where u is reachable from s in the residual graph but v is not. This is one of the most elegant duality results in combinatorial optimization.

### Min-Cost Flow

Generalize max flow by assigning a **cost** w(u, v) to each edge. The goal: find a flow satisfying demand constraints that minimizes total cost = sum of f(u, v) * w(u, v) over all edges.

**Successive Shortest Paths algorithm:**
1. While there exists a path from s to t in the residual graph:
   - Find shortest path (by cost) using SPFA (Bellman-Ford with queue)
   - Augment flow along this path
   - Accumulate cost
2. Key insight: reverse edges have cost = -w(original), so "undoing" expensive flow is correctly modeled as negative-cost edges
3. Complexity: O(V * E * max_flow) with SPFA, but practical for moderate-sized networks

SPFA works here because negative-cost edges exist in the residual graph (from reverse edges). Dijkstra requires non-negative edges (can be achieved with Johnson's potentials, but SPFA is simpler for a first implementation).

### Assignment Problem

A special case of min-cost flow on bipartite graphs: assign n workers to n jobs, minimizing total cost. The cost matrix C[i][j] = cost of assigning worker i to job j.

**Hungarian algorithm (Kuhn-Munkres):** O(n^3) algorithm using dual variables (potentials):
- Maintains potentials u[i] (worker) and v[j] (job) such that u[i] + v[j] <= C[i][j]
- Iteratively finds augmenting paths in the equality graph (edges where u[i] + v[j] = C[i][j])
- Updates potentials to expand the equality graph when no augmenting path exists

The potentials correspond to the **dual variables** of the LP formulation. Complementary slackness ensures optimality: if worker i is assigned to job j, then u[i] + v[j] = C[i][j] (the dual constraint is tight).

### Connection to Linear Programming

Network flow problems are special LPs where the constraint matrix is **totally unimodular** (TU): every square submatrix has determinant in {-1, 0, +1}. This guarantees that the LP relaxation always has **integer optimal solutions**, even without integrality constraints. This is why:

- Max flow, min-cost flow, assignment, shortest path, and bipartite matching are all polynomial-time solvable
- They can be modeled as LPs but solved by specialized combinatorial algorithms that are much faster
- The LP dual gives economic interpretations: shadow prices, reduced costs, potentials

The incidence matrix of a directed graph is TU (each column has exactly one +1 and one -1). This structural property is what makes network optimization so tractable compared to general integer programming.

### Applications

- **Transportation & logistics:** shipping goods from warehouses to stores at minimum cost
- **Scheduling:** assigning tasks to machines, workers to shifts
- **Bipartite matching:** job assignment, organ donor matching, stable matching
- **Network design:** maximum throughput, minimum cut (bottleneck identification)
- **Supply chain:** multi-echelon inventory routing, facility location with flow

### Key Concepts

| Concept | Definition |
|---------|------------|
| Flow network | Directed graph with capacities, source, and sink |
| Residual graph | Graph showing remaining capacity + flow-undoing edges |
| Augmenting path | s-to-t path in residual graph with positive capacity |
| Max flow | Maximum total flow from source to sink |
| Min cut | Minimum-capacity partition separating source from sink |
| Min-cost flow | Flow satisfying demands at minimum total edge cost |
| SPFA | Bellman-Ford with queue — handles negative edge costs |
| Assignment problem | Bipartite min-cost perfect matching |
| Hungarian algorithm | O(n^3) algorithm for assignment using dual potentials |
| Total unimodularity | Matrix property guaranteeing integer LP solutions |
| Dual variables | Potentials giving economic interpretation of optimality |

---

## Description

Implement three fundamental network optimization algorithms from scratch using only C++ STL:

1. **Max flow (Edmonds-Karp):** BFS-based augmenting paths in a residual graph, plus min-cut extraction
2. **Min-cost flow (Successive Shortest Paths):** SPFA-based shortest paths in a cost-weighted residual graph
3. **Hungarian algorithm:** O(n^3) optimal assignment using dual potentials

No external libraries — all algorithms implemented with `vector`, `queue`, and basic STL containers.

---

## Instructions

### Phase 1: Max Flow (Edmonds-Karp)

**File:** `src/max_flow.cpp`

The `FlowNetwork` struct, edge representation, `add_edge()`, BFS helper, sample networks, and `main()` are provided. You implement:

1. **`edmonds_karp()`** — The core max-flow algorithm. This teaches you the residual graph concept: every edge has a "shadow" reverse edge allowing flow to be rerouted. BFS guarantees shortest augmenting paths, giving the O(VE^2) bound. The key insight is that adding reverse edges lets the algorithm "undo" bad decisions — greedy choices become globally optimal.

2. **`find_min_cut()`** — After max flow, BFS from source in the residual graph partitions vertices into reachable (S) and unreachable (T). Edges crossing from S to T at full capacity form the min cut. This connects to the max-flow min-cut theorem — the most important duality result in network optimization.

### Phase 2: Min-Cost Flow (Successive Shortest Paths)

**File:** `src/min_cost_flow.cpp`

The `CostFlowNetwork` struct, edge representation with costs, `add_edge()`, sample transportation problem, and `main()` are provided. You implement:

3. **`spfa()`** — Shortest path in the residual graph by cost. Unlike Dijkstra, SPFA handles negative-cost edges (from reverse edges). This teaches the cost structure of residual graphs: sending flow on edge (u,v) with cost w creates a reverse edge (v,u) with cost -w, meaning "undoing" flow on an expensive edge effectively earns back its cost.

4. **`min_cost_max_flow()`** — Repeatedly find cheapest augmenting paths and send flow. This teaches the successive shortest paths principle: always augmenting along the cheapest available path guarantees global optimality. The connection to LP: each augmenting path corresponds to entering a variable into the simplex basis.

### Phase 3: Hungarian Algorithm

**File:** `src/hungarian.cpp`

The cost matrix, print helpers, sample problems, and `main()` are provided. You implement:

5. **`hungarian()`** — The O(n^3) Kuhn-Munkres algorithm for minimum-cost assignment. This teaches dual variables (potentials) in action: u[i] and v[j] provide a lower bound on the assignment cost, and the algorithm tightens this bound until a feasible assignment matching the bound is found. The CP-algorithms version uses a clever shortest-path formulation that processes one worker at a time.

---

## Motivation

Network flow sits at the intersection of graph algorithms and operations research. As someone with CP background (where max flow and bipartite matching appear as problem-solving tools), this practice deepens the understanding of *why* these algorithms work through the lens of LP duality and total unimodularity. The min-cost flow and Hungarian algorithm are directly applicable to logistics scheduling at AutoScheduler.AI. The LP connection — that network flows are integer-optimal LPs — is the key theoretical insight bridging combinatorial optimization with the LP/MIP foundations from practices 032-034.

---

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `cmake -S . -B build` | Configure CMake build (from practice root) |
| Setup | `cmake --build build --config Release` | Build all three executables |
| Phase 1 | `cmake --build build --config Release --target phase1_max_flow` | Build max flow only |
| Phase 1 | `./build/Release/phase1_max_flow.exe` | Run max flow examples |
| Phase 2 | `cmake --build build --config Release --target phase2_min_cost_flow` | Build min-cost flow only |
| Phase 2 | `./build/Release/phase2_min_cost_flow.exe` | Run min-cost flow transportation |
| Phase 3 | `cmake --build build --config Release --target phase3_hungarian` | Build Hungarian algorithm only |
| Phase 3 | `./build/Release/phase3_hungarian.exe` | Run assignment problems |
| All | `cmake --build build --config Release --target all_phases` | Build all three phases |

**Note:** On Windows with MSVC, executables are in `build\Release\`. On Linux/Mac with Make, they are in `build/`.
