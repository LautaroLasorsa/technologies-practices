#include <bits/stdc++.h>
using namespace std;

const int INF = 1e9;

// =============================================================================
// Edge in the cost-flow network
// Each edge has a paired reverse edge at index (idx ^ 1).
// Reverse edges have capacity 0 and cost = -cost(original).
// =============================================================================
struct Edge {
    int to;       // destination vertex
    int cap;      // capacity
    int cost;     // cost per unit of flow
    int flow;     // current flow
};

// =============================================================================
// Cost-Flow Network — adjacency list with costs
//
// Same structure as FlowNetwork but edges also carry per-unit costs.
// Reverse edges have negative cost: sending flow back "earns" the cost.
// =============================================================================
struct CostFlowNetwork {
    int n;
    vector<Edge> edges;
    vector<vector<int>> graph;

    explicit CostFlowNetwork(int n) : n(n), graph(n) {}

    // Add directed edge (from -> to) with given capacity and cost.
    // Also adds reverse edge (to -> from) with capacity 0 and cost -cost.
    int add_edge(int from, int to, int capacity, int cost) {
        int idx = (int)edges.size();
        edges.push_back({to, capacity, cost, 0});
        graph[from].push_back(idx);
        edges.push_back({from, 0, -cost, 0});    // reverse: cap=0, cost negated
        graph[to].push_back(idx + 1);
        return idx;
    }

    int residual(int idx) const {
        return edges[idx].cap - edges[idx].flow;
    }
};

// =============================================================================
// TODO(human): SPFA — shortest path in residual graph by cost
//
// SPFA (Shortest Path Faster Algorithm) in residual cost graph:
//   Initialize dist[source] = 0, dist[others] = INF
//   Queue: push source
//   While queue not empty:
//     u = queue.front(); pop
//     For each edge (u, v) with residual > 0:
//       If dist[u] + cost(u,v) < dist[v]:
//         dist[v] = dist[u] + cost(u,v)
//         parent[v] = u, parent_edge[v] = edge_index
//         Push v to queue (if not already in queue)
//   Return dist[sink] < INF
//
// Note: reverse edges have cost = -cost(original). This correctly
// accounts for "undoing" flow on costly edges.
//
// Hint: maintain an in_queue[] boolean array to avoid pushing the same
// vertex multiple times. dist[] is a vector<int> passed by reference.
// parent[] and parent_edge[] are output parameters for path reconstruction.
// =============================================================================
bool spfa(CostFlowNetwork& net, int source, int sink,
          vector<int>& dist, vector<int>& parent, vector<int>& parent_edge) {
    throw std::runtime_error("TODO(human): not implemented");
}

// =============================================================================
// TODO(human): Min-Cost Max-Flow via successive shortest paths
//
// Successive Shortest Paths algorithm:
//   total_flow = 0, total_cost = 0
//   While SPFA finds a path from source to sink in residual graph:
//     Find bottleneck along the path (trace parent_edge from sink to source)
//     Augment flow along the path:
//       For each edge on path: edges[idx].flow += bottleneck
//                              edges[idx^1].flow -= bottleneck
//     total_flow += bottleneck
//     total_cost += bottleneck * dist[sink]  (cost of this augmenting path)
//   Return (total_flow, total_cost)
//
// This finds the min-cost flow by always augmenting along the cheapest path.
// The key insight: each augmenting path uses the current shortest (cheapest)
// path in the residual graph. Negative-cost reverse edges allow "reassigning"
// flow from expensive edges to cheaper ones.
//
// Hint: path reconstruction is identical to Edmonds-Karp. Use parent_edge[v]
// to walk backward from sink to source. The bottleneck is the minimum
// residual capacity along this path.
// =============================================================================
pair<int,int> min_cost_max_flow(CostFlowNetwork& net, int source, int sink) {
    throw std::runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Print flow on each edge (skip reverse edges)
// =============================================================================
void print_flow(const CostFlowNetwork& net) {
    cout << "  Edge flows:\n";
    for (int u = 0; u < net.n; u++) {
        for (int idx : net.graph[u]) {
            const Edge& e = net.edges[idx];
            if (e.cap > 0) {
                cout << "    " << u << " -> " << e.to
                     << "  flow=" << e.flow << "/" << e.cap
                     << "  cost=" << e.cost << "/unit"
                     << "  (subtotal=" << e.flow * e.cost << ")";
                if (e.flow == e.cap) cout << "  [saturated]";
                cout << "\n";
            }
        }
    }
}

// =============================================================================
// Sample: Transportation problem
//
// Two suppliers (S1, S2) with supply, three consumers (C1, C2, C3) with demand.
// Edges from suppliers to consumers with capacity and cost.
//
// Network layout (8 nodes):
//   0 = super-source
//   1 = Supplier 1 (supply = 20)
//   2 = Supplier 2 (supply = 30)
//   3 = Consumer 1 (demand = 10)
//   4 = Consumer 2 (demand = 25)
//   5 = Consumer 3 (demand = 15)
//   6 = super-sink
//
// Source -> Suppliers (capacity = supply, cost = 0)
// Suppliers -> Consumers (capacity = INF, cost = shipping cost)
// Consumers -> Sink (capacity = demand, cost = 0)
//
// Cost matrix:
//          C1  C2  C3
//   S1     8   6   10
//   S2     9   12   7
//
// Expected: total flow = 50 (all demand met), min cost to be computed
// =============================================================================
CostFlowNetwork make_transportation() {
    CostFlowNetwork net(7);  // 0=source, 1=S1, 2=S2, 3=C1, 4=C2, 5=C3, 6=sink

    // Source to suppliers (capacity = supply, cost = 0)
    net.add_edge(0, 1, 20, 0);   // S1 supply = 20
    net.add_edge(0, 2, 30, 0);   // S2 supply = 30

    // Suppliers to consumers (capacity = large, cost = shipping)
    net.add_edge(1, 3, 50, 8);   // S1 -> C1, cost=8
    net.add_edge(1, 4, 50, 6);   // S1 -> C2, cost=6
    net.add_edge(1, 5, 50, 10);  // S1 -> C3, cost=10
    net.add_edge(2, 3, 50, 9);   // S2 -> C1, cost=9
    net.add_edge(2, 4, 50, 12);  // S2 -> C2, cost=12
    net.add_edge(2, 5, 50, 7);   // S2 -> C3, cost=7

    // Consumers to sink (capacity = demand, cost = 0)
    net.add_edge(3, 6, 10, 0);   // C1 demand = 10
    net.add_edge(4, 6, 25, 0);   // C2 demand = 25
    net.add_edge(5, 6, 15, 0);   // C3 demand = 15

    return net;
}

// =============================================================================
// Sample: Small 4-node min-cost flow
//
//   s(0) --cap=4,cost=1--> a(1) --cap=2,cost=3--> t(3)
//   s(0) --cap=2,cost=5--> b(2) --cap=3,cost=2--> t(3)
//   a(1) --cap=3,cost=1--> b(2)
//
// Expected max flow: 5 (limited by s outgoing = 4+2 = 6, but t incoming = 2+3 = 5)
// =============================================================================
CostFlowNetwork make_small_cost_network() {
    CostFlowNetwork net(4);
    net.add_edge(0, 1, 4, 1);  // s -> a: cap=4, cost=1
    net.add_edge(0, 2, 2, 5);  // s -> b: cap=2, cost=5
    net.add_edge(1, 3, 2, 3);  // a -> t: cap=2, cost=3
    net.add_edge(2, 3, 3, 2);  // b -> t: cap=3, cost=2
    net.add_edge(1, 2, 3, 1);  // a -> b: cap=3, cost=1
    return net;
}

// =============================================================================
// Main — solve sample problems
// =============================================================================
int main() {
    cout << "=== Phase 2: Min-Cost Flow (Successive Shortest Paths) ===\n\n";

    // Test 1: Small network
    {
        cout << "Test 1: Small 4-node cost network\n";
        cout << "  s --cap=4,cost=1--> a --cap=2,cost=3--> t\n";
        cout << "  s --cap=2,cost=5--> b --cap=3,cost=2--> t\n";
        cout << "  a --cap=3,cost=1--> b\n";
        auto net = make_small_cost_network();
        auto [flow, cost] = min_cost_max_flow(net, 0, 3);
        cout << "  Max flow: " << flow << ", Min cost: " << cost << "\n";
        print_flow(net);
        cout << "\n";
    }

    // Test 2: Transportation problem
    {
        cout << "Test 2: Transportation problem\n";
        cout << "  Suppliers: S1 (supply=20), S2 (supply=30)\n";
        cout << "  Consumers: C1 (demand=10), C2 (demand=25), C3 (demand=15)\n";
        cout << "  Cost matrix:\n";
        cout << "           C1  C2  C3\n";
        cout << "    S1      8   6  10\n";
        cout << "    S2      9  12   7\n";
        auto net = make_transportation();
        auto [flow, cost] = min_cost_max_flow(net, 0, 6);
        cout << "  Total flow: " << flow << " (expected: 50)\n";
        cout << "  Total cost: " << cost << "\n";
        print_flow(net);

        // Interpret the assignment
        cout << "\n  Shipping plan:\n";
        // Edges 2-7 are supplier-consumer edges (indices 4,6,8,10,12,14 in edges array)
        for (int u = 1; u <= 2; u++) {
            for (int idx : net.graph[u]) {
                const Edge& e = net.edges[idx];
                if (e.cap > 0 && e.to >= 3 && e.to <= 5 && e.flow > 0) {
                    string supplier = "S" + to_string(u);
                    string consumer = "C" + to_string(e.to - 2);
                    cout << "    " << supplier << " -> " << consumer
                         << ": " << e.flow << " units at cost " << e.cost
                         << "/unit (subtotal: " << e.flow * e.cost << ")\n";
                }
            }
        }
        cout << "\n";
    }

    cout << "Phase 2 complete.\n";
    return 0;
}
