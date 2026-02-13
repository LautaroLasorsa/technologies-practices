#include <bits/stdc++.h>
using namespace std;

// =============================================================================
// Edge in the flow network
// Each edge has a paired reverse edge at index (idx ^ 1) in the edge list.
// =============================================================================
struct Edge {
    int to;       // destination vertex
    int cap;      // capacity
    int flow;     // current flow
};

// =============================================================================
// Flow Network — adjacency list representation
//
// Edges are stored in a flat vector. For each vertex, graph[v] holds the
// indices into 'edges' of all edges leaving v.
//
// Key invariant: edges are added in pairs (forward + reverse). For edge at
// index i, its reverse is at index i^1 (XOR with 1). This makes residual
// graph updates trivial: augmenting forward edge i means also updating i^1.
// =============================================================================
struct FlowNetwork {
    int n;                          // number of vertices
    vector<Edge> edges;             // flat edge storage
    vector<vector<int>> graph;      // graph[v] = list of edge indices from v

    explicit FlowNetwork(int n) : n(n), graph(n) {}

    // Add a directed edge (from -> to) with given capacity.
    // Also adds the reverse edge (to -> from) with capacity 0.
    // Returns the index of the forward edge.
    int add_edge(int from, int to, int capacity) {
        int idx = (int)edges.size();
        edges.push_back({to, capacity, 0});
        graph[from].push_back(idx);
        edges.push_back({from, 0, 0});       // reverse edge, cap=0
        graph[to].push_back(idx + 1);
        return idx;
    }

    // Residual capacity of edge at index idx
    int residual(int idx) const {
        return edges[idx].cap - edges[idx].flow;
    }
};

// =============================================================================
// BFS in the residual graph — finds shortest augmenting path from source to
// sink. Returns true if a path exists, filling parent_edge so the path can
// be reconstructed.
//
// parent_edge[v] = index of the edge used to reach v (or -1 if not reached).
// =============================================================================
bool bfs(FlowNetwork& net, int source, int sink, vector<int>& parent_edge) {
    fill(parent_edge.begin(), parent_edge.end(), -1);
    parent_edge[source] = -2;  // mark source as visited (but no parent edge)

    queue<int> q;
    q.push(source);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int idx : net.graph[u]) {
            int v = net.edges[idx].to;
            if (parent_edge[v] == -1 && net.residual(idx) > 0) {
                parent_edge[v] = idx;
                if (v == sink) return true;
                q.push(v);
            }
        }
    }
    return false;
}

// =============================================================================
// TODO(human): Edmonds-Karp — BFS-based max flow
//
// Edmonds-Karp (BFS-based Ford-Fulkerson):
//
// While there exists an augmenting path from source to sink in the residual graph:
//   1. BFS from source to sink in residual graph
//      (edge exists in residual if capacity - flow > 0)
//   2. Find bottleneck: min residual capacity along the path
//   3. Augment: for each edge on path:
//      - Increase flow on forward edge by bottleneck
//      - Decrease flow on reverse edge by bottleneck (equivalently, increase reverse capacity)
//
// The residual graph trick: for each original edge (u,v) with capacity c,
// maintain a reverse edge (v,u) with capacity 0. When we send flow f on (u,v):
//   - residual capacity of (u,v) = c - f
//   - residual capacity of (v,u) = f  (allows "undoing" flow)
//
// BFS guarantees shortest augmenting path -> O(VE^2) complexity.
//
// After max flow: min-cut = edges (u,v) where u is reachable from source
// in residual graph but v is not.
//
// Hint: use parent_edge to trace the path backward from sink to source.
// For edge at index idx, the reverse edge is at idx ^ 1.
// To augment, add bottleneck to edges[idx].flow and subtract from edges[idx^1].flow.
// =============================================================================
int edmonds_karp(FlowNetwork& net, int source, int sink) {
    throw std::runtime_error("TODO(human): not implemented");
}

// =============================================================================
// TODO(human): Find min cut after running max flow
//
// After max flow, perform BFS from source in the residual graph.
// All vertices reachable from source form set S; the rest form set T.
// The min cut consists of all original edges (u, v) where u in S and v in T.
//
// Return the list of (u, v) pairs forming the min cut.
//
// Hint: BFS from source using only edges with residual > 0.
// Then iterate all edges: if edges[idx].cap > 0 (original edge, not reverse),
// check if 'from' is reachable and 'to' is not reachable.
// To get 'from' for edge at index idx: look through graph to find which vertex
// owns it, OR track it when iterating graph[u] for each u.
// =============================================================================
vector<pair<int,int>> find_min_cut(FlowNetwork& net, int source) {
    throw std::runtime_error("TODO(human): not implemented");
}

// =============================================================================
// Print flow on each edge (skip reverse edges with cap=0)
// =============================================================================
void print_flow(const FlowNetwork& net) {
    cout << "  Edge flows:\n";
    for (int u = 0; u < net.n; u++) {
        for (int idx : net.graph[u]) {
            const Edge& e = net.edges[idx];
            if (e.cap > 0) {  // original edge (not reverse)
                cout << "    " << u << " -> " << e.to
                     << "  flow=" << e.flow << "/" << e.cap;
                if (e.flow == e.cap) cout << "  [saturated]";
                cout << "\n";
            }
        }
    }
}

// =============================================================================
// Print min cut edges
// =============================================================================
void print_min_cut(const vector<pair<int,int>>& cut) {
    cout << "  Min cut edges:\n";
    for (auto [u, v] : cut) {
        cout << "    " << u << " -> " << v << "\n";
    }
}

// =============================================================================
// Sample networks
// =============================================================================

// Simple 4-node network:
//   0 --10--> 1 --10--> 3
//   0 --10--> 2 --10--> 3
//   1 ---1--> 2
// Expected max flow: 20
FlowNetwork make_simple_network() {
    FlowNetwork net(4);
    net.add_edge(0, 1, 10);
    net.add_edge(0, 2, 10);
    net.add_edge(1, 3, 10);
    net.add_edge(2, 3, 10);
    net.add_edge(1, 2, 1);
    return net;
}

// Medium 6-node transportation network:
//   Source(0) -> A(1): 16
//   Source(0) -> B(2): 13
//   A(1) -> B(2): 4
//   A(1) -> C(3): 12
//   B(2) -> A(1): 10
//   B(2) -> D(4): 14
//   C(3) -> B(2): 9
//   C(3) -> Sink(5): 20
//   D(4) -> C(3): 7
//   D(4) -> Sink(5): 4
// This is the classic CLRS max-flow example.
// Expected max flow: 23
FlowNetwork make_clrs_network() {
    FlowNetwork net(6);
    net.add_edge(0, 1, 16);
    net.add_edge(0, 2, 13);
    net.add_edge(1, 2, 4);
    net.add_edge(1, 3, 12);
    net.add_edge(2, 1, 10);
    net.add_edge(2, 4, 14);
    net.add_edge(3, 2, 9);
    net.add_edge(3, 5, 20);
    net.add_edge(4, 3, 7);
    net.add_edge(4, 5, 4);
    return net;
}

// Wikipedia max-flow example (7 nodes: s=0, a=1, b=2, c=3, d=4, e=5, t=6):
//   s->a:3, s->b:3, a->b:2, a->c:3, b->d:3, c->d:1, c->t:2, d->e:2, d->t:3, e->t:1 (modified capacities for non-trivial cut)
// We use a slightly adapted version for a clean example.
// Expected max flow: 5
FlowNetwork make_wikipedia_network() {
    FlowNetwork net(7);
    net.add_edge(0, 1, 3);  // s -> a
    net.add_edge(0, 2, 3);  // s -> b
    net.add_edge(1, 2, 2);  // a -> b
    net.add_edge(1, 3, 3);  // a -> c
    net.add_edge(2, 4, 3);  // b -> d
    net.add_edge(3, 4, 1);  // c -> d
    net.add_edge(3, 6, 2);  // c -> t
    net.add_edge(4, 5, 2);  // d -> e
    net.add_edge(4, 6, 3);  // d -> t (modified to make flow=5 achievable)
    net.add_edge(5, 6, 1);  // e -> t (modified)
    return net;
}

// =============================================================================
// Main — run all sample networks
// =============================================================================
int main() {
    cout << "=== Phase 1: Max Flow (Edmonds-Karp) ===\n\n";

    // Test 1: Simple 4-node
    {
        cout << "Test 1: Simple 4-node network\n";
        cout << "  0 --10--> 1 --10--> 3\n";
        cout << "  0 --10--> 2 --10--> 3\n";
        cout << "  1 ---1--> 2\n";
        auto net = make_simple_network();
        int flow = edmonds_karp(net, 0, 3);
        cout << "  Max flow: " << flow << " (expected: 20)\n";
        print_flow(net);
        auto cut = find_min_cut(net, 0);
        print_min_cut(cut);
        cout << "\n";
    }

    // Test 2: CLRS 6-node
    {
        cout << "Test 2: CLRS 6-node network\n";
        auto net = make_clrs_network();
        int flow = edmonds_karp(net, 0, 5);
        cout << "  Max flow: " << flow << " (expected: 23)\n";
        print_flow(net);
        auto cut = find_min_cut(net, 0);
        print_min_cut(cut);
        cout << "\n";
    }

    // Test 3: Wikipedia-style 7-node
    {
        cout << "Test 3: Wikipedia-style 7-node network\n";
        auto net = make_wikipedia_network();
        int flow = edmonds_karp(net, 0, 6);
        cout << "  Max flow: " << flow << " (expected: 5)\n";
        print_flow(net);
        auto cut = find_min_cut(net, 0);
        print_min_cut(cut);
        cout << "\n";
    }

    cout << "Phase 1 complete.\n";
    return 0;
}
