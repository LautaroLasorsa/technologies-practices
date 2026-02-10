#include "graph_algo.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/visitors.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <vector>

// ============================================================================
// Graph generation (provided -- not the learning focus)
// ============================================================================

Graph generate_random_graph(int num_vertices, int num_edges) {
  Graph g(num_vertices);
  std::mt19937 rng(42); // fixed seed for reproducibility
  std::uniform_int_distribution<int> vert_dist(0, num_vertices - 1);
  std::uniform_int_distribution<int> weight_dist(1, 100);

  std::set<std::pair<int, int>> existing;
  int added = 0;

  while (added < num_edges) {
    int u = vert_dist(rng);
    int v = vert_dist(rng);
    if (u == v)
      continue; // no self-loops
    if (existing.count({u, v}))
      continue; // no parallel edges
    existing.insert({u, v});

    boost::add_edge(u, v, weight_dist(rng), g);
    ++added;
  }

  std::cout << "Generated graph: " << num_vertices << " vertices, " << added
            << " edges\n";
  return g;
}

// ============================================================================
// BFS -- discovery order
// ============================================================================

// A BFS visitor that records the order in which vertices are discovered.
// BGL visitors are the idiomatic way to hook into algorithm events.
//
// Concept: BGL algorithms (bfs, dfs, dijkstra) accept a "visitor" object
// whose methods are called at specific algorithm events:
//   - discover_vertex(v, g)  -- when v is first encountered
//   - examine_edge(e, g)     -- when an edge is examined
//   - finish_vertex(v, g)    -- when all of v's neighbors are processed
//
// This is the Visitor pattern from GoF, applied generically via templates.

class BfsDiscoveryRecorder : public boost::default_bfs_visitor {
public:
  explicit BfsDiscoveryRecorder(std::vector<int> &order) : order_(order) {}

  // ========================================================================
  // TODO(human): Implement discover_vertex
  // ========================================================================
  //
  // This method is called by BGL's BFS each time a NEW vertex is discovered.
  //
  // What to do:
  //   Push the vertex `v` (cast to int) into order_.
  //
  // Signature (must match exactly for BGL to call it):
  //   template <typename G>
  //   void discover_vertex(Vertex v, const G& /* g */) { ... }
  //
  // Hint: order_.push_back(static_cast<int>(v));
  //
  // Docs: https://www.boost.org/doc/libs/release/libs/graph/doc/BFSVisitor.html
  // ========================================================================

  // --- YOUR CODE HERE ---

private:
  std::vector<int> &order_;
};

std::vector<int> run_bfs(const Graph &g, Vertex source) {
  std::vector<int> discovery_order;

  // ========================================================================
  // TODO(human): Run BFS using boost::breadth_first_search
  // ========================================================================
  //
  // Steps:
  //   1. Create a BfsDiscoveryRecorder visitor, passing discovery_order
  //   2. Call boost::breadth_first_search(g, source, boost::visitor(recorder))
  //
  // That's it! BGL handles the queue, color map, and traversal internally.
  // Your visitor's discover_vertex() is called at each discovery event.
  //
  // Docs:
  // https://www.boost.org/doc/libs/release/libs/graph/doc/breadth_first_search.html
  // ========================================================================

  // --- YOUR CODE HERE ---

  std::cout << "BFS discovery order from vertex " << source << ": ";
  for (int v : discovery_order) {
    std::cout << v << " ";
  }
  std::cout << "\n";

  return discovery_order;
}

// ============================================================================
// Dijkstra's shortest paths
// ============================================================================

DijkstraResult run_dijkstra(const Graph &g, Vertex source) {
  int n = static_cast<int>(boost::num_vertices(g));
  DijkstraResult result;
  result.source_vertex = static_cast<int>(source);
  result.distances.resize(n, std::numeric_limits<int>::max());
  result.predecessors.resize(n);

  // ========================================================================
  // TODO(human): Run Dijkstra's algorithm using boost::dijkstra_shortest_paths
  // ========================================================================
  //
  // Call boost::dijkstra_shortest_paths with named parameters:
  //
  //   boost::dijkstra_shortest_paths(
  //       g,
  //       source,
  //       boost::predecessor_map(
  //           boost::make_iterator_property_map(
  //               result.predecessors.begin(),
  //               boost::get(boost::vertex_index, g)
  //           )
  //       )
  //       .distance_map(
  //           boost::make_iterator_property_map(
  //               result.distances.begin(),
  //               boost::get(boost::vertex_index, g)
  //           )
  //       )
  //   );
  //
  // Key concepts:
  //   - BGL uses "property maps" to associate data with vertices/edges
  //   - make_iterator_property_map wraps a vector iterator as a property map
  //   - vertex_index is the built-in index for vecS-based graphs (0, 1, 2...)
  //   - The named-parameter idiom (.predecessor_map(...).distance_map(...))
  //     chains configuration fluently
  //
  // After this call, result.distances[v] = shortest distance from source to v,
  // and result.predecessors[v] = previous vertex on the shortest path to v.
  //
  // CP analogy: This is exactly like your Dijkstra, but instead of
  //   dist[] and par[] arrays with a priority_queue, BGL manages
  //   the internal state and exposes results via property maps.
  //
  // Docs:
  // https://www.boost.org/doc/libs/release/libs/graph/doc/dijkstra_shortest_paths.html
  // ========================================================================

  // --- YOUR CODE HERE ---

  std::cout << "Dijkstra from vertex " << source << ":\n";
  for (int i = 0; i < n; ++i) {
    if (result.distances[i] == std::numeric_limits<int>::max()) {
      std::cout << "  -> " << i << ": unreachable\n";
    } else {
      std::cout << "  -> " << i << ": distance = " << result.distances[i]
                << "\n";
    }
  }

  return result;
}

// ============================================================================
// Path reconstruction (provided)
// ============================================================================

void print_path(const DijkstraResult &result, int target) {
  if (target < 0 || target >= static_cast<int>(result.distances.size())) {
    std::cout << "Invalid target vertex\n";
    return;
  }

  if (result.distances[target] == std::numeric_limits<int>::max()) {
    std::cout << "No path from " << result.source_vertex << " to " << target
              << "\n";
    return;
  }

  // Reconstruct path by following predecessors backward
  std::vector<int> path;
  for (int v = target; v != result.source_vertex; v = result.predecessors[v]) {
    path.push_back(v);
    if (static_cast<int>(path.size()) >
        static_cast<int>(result.distances.size())) {
      std::cout << "Cycle detected in predecessor map (bug)\n";
      return;
    }
  }
  path.push_back(result.source_vertex);
  std::reverse(path.begin(), path.end());

  std::cout << "Path " << result.source_vertex << " -> " << target
            << " (distance " << result.distances[target] << "): ";
  for (size_t i = 0; i < path.size(); ++i) {
    if (i > 0)
      std::cout << " -> ";
    std::cout << path[i];
  }
  std::cout << "\n";
}
