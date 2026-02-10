#pragma once

#include "types.hpp"
#include <vector>

/// Generate a random directed weighted graph with `num_vertices` vertices
/// and approximately `num_edges` edges. Weights are in [1, 100].
Graph generate_random_graph(int num_vertices, int num_edges);

/// Run BFS from `source` and return the discovery order (list of vertex IDs).
std::vector<int> run_bfs(const Graph& g, Vertex source);

/// Run Dijkstra's shortest paths from `source`.
/// Returns a DijkstraResult with distances and predecessors for every vertex.
DijkstraResult run_dijkstra(const Graph& g, Vertex source);

/// Print the shortest path from `source` to `target` using the predecessor map.
void print_path(const DijkstraResult& result, int target);
