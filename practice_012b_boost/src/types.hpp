#pragma once

#include <boost/graph/adjacency_list.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

#include <cstdint>
#include <string>
#include <vector>

// ============================================================================
// Graph type aliases
// ============================================================================

// Edge weight property
using EdgeWeightProperty = boost::property<boost::edge_weight_t, int>;

// Directed weighted graph: adjacency_list with vecS for both containers
//   OutEdgeList = vecS  (vector of out-edges per vertex)
//   VertexList  = vecS  (vertices stored in a vector, indexed by int)
//   Directed    = directedS
//   VertexProp  = no_property
//   EdgeProp    = EdgeWeightProperty (int weight on each edge)
using Graph = boost::adjacency_list<
    boost::vecS,
    boost::vecS,
    boost::directedS,
    boost::no_property,
    EdgeWeightProperty
>;

using Vertex = boost::graph_traits<Graph>::vertex_descriptor;
using Edge = boost::graph_traits<Graph>::edge_descriptor;

// ============================================================================
// Dijkstra results -- serializable
// ============================================================================

struct DijkstraResult {
    int source_vertex = 0;
    std::vector<int> distances;
    std::vector<int> predecessors;

    // Boost.Serialization intrusive method.
    // The template allows this single function to handle both
    // serialization (saving) and deserialization (loading).
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int /* version */) {
        ar & source_vertex;
        ar & distances;
        ar & predecessors;
    }
};

// ============================================================================
// CLI configuration -- populated by program_options parsing
// ============================================================================

struct CliConfig {
    int num_vertices = 10;
    int num_edges = 20;
    int source_vertex = 0;
    std::string mode = "all";   // "all", "graph", "server"
    uint16_t port = 9090;
    std::string output_file = "dijkstra_results.txt";
};
