#include "cli.hpp"
#include "graph_algo.hpp"
#include "serialization.hpp"
#include "server.hpp"
#include "types.hpp"

#include <iostream>

// ============================================================================
// Phase orchestration
// ============================================================================

static void run_graph_phase(const CliConfig& config) {
    std::cout << "\n=== Graph Construction & Algorithms ===\n";
    auto graph = generate_random_graph(config.num_vertices, config.num_edges);

    std::cout << "\n--- BFS ---\n";
    run_bfs(graph, config.source_vertex);

    std::cout << "\n--- Dijkstra ---\n";
    auto result = run_dijkstra(graph, config.source_vertex);

    // Print a few example paths
    int n = config.num_vertices;
    for (int target : {n / 4, n / 2, n - 1}) {
        if (target > 0 && target < n) {
            print_path(result, target);
        }
    }

    std::cout << "\n--- Serialization ---\n";
    save_results(result, config.output_file);
    auto loaded = load_results(config.output_file);
    verify_roundtrip(result, loaded);
}

static void run_server_phase(const CliConfig& config) {
    std::cout << "\n=== TCP Server ===\n";
    auto graph = generate_random_graph(config.num_vertices, config.num_edges);
    run_server(graph, config.port, 30);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    auto maybe_config = parse_cli(argc, argv);
    if (!maybe_config) {
        return 0; // --help was shown, or validation failed
    }
    const auto& config = *maybe_config;

    if (config.mode == "all" || config.mode == "graph") {
        run_graph_phase(config);
    }

    if (config.mode == "all" || config.mode == "server") {
        run_server_phase(config);
    }

    return 0;
}
