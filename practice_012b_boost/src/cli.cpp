#include "cli.hpp"

#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <cstdint>
#include <iostream>

namespace po = boost::program_options;

std::optional<CliConfig> parse_cli(int argc, char *argv[]) {
  CliConfig config;

  // ========================================================================
  // TODO(human): Define the command-line options using Boost.Program_options
  // ========================================================================
  //
  // Create a po::options_description named "Graph Analysis Tool" and add
  // the following options:
  //
  //   --help, -h          : produce help message (implicit bool)
  //   --vertices, -v      : number of vertices (int, default 10)
  //   --edges, -e         : number of edges (int, default 20)
  //   --source, -s        : source vertex for Dijkstra (int, default 0)
  //   --mode, -m          : run mode: "all", "graph", or "server" (string,
  //   default "all")
  //   --port, -p          : TCP server port (uint16_t, default 9090)
  //   --output, -o        : output file for serialized results (string, default
  //   "dijkstra_results.txt")
  //
  // Hint: Use desc.add_options()("name,abbrev",
  // po::value<type>(&config.field)->default_value(x), "description")
  //
  // Docs:
  // https://www.boost.org/doc/libs/release/doc/html/program_options/tutorial.html
  // ========================================================================

  po::options_description desc("Graph Analysis Tool");

  // TODO(human): Define options + parse CLI
  //
  // 1. Use desc.add_options() to register each CLI flag. Chain calls like:
  //      desc.add_options()("help,h", "Produce help message");
  //      desc.add_options()("vertices,v", po::value<int>(&config.num_vertices)->default_value(10), "...");
  //    Add options for: help, vertices, edges, source, mode, port, output
  //    (see the TODO block at the top of this function for the full list).
  //
  // 2. Create a po::variables_map, then:
  //      po::store(po::parse_command_line(argc, argv, desc), vm);
  //      po::notify(vm);
  //    notify() applies default values and triggers any value-semantic callbacks.
  //
  // 3. Check vm.count("help") -- if set, print `desc` and return std::nullopt.
  //
  // Docs: https://www.boost.org/doc/libs/release/doc/html/program_options/tutorial.html

  // --- YOUR CODE HERE ---

  po::variables_map vm;

  // ========================================================================
  // Validation (provided) -- ensures source vertex is in range
  // ========================================================================

  if (config.source_vertex < 0 || config.source_vertex >= config.num_vertices) {
    std::cerr << "Error: --source must be in [0, --vertices)\n";
    return std::nullopt;
  }

  return config;
}
