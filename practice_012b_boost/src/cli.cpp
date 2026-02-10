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

  // TODO(human): Define options + parse CLI (see guidance below)

  desc.add_options()("help,h", "Produce help message");
  desc.add_options()("vertices,v",
                     po::value<int>(&config.num_vertices)->default_value(10),
                     "Number of vertices");
  desc.add_options()("edges,e",
                     po::value<int>(&config.num_edges)->default_value(20),
                     "Number of edges");
  desc.add_options()("source,s",
                     po::value<int>(&config.source_vertex)->default_value(0),
                     "Source vertex");
  desc.add_options()("mode,m",
                     po::value<std::string>(&config.mode)->default_value("all"),
                     "Run mode - all, graph or server -");
  desc.add_options()("port,p",
                     po::value<uint16_t>(&config.port)->default_value(9090),
                     "TCP server port");
  desc.add_options()("output,o",
                     po::value<std::string>(&config.output_file)
                         ->default_value("dijkstra_results.txt"),
                     "File to save the output");
  po::variables_map vm;

  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << "\n\n";
    return std::nullopt;
  }

  // ========================================================================
  // Validation (provided) -- ensures source vertex is in range
  // ========================================================================

  if (config.source_vertex < 0 || config.source_vertex >= config.num_vertices) {
    std::cerr << "Error: --source must be in [0, --vertices)\n";
    return std::nullopt;
  }

  return config;
}
