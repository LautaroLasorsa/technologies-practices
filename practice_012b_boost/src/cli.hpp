#pragma once

#include "types.hpp"
#include <optional>

/// Parse command-line arguments into a CliConfig.
/// Returns std::nullopt if the program should exit (e.g., --help was printed).
std::optional<CliConfig> parse_cli(int argc, char* argv[]);
