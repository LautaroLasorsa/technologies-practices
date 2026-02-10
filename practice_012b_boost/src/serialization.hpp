#pragma once

#include "types.hpp"
#include <string>

/// Serialize a DijkstraResult to a text file using Boost.Serialization.
void save_results(const DijkstraResult& result, const std::string& filename);

/// Deserialize a DijkstraResult from a text file.
DijkstraResult load_results(const std::string& filename);

/// Verify that two DijkstraResult structs are identical (round-trip test).
bool verify_roundtrip(const DijkstraResult& original, const DijkstraResult& loaded);
