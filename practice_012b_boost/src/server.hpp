#pragma once

#include "types.hpp"
#include <cstdint>

/// Start an async TCP server on the given port.
/// Clients send a single line with a source vertex ID (e.g., "3\n").
/// The server runs Dijkstra from that vertex on `graph` and responds
/// with the distance to every other vertex, one per line.
///
/// The server shuts down automatically after `timeout_seconds`.
void run_server(const Graph& graph, uint16_t port, int timeout_seconds = 30);
