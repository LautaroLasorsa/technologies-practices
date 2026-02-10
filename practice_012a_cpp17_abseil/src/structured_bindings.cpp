// ============================================================================
// Phase 1: Structured Bindings & Modern Control Flow
// ============================================================================
//
// C++17 introduced structured bindings (auto [a, b] = ...) and
// init-statements in if/switch (if (auto x = ...; condition)).
//
// You already use structured bindings in CP:
//   auto [lo, hi] = equal_range(ALL(v), target);
//
// This phase goes deeper: binding to maps, custom structs, and combining
// with init-statements for cleaner control flow.
//
// Docs:
//   https://en.cppreference.com/w/cpp/language/structured_binding
//   https://en.cppreference.com/w/cpp/language/if
// ============================================================================

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

// ─── Data types ──────────────────────────────────────────────────────────────

struct LogEntry {
  std::string timestamp; // e.g. "2024-03-15T10:30:00"
  std::string level;     // e.g. "ERROR"
  std::string service;   // e.g. "auth-service"
  std::string message;   // e.g. "Connection refused"
};

// Allow printing a LogEntry
std::ostream &operator<<(std::ostream &os, const LogEntry &entry) {
  return os << "[" << entry.timestamp << "] " << entry.level << " ("
            << entry.service << "): " << entry.message;
}

// ─── Sample data ─────────────────────────────────────────────────────────────

const std::vector<std::string> kRawLogLines = {
    "2024-03-15T10:30:00 ERROR auth-service Connection refused",
    "2024-03-15T10:30:01 INFO  api-gateway Request processed in 42ms",
    "2024-03-15T10:30:02 WARN  db-service Slow query detected (1200ms)",
    "2024-03-15T10:30:03 ERROR payment-svc Timeout waiting for response",
    "2024-03-15T10:30:04 INFO  auth-service User login successful",
    "2024-03-15T10:30:05 DEBUG api-gateway Cache hit ratio: 0.87",
};

// ─── Helpers (fully implemented) ─────────────────────────────────────────────

// Splits a string at the first N spaces, returning a tuple of substrings.
// Example: split_first_n("a b c d e", 3) -> ("a", "b", "c", "d e")
std::tuple<std::string, std::string, std::string, std::string>
split_log_line(const std::string &line) {
  auto p1 = line.find(' ');
  auto p2 = line.find(' ', p1 + 1);
  auto p3 = line.find(' ', p2 + 1);

  return {line.substr(0, p1), line.substr(p1 + 1, p2 - p1 - 1),
          line.substr(p2 + 1, p3 - p2 - 1), line.substr(p3 + 1)};
}

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// ─── Exercise 1: Parse log lines using structured bindings ───────────────────
//
// Use structured bindings to destructure the tuple returned by split_log_line
// into a LogEntry.
//
// Hint: auto [ts, lvl, svc, msg] = split_log_line(line);
//       Then construct a LogEntry from those variables.

LogEntry parse_log_line(const std::string &line) {
  // TODO(human): Use structured bindings to destructure the tuple
  //              from split_log_line() into individual variables,
  //              then return a LogEntry{...} from them.
  //
  // Pattern:
  //   auto [a, b, c, d] = some_tuple_returning_function();
  //   return LogEntry{a, b, c, d};
  return {};
}

// ─── Exercise 2: Iterate a map with structured bindings ──────────────────────
//
// Given a map of service -> error count, print each entry using structured
// bindings in the range-for loop.
//
// In CP you'd write: for (auto& [k, v] : mp) ...
// Same idea here, but build the map from parsed LogEntries.

void count_errors_by_service(const std::vector<LogEntry> &entries) {
  // TODO(human): 1. Build a std::map<std::string, int> counting how many
  //                 entries have level == "ERROR" for each service.
  //              2. Iterate the map with structured bindings:
  //                 for (const auto& [service, count] : error_counts)
  //              3. Print each: "  <service>: <count> errors"
  //
  // This is straightforward -- the point is practicing the syntax
  // in a non-CP context (string keys, real data).
}

// ─── Exercise 3: if-with-initializer ─────────────────────────────────────────
//
// C++17 lets you write:
//   if (auto it = map.find(key); it != map.end()) { use it->second; }
//
// This keeps `it` scoped to the if/else block -- no leaking variables.
//
// Implement a function that looks up a service in a config map and prints
// its config if found, or a default message if not.

void lookup_service_config(
    const absl::flat_hash_map<std::string, std::string> &config,
    const std::string &service) {
  // TODO(human): Use if-with-initializer to find `service` in `config`.
  //
  //   if (auto it = config.find(service); it != config.end()) {
  //       // Print: "Config for <service>: <it->second>"
  //   } else {
  //       // Print: "No config found for <service>, using defaults"
  //   }
  //
  // Bonus: also try the same pattern with a switch-with-initializer:
  //   switch (auto len = service.size(); len) { case 0: ...; default: ...; }

}

// ─── Exercise 4: Structured bindings with custom struct ──────────────────────
//
// Structured bindings work with ANY aggregate type (struct with all public
// members), not just tuples and pairs.

struct QueryResult {
  bool found;
  int match_count;
  std::string first_match;
};

// Simulates searching logs for a keyword
QueryResult search_logs(const std::vector<LogEntry> &entries,
                        const std::string &keyword) {
  // TODO(human): Search through entries for any whose message contains
  //              `keyword` (use std::string::find != std::string::npos).
  //              Return a QueryResult with:
  //                - found: true if at least one match
  //                - match_count: total matches
  //                - first_match: the message of the first matching entry
  //                  (or "" if none found)

  return {};
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 1: Structured Bindings & Modern Control Flow\n";
  std::cout << std::string(55, '=') << "\n";

  // --- Parse all log lines ---
  print_section("Exercise 1: Parse log lines");
  std::vector<LogEntry> entries;
  for (const auto &line : kRawLogLines) {
    entries.push_back(parse_log_line(line));
    std::cout << "  " << entries.back() << "\n";
  }

  // --- Count errors by service ---
  print_section("Exercise 2: Error counts by service");
  count_errors_by_service(entries);

  // --- Lookup service config ---
  print_section("Exercise 3: if-with-initializer");
  absl::flat_hash_map<std::string, std::string> service_config = {
      {"auth-service", "max_retries=3, timeout=5s"},
      {"api-gateway", "rate_limit=1000/min, cache=enabled"},
      {"db-service", "pool_size=20, query_timeout=30s"},
  };
  lookup_service_config(service_config, "auth-service");
  lookup_service_config(service_config, "payment-svc");

  // --- Structured bindings with custom struct ---
  print_section("Exercise 4: Search logs with structured binding on result");

  auto result = search_logs(entries, "refused");

  // Destructure the result -- this is the structured binding on YOUR struct
  auto [found, count, first] = result;
  if (found) {
    std::cout << "  Found " << count << " match(es). First: \"" << first
              << "\"\n";
  } else {
    std::cout << "  No matches found.\n";
  }

  // Try another search
  auto [found2, count2, first2] = search_logs(entries, "timeout");
  if (found2) {
    std::cout << "  Found " << count2 << " match(es). First: \"" << first2
              << "\"\n";
  } else {
    std::cout << "  No matches found.\n";
  }

  return 0;
}
