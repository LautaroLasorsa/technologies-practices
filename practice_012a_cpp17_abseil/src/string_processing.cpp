// ============================================================================
// Phase 3: Abseil String Processing
// ============================================================================
//
// Abseil's string utilities are arguably its most popular component.
// They're used pervasively in Google's codebase and in projects like
// gRPC, Protobuf, and TensorFlow.
//
// Key functions:
//   absl::StrSplit  -- Split strings by delimiter (returns lazy view)
//   absl::StrJoin   -- Join elements with separator (supports formatters)
//   absl::StrCat    -- Concatenate heterogeneous types in one allocation
//   absl::StrFormat  -- Type-safe printf (like Rust's format!)
//
// Why not just use std::string::find + substr?
//   - absl::StrSplit is lazy, allocation-free when using string_view
//   - absl::StrCat computes total size first, then does ONE allocation
//   - absl::StrJoin supports custom formatters per element
//
// Docs:
//   https://abseil.io/docs/cpp/guides/strings
//   https://abseil.io/docs/cpp/guides/format
// ============================================================================

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

// ─── Sample data ─────────────────────────────────────────────────────────────

// CSV-like log metadata (pipe-delimited, because real logs use weird formats)
const std::vector<std::string> kMetadataLines = {
    "auth-service|production|us-east-1|max_retries=3,timeout=5s,tls=true",
    "api-gateway|staging|eu-west-1|rate_limit=500/min,cache=disabled",
    "db-service|production|us-west-2|pool_size=20,query_timeout=30s",
    "payment-svc|production|ap-south-1|provider=stripe,currency=USD,retry=2",
};

// Service health data for report generation
struct ServiceHealth {
  std::string name;
  std::string status; // "healthy", "degraded", "down"
  int uptime_pct;
  double avg_latency_ms;
};

const std::vector<ServiceHealth> kHealthData = {
    {"auth-service", "healthy", 99, 12.3},
    {"api-gateway", "degraded", 95, 45.7},
    {"db-service", "healthy", 99, 8.1},
    {"payment-svc", "down", 0, 0.0},
    {"cache-service", "healthy", 99, 2.4},
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// ─── Exercise 1: Parse structured metadata with absl::StrSplit ───────────────
//
// Parse a pipe-delimited line into its components, then parse the
// key=value config pairs.
//
// Input:  "auth-service|production|us-east-1|max_retries=3,timeout=5s,tls=true"
// Output: service="auth-service", env="production", region="us-east-1",
//         config={max_retries: "3", timeout: "5s", tls: "true"}

struct ServiceMetadata {
  std::string service;
  std::string environment;
  std::string region;
  std::map<std::string, std::string> config;
};

std::ostream &operator<<(std::ostream &os, const ServiceMetadata &m) {
  os << "  Service: " << m.service << " | Env: " << m.environment
     << " | Region: " << m.region << "\n";
  os << "  Config:\n";
  for (const auto &[key, val] : m.config) {
    os << "    " << key << " = " << val << "\n";
  }
  return os;
}

ServiceMetadata parse_metadata(const std::string &line) {
  // TODO(human): Implement this in two steps.
  //
  // Step 1: Split `line` by '|' into 4 parts.
  //   std::vector<std::string> parts = absl::StrSplit(line, '|');
  //   parts[0] = service, parts[1] = env, parts[2] = region, parts[3] =
  //   config_str
  //
  // Step 2: Split the config string (parts[3]) by ',' to get key=value pairs.
  //   For each pair, split by '=' to get key and value.
  //   Store them in a std::map<std::string, std::string>.
  //
  //   for (absl::string_view pair : absl::StrSplit(config_str, ',')) {
  //       std::vector<std::string> kv = absl::StrSplit(pair, '=');
  //       ...
  //   }
  //
  // Tip: absl::StrSplit returns a type that's implicitly convertible to
  //      std::vector<std::string> or iterable as absl::string_view.
  //      The string_view version avoids allocations -- try both!

  auto pipe_split = std::vector<std::string>(absl::StrSplit(line, '|'));
  auto sm = ServiceMetadata{pipe_split[0], pipe_split[1], pipe_split[2], {}};
  for (absl::string_view item : absl::StrSplit(pipe_split[3], ',')) {
    auto kv = std::vector<std::string>(absl::StrSplit(item, '='));
    sm.config[kv[0]] = kv[1];
  }
  return sm;
}

// ─── Exercise 2: Build a report with absl::StrJoin ───────────────────────────
//
// absl::StrJoin concatenates elements with a separator. Its power comes from
// custom formatters -- you can control how each element is rendered.
//
// Simple usage:  absl::StrJoin({"a", "b", "c"}, ", ") -> "a, b, c"
// With formatter: absl::StrJoin(vec, "\n", custom_formatter)

std::string build_health_report(const std::vector<ServiceHealth> &data) {
  // TODO(human): Build a formatted health report string.
  //
  // Expected output (one line per service):
  //   "=== Health Report ===\n"
  //   "  auth-service   : HEALTHY  (uptime: 99%, latency: 12.3ms)\n"
  //   "  api-gateway    : DEGRADED (uptime: 95%, latency: 45.7ms)\n"
  //   ...
  //
  // Use absl::StrJoin with a custom formatter lambda:
  //
  //   auto formatter = [](std::string* out, const ServiceHealth& h) {
  //       absl::StrAppend(out, "  ", h.name, ...);
  //       // or use absl::StrAppendFormat(out, "  %-15s: %-8s ...", ...);
  //   };
  //
  //   std::string body = absl::StrJoin(data, "\n", formatter);
  //   return absl::StrCat("=== Health Report ===\n", body, "\n");
  //
  // The formatter signature is: void(std::string* out, const Element& elem)
  // You append to `out` -- StrJoin handles the separators.
  auto formatter = [](std::string *out, const ServiceHealth &h) {
    absl::StrAppendFormat(out,
                          "   %-15s: %-8s (uptime: %3d%%, latency: %5.1fms)",
                          h.name, h.status, h.uptime_pct, h.avg_latency_ms);
  };

  return absl::StrCat("=== Health report ===\n",
                      absl::StrJoin(data, "\n", formatter), "\n");
}

// ─── Exercise 3: Efficient concatenation with absl::StrCat ───────────────────
//
// absl::StrCat computes the total length of all arguments FIRST, allocates
// once, then copies. This beats repeated operator+ which allocates per step.
//
// absl::StrCat(a, b, c, d) == a + b + c + d  (but faster)
//
// It also handles mixed types: int, double, string, string_view, etc.

std::string build_log_line(absl::string_view timestamp, absl::string_view level,
                           absl::string_view service, int request_id,
                           double duration_ms, absl::string_view message) {
  // TODO(human): Build a single log line string using absl::StrCat.
  //
  // Format: "[<timestamp>] <level> <service> req=<request_id>
  // (<duration_ms>ms): <message>"
  //
  // Example output:
  //   "[2024-03-15T10:30:00] ERROR auth-service req=42 (12.5ms): Connection
  //   refused"
  //
  // Use absl::StrCat to combine all parts in ONE call:
  //   return absl::StrCat("[", timestamp, "] ", level, " ", service,
  //                       " req=", request_id, " (", duration_ms, "ms): ",
  //                       message);
  //
  // Key insight: StrCat accepts any "AlphaNum" type -- strings, ints, floats
  // are all converted efficiently without intermediate std::to_string calls.

  return absl::StrCat("[", timestamp, "] ", level, " ", service,
                      " req=", request_id, " (", duration_ms, "ms): ", message);
}

// ─── Exercise 4: String view as zero-copy reference ──────────────────────────
//
// absl::string_view (or std::string_view in C++17) is a non-owning reference
// to a string. Like Rust's &str.
//
// Key rule: the original string must outlive the string_view.

bool starts_with_service(absl::string_view log_line, absl::string_view prefix) {
  // TODO(human): Check if log_line starts with prefix.
  //
  // Use absl::StartsWith(log_line, prefix) or the manual way:
  //   return log_line.size() >= prefix.size()
  //       && log_line.substr(0, prefix.size()) == prefix;
  //
  // Note: absl::ConsumePrefix(&sv, prefix) returns true AND advances sv
  // past the prefix -- useful for parsing.

  return absl::StartsWith(log_line, prefix);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 3: Abseil String Processing\n";
  std::cout << std::string(55, '=') << "\n";

  // --- Exercise 1: Parse metadata ---
  print_section("Exercise 1: Parse pipe-delimited metadata");
  for (const auto &line : kMetadataLines) {
    auto meta = parse_metadata(line);
    std::cout << meta << "\n";
  }

  // --- Exercise 2: Build health report ---
  print_section("Exercise 2: Health report with StrJoin");
  std::cout << build_health_report(kHealthData) << "\n";

  // --- Exercise 3: Efficient concatenation ---
  print_section("Exercise 3: Build log lines with StrCat");

  struct LogLineInput {
    std::string ts, level, service, message;
    int req_id;
    double duration;
  };
  std::vector<LogLineInput> inputs = {
      {"2024-03-15T10:30:00", "ERROR", "auth-service", "Connection refused",
       1001, 12.5},
      {"2024-03-15T10:30:01", "INFO", "api-gateway", "Request processed", 1002,
       3.2},
      {"2024-03-15T10:30:02", "WARN", "db-service", "Slow query", 1003, 1200.0},
  };
  for (const auto &in : inputs) {
    std::cout << "  "
              << build_log_line(in.ts, in.level, in.service, in.req_id,
                                in.duration, in.message)
              << "\n";
  }

  // --- Exercise 4: string_view prefix matching ---
  print_section("Exercise 4: string_view prefix matching");

  std::vector<std::string> raw_lines = {
      "2024-03-15T10:30:00 ERROR auth-service Connection refused",
      "2024-03-15T10:30:01 INFO  api-gateway Request processed",
      "auth-service started successfully",
      "2024-03-15T10:30:02 WARN  db-service Slow query",
  };

  for (const auto &line : raw_lines) {
    bool match = starts_with_service(line, "auth");
    std::cout << "  \"" << line.substr(0, 40)
              << "...\" starts with 'auth': " << (match ? "YES" : "NO") << "\n";
  }

  // --- Bonus: demonstrate StrSplit with different delimiters ---
  print_section("Bonus: StrSplit delimiter types");
  std::cout << "  (Try splitting by absl::ByAnyChar(\",;\") or "
               "absl::MaxSplits('|', 2)\n"
            << "   in your own experiments!)\n";

  return 0;
}
