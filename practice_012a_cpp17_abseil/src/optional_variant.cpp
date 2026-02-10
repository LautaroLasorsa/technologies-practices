// ============================================================================
// Phase 2: std::optional, std::variant & std::visit
// ============================================================================
//
// std::optional<T> is C++17's "nullable value" -- like Rust's Option<T>.
// std::variant<Ts...> is a type-safe tagged union -- like Rust's enum.
// std::visit lets you pattern-match on a variant -- like Rust's match.
//
// In CP you rarely use these (you'd just use -1 or a sentinel).
// In production code they eliminate entire classes of bugs.
//
// Docs:
//   https://en.cppreference.com/w/cpp/utility/optional
//   https://en.cppreference.com/w/cpp/utility/variant
//   https://en.cppreference.com/w/cpp/utility/variant/visit
// ============================================================================

#include <iostream>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/str_split.h"

// ─── Data types ──────────────────────────────────────────────────────────────

struct LogEntry {
  std::string timestamp;
  std::string level;
  std::string service;
  std::string message;
};

std::ostream &operator<<(std::ostream &os, const LogEntry &e) {
  return os << "[" << e.timestamp << "] " << e.level << " (" << e.service
            << "): " << e.message;
}

// A log field can hold different types of values
using LogValue = std::variant<int, double, std::string>;

// ─── Sample data ─────────────────────────────────────────────────────────────

const std::vector<std::string> kRawLines = {
    "2024-03-15T10:30:00 ERROR auth-service Connection refused",
    "2024-03-15T10:30:01 INFO  api-gateway Request processed in 42ms",
    "BAD LINE WITH NO STRUCTURE",
    "2024-03-15T10:30:02 WARN  db-service Slow query detected",
    "",
    "2024-03-15T10:30:03 ERROR payment-svc Timeout",
    "ANOTHER BAD LINE",
};

// ─── Helpers (fully implemented) ─────────────────────────────────────────────

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// The "overloaded" pattern: a helper struct for creating a visitor from
// lambdas. This is a well-known C++17 idiom. Study it -- you'll use it a lot.
//
// How it works:
//   1. `overloaded` inherits from all the lambda types you pass in.
//   2. `using Fs::operator()...` brings all the call operators into scope.
//   3. CTAD (class template argument deduction) lets you write:
//      overloaded{lambda1, lambda2, lambda3} without specifying template args.
//
// Usage:
//   std::visit(overloaded{
//       [](int i)    { ... },
//       [](double d) { ... },
//       [](const std::string& s) { ... },
//   }, some_variant);
template <class... Fs> struct overloaded : Fs... {
  using Fs::operator()...;
};
// CTAD deduction guide (C++17)
template <class... Fs> overloaded(Fs...) -> overloaded<Fs...>;

// ─── Exercise 1: Safe parsing with std::optional ─────────────────────────────
//
// Parse a log line into a LogEntry. Return std::nullopt if the line is
// malformed (fewer than 4 space-separated tokens).
//
// This is like returning None in Python or None in Rust's Option.
//
// Hint: Use absl::StrSplit to split on spaces (with absl::MaxSplits if you
//       want, or just check the resulting vector's size).

std::optional<LogEntry> try_parse_log(const std::string &line) {
  // TODO(human): Implement this function.
  //
  // Steps:
  //   1. Split `line` by spaces: std::vector<std::string> parts =
  //   absl::StrSplit(line, ' ');
  //      (Note: absl::StrSplit returns something convertible to vector<string>)
  //   2. If parts.size() < 4, return std::nullopt
  //   3. Otherwise, build a LogEntry:
  //      - timestamp = parts[0]
  //      - level = parts[1]
  //      - service = parts[2]
  //      - message = everything from parts[3] onward (join with spaces,
  //        or use absl::StrJoin on a subrange, or just re-split with MaxSplits)
  //   4. Return the LogEntry (implicit conversion to std::optional)
  //
  // Key insight: returning a LogEntry from this function automatically wraps
  // it in std::optional -- no explicit std::optional<LogEntry>{entry} needed.

  return std::nullopt;
}

// ─── Exercise 2: Using std::optional results ─────────────────────────────────
//
// Process a batch of raw log lines. For each line, try to parse it.
// Collect only the successfully parsed entries.
//
// Demonstrates: checking .has_value() or using the bool conversion,
// and extracting with .value() or operator*.

std::vector<LogEntry>
parse_all_logs(const std::vector<std::string> &raw_lines) {
  // TODO(human): Implement this function.
  //
  // For each line in raw_lines:
  //   1. Call try_parse_log(line)
  //   2. If the result has a value (if (auto entry = try_parse_log(line)))
  //      push it into the output vector
  //   3. If not, print a warning: "  SKIP: malformed line: \"<line>\""
  //
  // Notice the if-with-initializer + optional pattern:
  //   if (auto entry = try_parse_log(line)) {
  //       // entry is truthy, use *entry or entry.value()
  //   }
  //
  // Return the vector of successfully parsed entries.

  return {};
}

// ─── Exercise 3: std::variant & the overloaded visitor ───────────────────────
//
// Log fields can have different types. Model this with std::variant.
//
// Given a vector of (field_name, LogValue) pairs, format each value
// for display using std::visit with the overloaded pattern.

std::string format_log_value(const LogValue &value) {
  // TODO(human): Use std::visit with the overloaded pattern to format
  //              the value based on its actual type.
  //
  // Pattern:
  //   return std::visit(overloaded{
  //       [](int i)                -> std::string { return "int:" +
  //       std::to_string(i); },
  //       [](double d)             -> std::string { return "float:" +
  //       std::to_string(d); },
  //       [](const std::string& s) -> std::string { return "str:\"" + s + "\"";
  //       },
  //   }, value);
  //
  // The overloaded struct (defined above) makes this work by combining
  // multiple lambdas into one callable object.

  return {};
}

// ─── Exercise 4: Variant-based log severity with behavior ────────────────────
//
// Model log severity as a variant where each level carries different data.
// This mirrors Rust's enum with associated data.

struct DebugInfo {
  std::string module;
  int verbosity;
};
struct ErrorInfo {
  std::string error_code;
  std::string stack_trace;
};
struct MetricInfo {
  std::string name;
  double value;
  std::string unit;
};

using SeverityPayload = std::variant<DebugInfo, ErrorInfo, MetricInfo>;

// Format a severity payload for display
std::string describe_payload(const SeverityPayload &payload) {
  // TODO(human): Use std::visit with the overloaded pattern.
  //
  //   For DebugInfo:  return "DEBUG [<module>] verbosity=<verbosity>"
  //   For ErrorInfo:  return "ERROR <error_code>: <stack_trace>"
  //   For MetricInfo: return "METRIC <name>=<value><unit>"
  //
  // Example outputs:
  //   "DEBUG [auth] verbosity=3"
  //   "ERROR E_CONN_REFUSED: at line 42"
  //   "METRIC latency_ms=42.5ms"

  return {};
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 2: std::optional, std::variant & std::visit\n";
  std::cout << std::string(55, '=') << "\n";

  // --- Exercise 1 & 2: Parse with optional ---
  print_section("Exercise 1 & 2: Safe parsing with std::optional");
  auto entries = parse_all_logs(kRawLines);
  std::cout << "\n  Successfully parsed " << entries.size() << " out of "
            << kRawLines.size() << " lines:\n";
  for (const auto &e : entries) {
    std::cout << "    " << e << "\n";
  }

  // --- Exercise 3: Variant values ---
  print_section("Exercise 3: Format variant log values");

  std::vector<std::pair<std::string, LogValue>> fields = {
      {"request_count", 42},
      {"avg_latency", 3.14},
      {"user_agent", std::string("Mozilla/5.0")},
      {"status_code", 200},
      {"error_rate", 0.02},
  };

  for (const auto &[name, value] : fields) {
    std::cout << "  " << name << " = " << format_log_value(value) << "\n";
  }

  // --- Exercise 4: Severity payloads ---
  print_section("Exercise 4: Variant-based severity payloads");

  std::vector<SeverityPayload> payloads = {
      DebugInfo{"auth", 3},
      ErrorInfo{"E_CONN_REFUSED", "at connection_pool.cpp:42"},
      MetricInfo{"latency_ms", 42.5, "ms"},
      ErrorInfo{"E_TIMEOUT", "at http_client.cpp:128"},
      MetricInfo{"cpu_usage", 0.73, "%"},
  };

  for (const auto &p : payloads) {
    std::cout << "  " << describe_payload(p) << "\n";
  }

  // --- Bonus: check which type a variant holds ---
  print_section("Bonus: std::holds_alternative");
  for (const auto &p : payloads) {
    if (std::holds_alternative<ErrorInfo>(p)) {
      const auto &err = std::get<ErrorInfo>(p);
      std::cout << "  Found error: " << err.error_code << "\n";
    }
  }

  return 0;
}
