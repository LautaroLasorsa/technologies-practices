// ============================================================================
// Phase 5: Error Handling with absl::StatusOr
// ============================================================================
//
// absl::StatusOr<T> is C++'s answer to Rust's Result<T, E>.
// It holds EITHER a value of type T OR an absl::Status error.
//
// In CP you'd never think about error handling. In production code,
// "what happens when this fails?" is the FIRST question you ask.
//
// absl::Status uses canonical error codes (the same ones used by gRPC):
//   - absl::StatusCode::kNotFound
//   - absl::StatusCode::kInvalidArgument
//   - absl::StatusCode::kInternal
//   - etc.
//
// Convenience constructors:
//   absl::NotFoundError("user not found")
//   absl::InvalidArgumentError("port must be > 0")
//   absl::InternalError("unexpected null pointer")
//
// Docs:
//   https://abseil.io/docs/cpp/guides/status
//   https://abseil.io/docs/cpp/guides/status-codes
// ============================================================================

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

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

// ─── Helpers ─────────────────────────────────────────────────────────────────

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// Simulated file contents (since we don't want real filesystem dependencies)
const std::string kValidFileContents =
    R"(2024-03-15T10:30:00 ERROR auth-service Connection refused
2024-03-15T10:30:01 INFO  api-gateway Request processed
2024-03-15T10:30:02 WARN  db-service Slow query detected
2024-03-15T10:30:03 ERROR payment-svc Timeout waiting)";

const std::string kFileWithBadLines =
    R"(2024-03-15T10:30:00 ERROR auth-service Connection refused
BAD LINE
2024-03-15T10:30:02 WARN  db-service Slow query detected
ANOTHER BAD LINE
2024-03-15T10:30:04 INFO  api-gateway OK)";

// ─── Exercise 1: Return absl::StatusOr from a parser ─────────────────────────
//
// Write a function that parses a single log line.
// On success, return the LogEntry.
// On failure, return a descriptive absl::Status error.
//
// This is like Rust's: fn parse(line: &str) -> Result<LogEntry, Status>

absl::StatusOr<LogEntry> parse_log_line(absl::string_view line) {
  // TODO(human): Implement this function.
  //
  // Steps:
  //   1. If `line` is empty, return absl::InvalidArgumentError("empty line")
  //
  //   2. Split the line by spaces.
  //      std::vector<std::string> parts = absl::StrSplit(line, ' ');
  //
  //   3. If parts.size() < 4, return:
  //      absl::InvalidArgumentError(
  //          absl::StrCat("malformed log line, expected >= 4 fields, got ",
  //                       parts.size(), ": \"", line, "\""))
  //
  //   4. Construct the LogEntry from the parts.
  //      For the message, join parts[3..] with spaces (or just take the
  //      substring after the third space).
  //
  //   5. Return the LogEntry. (Implicit conversion to StatusOr<LogEntry>.)
  //
  // Key insight: returning a T from a StatusOr<T> function just works.
  // Returning an absl::Status error also just works.
  // The compiler won't let you accidentally ignore the error.

  return absl::InvalidArgumentError("not implemented");
}

// ─── Exercise 2: Load and parse a "file" (simulated) ─────────────────────────
//
// Chain StatusOr-returning functions: load contents, parse each line,
// collect results or propagate errors.

absl::StatusOr<std::string> load_file(absl::string_view filename) {
  // Simulated file loading -- pretend this reads from disk.
  if (filename == "valid.log") {
    return std::string(kValidFileContents);
  }
  if (filename == "mixed.log") {
    return std::string(kFileWithBadLines);
  }
  return absl::NotFoundError(
      absl::StrCat("file not found: \"", filename, "\""));
}

absl::StatusOr<std::vector<LogEntry>>
load_and_parse(absl::string_view filename) {
  // TODO(human): Implement this function by chaining StatusOr results.
  //
  // Pattern (early return on error -- like Rust's ? operator):
  //
  //   // Step 1: Load file contents
  //   absl::StatusOr<std::string> contents = load_file(filename);
  //   if (!contents.ok()) {
  //       return contents.status();  // propagate the error
  //   }
  //
  //   // Step 2: Split into lines and parse each
  //   std::vector<LogEntry> entries;
  //   for (absl::string_view line : absl::StrSplit(*contents, '\n')) {
  //       absl::StatusOr<LogEntry> entry = parse_log_line(line);
  //       if (!entry.ok()) {
  //           // Option A: skip bad lines (lenient)
  //           // Option B: return entry.status(); (strict)
  //           // Choose one and explain why!
  //       }
  //       entries.push_back(*entry);  // or std::move(*entry)
  //   }
  //
  //   return entries;
  //
  // Note: *contents dereferences the StatusOr to get the string.
  // This is safe because we checked .ok() first.

  return absl::InternalError("not implemented");
}

// ─── Exercise 3: Status with rich error information ──────────────────────────
//
// Demonstrate different error codes and how to inspect them.

absl::StatusOr<LogEntry> find_first_error(const std::vector<LogEntry> &entries,
                                          absl::string_view service) {
  // TODO(human): Find the first log entry with level "ERROR" for the
  //              given service. Return it, or an appropriate error.
  //
  // Cases:
  //   - If entries is empty:
  //     return absl::FailedPreconditionError("log database is empty")
  //
  //   - If no entries match the service:
  //     return absl::NotFoundError(absl::StrCat("no logs for service: ",
  //     service))
  //
  //   - If the service exists but has no ERROR entries:
  //     return absl::NotFoundError(absl::StrCat("no errors for service: ",
  //     service))
  //
  //   - Otherwise: return the first ERROR entry
  //
  // This exercises choosing the RIGHT error code for each situation.
  // In gRPC, these codes get transmitted over the wire -- choosing
  // correctly matters.

  return absl::InternalError("not implemented");
}

// ─── Exercise 4: Consuming StatusOr results ──────────────────────────────────
//
// Show all the ways to use a StatusOr result.

void demonstrate_statusor_usage(absl::StatusOr<LogEntry> result) {
  // TODO(human): Demonstrate ALL the ways to consume a StatusOr.
  //
  // Way 1: Check with ok() and dereference
  //   if (result.ok()) {
  //       std::cout << "  Value: " << *result << "\n";
  //   } else {
  //       std::cout << "  Error: " << result.status() << "\n";
  //   }
  //
  // Way 2: Check the status code
  //   if (result.status().code() == absl::StatusCode::kNotFound) {
  //       std::cout << "  Not found!\n";
  //   }
  //
  // Way 3: Use value_or (not in abseil, but show the pattern)
  //   You can write your own:
  //   LogEntry entry = result.ok() ? *std::move(result) : default_entry;
  //
  // Way 4: Print the status message
  //   std::cout << "  Status: " << result.status().message() << "\n";
  //
  // Implement all four and print the results.

}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 5: Error Handling with absl::StatusOr\n";
  std::cout << std::string(55, '=') << "\n";

  // --- Exercise 1: Parse individual lines ---
  print_section("Exercise 1: Parse with StatusOr");

  std::vector<std::string> test_lines = {
      "2024-03-15T10:30:00 ERROR auth-service Connection refused",
      "",
      "BAD LINE",
      "2024-03-15T10:30:01 INFO  api-gateway Request processed in 42ms",
  };

  for (const auto &line : test_lines) {
    auto result = parse_log_line(line);
    if (result.ok()) {
      std::cout << "  OK: " << *result << "\n";
    } else {
      std::cout << "  ERR [" << result.status().code()
                << "]: " << result.status().message() << "\n";
    }
  }

  // --- Exercise 2: Load and parse files ---
  print_section("Exercise 2: Chain StatusOr (load + parse)");

  for (absl::string_view file : {"valid.log", "missing.log", "mixed.log"}) {
    std::cout << "  Loading \"" << file << "\"...\n";
    auto result = load_and_parse(file);
    if (result.ok()) {
      std::cout << "    Parsed " << result->size() << " entries.\n";
      for (const auto &e : *result) {
        std::cout << "      " << e << "\n";
      }
    } else {
      std::cout << "    Error: " << result.status() << "\n";
    }
    std::cout << "\n";
  }

  // --- Exercise 3: Find first error ---
  print_section("Exercise 3: Rich error codes");

  // First, parse the valid file to get entries
  auto parsed = load_and_parse("valid.log");
  if (parsed.ok()) {
    auto &entries = *parsed;

    // Should find an error
    auto r1 = find_first_error(entries, "auth-service");
    demonstrate_statusor_usage(std::move(r1));

    // Should return NotFound (no errors for this service)
    auto r2 = find_first_error(entries, "api-gateway");
    demonstrate_statusor_usage(std::move(r2));

    // Should return NotFound (service doesn't exist)
    auto r3 = find_first_error(entries, "nonexistent-svc");
    demonstrate_statusor_usage(std::move(r3));

    // Empty entries case
    std::vector<LogEntry> empty;
    auto r4 = find_first_error(empty, "any");
    demonstrate_statusor_usage(std::move(r4));
  }

  return 0;
}
