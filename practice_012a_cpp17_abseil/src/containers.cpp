// ============================================================================
// Phase 4: Abseil Containers & Hashing
// ============================================================================
//
// absl::flat_hash_map is a "Swiss table" -- Google's high-performance
// hash map that's 2-3x faster than std::unordered_map for most workloads.
//
// Why it's faster:
//   - Open addressing (no pointer chasing like std::unordered_map's chaining)
//   - Metadata bytes for fast probing (SSE2/NEON SIMD on supported platforms)
//   - Better cache locality (data stored inline, not in linked list nodes)
//
// In CP you'd use unordered_map or gp_hash_table. In production, you'd use
// absl::flat_hash_map -- same idea, better implementation.
//
// Docs:
//   https://abseil.io/docs/cpp/guides/container
//   https://abseil.io/docs/cpp/guides/hash
// ============================================================================

#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"

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

// ─── Helpers (fully implemented) ─────────────────────────────────────────────

void print_section(const std::string &title) {
  std::cout << "\n=== " << title << " ===\n\n";
}

// Generate N synthetic log entries for benchmarking
std::vector<LogEntry> generate_logs(int count) {
  static const std::vector<std::string> levels = {"DEBUG", "INFO", "WARN",
                                                  "ERROR"};
  static const std::vector<std::string> services = {
      "auth-service", "api-gateway",   "db-service",
      "payment-svc",  "cache-service", "user-service"};
  static const std::vector<std::string> messages = {
      "Request processed", "Connection refused", "Timeout",
      "Cache hit",         "Query executed",     "Auth failed",
      "Rate limited",      "Health check passed"};

  std::mt19937 rng(42); // deterministic seed
  std::vector<LogEntry> entries;
  entries.reserve(count);

  for (int i = 0; i < count; ++i) {
    entries.push_back({
        absl::StrCat("2024-03-15T10:", (10 + i / 3600) % 24, ":", (i / 60) % 60,
                     ":", i % 60),
        levels[rng() % levels.size()],
        services[rng() % services.size()],
        messages[rng() % messages.size()],
    });
  }
  return entries;
}

// Simple timer for benchmarks
class Timer {
public:
  Timer() : start_(std::chrono::high_resolution_clock::now()) {}

  double elapsed_ms() const {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(now - start_).count();
  }

  void reset() { start_ = std::chrono::high_resolution_clock::now(); }

private:
  std::chrono::high_resolution_clock::time_point start_;
};

// ─── Exercise 1: Store logs in absl::flat_hash_map ───────────────────────────
//
// Use a flat_hash_map keyed by a unique ID (e.g., index or timestamp)
// to store and look up LogEntry objects.

void exercise_basic_flat_hash_map(const std::vector<LogEntry> &logs) {
  // TODO(human): Implement these steps:
  //
  // 1. Create an absl::flat_hash_map<std::string, LogEntry> called `log_index`.
  //
  // 2. Insert all logs into the map, keyed by a unique ID.
  //    Use absl::StrCat("log_", i) as the key for the i-th entry.
  //    Use insert_or_assign or operator[] or try_emplace -- your choice.
  //
  // 3. Look up a specific entry:
  //    if (auto it = log_index.find("log_0"); it != log_index.end()) {
  //        std::cout << "  Found: " << it->second << "\n";
  //    }
  //
  // 4. Print the total number of entries in the map.
  //
  // Key insight: flat_hash_map has the SAME API as std::unordered_map.
  // The difference is purely in performance, not interface.

  absl::flat_hash_map<std::string, LogEntry> log_index;
  for (int i = 0; i < int(logs.size()); i++) {
    log_index[absl::StrCat("log_", i)] = logs[i];
  }
  if (auto it = log_index.find("log_0"); it != log_index.end()) {
    std::cout << "  Found: " << it->second << "\n";
  }
  std::cout << log_index.size() << "\n";
}

// ─── Exercise 2: Aggregate counts by level ───────────────────────────────────
//
// Count how many log entries exist for each severity level.
// Compare the same operation using std::unordered_map vs absl::flat_hash_map.

void exercise_aggregate_by_level(const std::vector<LogEntry> &logs) {
  // TODO(human): Implement aggregation using absl::flat_hash_map.
  //
  // 1. Create an absl::flat_hash_map<std::string, int> called `level_counts`.
  //
  // 2. For each log entry, increment the count for its level:
  //    level_counts[entry.level]++;
  //    (operator[] default-constructs the value (int -> 0) if not present)
  //
  // 3. Print results using structured bindings:
  //    for (const auto& [level, count] : level_counts) {
  //        std::cout << "  " << level << ": " << count << "\n";
  //    }

  absl::flat_hash_map<std::string, int> level_counts;
  for (const auto &log : logs) {
    level_counts[log.level]++;
  }
  for (const auto &[level, count] : level_counts) {
    std::cout << absl::StrFormat("   %-10s: %3d\n", level, count);
  }
}

// ─── Exercise 3: Custom hashing with absl::Hash ─────────────────────────────
//
// abseil's hash framework lets you make your own types hashable by
// implementing a friend function called AbslHashValue.
//
// This is cleaner than specializing std::hash<>.

struct ServiceKey {
  std::string name;
  std::string environment; // "production", "staging"

  bool operator==(const ServiceKey &other) const {
    return name == other.name && environment == other.environment;
  }

  // TODO(human): Implement AbslHashValue to make ServiceKey hashable.
  //
  // Pattern:
  //   template <typename H>
  //   friend H AbslHashValue(H h, const ServiceKey& key) {
  //       return H::combine(std::move(h), key.name, key.environment);
  //   }
  //
  // H::combine hashes multiple fields together. Abseil handles the
  // details (mixing, avalanche, etc.) -- you just list the fields.
  //
  // This is analogous to deriving Hash in Rust:
  //   #[derive(Hash)] struct ServiceKey { name: String, env: String }
  template <typename H> friend H AbslHashValue(H h, const ServiceKey &key) {
    return H::combine(std::move(h), key.name, key.environment);
  }
};

void exercise_custom_hash(const std::vector<LogEntry> &logs) {
  // TODO(human): After implementing AbslHashValue for ServiceKey:
  //
  // 1. Create an absl::flat_hash_map<ServiceKey, int> counting logs per
  //    (service, "production") pair. Use "production" as the environment
  //    for all entries (just to exercise the custom key).
  //
  // 2. Insert entries: map[ServiceKey{entry.service, "production"}]++;
  //
  // 3. Print results:
  //    for (const auto& [key, count] : map) {
  //        std::cout << "  " << key.name << "/" << key.environment
  //                  << ": " << count << "\n";
  //    }

  absl::flat_hash_map<ServiceKey, int> clogs;
  for (const auto &entry : logs)
    clogs[ServiceKey{entry.service, "production"}]++;
  for (const auto &[key, count] : clogs)
    printf("   %s / %s : %d \n", key.name.c_str(), key.environment.c_str(),
           count);
}

// ─── Exercise 4: Benchmark flat_hash_map vs unordered_map ────────────────────
//
// Insert N entries into both containers and measure time.
// This gives you intuition for the performance difference.

template <class HT, class K>
std::pair<double, double> benchmark(const std::vector<K> &v, HT &ht) {
  double insertion_time = 0, find_time = 0;
  {
    Timer t;
    for (size_t i = 0; i < v.size(); i++)
      ht[v[i]] = i;
    insertion_time = t.elapsed_ms();
  }
  {
    Timer t;
    for (size_t i = 0; i < v.size(); i++) {
      (void)ht.find(v[i]);
    }
    find_time = t.elapsed_ms();
  }
  return {insertion_time, find_time};
}

void exercise_benchmark(int n) {
  // TODO(human): Benchmark insertion and lookup for both containers.
  //
  // 1. Generate keys: std::vector<std::string> keys;
  //    for (int i = 0; i < n; i++) keys.push_back(absl::StrCat("key_", i));
  //
  // 2. Time insertion into std::unordered_map<std::string, int>:
  //    {
  //        Timer t;
  //        std::unordered_map<std::string, int> m;
  //        for (int i = 0; i < n; i++) m[keys[i]] = i;
  //        std::cout << "  unordered_map insert: " << t.elapsed_ms() << "ms\n";
  //    }
  //
  // 3. Time insertion into absl::flat_hash_map<std::string, int>:
  //    (same pattern, different container)
  //
  // 4. Time lookups: iterate all keys, call .find(key) on each.
  //    Measure both containers.
  //
  // 5. Print the speedup ratio.
  //
  // Expected: flat_hash_map should be noticeably faster, especially
  // for lookups (due to cache-friendly open addressing).
  std::vector<std::string> keys;
  for (int i = 0; i < n; i++)
    keys.push_back(absl::StrCat("key_", i));
  std::unordered_map<std::string, int> um;
  absl::flat_hash_map<std::string, int> fm;

  auto [umi, umf] = benchmark(keys, um);
  auto [fmi, fmf] = benchmark(keys, fm);

  printf("unordered_map: insert : %5.3f ms find: %5.3f ms\n", umi, umf);
  printf("flat_hash_map: insert : %5.3f ms find: %5.3f ms\n", fmi, fmf);
  printf("speed_up     : insert : %5.3f x  find: %5.3f x \n", umi / fmi,
         umf / fmf);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
  std::cout << "Phase 4: Abseil Containers & Hashing\n";
  std::cout << std::string(55, '=') << "\n";

  auto logs = generate_logs(1000);
  std::cout << "  Generated " << logs.size() << " synthetic log entries.\n";

  print_section("Exercise 1: Basic flat_hash_map");
  exercise_basic_flat_hash_map(logs);

  print_section("Exercise 2: Aggregate by level");
  exercise_aggregate_by_level(logs);

  print_section("Exercise 3: Custom hashing");
  exercise_custom_hash(logs);

  print_section("Exercise 4: Benchmark (20M entries)");
  exercise_benchmark(int(2e7));

  return 0;
}
