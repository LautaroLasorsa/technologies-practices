#pragma once
// =============================================================================
// bench_utils.h -- Shared benchmarking utilities for Practice 022
//
// Provides:
//   - Timer: high-resolution wall-clock timing
//   - ThroughputResult: ops/sec + latency calculation
//   - ComparisonTable: side-by-side benchmark result printer
//   - cpu_usage_hint(): rough CPU usage indicator
//
// CONCURRENCY CONTEXT:
//   When benchmarking concurrent data structures, timing must account for:
//   1. Thread startup overhead (~1-10us per std::thread construction)
//   2. Cache warm-up (first N operations are slower -- cold cache)
//   3. OS scheduler interference (pin to cores in production; here we accept jitter)
//   4. Compiler optimization (use do_not_optimize to prevent dead code elimination)
//
// RUST EQUIVALENT:
//   criterion::Bencher for benchmarks, std::time::Instant for manual timing.
//   C++ doesn't have a built-in benchmark framework, so we roll our own.
// =============================================================================

#include <chrono>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// =============================================================================
// Prevent compiler from optimizing away benchmark results
// =============================================================================
//
// Without this, the compiler may see that you enqueue values but never use
// the dequeued result, and optimize away the entire benchmark loop.
// This is the standard "escape" pattern from Google Benchmark.
// =============================================================================

template <typename T>
inline void do_not_optimize(T const& value) {
#ifdef _MSC_VER
    // MSVC: volatile read prevents dead code elimination
    volatile auto sink = value;
    (void)sink;
#else
    // GCC/Clang: inline asm with memory clobber
    asm volatile("" : : "r,m"(value) : "memory");
#endif
}

// =============================================================================
// Timer -- steady_clock wrapper for benchmark timing
// =============================================================================
//
// Uses std::chrono::steady_clock (~15-25ns overhead per call).
// For concurrent queue benchmarks, the overhead is negligible compared to
// the operation latency (~10-200ns per enqueue/dequeue).
//
// For sub-nanosecond timing (HFT hot path), use __rdtsc() instead (see 020a).
// =============================================================================

class Timer {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    void start() { start_ = Clock::now(); }
    void stop()  { end_ = Clock::now(); }

    // Elapsed time in various units
    double elapsed_ns() const {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(end_ - start_).count());
    }

    double elapsed_us() const { return elapsed_ns() / 1'000.0; }
    double elapsed_ms() const { return elapsed_ns() / 1'000'000.0; }
    double elapsed_sec() const { return elapsed_ns() / 1'000'000'000.0; }

private:
    TimePoint start_{};
    TimePoint end_{};
};

// =============================================================================
// ThroughputResult -- computed from total ops and elapsed time
// =============================================================================

struct ThroughputResult {
    std::string label;
    size_t total_ops;
    double elapsed_ms;
    double ops_per_sec;      // total throughput
    double ns_per_op;        // average latency per operation
    double m_ops_per_sec;    // millions of ops per second (easier to read)
};

inline ThroughputResult compute_throughput(
    const std::string& label,
    size_t total_ops,
    double elapsed_ms)
{
    ThroughputResult r;
    r.label = label;
    r.total_ops = total_ops;
    r.elapsed_ms = elapsed_ms;
    r.ops_per_sec = (elapsed_ms > 0.0)
        ? static_cast<double>(total_ops) / (elapsed_ms / 1000.0)
        : 0.0;
    r.ns_per_op = (total_ops > 0)
        ? (elapsed_ms * 1'000'000.0) / static_cast<double>(total_ops)
        : 0.0;
    r.m_ops_per_sec = r.ops_per_sec / 1'000'000.0;
    return r;
}

// =============================================================================
// Print a single throughput result
// =============================================================================

inline void print_throughput(const ThroughputResult& r) {
    std::cout << std::fixed;
    std::cout << "  " << std::left << std::setw(40) << r.label
              << std::right
              << std::setw(10) << std::setprecision(2) << r.m_ops_per_sec << " M ops/s"
              << std::setw(12) << std::setprecision(1) << r.ns_per_op << " ns/op"
              << std::setw(10) << std::setprecision(1) << r.elapsed_ms << " ms"
              << "\n";
}

// =============================================================================
// ComparisonTable -- collect and display benchmark results side-by-side
// =============================================================================
//
// Usage:
//   ComparisonTable table("Queue Throughput Comparison");
//   table.add(compute_throughput("std::queue + mutex", ops, ms));
//   table.add(compute_throughput("moodycamel (no tokens)", ops, ms));
//   table.print();
//
// Output:
//   === Queue Throughput Comparison ===
//   ---------------------------------------- ---------- ------------ ----------
//   std::queue + mutex                          12.34 M ops/s    81.0 ns/op   810.0 ms
//   moodycamel (no tokens)                      89.12 M ops/s    11.2 ns/op   112.0 ms
//   ---------------------------------------- ---------- ------------ ----------
//   Speedup vs baseline (std::queue + mutex): 7.2x
// =============================================================================

class ComparisonTable {
public:
    explicit ComparisonTable(const std::string& title) : title_(title) {}

    void add(const ThroughputResult& r) { results_.push_back(r); }

    void print() const {
        std::cout << "\n=== " << title_ << " ===\n";
        std::cout << std::string(82, '-') << "\n";

        for (const auto& r : results_) {
            print_throughput(r);
        }

        std::cout << std::string(82, '-') << "\n";

        // Show speedup relative to first entry (baseline)
        if (results_.size() >= 2 && results_[0].ops_per_sec > 0.0) {
            for (size_t i = 1; i < results_.size(); ++i) {
                double speedup = results_[i].ops_per_sec / results_[0].ops_per_sec;
                std::cout << "  " << results_[i].label
                          << " vs " << results_[0].label
                          << ": " << std::fixed << std::setprecision(1)
                          << speedup << "x\n";
            }
        }
        std::cout << "\n";
    }

private:
    std::string title_;
    std::vector<ThroughputResult> results_;
};

// =============================================================================
// Latency statistics (for per-operation latency measurements)
// =============================================================================
//
// Collect individual latency samples and compute percentiles.
// Useful for Phase 5 (blocking vs spinning CPU comparison).
// =============================================================================

struct LatencyStats {
    double min_ns   = 0.0;
    double max_ns   = 0.0;
    double mean_ns  = 0.0;
    double p50_ns   = 0.0;
    double p90_ns   = 0.0;
    double p99_ns   = 0.0;
    double p999_ns  = 0.0;
    double stddev_ns = 0.0;
    size_t count    = 0;
};

inline LatencyStats compute_latency_stats(std::vector<double>& samples_ns) {
    LatencyStats s;
    if (samples_ns.empty()) return s;

    std::sort(samples_ns.begin(), samples_ns.end());
    s.count = samples_ns.size();
    s.min_ns = samples_ns.front();
    s.max_ns = samples_ns.back();

    double sum = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0);
    s.mean_ns = sum / static_cast<double>(s.count);

    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * static_cast<double>(s.count - 1));
        return samples_ns[idx];
    };

    s.p50_ns  = pct(0.50);
    s.p90_ns  = pct(0.90);
    s.p99_ns  = pct(0.99);
    s.p999_ns = pct(0.999);

    double sq_sum = 0.0;
    for (double v : samples_ns) {
        double d = v - s.mean_ns;
        sq_sum += d * d;
    }
    s.stddev_ns = std::sqrt(sq_sum / static_cast<double>(s.count));

    return s;
}

inline void print_latency_stats(const std::string& label, const LatencyStats& s) {
    std::cout << "\n--- " << label << " (" << s.count << " samples) ---\n";
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Min:    " << std::setw(10) << s.min_ns    << " ns\n";
    std::cout << "  P50:    " << std::setw(10) << s.p50_ns    << " ns\n";
    std::cout << "  P90:    " << std::setw(10) << s.p90_ns    << " ns\n";
    std::cout << "  P99:    " << std::setw(10) << s.p99_ns    << " ns\n";
    std::cout << "  P99.9:  " << std::setw(10) << s.p999_ns   << " ns\n";
    std::cout << "  Max:    " << std::setw(10) << s.max_ns    << " ns\n";
    std::cout << "  Mean:   " << std::setw(10) << s.mean_ns   << " ns\n";
    std::cout << "  Stddev: " << std::setw(10) << s.stddev_ns << " ns\n";
}

// =============================================================================
// Separator / section header helpers
// =============================================================================

inline void print_header(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

inline void print_subheader(const std::string& title) {
    std::cout << "\n--- " << title << " ---\n";
}
