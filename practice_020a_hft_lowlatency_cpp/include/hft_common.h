#pragma once
// =============================================================================
// hft_common.h -- Shared utilities for HFT low-latency C++ practice
//
// This header provides platform-abstracted timing, alignment helpers, and
// benchmarking utilities used across all phases.
//
// HFT CONTEXT:
//   In a real trading system, you'd have a similar "platform.h" that abstracts
//   OS-specific APIs (RDTSC, huge pages, CPU affinity). The hot path never
//   calls into the OS directly -- everything goes through thin inlined wrappers.
// =============================================================================

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <string>

// =============================================================================
// Platform detection & intrinsics
// =============================================================================

#ifdef _WIN32
    #include <intrin.h>   // __rdtsc(), _mm_prefetch(), _mm_pause()
    #define HFT_WINDOWS 1
#else
    #include <x86intrin.h> // __rdtsc() on GCC/Clang
    #define HFT_WINDOWS 0
#endif

// =============================================================================
// Cache line size constant
// =============================================================================
//
// On virtually all x86/x64 CPUs (Intel since Pentium 4, AMD since K8):
//   - L1 cache line = 64 bytes
//   - L2/L3 cache line = 64 bytes
//
// ARM (Apple M1/M2) uses 128-byte cache lines, but we target x64 here.
// =============================================================================

inline constexpr std::size_t CACHE_LINE_SIZE = 64;

// =============================================================================
// Alignment helpers
// =============================================================================

// Compile-time check that a type fits exactly in N cache lines
#define HFT_ASSERT_CACHE_LINES(Type, N) \
    static_assert(sizeof(Type) == (N) * CACHE_LINE_SIZE, \
        #Type " must be exactly " #N " cache line(s) (" #N " * 64 bytes)")

// Compile-time check that a type is cache-line aligned
#define HFT_ASSERT_ALIGNED(Type) \
    static_assert(alignof(Type) >= CACHE_LINE_SIZE, \
        #Type " must be aligned to cache line boundary (64 bytes)")

// =============================================================================
// Prefetch hints
// =============================================================================
//
// _mm_prefetch fetches a cache line into L1/L2 before you need it.
// In HFT, you prefetch the *next* order/message while processing the current one.
//
// Locality hints:
//   _MM_HINT_T0  = prefetch into L1 (use for data you'll access very soon)
//   _MM_HINT_T1  = prefetch into L2 (use for data you'll access soon-ish)
//   _MM_HINT_T2  = prefetch into L3
//   _MM_HINT_NTA = non-temporal (streaming data you won't reuse)
// =============================================================================

inline void prefetch_l1(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T0);
}

inline void prefetch_l2(const void* addr) {
    _mm_prefetch(static_cast<const char*>(addr), _MM_HINT_T1);
}

// =============================================================================
// CPU pause (spin-wait hint)
// =============================================================================
//
// _mm_pause() tells the CPU you're in a spin-wait loop. It:
//   1. Reduces power consumption during spin
//   2. Avoids memory-order violations that cause pipeline flushes
//   3. On hyperthreaded cores, yields execution resources to the sibling thread
//
// Every spin-wait in HFT uses this. Without it, you waste ~100 cycles per
// iteration on pipeline flush penalties.
// =============================================================================

inline void cpu_pause() {
    _mm_pause();
}

// =============================================================================
// RDTSC -- Read Time Stamp Counter
// =============================================================================
//
// __rdtsc() reads the CPU's 64-bit cycle counter. On modern x64:
//   - Invariant TSC: runs at constant rate regardless of CPU frequency
//   - ~1 cycle resolution (sub-nanosecond at 3+ GHz)
//   - ~5-10 ns overhead (vs ~20ns for steady_clock, ~30ns for QPC)
//
// WARNING: On multi-socket systems, TSC may differ between sockets.
// HFT systems pin threads to specific cores (CPU affinity) to avoid this.
// =============================================================================

inline uint64_t rdtsc() {
    return __rdtsc();
}

// Serializing RDTSC -- forces all prior instructions to complete before reading.
// Use this for accurate START timestamps (prevents out-of-order execution from
// reading the counter before your measured code actually begins).
inline uint64_t rdtscp() {
    unsigned int aux;
    return __rdtscp(&aux);
}

// =============================================================================
// TSC frequency calibration
// =============================================================================

// Estimate TSC frequency by measuring cycles over a known time interval.
// Returns cycles per second. Call once at startup, cache the result.
inline uint64_t calibrate_tsc_frequency() {
    using Clock = std::chrono::steady_clock;

    // Warm up
    volatile uint64_t warmup = __rdtsc();
    (void)warmup;

    auto wall_start = Clock::now();
    uint64_t tsc_start = __rdtsc();

    // Spin for ~50ms to get a decent sample
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - wall_start).count() < 50) {
        cpu_pause();
    }

    uint64_t tsc_end = __rdtsc();
    auto wall_end = Clock::now();

    double wall_ns = static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            wall_end - wall_start).count());

    double cycles = static_cast<double>(tsc_end - tsc_start);
    double freq = cycles / (wall_ns / 1e9);

    return static_cast<uint64_t>(freq);
}

// Convert TSC cycles to nanoseconds given a known frequency
inline double tsc_to_ns(uint64_t cycles, uint64_t tsc_freq) {
    return static_cast<double>(cycles) * 1e9 / static_cast<double>(tsc_freq);
}

// =============================================================================
// Benchmark utilities
// =============================================================================

// Prevent compiler from optimizing away a value (DoNotOptimize)
// This is the standard benchmark escape pattern.
template <typename T>
inline void do_not_optimize(T const& value) {
    // MSVC: volatile read prevents optimization
    // GCC/Clang: asm volatile with memory clobber
#ifdef _MSC_VER
    volatile auto sink = value;
    (void)sink;
#else
    asm volatile("" : : "r,m"(value) : "memory");
#endif
}

// Prevent compiler from reordering reads/writes across this barrier
inline void compiler_fence() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

// =============================================================================
// Latency statistics
// =============================================================================

struct LatencyStats {
    double min_ns;
    double max_ns;
    double mean_ns;
    double median_ns;
    double p50_ns;
    double p90_ns;
    double p99_ns;
    double p999_ns;
    double stddev_ns;
    size_t count;
};

// Compute latency statistics from a vector of nanosecond measurements.
// The input vector is sorted in-place.
inline LatencyStats compute_latency_stats(std::vector<double>& latencies_ns) {
    LatencyStats stats{};
    if (latencies_ns.empty()) return stats;

    std::sort(latencies_ns.begin(), latencies_ns.end());

    stats.count = latencies_ns.size();
    stats.min_ns = latencies_ns.front();
    stats.max_ns = latencies_ns.back();

    double sum = std::accumulate(latencies_ns.begin(), latencies_ns.end(), 0.0);
    stats.mean_ns = sum / static_cast<double>(stats.count);

    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * static_cast<double>(stats.count - 1));
        return latencies_ns[idx];
    };

    stats.median_ns = percentile(0.50);
    stats.p50_ns = stats.median_ns;
    stats.p90_ns = percentile(0.90);
    stats.p99_ns = percentile(0.99);
    stats.p999_ns = percentile(0.999);

    double sq_sum = 0.0;
    for (double v : latencies_ns) {
        double diff = v - stats.mean_ns;
        sq_sum += diff * diff;
    }
    stats.stddev_ns = std::sqrt(sq_sum / static_cast<double>(stats.count));

    return stats;
}

// Pretty-print latency statistics
inline void print_latency_stats(const std::string& label, const LatencyStats& s) {
    std::cout << "\n=== " << label << " (" << s.count << " samples) ===\n";
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
// Simple scoped timer for quick benchmarks
// =============================================================================

class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << "[Timer] " << name_ << ": " << us << " us\n";
    }

    // Non-copyable
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};
