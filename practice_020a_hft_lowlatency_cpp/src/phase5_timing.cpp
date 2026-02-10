// =============================================================================
// Phase 5: Timestamps & Low-Latency Timing
// =============================================================================
//
// HFT CONTEXT:
//   In HFT, timing is everything -- literally. You need to:
//   1. Timestamp every market data message with ~nanosecond precision
//   2. Measure your own processing latency to detect regressions
//   3. Produce latency reports (P50, P99, P99.9) for compliance and monitoring
//
//   The problem: standard timing APIs are TOO SLOW for what you're measuring.
//
//   | Method                      | Overhead | Resolution  |
//   |-----------------------------|----------|-------------|
//   | __rdtsc()                   | ~5-10 ns | ~0.3 ns     |
//   | QueryPerformanceCounter     | ~15-30 ns| ~100 ns     |
//   | std::chrono::steady_clock   | ~15-25 ns| ~100 ns     |
//   | std::chrono::system_clock   | ~20-30 ns| ~1 us       |
//   | GetSystemTimePreciseAsFileT | ~20-30 ns| ~100 ns     |
//
//   When your entire processing pipeline takes 200ns, spending 25ns JUST TO
//   CHECK THE CLOCK is a 12.5% overhead. That's unacceptable.
//
//   RDTSC reads the CPU's cycle counter directly -- it's an instruction, not
//   a syscall. ~5-10ns overhead, sub-nanosecond resolution. Every HFT system
//   uses it for hot-path timing.
//
// CAVEATS:
//   - TSC frequency varies by CPU model (need calibration at startup)
//   - On multi-socket systems, TSCs may not be synchronized between sockets
//     (solution: pin threads to one socket, or use rdtscp which reads the CPU ID)
//   - Invariant TSC (constant rate) is standard on all modern x64 CPUs
//     Check: CPUID leaf 0x80000007, bit 8 (TscInvariant)
//
// RUST EQUIVALENT:
//   - std::time::Instant (wraps clock_gettime, similar to steady_clock)
//   - quanta crate (Rust's rdtsc wrapper, with calibration)
//   - minstant crate (lightweight, uses TSC on supported platforms)
// =============================================================================

#include "hft_common.h"
#include <thread>

#ifdef _WIN32
    #include <windows.h>  // QueryPerformanceCounter, QueryPerformanceFrequency
#endif

// =============================================================================
// Exercise 1: TSC Reader
// =============================================================================
//
// GOAL: Create a lightweight timestamp reader using __rdtsc().
//
// The rdtsc() function is already in hft_common.h. Here we wrap it in a
// more ergonomic class that handles calibration and conversion.
//
// TODO(human): Implement the TscClock class.
//
// DESIGN:
//   - Calibrate TSC frequency at construction time
//   - now() returns raw TSC cycles (fastest, use for relative measurements)
//   - elapsed_ns(start, end) converts cycle difference to nanoseconds
//   - to_ns(cycles) converts absolute cycle count to nanoseconds since calibration
//
// WHY RAW CYCLES ARE PREFERRED:
//   In the hot path, you store raw TSC values. Converting to nanoseconds
//   involves a multiply and divide -- do that OFFLINE when analyzing logs,
//   not on the hot path. Raw cycle storage = one uint64_t read, zero math.
// =============================================================================

class TscClock {
public:
    TscClock() {
        // TODO(human): Calibrate TSC frequency.
        // Use calibrate_tsc_frequency() from hft_common.h.
        // Store the result and also compute ns_per_cycle for conversions.
        //
        // tsc_freq_ = calibrate_tsc_frequency();
        // ns_per_cycle_ = 1e9 / static_cast<double>(tsc_freq_);
        // base_tsc_ = rdtsc();  // reference point for absolute timestamps

        tsc_freq_ = 1;       // placeholder
        ns_per_cycle_ = 1.0; // placeholder
        base_tsc_ = 0;       // placeholder

        // After implementation, uncomment:
        // std::cout << "TSC frequency: " << tsc_freq_ / 1'000'000 << " MHz\n";
        // std::cout << "ns per cycle: " << std::fixed << std::setprecision(4)
        //           << ns_per_cycle_ << "\n";
    }

    // Read current TSC value (raw cycles, fastest possible timestamp)
    uint64_t now() const {
        return rdtsc();
    }

    // Read current TSC value with serialization (for accurate START timestamps)
    uint64_t now_serialized() const {
        return rdtscp();
    }

    // Convert cycle count difference to nanoseconds
    double elapsed_ns(uint64_t start_cycles, uint64_t end_cycles) const {
        // TODO(human): Implement.
        // return static_cast<double>(end_cycles - start_cycles) * ns_per_cycle_;
        (void)start_cycles;
        (void)end_cycles;
        return 0.0;  // placeholder
    }

    // Convert cycles to nanoseconds since construction
    double to_ns_since_start(uint64_t tsc_value) const {
        return static_cast<double>(tsc_value - base_tsc_) * ns_per_cycle_;
    }

    uint64_t frequency() const { return tsc_freq_; }
    double ns_per_cycle() const { return ns_per_cycle_; }

private:
    uint64_t tsc_freq_;
    double ns_per_cycle_;
    uint64_t base_tsc_;
};

// =============================================================================
// Exercise 2: TSC Calibration Verification
// =============================================================================
//
// GOAL: Verify that your TSC calibration is accurate by comparing against
//       a known-duration sleep.
//
// TEST:
//   1. Record TSC start
//   2. Sleep for exactly 100ms (using std::this_thread::sleep_for)
//   3. Record TSC end
//   4. Convert (end - start) to milliseconds using your calibration
//   5. Check that the result is within ~1% of 100ms
//
// NOTE: sleep_for is not precise (OS scheduler jitter), so allow +-5ms tolerance.
// =============================================================================

void exercise2_calibration_test() {
    std::cout << "\n=== Exercise 2: TSC Calibration Verification ===\n\n";

    // TODO(human): Implement calibration verification.
    //
    // TscClock clock;
    // auto tsc_start = clock.now();
    // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    // auto tsc_end = clock.now();
    //
    // double measured_ms = clock.elapsed_ns(tsc_start, tsc_end) / 1e6;
    // std::cout << "Expected: 100.0 ms\n";
    // std::cout << "Measured: " << std::fixed << std::setprecision(2) << measured_ms << " ms\n";
    // std::cout << "Error:    " << std::abs(measured_ms - 100.0) << " ms\n";
    //
    // bool ok = std::abs(measured_ms - 100.0) < 5.0;  // 5ms tolerance
    // std::cout << "Calibration: " << (ok ? "PASS" : "FAIL") << "\n";

    std::cout << "[Placeholder] Implement TSC calibration verification.\n";
}

// =============================================================================
// Exercise 3: Timing Method Comparison
// =============================================================================
//
// GOAL: Measure the overhead of different timing methods to justify using RDTSC.
//
// FOR EACH METHOD:
//   1. Call the timing function 1M times in a tight loop
//   2. Measure total time with a reference clock
//   3. Compute per-call overhead
//
// METHODS TO COMPARE:
//   a) __rdtsc()
//   b) std::chrono::steady_clock::now()
//   c) QueryPerformanceCounter (Windows only)
//
// EXPECTED:
//   __rdtsc():               ~5-10 ns per call
//   steady_clock::now():     ~15-25 ns per call
//   QueryPerformanceCounter: ~15-30 ns per call
// =============================================================================

void exercise3_timing_comparison() {
    std::cout << "\n=== Exercise 3: Timing Method Comparison ===\n\n";

    constexpr size_t ITERS = 1'000'000;

    // --- __rdtsc() overhead ---
    {
        // TODO(human): Measure rdtsc overhead.
        //
        // uint64_t sum = 0;
        // auto wall_start = std::chrono::steady_clock::now();
        // for (size_t i = 0; i < ITERS; ++i) {
        //     sum += __rdtsc();
        // }
        // auto wall_end = std::chrono::steady_clock::now();
        // do_not_optimize(sum);
        //
        // double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        //     wall_end - wall_start).count();
        // std::cout << "__rdtsc: " << std::fixed << std::setprecision(1)
        //           << (total_ns / ITERS) << " ns/call\n";

        std::cout << "__rdtsc:  [Placeholder]\n";
    }

    // --- steady_clock overhead ---
    {
        // TODO(human): Measure steady_clock::now() overhead.
        //
        // volatile int64_t sum = 0;
        // auto wall_start = std::chrono::steady_clock::now();
        // for (size_t i = 0; i < ITERS; ++i) {
        //     auto t = std::chrono::steady_clock::now();
        //     sum += t.time_since_epoch().count();
        // }
        // auto wall_end = std::chrono::steady_clock::now();
        //
        // double total_ns = ...;
        // std::cout << "steady_clock: " << (total_ns / ITERS) << " ns/call\n";

        std::cout << "steady_clock: [Placeholder]\n";
    }

    // --- QueryPerformanceCounter overhead (Windows) ---
#ifdef _WIN32
    {
        // TODO(human): Measure QPC overhead.
        //
        // LARGE_INTEGER freq, counter;
        // QueryPerformanceFrequency(&freq);
        // volatile int64_t sum = 0;
        //
        // auto wall_start = std::chrono::steady_clock::now();
        // for (size_t i = 0; i < ITERS; ++i) {
        //     QueryPerformanceCounter(&counter);
        //     sum += counter.QuadPart;
        // }
        // auto wall_end = std::chrono::steady_clock::now();
        //
        // double total_ns = ...;
        // std::cout << "QPC: " << (total_ns / ITERS) << " ns/call\n";

        std::cout << "QPC:          [Placeholder]\n";
    }
#endif

    std::cout << "\nExpected: __rdtsc ~5-10ns, steady_clock ~15-25ns, QPC ~15-30ns\n";
}

// =============================================================================
// Exercise 4: Latency Histogram
// =============================================================================
//
// GOAL: Implement a fixed-bucket histogram for sub-microsecond latency
//       measurement. This is how HFT firms monitor their system in production.
//
// DESIGN:
//   - Bucket width: configurable (e.g., 10 ns per bucket)
//   - Number of buckets: configurable (e.g., 200 buckets = 0-2000 ns range)
//   - Overflow bucket: anything above max goes in the last bucket
//   - Thread-local (no synchronization needed)
//
// WHY A HISTOGRAM (not raw samples):
//   At 10M messages/sec, storing every latency sample = 80 MB/sec (8 bytes each).
//   A histogram with 200 buckets = 1.6 KB total, forever. And you can compute
//   P50/P99/P99.9 from the histogram without storing individual samples.
//
// USAGE PATTERN:
//   LatencyHistogram hist(10, 200);  // 10ns buckets, 200 buckets (0-2000ns)
//   for each message:
//       auto start = rdtsc();
//       process(msg);
//       auto end = rdtsc();
//       hist.record(tsc_to_ns(end - start, freq));
//   hist.print();
// =============================================================================

class LatencyHistogram {
public:
    // bucket_width_ns: width of each bucket in nanoseconds
    // num_buckets: number of buckets (last bucket = overflow)
    LatencyHistogram(double bucket_width_ns, size_t num_buckets)
        : bucket_width_(bucket_width_ns)
        , buckets_(num_buckets, 0)
        , total_count_(0)
        , overflow_count_(0)
    {}

    // -------------------------------------------------------------------------
    // record: Add a latency sample to the histogram.
    //
    // TODO(human): Implement.
    //   1. Compute bucket index: idx = static_cast<size_t>(latency_ns / bucket_width_)
    //   2. If idx >= buckets_.size(), increment overflow_count_
    //   3. Else, increment buckets_[idx]
    //   4. Increment total_count_
    //
    // PERFORMANCE: This is O(1) -- just an integer divide and array increment.
    //   No allocation, no branching (except the overflow check).
    // -------------------------------------------------------------------------
    void record(double latency_ns) {
        // --- TODO(human) ---
        (void)latency_ns;  // placeholder
    }

    // -------------------------------------------------------------------------
    // percentile: Compute the p-th percentile from the histogram.
    //
    // TODO(human): Implement.
    //   1. target = total_count_ * p (e.g., p=0.99 for P99)
    //   2. Walk buckets from 0 to N, accumulating counts
    //   3. When accumulated count >= target, return bucket_center = (idx + 0.5) * bucket_width_
    //
    // NOTE: This is an approximation -- the true value is somewhere in the bucket range.
    //       With 10ns buckets, your error is at most 10ns. Good enough for HFT monitoring.
    // -------------------------------------------------------------------------
    double percentile(double p) const {
        // --- TODO(human) ---
        (void)p;
        return 0.0;  // placeholder
    }

    void print(const std::string& label = "Latency Histogram") const {
        std::cout << "\n=== " << label << " ===\n";
        std::cout << "Total samples: " << total_count_
                  << ", Overflow: " << overflow_count_ << "\n";
        std::cout << "Bucket width: " << bucket_width_ << " ns\n\n";

        // Find max count for scaling the bar chart
        uint64_t max_count = *std::max_element(buckets_.begin(), buckets_.end());
        if (max_count == 0) {
            std::cout << "(no data)\n";
            return;
        }

        // Print non-empty buckets as a horizontal bar chart
        constexpr int BAR_WIDTH = 50;
        for (size_t i = 0; i < buckets_.size(); ++i) {
            if (buckets_[i] == 0) continue;

            double lo = i * bucket_width_;
            double hi = (i + 1) * bucket_width_;
            int bar_len = static_cast<int>(
                static_cast<double>(buckets_[i]) / static_cast<double>(max_count) * BAR_WIDTH);

            std::cout << std::fixed << std::setprecision(0);
            std::cout << std::setw(6) << lo << "-" << std::setw(6) << hi << " ns |";
            for (int j = 0; j < bar_len; ++j) std::cout << '#';
            std::cout << " " << buckets_[i] << "\n";
        }

        // Print percentiles
        // TODO(human): Uncomment after implementing percentile():
        // std::cout << "\nPercentiles:\n";
        // std::cout << "  P50:   " << std::fixed << std::setprecision(1) << percentile(0.50) << " ns\n";
        // std::cout << "  P90:   " << percentile(0.90) << " ns\n";
        // std::cout << "  P99:   " << percentile(0.99) << " ns\n";
        // std::cout << "  P99.9: " << percentile(0.999) << " ns\n";
    }

private:
    double bucket_width_;
    std::vector<uint64_t> buckets_;
    uint64_t total_count_;
    uint64_t overflow_count_;
};

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 5: Timestamps & Low-Latency Timing\n";
    std::cout << "========================================\n";

    // Exercise 1: Basic TSC reading
    {
        std::cout << "\n=== Exercise 1: TSC Reader ===\n\n";
        TscClock clock;
        std::cout << "TSC frequency: " << clock.frequency() << " Hz\n";
        std::cout << "ns per cycle:  " << std::fixed << std::setprecision(4)
                  << clock.ns_per_cycle() << "\n";

        // Quick test: measure a no-op
        auto start = clock.now();
        do_not_optimize(42);
        auto end = clock.now();
        std::cout << "No-op measurement: " << (end - start) << " cycles ("
                  << clock.elapsed_ns(start, end) << " ns)\n";

        // NOTE: frequency will show 1 Hz until you implement calibration.
        // After implementation, expect ~3-5 GHz depending on your CPU.
    }

    exercise2_calibration_test();
    exercise3_timing_comparison();

    // Exercise 4: Histogram quick demo
    {
        std::cout << "\n=== Exercise 4: Latency Histogram Demo ===\n\n";
        LatencyHistogram hist(10.0, 200);  // 10ns buckets, 0-2000ns range

        // Generate some fake latency data for demonstration
        // TODO(human): Replace with real measurements after implementing record()
        // and percentile(). For now, this just shows the histogram structure.
        //
        // Example with real data:
        //   TscClock clock;
        //   for (int i = 0; i < 100000; ++i) {
        //       auto start = clock.now();
        //       // ... some operation ...
        //       auto end = clock.now();
        //       hist.record(clock.elapsed_ns(start, end));
        //   }

        std::cout << "[Placeholder] Implement record() and percentile(), then\n";
        std::cout << "feed real timing data into the histogram.\n";
        hist.print("Demo Histogram (empty)");
    }

    std::cout << "\n========================================\n";
    std::cout << "Phase 5 complete.\n";
    std::cout << "========================================\n";

    return 0;
}
