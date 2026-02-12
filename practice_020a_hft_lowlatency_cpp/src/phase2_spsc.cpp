// =============================================================================
// Phase 2: Lock-Free SPSC Ring Buffer
// =============================================================================
//
// HFT CONTEXT:
//   The Single-Producer Single-Consumer (SPSC) queue is THE fundamental data
//   structure in HFT systems. Every trading system has this topology:
//
//     [Network thread] --SPSC--> [Strategy thread] --SPSC--> [Order thread]
//
//   Each thread owns one queue for input and one for output. No mutexes, no
//   locks, no contention. Just atomic load/store with acquire/release ordering.
//
//   A well-tuned SPSC queue can transfer messages in ~5-15ns per message.
//   A mutex-based queue: ~50-200ns per message. That's 10-40x slower.
//
// KEY CONCEPTS:
//   - Ring buffer: fixed-size circular array, no allocation
//   - Power-of-2 size: use bitwise AND for modulo (faster than %)
//   - Memory ordering: acquire on load, release on store
//   - False sharing: head and tail on separate cache lines
//
// REFERENCES:
//   - Preshing: https://preshing.com/20120612/an-introduction-to-lock-free-programming/
//   - Dmitry Vyukov's SPSC: https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
//   - LMAX Disruptor (Java, but same concepts): https://lmax-exchange.github.io/disruptor/
//
// RUST EQUIVALENT: crossbeam::channel::bounded(N) or rtrb::RingBuffer
// =============================================================================

#include "hft_common.h"
#include <atomic>
#include <thread>
#include <optional>
#include <cassert>
#include <new>

// =============================================================================
// Exercise 1: Basic SPSC Ring Buffer
// =============================================================================
//
// GOAL: Implement a lock-free SPSC queue using a ring buffer with atomic
//       head/tail indices.
//
// DESIGN:
//   - Fixed capacity (power of 2) set at compile time
//   - Producer writes to buffer[tail % capacity], then increments tail
//   - Consumer reads from buffer[head % capacity], then increments head
//   - Full when: tail - head == capacity
//   - Empty when: tail == head
//
// MEMORY ORDERING (the tricky part):
//   Producer (try_push):
//     1. Load head with memory_order_acquire  (see consumer's latest read position)
//     2. Check if full (tail - head >= capacity)
//     3. Write data to buffer[tail & mask]
//     4. Store tail+1 with memory_order_release (publish the write to consumer)
//
//   Consumer (try_pop):
//     1. Load tail with memory_order_acquire  (see producer's latest write position)
//     2. Check if empty (head == tail)
//     3. Read data from buffer[head & mask]
//     4. Store head+1 with memory_order_release (publish the read to producer)
//
//   WHY acquire/release is sufficient:
//     - Release on store guarantees: all writes BEFORE the store are visible
//       to any thread that does an acquire-load of the same variable.
//     - The buffer write (step 3 of push) happens BEFORE the tail store (step 4).
//     - The consumer's acquire-load of tail (step 1 of pop) sees the tail AND
//       all preceding writes (including the buffer data). This is the "happens-before" guarantee.
//     - seq_cst would add unnecessary full memory barriers (~5-10ns overhead).
//
// TEMPLATE PARAMETER:
//   Capacity must be a power of 2 so we can use (index & (Capacity - 1))
//   instead of (index % Capacity). Bitwise AND is 1 cycle, modulo is ~20 cycles.
// =============================================================================

template <typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(Capacity >= 2, "Capacity must be at least 2");

    static constexpr size_t MASK = Capacity - 1;

public:
    SPSCQueue() : head_(0), tail_(0) {}

    // Non-copyable, non-movable (contains atomics)
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    // -------------------------------------------------------------------------
    // try_push: Attempt to enqueue an element. Returns false if full.
    // Called ONLY by the producer thread.
    // -------------------------------------------------------------------------
    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches lock-free push using acquire/release memory ordering.
    // Understanding when relaxed vs acquire/release suffices is critical for writing
    // correct concurrent code without unnecessary synchronization overhead.
    //
    // TODO(human): Implement try_push.
    //
    // STEPS:
    //   1. Load tail_ (relaxed -- only producer writes tail_)
    //   2. Load head_ with memory_order_acquire
    //   3. If (tail - head >= Capacity) return false (full)
    //   4. Construct element: buffer_[tail & MASK] = std::move(value)
    //   5. Store tail_+1 with memory_order_release
    //   6. Return true
    //
    // HINT on step 1: The producer is the ONLY writer of tail_. So the
    //   producer can load its own tail_ with relaxed ordering -- it already
    //   knows the latest value because it wrote it.
    bool try_push(const T& value) {
        // --- TODO(human) ---
        (void)value;
        return false;  // placeholder
    }

    // Move version for efficiency
    bool try_push(T&& value) {
        // --- TODO(human) ---
        (void)value;
        return false;  // placeholder
    }

    // -------------------------------------------------------------------------
    // try_pop: Attempt to dequeue an element. Returns std::nullopt if empty.
    // Called ONLY by the consumer thread.
    // -------------------------------------------------------------------------
    // TODO(human): Implement try_pop.
    //
    // STEPS:
    //   1. Load head_ (relaxed -- only consumer writes head_)
    //   2. Load tail_ with memory_order_acquire
    //   3. If (head == tail) return std::nullopt (empty)
    //   4. Read element: T result = std::move(buffer_[head & MASK])
    //   5. Store head_+1 with memory_order_release
    //   6. Return result
    //
    // SYMMETRY: This is the mirror of try_push. Consumer owns head_,
    //   loads it relaxed, and acquire-loads the producer's tail_.
    std::optional<T> try_pop() {
        // --- TODO(human) ---
        return std::nullopt;  // placeholder
    }

    // Utility: approximate size (not exact under concurrency, for debugging only)
    size_t size_approx() const {
        auto t = tail_.load(std::memory_order_relaxed);
        auto h = head_.load(std::memory_order_relaxed);
        return static_cast<size_t>(t - h);
    }

    bool empty_approx() const { return size_approx() == 0; }
    static constexpr size_t capacity() { return Capacity; }

private:
    // Buffer storage -- array of T, aligned to cache line
    alignas(CACHE_LINE_SIZE) T buffer_[Capacity];

    // Producer-owned index (only producer writes, consumer reads)
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail_;

    // Consumer-owned index (only consumer writes, producer reads)
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head_;
};

// =============================================================================
// Exercise 2: Demonstrate False Sharing Problem
// =============================================================================
//
// GOAL: Show the performance impact of false sharing when head_ and tail_
//       are on the same cache line vs separate cache lines.
//
// FALSE SHARING:
//   When two threads write to different variables that happen to be on the
//   same cache line, the CPU's cache coherency protocol (MESI) bounces the
//   cache line between cores on every write. This costs ~40-100 cycles per
//   "false" sharing event.
//
//   In our SPSCQueue above, we use alignas(CACHE_LINE_SIZE) on tail_ and head_
//   to force them onto separate cache lines. This is critical.
//
// THE BAD VERSION (for comparison):
//   struct BadQueue {
//       std::atomic<uint64_t> head_;  // same cache line!
//       std::atomic<uint64_t> tail_;  // same cache line!
//   };
//
// BENCHMARK PLAN:
//   1. Our good queue (padded): run producer+consumer threads, measure throughput
//   2. A bad queue (unpadded): same test, measure throughput
//   3. Compare: padded should be 2-5x faster
// =============================================================================

// Bad version for comparison -- head and tail on the same cache line
template <typename T, size_t Capacity>
class SPSCQueueBad {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr size_t MASK = Capacity - 1;

public:
    SPSCQueueBad() : head_(0), tail_(0) {}

    // TODO(human): Copy your try_push/try_pop implementation here,
    // but note that head_ and tail_ are NOT padded -- they share a cache line.
    bool try_push(const T& value) {
        (void)value;
        return false;
    }

    std::optional<T> try_pop() {
        return std::nullopt;
    }

private:
    T buffer_[Capacity];

    // BAD: head_ and tail_ on the SAME cache line -- false sharing!
    std::atomic<uint64_t> head_;
    std::atomic<uint64_t> tail_;
};

// =============================================================================
// Exercise 3: Throughput Benchmark
// =============================================================================
//
// TODO(human): Implement this benchmark.
//
// STEPS:
//   1. Create an SPSCQueue<uint64_t, 65536>
//   2. Launch producer thread: push 10M sequential integers, spin on try_push failure
//   3. Launch consumer thread: pop 10M integers, spin on try_pop failure (with cpu_pause)
//   4. Measure wall-clock time for all 10M messages
//   5. Print throughput in M msgs/sec and latency in ns/msg
//   6. Repeat with SPSCQueueBad to compare
//   7. Repeat with memory_order_seq_cst instead of acquire/release to compare
//
// EXPECTED RESULTS (rough, depends on CPU):
//   Good queue (padded, acq/rel):    ~100-300 M msgs/sec (~3-10 ns/msg)
//   Bad queue (false sharing):       ~30-80 M msgs/sec
//   seq_cst version:                 ~80-200 M msgs/sec (slightly slower)
// =============================================================================

void benchmark_spsc() {
    std::cout << "\n=== Exercise 3: SPSC Throughput Benchmark ===\n\n";

    constexpr size_t NUM_MESSAGES = 10'000'000;
    constexpr size_t QUEUE_SIZE = 65536;  // 2^16

    SPSCQueue<uint64_t, QUEUE_SIZE> queue;

    // TODO(human): Implement the producer-consumer benchmark.
    //
    // PRODUCER THREAD:
    //   for (uint64_t i = 0; i < NUM_MESSAGES; ++i) {
    //       while (!queue.try_push(i)) {
    //           cpu_pause();  // spin-wait hint
    //       }
    //   }
    //
    // CONSUMER THREAD:
    //   uint64_t expected = 0;
    //   while (expected < NUM_MESSAGES) {
    //       auto val = queue.try_pop();
    //       if (val.has_value()) {
    //           assert(val.value() == expected);  // verify ordering
    //           ++expected;
    //       } else {
    //           cpu_pause();
    //       }
    //   }
    //
    // TIMING:
    //   auto start = std::chrono::steady_clock::now();
    //   // ... launch threads, join ...
    //   auto end = std::chrono::steady_clock::now();
    //   double ns_per_msg = duration_ns / NUM_MESSAGES;
    //   double throughput = NUM_MESSAGES / (duration_sec);

    std::cout << "Queue capacity: " << QUEUE_SIZE << "\n";
    std::cout << "Messages: " << NUM_MESSAGES << "\n";
    std::cout << "\n[Placeholder] Implement producer/consumer threads and measure throughput.\n";
    std::cout << "Expected: ~100-300M msgs/sec for padded acq/rel queue.\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 2: Lock-Free SPSC Ring Buffer\n";
    std::cout << "========================================\n";

    // Quick functionality check (single-threaded)
    {
        std::cout << "\n--- Single-threaded sanity check ---\n";
        SPSCQueue<int, 8> q;
        std::cout << "Push 1: " << (q.try_push(1) ? "ok" : "FAIL") << "\n";
        std::cout << "Push 2: " << (q.try_push(2) ? "ok" : "FAIL") << "\n";
        auto v1 = q.try_pop();
        auto v2 = q.try_pop();
        auto v3 = q.try_pop();
        std::cout << "Pop 1: " << (v1.has_value() ? std::to_string(*v1) : "empty") << "\n";
        std::cout << "Pop 2: " << (v2.has_value() ? std::to_string(*v2) : "empty") << "\n";
        std::cout << "Pop 3 (should be empty): " << (v3.has_value() ? std::to_string(*v3) : "empty") << "\n";

        // NOTE: These will print "empty" until you implement try_push/try_pop.
        // After implementation, they should print 1 and 2.
    }

    benchmark_spsc();

    std::cout << "\n========================================\n";
    std::cout << "Phase 2 complete.\n";
    std::cout << "========================================\n";

    return 0;
}
