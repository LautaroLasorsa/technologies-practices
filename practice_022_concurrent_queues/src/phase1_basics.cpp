// =============================================================================
// Phase 1: ConcurrentQueue Basics
// =============================================================================
//
// GOAL: Get comfortable with the basic moodycamel::ConcurrentQueue API.
//
// CONCURRENCY CONTEXT:
//   In competitive programming, you never need concurrent queues because
//   everything runs on a single thread. In production systems, threads are
//   how you use multiple CPU cores. A concurrent queue is the bridge:
//   one thread produces data, another consumes it, and the queue handles
//   the synchronization.
//
//   std::queue is NOT thread-safe. If two threads push/pop simultaneously,
//   you get undefined behavior (corrupted memory, crashes). The traditional
//   fix is std::mutex -- but that serializes access (only one thread at a time).
//
//   moodycamel::ConcurrentQueue is LOCK-FREE: multiple threads can push
//   and pop SIMULTANEOUSLY without any mutex. Internally it uses CAS
//   (Compare-And-Swap) atomic operations:
//
//     CAS(ptr, expected, desired):
//       if (*ptr == expected) { *ptr = desired; return true; }
//       else { expected = *ptr; return false; }  // someone else modified it
//
//   On x86, CAS compiles to the LOCK CMPXCHG instruction (~10-30ns).
//   A mutex lock/unlock is ~15-25ns uncontended, ~50-200ns under contention.
//
// RUST EQUIVALENT:
//   crossbeam::channel::unbounded() for MPMC
//   std::sync::mpsc::channel() for MPSC (stdlib)
//   flume::unbounded() for MPMC
//
// REFERENCE:
//   https://github.com/cameron314/concurrentqueue
//   https://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++.htm
// =============================================================================

#include "concurrentqueue.h"
#include "bench_utils.h"

#include <iostream>
#include <vector>
#include <cassert>

// =============================================================================
// Exercise 1: Basic Enqueue and Dequeue
// =============================================================================
//
// GOAL: Create a ConcurrentQueue<int>, enqueue 100 values from the main
//       thread, then dequeue all of them and verify you got them all.
//
// KEY API:
//   moodycamel::ConcurrentQueue<T> q;    // default constructor (starts empty)
//   q.enqueue(value);                     // always succeeds (unbounded growth)
//   bool ok = q.try_dequeue(item);        // returns false if empty
//   size_t n = q.size_approx();           // approximate size (not exact!)
//
// NOTE ON size_approx():
//   The queue is lock-free, so size() can't be exact without a global lock.
//   size_approx() returns a value that may be slightly stale. In a concurrent
//   context, by the time you read the size, another thread may have changed it.
//   Only use it for debugging/logging, never for control flow.
//
// WHY enqueue() ALWAYS SUCCEEDS:
//   Unlike a bounded queue (fixed capacity), ConcurrentQueue is unbounded by
//   default -- it dynamically allocates new blocks when it runs out of space.
//   This is fine for most use cases but bad for HFT (allocation on hot path).
//   Pre-sizing (Exercise 3) mitigates this.
// =============================================================================

void exercise1_basic_enqueue_dequeue() {
    print_subheader("Exercise 1: Basic Enqueue and Dequeue");

    // TODO(human): Implement this exercise.
    //
    // STEPS:
    //   1. Create a ConcurrentQueue<int>
    //   2. Enqueue integers 0..99 using q.enqueue(i) in a loop
    //   3. Print size_approx() (should be ~100)
    //   4. Dequeue all items using q.try_dequeue(item) in a loop
    //   5. Collect into a vector, sort it, and verify you got 0..99
    //   6. Print results
    //
    // HINT: try_dequeue returns bool. Loop until it returns false.
    //
    // WHY SORT? In single-threaded usage, items come out in FIFO order.
    // But in multi-threaded usage (later phases), ordering is NOT guaranteed
    // across producers. Sorting lets you verify completeness regardless.

    moodycamel::ConcurrentQueue<int> q;
    (void)q;  // remove this line when implementing

    std::cout << "[Placeholder] Enqueue 100 items, dequeue all, verify.\n";
    std::cout << "Expected: all 100 items recovered in order (single-threaded).\n";
}

// =============================================================================
// Exercise 2: try_enqueue vs enqueue
// =============================================================================
//
// GOAL: Understand the difference between enqueue() and try_enqueue().
//
// KEY API:
//   q.enqueue(value);            // always succeeds (allocates if needed)
//   bool ok = q.try_enqueue(value);  // may fail if no pre-allocated space
//
// WHEN try_enqueue FAILS:
//   try_enqueue only fails when:
//   1. The queue has no pre-allocated space left, AND
//   2. Memory allocation for a new block fails (extremely rare)
//
//   In practice, try_enqueue almost always succeeds because the queue
//   auto-allocates. The difference matters when you've configured the queue
//   with a specific initial capacity and want to detect overflow.
//
// ANALOGY TO CP:
//   Think of enqueue() as vector::push_back() -- it grows dynamically.
//   think of try_enqueue() as "push_back but tell me if reallocation failed."
// =============================================================================

void exercise2_try_enqueue() {
    print_subheader("Exercise 2: try_enqueue vs enqueue");

    // TODO(human): Implement this exercise.
    //
    // STEPS:
    //   1. Create a ConcurrentQueue<int>
    //   2. Use try_enqueue to push 50 items, count successes
    //   3. Use enqueue to push 50 more items (always succeeds)
    //   4. Dequeue all 100, verify count
    //   5. Print how many try_enqueue succeeded vs failed
    //
    // EXPECTED: All 100 try_enqueue + enqueue succeed (failure is very rare).
    //
    // ALSO TRY: q.try_dequeue(item) on an empty queue -- returns false.
    //   int item;
    //   bool got_one = q.try_dequeue(item);  // false when empty

    moodycamel::ConcurrentQueue<int> q;
    (void)q;

    std::cout << "[Placeholder] try_enqueue 50 items, enqueue 50 more, verify.\n";
    std::cout << "Expected: all 100 succeed.\n";
}

// =============================================================================
// Exercise 3: Pre-Sized Queue
// =============================================================================
//
// GOAL: Create a queue with pre-allocated capacity to avoid runtime allocation.
//
// KEY API:
//   moodycamel::ConcurrentQueue<T> q(initial_capacity);
//
// WHY PRE-SIZE?
//   The default queue starts with a small internal block (~32 elements).
//   When you exceed this, it allocates a new block (calls malloc/new).
//   In latency-sensitive code, allocation is the enemy:
//     - malloc can take 50-500ns (and has unbounded worst-case due to OS)
//     - In HFT, the hot-path budget is ~500ns TOTAL
//     - One allocation can blow your entire latency budget
//
//   Pre-sizing allocates all blocks upfront, so enqueue never allocates.
//   This is the same principle as std::vector::reserve() -- pay the cost
//   once at startup, not on the hot path.
//
// BLOCK SIZE ROUNDING:
//   The queue allocates in blocks (default: 32 elements per block).
//   If you pre-size with 1000, it allocates ceil(1000/32) = 32 blocks.
//   The actual capacity may be slightly more than requested.
//
// BENCHMARK PLAN:
//   Compare enqueue throughput with and without pre-sizing when pushing
//   many items. Pre-sized should be faster (fewer allocations).
// =============================================================================

void exercise3_presized_queue() {
    print_subheader("Exercise 3: Pre-Sized Queue");

    // TODO(human): Implement this exercise.
    //
    // STEPS:
    //   1. Create a default queue:      ConcurrentQueue<int> q_default;
    //   2. Create a pre-sized queue:    ConcurrentQueue<int> q_presized(100000);
    //   3. Benchmark: enqueue 100000 items into q_default, measure time
    //   4. Benchmark: enqueue 100000 items into q_presized, measure time
    //   5. Print comparison using ComparisonTable
    //
    // HINT: Use Timer from bench_utils.h:
    //   Timer t;
    //   t.start();
    //   // ... work ...
    //   t.stop();
    //   auto result = compute_throughput("label", ops, t.elapsed_ms());
    //
    // EXPECTED: Pre-sized is ~10-30% faster for large batches.
    // The difference is more dramatic under multi-threaded contention.

    constexpr size_t NUM_ITEMS = 100'000;

    moodycamel::ConcurrentQueue<int> q_default;
    moodycamel::ConcurrentQueue<int> q_presized(NUM_ITEMS);
    (void)q_default;
    (void)q_presized;

    std::cout << "[Placeholder] Benchmark default vs pre-sized queue (" << NUM_ITEMS << " items).\n";
    std::cout << "Expected: pre-sized ~10-30% faster (fewer allocations).\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 1: ConcurrentQueue Basics");

    std::cout << "\nThis phase introduces the moodycamel::ConcurrentQueue API.\n"
              << "All exercises run single-threaded -- multi-threading starts in Phase 2.\n"
              << "\n"
              << "KEY DIFFERENCE FROM std::queue:\n"
              << "  std::queue:       NOT thread-safe. Needs external mutex.\n"
              << "  ConcurrentQueue:  Lock-free. Safe for MPMC (Multi-Producer Multi-Consumer).\n"
              << "  Both:             FIFO ordering within a single thread.\n"
              << "\n"
              << "CAS (Compare-And-Swap) quick primer:\n"
              << "  x86 instruction: LOCK CMPXCHG\n"
              << "  Atomically: if (*addr == expected) *addr = desired; return success;\n"
              << "  Cost: ~10-30ns per operation (vs mutex ~50-200ns under contention)\n"
              << "  Lock-free = no thread can block another; CAS may retry but never waits.\n";

    exercise1_basic_enqueue_dequeue();
    exercise2_try_enqueue();
    exercise3_presized_queue();

    print_header("Phase 1 Complete");
    std::cout << "\nNext: Phase 2 -- Multi-threaded producer/consumer patterns.\n"
              << "Run: build\\Release\\phase2_multithreaded.exe\n\n";

    return 0;
}
