// =============================================================================
// Phase 5: BlockingConcurrentQueue
// =============================================================================
//
// GOAL: Use BlockingConcurrentQueue for CPU-efficient waiting.
//
// CONCURRENCY CONTEXT -- SPINNING vs BLOCKING:
//
//   SPINNING (ConcurrentQueue + try_dequeue loop):
//     while (true) {
//         if (q.try_dequeue(item)) { process(item); }
//         // CPU burns 100% on this core even when queue is empty
//     }
//     Latency: ~10-30ns (item available instantly)
//     CPU: 100% on the consumer core (wasteful when idle)
//     Use when: HFT hot path, real-time audio, game loop -- latency is king
//
//   BLOCKING (BlockingConcurrentQueue + wait_dequeue):
//     q.wait_dequeue(item);  // blocks until item available
//     process(item);
//     Latency: ~1-10us (OS wakeup latency after signal)
//     CPU: ~0% when idle (thread sleeps in kernel)
//     Use when: background workers, I/O handlers, services -- CPU efficiency matters
//
//   TIMED BLOCKING (wait_dequeue_timed):
//     if (q.wait_dequeue_timed(item, 100ms)) {
//         process(item);
//     } else {
//         handle_timeout();  // heartbeat, cleanup, shutdown check
//     }
//     Use when: you need periodic health checks even when no items arrive
//
// INTERNAL MECHANISM:
//   BlockingConcurrentQueue uses a lightweight semaphore internally:
//   - On Windows: CreateSemaphore / WaitForSingleObject
//   - On Linux: futex (fast userspace mutex)
//   - On macOS: dispatch_semaphore
//
//   The semaphore is signaled on every enqueue, and waited on for dequeue.
//   This avoids the "thundering herd" problem of condition_variable + mutex.
//
// RUST EQUIVALENT:
//   crossbeam::channel::bounded(N) -- recv() blocks until item available
//   std::sync::mpsc::Receiver::recv() -- blocks
//   flume::Receiver::recv_timeout() -- timed blocking
//
// REFERENCE:
//   https://github.com/cameron314/concurrentqueue#blocking-version
//   Header: blockingconcurrentqueue.h
// =============================================================================

#include "blockingconcurrentqueue.h"   // extends concurrentqueue.h
#include "bench_utils.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

// =============================================================================
// Exercise 1: Basic Blocking Dequeue
// =============================================================================
//
// GOAL: Use wait_dequeue() -- the consumer blocks until an item is available.
//       No spinning, no polling, no busy-wait.
//
// KEY API:
//   moodycamel::BlockingConcurrentQueue<int> q;
//
//   // Producer:
//   q.enqueue(42);
//
//   // Consumer (blocks until item available):
//   int item;
//   q.wait_dequeue(item);  // blocks here until enqueue happens
//
// DIFFERENCE FROM try_dequeue:
//   try_dequeue: returns false immediately if empty (caller must loop)
//   wait_dequeue: blocks the thread until an item is available (no loop needed)
//
// ADVANTAGE:
//   The consumer thread goes to sleep (kernel-level wait). The OS only
//   wakes it when the producer signals the semaphore. Zero CPU usage while
//   waiting. Perfect for services that process events "when they arrive."
// =============================================================================

void exercise1_blocking_dequeue() {
    print_subheader("Exercise 1: Basic Blocking Dequeue");

    // TODO(human): Implement a blocking producer-consumer.
    //
    // STEPS:
    //   1. Create BlockingConcurrentQueue<int> q;
    //   2. Launch a consumer thread:
    //        int item;
    //        for (int i = 0; i < NUM_ITEMS; ++i) {
    //            q.wait_dequeue(item);
    //            // item is guaranteed valid here (wait_dequeue always succeeds)
    //            do_not_optimize(item);
    //        }
    //   3. Launch a producer thread:
    //        for (int i = 0; i < NUM_ITEMS; ++i) {
    //            q.enqueue(i);
    //        }
    //   4. Join both threads
    //   5. Print success message
    //
    // NOTE: The consumer does NOT need a while/if loop around dequeue!
    //   wait_dequeue blocks until an item is available, then returns it.
    //   This is cleaner code than the try_dequeue spin loop.

    constexpr int NUM_ITEMS = 100'000;

    moodycamel::BlockingConcurrentQueue<int> q;
    (void)q;

    std::cout << "[Placeholder] Blocking dequeue: " << NUM_ITEMS << " items.\n";
    std::cout << "Expected: consumer blocks until producer pushes, no busy-wait.\n";
}

// =============================================================================
// Exercise 2: Timed Wait
// =============================================================================
//
// GOAL: Use wait_dequeue_timed() to block with a timeout.
//       This is critical for services that need periodic health checks.
//
// KEY API:
//   bool got_item = q.wait_dequeue_timed(item, std::chrono::milliseconds(100));
//   if (got_item) {
//       process(item);
//   } else {
//       // Timeout! No item arrived in 100ms.
//       // Good time for: heartbeat, metrics flush, shutdown check, etc.
//   }
//
// REAL-WORLD PATTERN:
//   while (!shutdown.load()) {
//       int item;
//       if (q.wait_dequeue_timed(item, std::chrono::seconds(1))) {
//           process(item);
//       } else {
//           send_heartbeat();
//           flush_metrics();
//       }
//   }
//
// WHY TIMED WAIT?
//   Pure wait_dequeue has no way to check for shutdown. If the producer
//   crashes, the consumer blocks forever. With timed wait, the consumer
//   periodically wakes up, checks the shutdown flag, and can exit gracefully.
// =============================================================================

void exercise2_timed_wait() {
    print_subheader("Exercise 2: Timed Wait");

    // TODO(human): Implement timed wait with timeout handling.
    //
    // STEPS:
    //   1. Create BlockingConcurrentQueue<int> q;
    //   2. Create atomic<bool> producer_done{false};
    //   3. Launch consumer thread:
    //        int item;
    //        size_t received = 0;
    //        size_t timeouts = 0;
    //        while (true) {
    //            if (q.wait_dequeue_timed(item, std::chrono::milliseconds(50))) {
    //                ++received;
    //                do_not_optimize(item);
    //            } else {
    //                ++timeouts;
    //                if (producer_done.load() && q.size_approx() == 0) break;
    //            }
    //        }
    //        // Print received count and timeout count
    //
    //   4. Launch producer thread (with deliberate pauses to trigger timeouts):
    //        for (int batch = 0; batch < 10; ++batch) {
    //            for (int i = 0; i < 1000; ++i) q.enqueue(i);
    //            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    //        }
    //        producer_done.store(true);
    //
    //   5. Join both threads
    //   6. Print total received, total timeouts

    std::cout << "[Placeholder] Timed wait with periodic producer pauses.\n";
    std::cout << "Expected: some timeouts during producer pauses, all items received.\n";
}

// =============================================================================
// Exercise 3: Graceful Shutdown -- Poison Pill Pattern
// =============================================================================
//
// GOAL: Signal consumers to shut down using a sentinel/poison value.
//
// POISON PILL PATTERN:
//   Define a special value that means "stop consuming." When the producer
//   is done, it enqueues one poison pill per consumer. When a consumer
//   dequeues the poison pill, it exits its loop.
//
//   For int queues, use a sentinel like INT_MIN or -1.
//   For struct queues, add a bool `is_poison` field.
//   For pointer queues, use nullptr.
//
// ALTERNATIVE: std::optional<T> as queue element.
//   Queue of std::optional<int>:
//     Producer sends nullopt as poison pill.
//     Consumer: auto item = q.wait_dequeue(...);
//               if (!item.has_value()) break;  // poison pill
//
// WHY NOT JUST USE A SHUTDOWN FLAG?
//   A flag requires either:
//   a) Timed wait (Exercise 2) -- periodic wakeup to check flag
//   b) Interrupt/cancel (not available in moodycamel)
//   Poison pill lets you use pure wait_dequeue (no timeout overhead).
// =============================================================================

void exercise3_poison_pill() {
    print_subheader("Exercise 3: Graceful Shutdown (Poison Pill)");

    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_CONSUMER = 25'000;
    constexpr int TOTAL_ITEMS = NUM_CONSUMERS * ITEMS_PER_CONSUMER;
    constexpr int POISON = -1;  // sentinel value

    // TODO(human): Implement poison pill shutdown.
    //
    // STEPS:
    //   1. Create BlockingConcurrentQueue<int> q;
    //   2. Launch NUM_CONSUMERS consumer threads:
    //        int item;
    //        size_t my_count = 0;
    //        while (true) {
    //            q.wait_dequeue(item);
    //            if (item == POISON) break;  // exit on poison pill
    //            ++my_count;
    //        }
    //
    //   3. Launch producer thread:
    //        // Send real items
    //        for (int i = 0; i < TOTAL_ITEMS; ++i) {
    //            q.enqueue(i);
    //        }
    //        // Send one poison pill per consumer
    //        for (int c = 0; c < NUM_CONSUMERS; ++c) {
    //            q.enqueue(POISON);
    //        }
    //
    //   4. Join all threads
    //   5. Verify total consumed == TOTAL_ITEMS
    //   6. Print per-consumer counts

    std::cout << "[Placeholder] Poison pill shutdown with " << NUM_CONSUMERS << " consumers.\n";
    std::cout << "Expected: all consumers exit cleanly after receiving poison pill.\n";
    std::cout << "Total items: " << TOTAL_ITEMS << " (+ " << NUM_CONSUMERS << " poison pills).\n";
}

// =============================================================================
// Exercise 4: CPU Usage -- Spinning vs Blocking
// =============================================================================
//
// GOAL: Demonstrate that blocking queues use dramatically less CPU when idle.
//
// BENCHMARK DESIGN:
//   Scenario: Producer sends items at a LOW RATE (1 item per 10ms).
//   Consumer tries to keep up using two strategies:
//
//   Strategy A -- SPINNING (ConcurrentQueue):
//     while (!done) {
//         if (q.try_dequeue(item)) process(item);
//         // Burns 100% CPU between items
//     }
//
//   Strategy B -- BLOCKING (BlockingConcurrentQueue):
//     while (!done) {
//         q.wait_dequeue(item);  // sleeps between items
//         process(item);
//     }
//
//   Measure wall-clock time (should be similar) but observe CPU usage difference.
//   Strategy A: ~100% CPU on consumer core
//   Strategy B: ~0% CPU on consumer core (sleeps between items)
//
// HOW TO OBSERVE CPU:
//   On Windows, open Task Manager while the benchmark runs.
//   Or measure programmatically with GetProcessTimes() / clock().
//   We'll use clock() as a rough proxy: it measures CPU time, not wall time.
//   If CPU time << wall time, the thread was sleeping (blocking).
//   If CPU time ~= wall time, the thread was spinning.
//
// WHEN TO USE EACH:
//   SPINNING: when you need <1us latency and can afford dedicated CPU cores
//             (HFT, game engine main loop, audio DSP)
//   BLOCKING: when latency tolerance is >1ms and CPU efficiency matters
//             (web servers, background workers, microservices)
// =============================================================================

void exercise4_cpu_usage_comparison() {
    print_subheader("Exercise 4: CPU Usage -- Spinning vs Blocking");

    constexpr int NUM_ITEMS = 100;
    constexpr auto ITEM_INTERVAL = std::chrono::milliseconds(5);
    // Total time: ~500ms (100 items * 5ms/item)

    // TODO(human): Compare CPU usage between spinning and blocking.
    //
    // STRATEGY A -- SPINNING:
    //   moodycamel::ConcurrentQueue<int> q_spin;
    //   atomic<bool> done_spin{false};
    //
    //   auto t_cpu_start = clock();  // CPU time
    //   auto t_wall_start = chrono::steady_clock::now();
    //
    //   // Consumer (spinning):
    //   thread consumer_spin([&]() {
    //       int item;
    //       size_t count = 0;
    //       while (count < NUM_ITEMS) {
    //           if (q_spin.try_dequeue(item)) ++count;
    //           // No sleep, no pause -- pure spin
    //       }
    //   });
    //
    //   // Producer (slow):
    //   thread producer_spin([&]() {
    //       for (int i = 0; i < NUM_ITEMS; ++i) {
    //           q_spin.enqueue(i);
    //           this_thread::sleep_for(ITEM_INTERVAL);
    //       }
    //   });
    //
    //   producer_spin.join();
    //   consumer_spin.join();
    //   auto cpu_time_spin = clock() - t_cpu_start;
    //   auto wall_time_spin = chrono::steady_clock::now() - t_wall_start;
    //
    // STRATEGY B -- BLOCKING:
    //   BlockingConcurrentQueue<int> q_block;
    //   // Same structure but consumer uses q_block.wait_dequeue(item);
    //   // CPU time should be much lower
    //
    // PRINT:
    //   "Spinning: wall=XXms, cpu=XXms (ratio=X.XX)"
    //   "Blocking: wall=XXms, cpu=XXms (ratio=X.XX)"
    //   Where ratio ~1.0 means burning CPU, ratio ~0.0 means sleeping

    std::cout << "[Placeholder] " << NUM_ITEMS << " items at "
              << ITEM_INTERVAL.count() << "ms intervals.\n";
    std::cout << "Expected:\n";
    std::cout << "  Spinning: cpu_time ~= wall_time (ratio ~1.0, 100% CPU)\n";
    std::cout << "  Blocking: cpu_time << wall_time (ratio ~0.01, ~0% CPU)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 5: BlockingConcurrentQueue");

    std::cout << "\nWhen to use each queue type:\n"
              << "\n"
              << "  +-----------------------+-------------------+------------------+\n"
              << "  |                       | ConcurrentQueue   | BlockingConc.Q   |\n"
              << "  +-----------------------+-------------------+------------------+\n"
              << "  | Dequeue when empty    | returns false     | blocks (sleeps)  |\n"
              << "  | CPU usage when idle   | 100% (spinning)   | ~0% (sleeping)   |\n"
              << "  | Wakeup latency        | ~0ns (already up) | ~1-10us (kernel) |\n"
              << "  | Best for              | HFT, real-time    | services, I/O    |\n"
              << "  | Header                | concurrentqueue.h | blockingconc...h |\n"
              << "  +-----------------------+-------------------+------------------+\n"
              << "\n"
              << "  BlockingConcurrentQueue supports ALL ConcurrentQueue operations\n"
              << "  PLUS: wait_dequeue(), wait_dequeue_timed()\n"
              << "  It inherits from ConcurrentQueue and adds a semaphore.\n";

    exercise1_blocking_dequeue();
    exercise2_timed_wait();
    exercise3_poison_pill();
    exercise4_cpu_usage_comparison();

    print_header("Phase 5 Complete");
    std::cout << "\nNext: Phase 6 -- Real-world concurrent patterns.\n"
              << "Run: build\\Release\\phase6_patterns.exe\n\n";

    return 0;
}
