// =============================================================================
// Phase 2: Multi-Threaded Producer/Consumer
// =============================================================================
//
// GOAL: Use ConcurrentQueue with real threads -- multiple producers and
//       consumers operating simultaneously.
//
// CONCURRENCY CONTEXT:
//   This is where lock-free queues earn their keep. With a mutex-based queue,
//   every enqueue/dequeue takes the same lock. Under 4+ threads, lock
//   contention becomes the bottleneck:
//
//     Mutex contention cost (4 threads): ~50-200ns per operation
//     ConcurrentQueue (4 threads):       ~15-40ns per operation
//     Speedup:                           ~3-10x
//
//   The speedup grows with thread count because mutex serializes ALL access,
//   while the lock-free queue allows true parallelism. Each producer and
//   consumer can operate independently.
//
// CORRECTNESS CHALLENGE:
//   With multiple producers, items from different threads interleave.
//   You CANNOT expect FIFO ordering across producers. You CAN expect:
//   - No items lost (total dequeued == total enqueued)
//   - No items duplicated
//   - FIFO within a single producer's items (but interleaved with others)
//
// RUST EQUIVALENT:
//   use crossbeam::channel::unbounded;
//   let (tx, rx) = unbounded();
//   // Clone tx for each producer, share rx among consumers
//   // crossbeam guarantees: no lost items, no duplicates
//
// REFERENCE:
//   https://github.com/cameron314/concurrentqueue#basic-use
// =============================================================================

#include "concurrentqueue.h"
#include "bench_utils.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>
#include <queue>
#include <cassert>

// =============================================================================
// Exercise 1: N Producers, 1 Consumer
// =============================================================================
//
// GOAL: Verify that no items are lost when multiple producers push into
//       a single queue and one consumer drains it.
//
// DESIGN:
//   - N producer threads, each enqueuing M integers
//   - 1 consumer thread, dequeuing until it receives N*M items
//   - Verify: total dequeued == N * M
//
// SHUTDOWN COORDINATION:
//   The consumer needs to know when to stop. Options:
//   a) Use an atomic counter of total items produced (done here)
//   b) Poison pill / sentinel value (covered in Phase 5)
//   c) External flag + final drain
//
// MEMORY ORDERING OF std::atomic<size_t> counter:
//   We use memory_order_relaxed for the counter because we only need
//   the eventual total count, not ordering relative to queue operations.
//   The queue itself handles its own internal synchronization.
// =============================================================================

void exercise1_n_producers_1_consumer() {
    print_subheader("Exercise 1: N Producers, 1 Consumer");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    // TODO(human): Implement multi-producer single-consumer.
    //
    // STEPS:
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS);  // pre-sized
    //   2. Create atomic<bool> producers_done{false};
    //   3. Launch NUM_PRODUCERS producer threads:
    //        for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
    //            q.enqueue(producer_id * ITEMS_PER_PRODUCER + i);
    //        }
    //   4. Launch 1 consumer thread:
    //        int item;
    //        size_t count = 0;
    //        while (count < TOTAL_ITEMS) {
    //            if (q.try_dequeue(item)) {
    //                ++count;
    //            }
    //            // Optionally: std::this_thread::yield() if not found
    //        }
    //   5. Join all threads
    //   6. Verify count == TOTAL_ITEMS
    //   7. Print results
    //
    // HINT: Use std::vector<std::thread> for the producers.
    //   std::vector<std::thread> producers;
    //   for (int p = 0; p < NUM_PRODUCERS; ++p) {
    //       producers.emplace_back([&, p]() { ... });
    //   }
    //   for (auto& t : producers) t.join();

    std::cout << "[Placeholder] " << NUM_PRODUCERS << " producers x "
              << ITEMS_PER_PRODUCER << " items = " << TOTAL_ITEMS << " total.\n";
    std::cout << "Expected: consumer receives exactly " << TOTAL_ITEMS << " items.\n";
}

// =============================================================================
// Exercise 2: N Producers, N Consumers
// =============================================================================
//
// GOAL: Multiple consumers compete for items. Each consumer counts how
//       many it received. The global total must equal N*M.
//
// KEY INSIGHT:
//   With multiple consumers, each item goes to EXACTLY ONE consumer.
//   This is "competing consumers" pattern (vs "pub/sub" where all get a copy).
//   Items are NOT duplicated across consumers.
//
// SHUTDOWN:
//   Use an atomic counter. When total_consumed == TOTAL_ITEMS, all stop.
//   Each consumer atomically increments the counter after each successful dequeue.
//
// LOAD BALANCING:
//   moodycamel doesn't guarantee equal distribution. Under contention,
//   some consumers may get more items than others. This is normal --
//   the queue is optimized for throughput, not fairness.
// =============================================================================

void exercise2_n_producers_n_consumers() {
    print_subheader("Exercise 2: N Producers, N Consumers");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    // TODO(human): Implement multi-producer multi-consumer.
    //
    // STEPS:
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS);
    //   2. Create atomic<size_t> total_consumed{0};
    //   3. Create a vector<size_t> per_consumer_count(NUM_CONSUMERS, 0);
    //   4. Launch NUM_PRODUCERS producer threads (same as Exercise 1)
    //   5. Launch NUM_CONSUMERS consumer threads:
    //        int item;
    //        while (total_consumed.load(std::memory_order_relaxed) < TOTAL_ITEMS) {
    //            if (q.try_dequeue(item)) {
    //                ++per_consumer_count[consumer_id];
    //                total_consumed.fetch_add(1, std::memory_order_relaxed);
    //            }
    //        }
    //   6. Join all threads
    //   7. Verify sum(per_consumer_count) == TOTAL_ITEMS
    //   8. Print per-consumer distribution
    //
    // EXPECTED OUTPUT (example, distribution varies):
    //   Consumer 0: 262,145 items (26.2%)
    //   Consumer 1: 248,903 items (24.9%)
    //   Consumer 2: 241,567 items (24.2%)
    //   Consumer 3: 247,385 items (24.7%)
    //   Total: 1,000,000 == 1,000,000 OK

    std::cout << "[Placeholder] " << NUM_PRODUCERS << "P x " << NUM_CONSUMERS << "C, "
              << TOTAL_ITEMS << " total items.\n";
    std::cout << "Expected: sum of all consumer counts == " << TOTAL_ITEMS << ".\n";
}

// =============================================================================
// Exercise 3: Throughput Benchmark (varying ratios)
// =============================================================================
//
// GOAL: Measure ops/sec for different producer:consumer ratios.
//       This reveals how the queue scales with contention.
//
// RATIOS TO TEST:
//   1:1  -- minimal contention (like SPSC but through MPMC queue)
//   4:1  -- many-to-one (fan-in / aggregation pattern)
//   1:4  -- one-to-many (fan-out / work distribution pattern)
//   4:4  -- symmetric MPMC (maximum contention)
//
// EXPECTED RESULTS (rough):
//   1:1  -- ~50-100 M ops/sec (minimal contention)
//   4:1  -- ~40-80 M ops/sec (producers contend with each other)
//   1:4  -- ~30-60 M ops/sec (consumers contend with each other)
//   4:4  -- ~20-50 M ops/sec (maximum contention on both sides)
//
// KEY INSIGHT:
//   Even 4:4 with maximum contention is still faster than mutex-based
//   queues because lock-free allows partial parallelism. With a mutex,
//   8 threads sharing one lock degrades to nearly sequential.
// =============================================================================

void exercise3_throughput_ratios() {
    print_subheader("Exercise 3: Throughput Benchmark");

    constexpr size_t TOTAL_ITEMS = 1'000'000;

    // TODO(human): Implement throughput benchmark for different ratios.
    //
    // For each ratio (num_producers, num_consumers):
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS)
    //   2. items_per_producer = TOTAL_ITEMS / num_producers
    //   3. Start timer
    //   4. Launch producer threads (each enqueues items_per_producer items)
    //   5. Launch consumer threads (use atomic counter for shutdown)
    //   6. Join all, stop timer
    //   7. Compute throughput: TOTAL_ITEMS / elapsed_sec
    //
    // USE ComparisonTable:
    //   ComparisonTable table("Throughput vs Producer:Consumer Ratio");
    //   table.add(compute_throughput("1P : 1C", TOTAL_ITEMS, elapsed_ms_1_1));
    //   table.add(compute_throughput("4P : 1C", TOTAL_ITEMS, elapsed_ms_4_1));
    //   ...
    //   table.print();
    //
    // HINT: Extract a helper function:
    //   double run_benchmark(int num_producers, int num_consumers, size_t total_items);
    //   // Returns elapsed_ms

    struct Ratio { int producers; int consumers; std::string label; };
    std::vector<Ratio> ratios = {
        {1, 1, "1P : 1C"},
        {4, 1, "4P : 1C"},
        {1, 4, "1P : 4C"},
        {4, 4, "4P : 4C"},
    };

    std::cout << "[Placeholder] Benchmark " << ratios.size() << " ratios, "
              << TOTAL_ITEMS << " items each.\n";
    for (const auto& r : ratios) {
        std::cout << "  " << r.label << ": [implement me]\n";
    }
}

// =============================================================================
// Exercise 4: Comparison with std::queue + std::mutex
// =============================================================================
//
// GOAL: Quantify the speedup of lock-free vs mutex-based queue.
//
// MUTEX-BASED QUEUE:
//   template <typename T>
//   class MutexQueue {
//       std::queue<T> q_;
//       std::mutex mtx_;
//   public:
//       void enqueue(const T& val) {
//           std::lock_guard<std::mutex> lock(mtx_);
//           q_.push(val);
//       }
//       bool try_dequeue(T& val) {
//           std::lock_guard<std::mutex> lock(mtx_);
//           if (q_.empty()) return false;
//           val = q_.front();
//           q_.pop();
//           return true;
//       }
//   };
//
// WHY MUTEX IS SLOWER:
//   1. Lock acquisition: even uncontended, ~15-25ns (atomic exchange + memory barrier)
//   2. Under contention: OS futex wait/wake, context switches (~1-10us each!)
//   3. Serialization: only one thread operates at a time -- zero parallelism
//   4. Cache line bouncing: the mutex word bounces between cores
//
//   ConcurrentQueue avoids all of this with CAS-based progress.
//   Even when CAS retries, the thread keeps running (no context switch).
//
// EXPECTED SPEEDUP (4P:4C):
//   Mutex: ~5-15 M ops/sec
//   ConcurrentQueue: ~20-50 M ops/sec
//   Speedup: ~3-10x
// =============================================================================

// Provided: simple mutex-based queue for comparison
template <typename T>
class MutexQueue {
public:
    void enqueue(const T& val) {
        std::lock_guard<std::mutex> lock(mtx_);
        q_.push(val);
    }

    bool try_dequeue(T& val) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (q_.empty()) return false;
        val = q_.front();
        q_.pop();
        return true;
    }

private:
    std::queue<T> q_;
    std::mutex mtx_;
};

void exercise4_mutex_comparison() {
    print_subheader("Exercise 4: ConcurrentQueue vs std::queue + mutex");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr size_t TOTAL_ITEMS = 1'000'000;

    // TODO(human): Benchmark MutexQueue vs ConcurrentQueue.
    //
    // STEPS:
    //   1. Run the 4P:4C benchmark with MutexQueue<int>, measure elapsed_ms
    //   2. Run the 4P:4C benchmark with ConcurrentQueue<int>, measure elapsed_ms
    //   3. Use ComparisonTable to display results
    //
    // IMPLEMENTATION HINT:
    //   For MutexQueue, the producer/consumer loop is the same structure as
    //   Exercise 2, but uses mutex_queue.enqueue(val) and mutex_queue.try_dequeue(val).
    //
    //   For the ConcurrentQueue version, reuse your Exercise 2 code.
    //
    // TEMPLATE for a generic benchmark function:
    //   template <typename Queue>
    //   double benchmark_queue(Queue& q, int num_producers, int num_consumers, size_t total) {
    //       Timer t;
    //       t.start();
    //       // ... spawn producers/consumers, join ...
    //       t.stop();
    //       return t.elapsed_ms();
    //   }

    MutexQueue<int> mq;
    moodycamel::ConcurrentQueue<int> cq(TOTAL_ITEMS);
    (void)mq;
    (void)cq;

    std::cout << "[Placeholder] " << NUM_PRODUCERS << "P:" << NUM_CONSUMERS << "C comparison.\n";
    std::cout << "Expected: ConcurrentQueue ~3-10x faster than mutex under contention.\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 2: Multi-Threaded Producer/Consumer");

    auto hw_threads = std::thread::hardware_concurrency();
    std::cout << "\nHardware threads available: " << hw_threads << "\n"
              << "\nThis phase introduces real concurrency. Key principles:\n"
              << "  - No items lost (total dequeued == total enqueued)\n"
              << "  - No items duplicated (each item consumed exactly once)\n"
              << "  - No ordering guarantee across producers (only within one producer)\n"
              << "  - Lock-free >> mutex under contention (3-10x)\n";

    exercise1_n_producers_1_consumer();
    exercise2_n_producers_n_consumers();
    exercise3_throughput_ratios();
    exercise4_mutex_comparison();

    print_header("Phase 2 Complete");
    std::cout << "\nNext: Phase 3 -- Producer/Consumer tokens for even faster throughput.\n"
              << "Run: build\\Release\\phase3_tokens.exe\n\n";

    return 0;
}
