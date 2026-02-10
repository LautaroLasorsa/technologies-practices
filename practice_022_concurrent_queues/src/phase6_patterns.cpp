// =============================================================================
// Phase 6: Real-World Concurrent Patterns
// =============================================================================
//
// GOAL: Build common concurrent architectures using moodycamel queues.
//
// CONCURRENCY CONTEXT:
//   Most production systems are PIPELINES with multiple stages connected
//   by queues. The queue between stages determines:
//     - Throughput: how many items/sec flow through the system
//     - Latency: how long one item takes from input to output
//     - Backpressure: what happens when a stage is slower than upstream
//
//   Common patterns:
//     FAN-OUT:   1 dispatcher -> N workers (load distribution)
//     FAN-IN:    N producers -> 1 aggregator (event collection)
//     PIPELINE:  Stage1 -> Q -> Stage2 -> Q -> Stage3 (assembly line)
//
//   These patterns compose. A real system might look like:
//     [Network] --fan-in--> [Parser] --pipeline--> [Processor] --fan-out--> [Writers]
//
// REAL-WORLD EXAMPLES:
//   - Kafka:     producers fan-in to topic, consumers fan-out from partitions
//   - gRPC:      requests fan-in, thread pool processes, responses fan-out
//   - Game:      input -> physics -> rendering pipeline, each stage threaded
//   - HFT:       market data -> signal -> risk -> order (strict pipeline)
//   - Spark:     map -> shuffle (fan-out/fan-in) -> reduce
//
// RUST EQUIVALENT:
//   crossbeam::scope for scoped threads
//   rayon::ThreadPool for work-stealing fan-out
//   tokio::select! for fan-in from multiple channels
//
// REFERENCE:
//   https://en.wikipedia.org/wiki/Pipeline_(computing)
//   "Designing Data-Intensive Applications" Ch. 11 (stream processing)
// =============================================================================

#include "concurrentqueue.h"
#include "blockingconcurrentqueue.h"
#include "bench_utils.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <queue>
#include <mutex>
#include <functional>
#include <chrono>
#include <numeric>
#include <cmath>
#include <string>

// =============================================================================
// Shared types for all exercises
// =============================================================================

// Simulated work item -- represents a task flowing through the pipeline.
// In a real system, this might be a network packet, market data tick,
// game event, or database query.
struct WorkItem {
    int id = 0;
    int stage = 0;          // which pipeline stage processed this
    int producer_id = 0;    // which producer created it
    int64_t value = 0;      // payload (simulated processing result)
};

// Simulate CPU work (prevents compiler from optimizing away the "processing")
inline void simulate_work(int64_t& value, int iterations = 10) {
    for (int i = 0; i < iterations; ++i) {
        value = value * 6364136223846793005LL + 1442695040888963407LL;  // LCG
    }
    do_not_optimize(value);
}

// =============================================================================
// Exercise 1: Fan-Out (Dispatcher -> N Workers)
// =============================================================================
//
// PATTERN:
//   One dispatcher thread distributes tasks to N worker queues.
//   Each worker has its own queue and processes items independently.
//
//   +----------+     +-------+     +--------+
//   |          | --> | Q[0]  | --> | Worker0 |
//   |          |     +-------+     +--------+
//   | Dispatch | --> | Q[1]  | --> | Worker1 |
//   |          |     +-------+     +--------+
//   |          | --> | Q[2]  | --> | Worker2 |
//   |          |     +-------+     +--------+
//   |          | --> | Q[3]  | --> | Worker3 |
//   +----------+     +-------+     +--------+
//
// DISPATCH STRATEGY:
//   Round-robin: item i goes to worker (i % N).
//   Simple, deterministic, but doesn't account for worker speed differences.
//
//   In production, you might also use:
//   - Hash-based: hash(item.key) % N (ensures same-key items go to same worker)
//   - Least-loaded: pick the worker with the shortest queue
//   - Work-stealing: workers steal from each other when idle (rayon/tokio pattern)
//
// WHY FAN-OUT WITH SEPARATE QUEUES?
//   With one shared MPMC queue, all workers compete on dequeue (contention).
//   With per-worker queues, each worker has exclusive access (SPSC-like).
//   The dispatcher is the only producer for each queue -- no producer contention.
//   Result: higher throughput when work is evenly distributable.
// =============================================================================

void exercise1_fan_out() {
    print_subheader("Exercise 1: Fan-Out (Dispatcher -> N Workers)");

    constexpr int NUM_WORKERS = 4;
    constexpr int TOTAL_ITEMS = 100'000;

    // TODO(human): Implement fan-out pattern.
    //
    // STEPS:
    //   1. Create N worker queues:
    //      std::vector<moodycamel::BlockingConcurrentQueue<WorkItem>> worker_queues(NUM_WORKERS);
    //
    //   2. Launch N worker threads:
    //      for (int w = 0; w < NUM_WORKERS; ++w) {
    //          workers.emplace_back([&, w]() {
    //              WorkItem item;
    //              size_t count = 0;
    //              while (true) {
    //                  worker_queues[w].wait_dequeue(item);
    //                  if (item.id == -1) break;  // poison pill
    //                  simulate_work(item.value);
    //                  ++count;
    //              }
    //              // Print: "Worker W processed COUNT items"
    //          });
    //      }
    //
    //   3. Dispatcher: round-robin distribute items
    //      Timer t; t.start();
    //      for (int i = 0; i < TOTAL_ITEMS; ++i) {
    //          WorkItem item{i, 0, 0, (int64_t)i};
    //          worker_queues[i % NUM_WORKERS].enqueue(item);
    //      }
    //      // Send poison pills
    //      for (int w = 0; w < NUM_WORKERS; ++w) {
    //          worker_queues[w].enqueue(WorkItem{-1, 0, 0, 0});
    //      }
    //      t.stop();
    //
    //   4. Join all workers, print throughput

    std::cout << "[Placeholder] Fan-out: 1 dispatcher -> " << NUM_WORKERS << " workers, "
              << TOTAL_ITEMS << " items.\n";
    std::cout << "Expected: ~" << TOTAL_ITEMS / NUM_WORKERS << " items per worker.\n";
}

// =============================================================================
// Exercise 2: Fan-In / Aggregation (N Producers -> 1 Consumer)
// =============================================================================
//
// PATTERN:
//   Multiple producer threads push events into a single shared queue.
//   One aggregator thread consumes all events and merges/reduces them.
//
//   +----------+     +-------+     +------------+
//   | Sensor 0 | --> |       |     |            |
//   +----------+     |       |     |            |
//   | Sensor 1 | --> | Shared| --> | Aggregator |
//   +----------+     | Queue |     |            |
//   | Sensor 2 | --> |       |     |            |
//   +----------+     |       |     |            |
//   | Sensor 3 | --> |       |     +------------+
//   +----------+     +-------+
//
// REAL-WORLD EXAMPLES:
//   - Log aggregation: N services -> one log collector
//   - Market data: N exchange feeds -> one order book updater
//   - Metrics: N application threads -> one metrics exporter
//   - Event sourcing: N command handlers -> one event store writer
//
// THIS IS WHERE MPMC SHINES:
//   With SPSC, you'd need N queues + a selector (complex).
//   With MPMC, all producers push to ONE queue. Simple and fast.
//   moodycamel handles the multi-producer contention efficiently.
// =============================================================================

void exercise2_fan_in() {
    print_subheader("Exercise 2: Fan-In / Aggregation");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 25'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    // TODO(human): Implement fan-in pattern.
    //
    // STEPS:
    //   1. Create one shared queue:
    //      moodycamel::ConcurrentQueue<WorkItem> shared_q(TOTAL_ITEMS);
    //
    //   2. Launch NUM_PRODUCERS producer threads, each with a ProducerToken:
    //      ProducerToken ptok(shared_q);
    //      for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
    //          WorkItem item{i, 0, producer_id, (int64_t)(producer_id * 1000 + i)};
    //          shared_q.enqueue(ptok, item);
    //      }
    //
    //   3. Launch 1 aggregator thread:
    //      - Dequeue all items
    //      - Aggregate: count per producer, sum of values, min/max id
    //      - Use ConsumerToken for efficiency
    //
    //   4. Print aggregation results:
    //      "Producer 0: 25000 items, value_sum=XXX"
    //      "Producer 1: 25000 items, value_sum=XXX"
    //      etc.

    std::cout << "[Placeholder] Fan-in: " << NUM_PRODUCERS << " producers -> 1 aggregator, "
              << TOTAL_ITEMS << " items.\n";
    std::cout << "Expected: aggregator receives all items, counts per producer.\n";
}

// =============================================================================
// Exercise 3: Pipeline Stages (Parse -> Process -> Output)
// =============================================================================
//
// PATTERN:
//   Three stages connected by queues. Each stage can be a thread pool.
//   Items flow through: input -> stage1 -> queue -> stage2 -> queue -> stage3
//
//   +--------+   Q1   +---------+   Q2   +--------+
//   | Parse  | -----> | Process | -----> | Output |
//   | (2 th) |        | (4 th)  |        | (1 th) |
//   +--------+        +---------+        +--------+
//
// PIPELINE THROUGHPUT:
//   Total throughput = throughput of SLOWEST stage (bottleneck).
//   If Parse does 100K/s, Process does 50K/s, Output does 200K/s:
//     Pipeline throughput = 50K/s (bottleneck: Process)
//     Fix: add more Process threads (scale the bottleneck)
//
// PIPELINE LATENCY:
//   End-to-end latency = sum of all stage latencies + queue wait times.
//   Queue wait = time item sits in queue before being dequeued.
//   Under load: queue wait depends on consumer speed relative to producer.
//
// MEASUREMENT:
//   - Attach a timestamp to each WorkItem at creation
//   - At each stage, record stage entry/exit time
//   - At output, compute total latency = output_time - creation_time
//   - Compute per-stage latency = stage_exit - stage_entry
//
// DESIGN DECISION:
//   Use BlockingConcurrentQueue for inter-stage queues (workers sleep when idle).
//   Use ProducerToken/ConsumerToken for higher throughput within each stage.
// =============================================================================

void exercise3_pipeline() {
    print_subheader("Exercise 3: Multi-Stage Pipeline");

    constexpr int NUM_PARSERS    = 2;
    constexpr int NUM_PROCESSORS = 4;
    constexpr int NUM_OUTPUTS    = 1;
    constexpr int TOTAL_ITEMS    = 100'000;
    constexpr int POISON_ID      = -1;

    // TODO(human): Implement 3-stage pipeline.
    //
    // QUEUE SETUP:
    //   BlockingConcurrentQueue<WorkItem> q_parse_to_process;
    //   BlockingConcurrentQueue<WorkItem> q_process_to_output;
    //
    // STAGE 1 -- PARSE (simulates parsing raw input):
    //   for each item:
    //     item.value = hash(item.id)   // simulate parsing
    //     item.stage = 1
    //     q_parse_to_process.enqueue(ptok, item)
    //   Then send NUM_PROCESSORS poison pills to q_parse_to_process
    //
    // STAGE 2 -- PROCESS (simulates heavy computation):
    //   while (true):
    //     q_parse_to_process.wait_dequeue(item)
    //     if (item.id == POISON_ID) break
    //     simulate_work(item.value, 50)  // heavier work
    //     item.stage = 2
    //     q_process_to_output.enqueue(ptok, item)
    //   Then send NUM_OUTPUTS poison pills to q_process_to_output
    //   (only last processor to finish sends the pills -- use atomic counter)
    //
    // STAGE 3 -- OUTPUT (simulates writing results):
    //   while (true):
    //     q_process_to_output.wait_dequeue(item)
    //     if (item.id == POISON_ID) break
    //     item.stage = 3
    //     ++output_count
    //
    // TIMING:
    //   auto start = steady_clock::now();
    //   // launch all stages
    //   // join all threads
    //   auto end = steady_clock::now();
    //   // pipeline_throughput = TOTAL_ITEMS / elapsed_sec
    //
    // POISON PILL PROPAGATION:
    //   Parsers send pills to processors. Last processor sends pills to output.
    //   Use atomic<int> processors_done to coordinate.

    std::cout << "[Placeholder] Pipeline: "
              << NUM_PARSERS << " parsers -> "
              << NUM_PROCESSORS << " processors -> "
              << NUM_OUTPUTS << " output, "
              << TOTAL_ITEMS << " items.\n";
    std::cout << "Expected: all " << TOTAL_ITEMS << " items flow through all 3 stages.\n";
}

// =============================================================================
// Exercise 4: Comparison Matrix -- All Queue Types
// =============================================================================
//
// GOAL: Run the pipeline (Exercise 3 simplified) with 4 different queue
//       implementations and compare throughput.
//
// QUEUE TYPES:
//   1. std::queue<T> + std::mutex (baseline -- simple but slow)
//   2. moodycamel::ConcurrentQueue (no tokens)
//   3. moodycamel::ConcurrentQueue (with ProducerToken + ConsumerToken)
//   4. moodycamel::BlockingConcurrentQueue (with tokens)
//
// SIMPLIFIED PIPELINE FOR BENCHMARKING:
//   4 producers -> [queue] -> 4 consumers
//   Each producer enqueues 250K items
//   Each consumer dequeues until total == 1M
//
// EXPECTED RANKING (fastest to slowest):
//   1. ConcurrentQueue + tokens  (~50-150 M ops/sec)
//   2. ConcurrentQueue no tokens (~20-50 M ops/sec)
//   3. BlockingConcurrentQueue   (~15-40 M ops/sec, semaphore overhead)
//   4. std::queue + mutex        (~5-15 M ops/sec)
//
// NOTE: BlockingConcurrentQueue is slower than ConcurrentQueue for pure
//   throughput because of the semaphore signal/wait. But for CPU efficiency
//   and idle-period behavior, it's vastly superior (see Phase 5 Exercise 4).
//   Choose based on your priority: throughput vs CPU efficiency.
// =============================================================================

// Provided: MutexQueue for baseline comparison
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

void exercise4_comparison_matrix() {
    print_subheader("Exercise 4: Comparison Matrix");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr size_t TOTAL_ITEMS = 1'000'000;

    // TODO(human): Implement the 4-way comparison.
    //
    // HINT: Create a template benchmark function that works with any queue type.
    //   The tricky part is that MutexQueue has different API than ConcurrentQueue.
    //   Options:
    //     a) Write separate functions for each queue type
    //     b) Use if constexpr with type traits
    //     c) Use lambdas that wrap each queue's API
    //
    // APPROACH (c) -- Lambda wrappers:
    //
    //   auto benchmark = [&](auto enqueue_fn, auto dequeue_fn,
    //                        const std::string& label) -> double {
    //       atomic<size_t> consumed{0};
    //       Timer t; t.start();
    //       // Launch producers using enqueue_fn(value)
    //       // Launch consumers using dequeue_fn(value) -> bool
    //       // Join all
    //       t.stop();
    //       return t.elapsed_ms();
    //   };
    //
    //   // Mutex version:
    //   MutexQueue<int> mq;
    //   auto ms_mutex = benchmark(
    //       [&](int v) { mq.enqueue(v); },
    //       [&](int& v) { return mq.try_dequeue(v); },
    //       "std::queue + mutex"
    //   );
    //
    //   // ConcurrentQueue (no tokens):
    //   ConcurrentQueue<int> cq(TOTAL_ITEMS);
    //   auto ms_cq = benchmark(
    //       [&](int v) { cq.enqueue(v); },
    //       [&](int& v) { return cq.try_dequeue(v); },
    //       "ConcurrentQueue"
    //   );
    //
    //   // ... etc for token and blocking versions

    ComparisonTable table("Queue Type Comparison (4P:4C, 1M items)");
    // table.add(compute_throughput("std::queue + mutex",          TOTAL_ITEMS, ms_mutex));
    // table.add(compute_throughput("ConcurrentQueue (no tokens)", TOTAL_ITEMS, ms_cq));
    // table.add(compute_throughput("ConcurrentQueue (tokens)",    TOTAL_ITEMS, ms_cq_tok));
    // table.add(compute_throughput("BlockingConcurrentQueue",     TOTAL_ITEMS, ms_bcq));
    // table.print();

    std::cout << "[Placeholder] 4-way comparison: " << TOTAL_ITEMS << " items, "
              << NUM_PRODUCERS << "P:" << NUM_CONSUMERS << "C.\n";
    std::cout << "[Implement the benchmark, then uncomment the table code above.]\n\n";

    std::cout << "Expected ranking (fastest to slowest):\n";
    std::cout << "  1. ConcurrentQueue + tokens  (~50-150 M ops/sec)\n";
    std::cout << "  2. ConcurrentQueue no tokens (~20-50 M ops/sec)\n";
    std::cout << "  3. BlockingConcurrentQueue   (~15-40 M ops/sec)\n";
    std::cout << "  4. std::queue + mutex         (~5-15 M ops/sec)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 6: Real-World Concurrent Patterns");

    std::cout << "\nThis phase builds on everything from Phases 1-5 to construct\n"
              << "production-grade concurrent architectures.\n"
              << "\n"
              << "Pattern overview:\n"
              << "  FAN-OUT:   1 -> N  (distribute work to parallel workers)\n"
              << "  FAN-IN:    N -> 1  (collect/aggregate from multiple sources)\n"
              << "  PIPELINE:  A -> B -> C  (assembly line, each stage threaded)\n"
              << "\n"
              << "Every distributed system (Kafka, gRPC, Spark) uses these patterns.\n"
              << "The queue between stages is the critical component:\n"
              << "  - Too slow? Bottleneck. System throughput drops.\n"
              << "  - Too small? Backpressure. Producers stall.\n"
              << "  - Wrong type? CPU waste (spinning) or latency (blocking).\n"
              << "\n"
              << "After this phase, you'll be able to choose the right queue for\n"
              << "any concurrent architecture.\n";

    exercise1_fan_out();
    exercise2_fan_in();
    exercise3_pipeline();
    exercise4_comparison_matrix();

    print_header("Phase 6 Complete -- Practice 022 Done!");
    std::cout << "\nSummary of what you've learned:\n"
              << "  Phase 1: ConcurrentQueue basics (enqueue, dequeue, pre-sizing)\n"
              << "  Phase 2: Multi-threaded producer/consumer, mutex comparison\n"
              << "  Phase 3: ProducerToken/ConsumerToken for 2-3x speedup\n"
              << "  Phase 4: Bulk operations for 5-15x less per-item overhead\n"
              << "  Phase 5: BlockingConcurrentQueue for CPU-efficient waiting\n"
              << "  Phase 6: Fan-out, fan-in, pipeline, comparison matrix\n"
              << "\n"
              << "Key takeaway: choose the right tool:\n"
              << "  - SPSC ring (020a): minimum latency, single producer/consumer\n"
              << "  - ConcurrentQueue + tokens: best throughput MPMC\n"
              << "  - ConcurrentQueue + bulk: best batch throughput\n"
              << "  - BlockingConcurrentQueue: best CPU efficiency\n"
              << "  - std::queue + mutex: simplest, slowest (fine for low contention)\n\n";

    return 0;
}
