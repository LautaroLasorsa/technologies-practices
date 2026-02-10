// =============================================================================
// Phase 4: Bulk Operations
// =============================================================================
//
// GOAL: Use enqueue_bulk / try_dequeue_bulk for batch processing.
//
// CONCURRENCY CONTEXT:
//   Individual enqueue/dequeue operations each pay a fixed overhead:
//     - Memory barrier (fence instruction): ~5-10ns
//     - CAS attempt (LOCK CMPXCHG): ~10-30ns
//     - Cache line transfer between cores: ~40-100ns (if contended)
//
//   Total: ~15-40ns per item.
//
//   Bulk operations AMORTIZE this overhead across N items:
//     - One memory barrier for the whole batch
//     - One CAS to claim space for N items
//     - One cache line transfer
//
//   Total: ~50-100ns for the whole batch = ~1-3ns per item (batch of 64).
//   That's 10-30x less overhead per item!
//
//   ANALOGY:
//     Individual ops:  Post office, mailing 64 letters one at a time.
//                      Each letter = wait in line + stamp + send.
//     Bulk ops:        Bundle 64 letters in one package.
//                      One wait + one stamp + one send. Same total mail.
//
// KEY API:
//   // Enqueue N items from a contiguous array/iterator
//   q.enqueue_bulk(items, count);
//   q.enqueue_bulk(ptok, items, count);           // with token
//
//   // Dequeue up to N items into a buffer, returns actual count dequeued
//   size_t actual = q.try_dequeue_bulk(buffer, max_count);
//   size_t actual = q.try_dequeue_bulk(ctok, buffer, max_count);  // with token
//
// IMPORTANT: try_dequeue_bulk may return FEWER items than requested.
//   It returns however many are immediately available (up to max_count).
//   You must check the return value and loop if you need more.
//
// RUST EQUIVALENT:
//   crossbeam doesn't have bulk ops. You'd batch manually:
//     let batch: Vec<T> = (0..64).map(|_| rx.try_recv()).flatten().collect();
//   flume has try_drain() for bulk dequeue.
//
// REFERENCE:
//   https://github.com/cameron314/concurrentqueue#bulk-operations
// =============================================================================

#include "concurrentqueue.h"
#include "bench_utils.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>
#include <array>
#include <numeric>

// =============================================================================
// Exercise 1: enqueue_bulk vs Individual Enqueue
// =============================================================================
//
// GOAL: Benchmark the difference between enqueuing 1000 items one at a time
//       vs enqueue_bulk with an array of 1000.
//
// SETUP:
//   - Create array: int items[1000]; (fill with 0..999)
//   - Method A: for (int i = 0; i < 1000; ++i) q.enqueue(i);
//   - Method B: q.enqueue_bulk(items, 1000);
//   - Run each 1000 times, measure total
//
// EXPECTED:
//   enqueue_bulk should be ~5-15x faster per item because it amortizes:
//   - CAS operations (one claim for all 1000 items)
//   - Memory barriers (one sfence instead of 1000)
//   - Internal bookkeeping (block traversal, tail updates)
// =============================================================================

void exercise1_bulk_enqueue_benchmark() {
    print_subheader("Exercise 1: enqueue_bulk vs Individual Enqueue");

    constexpr size_t BATCH_SIZE = 1000;
    constexpr size_t NUM_BATCHES = 1000;
    constexpr size_t TOTAL_ITEMS = BATCH_SIZE * NUM_BATCHES;

    // TODO(human): Implement the benchmark.
    //
    // STEPS:
    //   1. Prepare a source array: std::array<int, BATCH_SIZE> items;
    //      std::iota(items.begin(), items.end(), 0);  // fill with 0..999
    //
    //   2. Benchmark individual enqueue:
    //      ConcurrentQueue<int> q1(TOTAL_ITEMS);
    //      Timer t1;
    //      t1.start();
    //      for (size_t batch = 0; batch < NUM_BATCHES; ++batch) {
    //          for (size_t i = 0; i < BATCH_SIZE; ++i) {
    //              q1.enqueue(items[i]);
    //          }
    //      }
    //      t1.stop();
    //
    //   3. Benchmark bulk enqueue:
    //      ConcurrentQueue<int> q2(TOTAL_ITEMS);
    //      Timer t2;
    //      t2.start();
    //      for (size_t batch = 0; batch < NUM_BATCHES; ++batch) {
    //          q2.enqueue_bulk(items.data(), BATCH_SIZE);
    //      }
    //      t2.stop();
    //
    //   4. ComparisonTable with both results
    //   5. Drain both queues to verify item count

    std::cout << "[Placeholder] " << TOTAL_ITEMS << " items: individual vs bulk enqueue.\n";
    std::cout << "Expected: enqueue_bulk ~5-15x faster per item.\n";
}

// =============================================================================
// Exercise 2: try_dequeue_bulk -- Batch Dequeue
// =============================================================================
//
// GOAL: Dequeue items in batches using try_dequeue_bulk.
//
// KEY API:
//   int buffer[64];
//   size_t count = q.try_dequeue_bulk(buffer, 64);
//   // count <= 64, could be 0 if queue is empty
//   // buffer[0..count-1] contains the dequeued items
//
// BATCH SIZE CHOICE:
//   - Too small (1-4): negligible benefit over individual dequeue
//   - Sweet spot (32-128): amortizes overhead without excessive latency
//   - Too large (10000+): may cause latency spikes (long batch = long pause)
//
//   In practice, 64 is a common choice because:
//   1. Fits in one cache line (64 * 4 bytes = 256 bytes = 4 cache lines for int)
//   2. Good amortization ratio
//   3. Small enough for stack allocation
//
// PATTERN FOR DRAINING A QUEUE:
//   int buffer[64];
//   size_t total = 0;
//   while (true) {
//       size_t got = q.try_dequeue_bulk(buffer, 64);
//       if (got == 0) break;  // queue empty
//       for (size_t i = 0; i < got; ++i) {
//           process(buffer[i]);
//       }
//       total += got;
//   }
// =============================================================================

void exercise2_bulk_dequeue() {
    print_subheader("Exercise 2: try_dequeue_bulk");

    constexpr size_t TOTAL_ITEMS = 1'000'000;
    constexpr size_t DEQUEUE_BATCH = 64;

    // TODO(human): Implement bulk dequeue benchmark.
    //
    // STEPS:
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS);
    //   2. Enqueue TOTAL_ITEMS items (use enqueue_bulk for speed)
    //   3. Benchmark individual dequeue:
    //      Timer t1; t1.start();
    //      int item;
    //      size_t count1 = 0;
    //      while (q.try_dequeue(item)) ++count1;
    //      t1.stop();
    //
    //   4. Re-enqueue TOTAL_ITEMS items
    //   5. Benchmark bulk dequeue:
    //      Timer t2; t2.start();
    //      int buffer[DEQUEUE_BATCH];
    //      size_t count2 = 0;
    //      while (true) {
    //          size_t got = q.try_dequeue_bulk(buffer, DEQUEUE_BATCH);
    //          if (got == 0) break;
    //          count2 += got;
    //      }
    //      t2.stop();
    //
    //   6. Verify count1 == count2 == TOTAL_ITEMS
    //   7. ComparisonTable

    std::cout << "[Placeholder] Dequeue " << TOTAL_ITEMS << " items: individual vs bulk ("
              << DEQUEUE_BATCH << " per batch).\n";
    std::cout << "Expected: bulk dequeue ~3-10x faster.\n";
}

// =============================================================================
// Exercise 3: Batched Pipeline -- Bulk Enqueue + Bulk Dequeue
// =============================================================================
//
// GOAL: Build a producer-consumer pipeline where BOTH sides use bulk ops.
//       This is how real data pipelines work:
//         - Producer collects a batch of events, enqueues them in one shot
//         - Consumer dequeues a batch, processes all, then dequeues next batch
//
// DESIGN:
//   Producer thread:
//     int batch[64];
//     while (has_more_data) {
//         fill_batch(batch, 64);
//         q.enqueue_bulk(ptok, batch, 64);
//     }
//
//   Consumer thread:
//     int batch[64];
//     while (!done) {
//         size_t got = q.try_dequeue_bulk(ctok, batch, 64);
//         for (size_t i = 0; i < got; ++i) process(batch[i]);
//     }
//
// REAL-WORLD EXAMPLE:
//   Network recv() returns up to N bytes -> parse into messages -> bulk enqueue
//   Consumer bulk dequeues -> batch process -> bulk write to disk/network
//   This is the pattern used in Kafka consumers, Redis event loops, game tick processors.
//
// BENCHMARK:
//   Compare 4 configurations (all 4P:4C):
//     A. Individual enqueue + individual dequeue (baseline)
//     B. Bulk enqueue + individual dequeue
//     C. Individual enqueue + bulk dequeue
//     D. Bulk enqueue + bulk dequeue (expected fastest)
// =============================================================================

void exercise3_batched_pipeline() {
    print_subheader("Exercise 3: Batched Pipeline");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr size_t TOTAL_ITEMS = 1'000'000;
    constexpr size_t BATCH_SIZE = 64;

    // TODO(human): Implement the 4-way batched pipeline benchmark.
    //
    // HINT: Create a generic benchmark function:
    //
    //   double run_pipeline(bool bulk_enqueue, bool bulk_dequeue,
    //                       int num_p, int num_c, size_t total, size_t batch) {
    //       ConcurrentQueue<int> q(total);
    //       atomic<size_t> consumed{0};
    //       Timer t; t.start();
    //
    //       // Producers:
    //       size_t per_producer = total / num_p;
    //       for (int p = 0; p < num_p; ++p) {
    //           producers.emplace_back([&]() {
    //               ProducerToken ptok(q);
    //               if (bulk_enqueue) {
    //                   vector<int> buf(batch);
    //                   for (size_t sent = 0; sent < per_producer; sent += batch) {
    //                       size_t n = min(batch, per_producer - sent);
    //                       iota(buf.begin(), buf.begin() + n, (int)sent);
    //                       q.enqueue_bulk(ptok, buf.data(), n);
    //                   }
    //               } else {
    //                   for (size_t i = 0; i < per_producer; ++i) {
    //                       q.enqueue(ptok, (int)i);
    //                   }
    //               }
    //           });
    //       }
    //
    //       // Consumers (similar pattern with try_dequeue_bulk)...
    //       t.stop();
    //       return t.elapsed_ms();
    //   }
    //
    // Then build ComparisonTable with 4 rows.

    std::cout << "[Placeholder] Batched pipeline: " << NUM_PRODUCERS << "P:"
              << NUM_CONSUMERS << "C, batch=" << BATCH_SIZE << ".\n";

    ComparisonTable table("Batched Pipeline (4P:4C, 1M items, batch=64)");
    // table.add(compute_throughput("Individual + Individual", TOTAL_ITEMS, ms_a));
    // table.add(compute_throughput("Bulk enq + Individual",   TOTAL_ITEMS, ms_b));
    // table.add(compute_throughput("Individual + Bulk deq",   TOTAL_ITEMS, ms_c));
    // table.add(compute_throughput("Bulk enq + Bulk deq",     TOTAL_ITEMS, ms_d));
    // table.print();

    std::cout << "[Implement the benchmark, then uncomment the table code above.]\n";
    std::cout << "Expected: Bulk+Bulk ~5-15x faster than Individual+Individual.\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 4: Bulk Operations");

    std::cout << "\nBulk operations amortize per-item overhead:\n"
              << "\n"
              << "  Individual enqueue (N items):\n"
              << "    N * (CAS + memory barrier + bookkeeping) = N * ~20ns = ~20Nns\n"
              << "\n"
              << "  Bulk enqueue (N items):\n"
              << "    1 * (CAS + memory barrier + bookkeeping) = ~50ns total\n"
              << "    Per-item cost: ~50/N ns  (for batch of 64: ~0.8ns/item)\n"
              << "\n"
              << "  In HFT, market data arrives in bursts. You batch-process\n"
              << "  the burst rather than handling each tick individually.\n"
              << "  Same principle in Kafka (batch fetch), Redis (pipeline),\n"
              << "  and network I/O (scatter-gather / readv/writev).\n";

    exercise1_bulk_enqueue_benchmark();
    exercise2_bulk_dequeue();
    exercise3_batched_pipeline();

    print_header("Phase 4 Complete");
    std::cout << "\nNext: Phase 5 -- BlockingConcurrentQueue for CPU-friendly waiting.\n"
              << "Run: build\\Release\\phase5_blocking.exe\n\n";

    return 0;
}
