// =============================================================================
// Phase 3: Producer/Consumer Tokens
// =============================================================================
//
// GOAL: Understand and use ProducerToken / ConsumerToken for faster throughput.
//
// CONCURRENCY CONTEXT -- HOW TOKENS WORK INTERNALLY:
//
//   Without tokens, moodycamel::ConcurrentQueue uses a single pool of
//   internal "sub-queues" (blocks). When a producer enqueues, it must:
//     1. Find an available sub-queue (CAS on a shared counter)
//     2. Enqueue into that sub-queue (CAS on the sub-queue's tail)
//   Step 1 is the bottleneck -- all producers compete for the same counter.
//
//   With ProducerToken, each producer gets a DEDICATED sub-queue:
//     1. Skip the shared counter entirely (already have own sub-queue)
//     2. Enqueue into own sub-queue (CAS, but no contention from others)
//   This is like giving each producer its own SPSC lane within the MPMC queue.
//
//   With ConsumerToken, the consumer remembers which sub-queue it last
//   dequeued from, reducing search overhead on the next dequeue.
//
//   ANALOGY:
//     Without tokens: 4 cashiers share 1 pile of receipts. Each must grab
//     the pile lock before adding their receipt. Contention = slow.
//     With tokens: each cashier has their own receipt box. A collector
//     (consumer) rounds through all boxes. No contention = fast.
//
//   PERFORMANCE IMPACT:
//     Without tokens (4P:4C): ~20-50 M ops/sec
//     With tokens (4P:4C):    ~50-150 M ops/sec
//     Speedup:                ~2-3x
//
// TOKEN LIFETIME RULES:
//   - ProducerToken must outlive all enqueue() calls using it
//   - ConsumerToken must outlive all try_dequeue() calls using it
//   - Both hold a reference to the queue -- queue must outlive tokens
//   - NOT thread-safe: one token per thread (don't share tokens)
//
// RUST EQUIVALENT:
//   No direct equivalent in crossbeam. The closest is using separate channels
//   per producer and selecting across them. tokio::sync::mpsc also internally
//   uses per-producer slots (similar concept).
//
// REFERENCE:
//   https://github.com/cameron314/concurrentqueue#advanced-usage
// =============================================================================

#include "concurrentqueue.h"
#include "bench_utils.h"

#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

// =============================================================================
// Exercise 1: ProducerToken -- One Token Per Producer Thread
// =============================================================================
//
// GOAL: Refactor the Phase 2 multi-producer pattern to use ProducerToken.
//
// KEY API:
//   moodycamel::ConcurrentQueue<int> q;
//   moodycamel::ProducerToken ptok(q);    // create token bound to queue
//   q.enqueue(ptok, value);               // enqueue using token (fast path)
//
// IMPORTANT: Create the token INSIDE the producer thread function.
//   Each thread must have its own token. DO NOT share tokens across threads.
//
//   // CORRECT:
//   auto producer_fn = [&q]() {
//       moodycamel::ProducerToken ptok(q);   // token per thread
//       for (int i = 0; i < N; ++i) {
//           q.enqueue(ptok, i);
//       }
//   };
//
//   // WRONG (undefined behavior):
//   moodycamel::ProducerToken shared_ptok(q);
//   auto producer_fn = [&q, &shared_ptok]() {
//       q.enqueue(shared_ptok, i);  // RACE CONDITION!
//   };
// =============================================================================

void exercise1_producer_tokens() {
    print_subheader("Exercise 1: ProducerToken");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches ProducerToken—the key optimization that makes moodycamel shine.
    // Tokens give each producer a dedicated sub-queue, eliminating CAS contention on the
    // shared enqueue counter. Under multi-core contention, this yields 2-3× throughput gains.
    //
    // TODO(human): Implement producers with ProducerToken.
    //
    // STEPS:
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS);
    //   2. Launch NUM_PRODUCERS producer threads, each:
    //        moodycamel::ProducerToken ptok(q);  // one per thread!
    //        for (int i = 0; i < ITEMS_PER_PRODUCER; ++i) {
    //            q.enqueue(ptok, producer_id * ITEMS_PER_PRODUCER + i);
    //        }
    //   3. Launch 1 consumer (no token for now):
    //        int item;
    //        size_t count = 0;
    //        while (count < TOTAL_ITEMS) {
    //            if (q.try_dequeue(item)) ++count;
    //        }
    //   4. Verify count == TOTAL_ITEMS
    //   5. Print results

    moodycamel::ConcurrentQueue<int> q(TOTAL_ITEMS);
    (void)q;

    std::cout << "[Placeholder] " << NUM_PRODUCERS << " producers with ProducerToken.\n";
    std::cout << "Expected: " << TOTAL_ITEMS << " items dequeued successfully.\n";
}

// =============================================================================
// Exercise 2: ConsumerToken -- One Token Per Consumer Thread
// =============================================================================
//
// GOAL: Add ConsumerToken to the consumer side for faster dequeue.
//
// KEY API:
//   moodycamel::ConsumerToken ctok(q);       // create token bound to queue
//   bool ok = q.try_dequeue(ctok, item);     // dequeue using token (fast path)
//
// HOW ConsumerToken HELPS:
//   Without a token, try_dequeue must scan all sub-queues to find one with items.
//   With a token, it remembers which sub-queue had items last time and starts
//   there. This "locality hint" reduces the number of CAS operations per dequeue.
//
//   Think of it like a B-tree finger: instead of searching from the root each time,
//   you keep a pointer to where you last read.
//
// COMBINED PATTERN:
//   Producer thread:
//     ProducerToken ptok(q);
//     q.enqueue(ptok, val);
//
//   Consumer thread:
//     ConsumerToken ctok(q);
//     q.try_dequeue(ctok, val);
// =============================================================================

void exercise2_consumer_tokens() {
    print_subheader("Exercise 2: ConsumerToken");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr int ITEMS_PER_PRODUCER = 250'000;
    constexpr int TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    // TODO(human): Implement with both ProducerToken and ConsumerToken.
    //
    // STEPS:
    //   1. Create ConcurrentQueue<int> q(TOTAL_ITEMS);
    //   2. Create atomic<size_t> total_consumed{0};
    //   3. Launch producer threads (with ProducerToken, same as Exercise 1)
    //   4. Launch consumer threads, each:
    //        moodycamel::ConsumerToken ctok(q);  // one per consumer thread!
    //        int item;
    //        size_t my_count = 0;
    //        while (total_consumed.load(std::memory_order_relaxed) < TOTAL_ITEMS) {
    //            if (q.try_dequeue(ctok, item)) {
    //                ++my_count;
    //                total_consumed.fetch_add(1, std::memory_order_relaxed);
    //            }
    //        }
    //   5. Verify total_consumed == TOTAL_ITEMS
    //   6. Print per-consumer distribution

    moodycamel::ConcurrentQueue<int> q(TOTAL_ITEMS);
    (void)q;

    std::cout << "[Placeholder] " << NUM_PRODUCERS << "P (ProducerToken) : "
              << NUM_CONSUMERS << "C (ConsumerToken).\n";
    std::cout << "Expected: " << TOTAL_ITEMS << " items total.\n";
}

// =============================================================================
// Exercise 3: Benchmark -- Tokens vs No Tokens
// =============================================================================
//
// GOAL: Quantify the speedup from using tokens.
//
// TEST MATRIX:
//   1. No tokens:                      q.enqueue(val) / q.try_dequeue(val)
//   2. ProducerToken only:             q.enqueue(ptok, val) / q.try_dequeue(val)
//   3. ConsumerToken only:             q.enqueue(val) / q.try_dequeue(ctok, val)
//   4. Both tokens:                    q.enqueue(ptok, val) / q.try_dequeue(ctok, val)
//
// Run each configuration with 4P:4C and 1M items. Compare throughput.
//
// EXPECTED RESULTS:
//   No tokens:       ~20-50 M ops/sec (baseline)
//   ProducerToken:   ~40-80 M ops/sec (~2x, biggest impact)
//   ConsumerToken:   ~25-60 M ops/sec (~1.2x, modest impact)
//   Both tokens:     ~50-150 M ops/sec (~2-3x, best)
//
// WHY ProducerToken HAS MORE IMPACT:
//   Producers contend on finding a sub-queue. ProducerToken eliminates this.
//   Consumers only contend on dequeuing from sub-queues -- less overhead to begin with.
//   ProducerToken removes the #1 bottleneck; ConsumerToken removes the #2.
// =============================================================================

void exercise3_benchmark_tokens() {
    print_subheader("Exercise 3: Tokens vs No Tokens Benchmark");

    constexpr int NUM_PRODUCERS = 4;
    constexpr int NUM_CONSUMERS = 4;
    constexpr size_t TOTAL_ITEMS = 1'000'000;

    // TODO(human): Implement the 4-way benchmark.
    //
    // HINT: Create a template function that takes booleans for token usage:
    //
    //   double run_benchmark(bool use_ptok, bool use_ctok,
    //                        int num_producers, int num_consumers,
    //                        size_t total_items) {
    //       ConcurrentQueue<int> q(total_items);
    //       Timer t;
    //       t.start();
    //
    //       // Producers:
    //       for (int p = 0; p < num_producers; ++p) {
    //           producers.emplace_back([&, p]() {
    //               std::unique_ptr<ProducerToken> ptok;
    //               if (use_ptok) ptok = std::make_unique<ProducerToken>(q);
    //
    //               for (size_t i = 0; i < items_per_producer; ++i) {
    //                   if (ptok) q.enqueue(*ptok, (int)i);
    //                   else      q.enqueue((int)i);
    //               }
    //           });
    //       }
    //
    //       // Similar for consumers with ConsumerToken...
    //       t.stop();
    //       return t.elapsed_ms();
    //   }
    //
    // Then build a ComparisonTable with 4 rows.

    std::cout << "[Placeholder] 4-way benchmark: " << TOTAL_ITEMS << " items, "
              << NUM_PRODUCERS << "P:" << NUM_CONSUMERS << "C.\n";
    std::cout << "Expected speedup with both tokens: ~2-3x vs no tokens.\n";

    ComparisonTable table("Token Impact (4P:4C, 1M items)");
    // table.add(compute_throughput("No tokens",       TOTAL_ITEMS, elapsed_none));
    // table.add(compute_throughput("ProducerToken",   TOTAL_ITEMS, elapsed_ptok));
    // table.add(compute_throughput("ConsumerToken",   TOTAL_ITEMS, elapsed_ctok));
    // table.add(compute_throughput("Both tokens",     TOTAL_ITEMS, elapsed_both));
    // table.print();

    std::cout << "[Implement the benchmark, then uncomment the table code above.]\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    print_header("Phase 3: Producer/Consumer Tokens");

    std::cout << "\nTokens are moodycamel's killer feature. They give each thread a\n"
              << "dedicated sub-queue, turning MPMC contention into per-thread SPSC.\n"
              << "\n"
              << "Internal architecture:\n"
              << "  +-----------+     +-----------+     +-----------+\n"
              << "  | Producer0 | --> | SubQueue0 | --> |           |\n"
              << "  +-----------+     +-----------+     |           |\n"
              << "  | Producer1 | --> | SubQueue1 | --> | Consumer  |\n"
              << "  +-----------+     +-----------+     |  (scans   |\n"
              << "  | Producer2 | --> | SubQueue2 | --> |   all     |\n"
              << "  +-----------+     +-----------+     | sub-qs)   |\n"
              << "  | Producer3 | --> | SubQueue3 | --> |           |\n"
              << "  +-----------+     +-----------+     +-----------+\n"
              << "\n"
              << "Each ProducerToken owns one SubQueue. No cross-producer contention.\n"
              << "ConsumerToken remembers which SubQueue had items -- avoids re-scanning.\n";

    exercise1_producer_tokens();
    exercise2_consumer_tokens();
    exercise3_benchmark_tokens();

    print_header("Phase 3 Complete");
    std::cout << "\nNext: Phase 4 -- Bulk operations for even higher throughput.\n"
              << "Run: build\\Release\\phase4_bulk.exe\n\n";

    return 0;
}
