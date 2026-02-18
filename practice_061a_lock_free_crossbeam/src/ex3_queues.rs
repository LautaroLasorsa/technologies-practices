//! Exercise 3: Lock-Free Queues (SegQueue & ArrayQueue)
//!
//! crossbeam provides two lock-free MPMC queue implementations:
//!
//! - `SegQueue`: Unbounded, backed by linked segments. Grows dynamically.
//!   push() never blocks or fails. pop() returns None if empty.
//!   Internally uses a linked list of small arrays (segments) — when one fills up,
//!   a new segment is allocated. This amortizes allocation cost.
//!
//! - `ArrayQueue`: Bounded, backed by a pre-allocated contiguous array.
//!   push() returns Err if full. pop() returns None if empty.
//!   No allocations after construction — lower per-operation overhead than SegQueue
//!   but requires knowing the capacity upfront.
//!
//! Both are MPMC (multi-producer multi-consumer) and lock-free:
//! - Multiple threads can push simultaneously without locking
//! - Multiple threads can pop simultaneously without locking
//! - At least one thread always makes progress (lock-free guarantee)
//!
//! Compare with Mutex<VecDeque<T>>:
//! - Mutex blocks all other threads during push/pop
//! - SegQueue/ArrayQueue allow truly concurrent push AND pop operations

use crossbeam_queue::{ArrayQueue, SegQueue};
use crossbeam_utils::Backoff;
use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Multi-producer multi-consumer pattern with SegQueue.
///
/// SegQueue is ideal when:
/// - You don't know the maximum queue size upfront
/// - You want push() to always succeed (no backpressure needed)
/// - Allocation overhead is acceptable (one allocation per segment, amortized)
///
/// SegQueue is NOT ideal when:
/// - You need backpressure (bounded capacity)
/// - You're in a no-alloc environment (real-time systems, kernel code)
/// - You need very low per-operation latency (ArrayQueue has less overhead)
pub fn segqueue_mpmc() {
    let num_producers = 4;
    let num_consumers = 4;
    let items_per_producer = 1000;

    // TODO(human): Implement MPMC work distribution with SegQueue.
    //
    // This exercise teaches the fundamental MPMC queue pattern: multiple producers
    // feed work into a shared queue, multiple consumers drain it. This is the backbone
    // of work-stealing thread pools (like rayon) and concurrent task schedulers.
    //
    // Steps:
    //
    // 1. Create a shared SegQueue: `let queue = Arc::new(SegQueue::<u64>::new());`
    //    SegQueue is Send + Sync, so it can be shared via Arc across threads.
    //
    // 2. Create a shared `AtomicBool` flag for signaling completion:
    //    `let producers_done = Arc::new(AtomicBool::new(false));`
    //    This tells consumers when to stop trying to pop.
    //
    // 3. Create a results collector: `let results = Arc::new(SegQueue::<u64>::new());`
    //    Consumers push their consumed items here for verification.
    //
    // 4. Spawn `num_producers` producer threads. Each producer:
    //    - Pushes `items_per_producer` unique items to the queue
    //    - Items should be globally unique: e.g., `producer_id * items_per_producer + i`
    //    - Uses `queue.push(item)` — this NEVER blocks or fails (unbounded queue)
    //
    // 5. Spawn `num_consumers` consumer threads. Each consumer:
    //    - Loops: tries `queue.pop()`:
    //      - If `Some(item)` → push to results queue
    //      - If `None` → check if producers are done AND queue is still empty → break
    //    - IMPORTANT: Use Backoff when the queue is empty to avoid busy-waiting:
    //      ```
    //      let backoff = Backoff::new();
    //      loop {
    //          match queue.pop() {
    //              Some(item) => {
    //                  results.push(item);
    //                  backoff.reset();  // reset after successful pop
    //              }
    //              None => {
    //                  if producers_done.load(Ordering::Acquire) && queue.is_empty() {
    //                      break;
    //                  }
    //                  backoff.snooze();  // exponential backoff on empty queue
    //              }
    //          }
    //      }
    //      ```
    //    The Backoff is crucial: without it, consumers burn 100% CPU on an empty queue.
    //    reset() after a successful pop restarts the backoff sequence.
    //
    // 6. Join all producer threads, then set producers_done to true (Release ordering).
    //    Join all consumer threads.
    //
    // 7. Collect all results into a HashSet and verify:
    //    - `results.len() == num_producers * items_per_producer` (no items lost)
    //    - All items unique (no duplicates)
    //    ```
    //    let mut seen = HashSet::new();
    //    while let Some(item) = results_queue.pop() {
    //        assert!(seen.insert(item), "Duplicate item: {}", item);
    //    }
    //    assert_eq!(seen.len(), num_producers * items_per_producer);
    //    ```
    //
    // 8. Print: "  Produced: {}, Consumed: {}, All items accounted for: {}"
    //
    // Key insight: SegQueue's lock-free property means producers and consumers make
    // progress independently. A slow consumer doesn't block producers (unlike Mutex),
    // and a slow producer doesn't block consumers. The only synchronization is the
    // atomic CAS operations inside push/pop.

    todo!("Exercise 3a: Implement MPMC pattern with SegQueue")
}

/// Bounded queue with backpressure using ArrayQueue.
///
/// ArrayQueue is ideal when:
/// - You want bounded memory usage (capacity fixed at construction)
/// - You need backpressure (push fails when full → producer must slow down)
/// - You're in a latency-sensitive path (no allocations, cache-friendly array layout)
///
/// The key difference from SegQueue: `push()` returns `Err(value)` when full,
/// giving the caller the item back. The caller must decide: retry? drop? block?
/// This is **non-blocking backpressure** — the caller chooses the policy.
pub fn arrayqueue_bounded() {
    let capacity = 16; // Small capacity to demonstrate backpressure
    let num_producers = 4;
    let items_per_producer = 500;

    // TODO(human): Implement bounded MPMC with ArrayQueue and retry-with-backoff.
    //
    // This exercise teaches how to handle a bounded queue where push can fail.
    // In real systems, bounded queues are essential for:
    // - Preventing OOM under burst load (the queue doesn't grow unboundedly)
    // - Applying backpressure to upstream producers (slow consumers slow down producers)
    // - Ring buffers in HFT systems (fixed-size, no allocation, predictable latency)
    //
    // Steps:
    //
    // 1. Create a shared ArrayQueue: `let queue = Arc::new(ArrayQueue::<u64>::new(capacity));`
    //    The capacity is set at construction and cannot change. Internally it's a
    //    contiguous array with head/tail atomic indices — very cache-friendly.
    //
    // 2. Create a completion flag and results queue (same as Exercise 3a).
    //
    // 3. Spawn `num_producers` producer threads. Each producer:
    //    - For each item, attempts `queue.push(item)`:
    //      - If `Ok(())` → item enqueued, continue
    //      - If `Err(item)` → queue is full, retry with backoff:
    //        ```
    //        let backoff = Backoff::new();
    //        let mut item = initial_item;
    //        loop {
    //            match queue.push(item) {
    //                Ok(()) => break,
    //                Err(returned_item) => {
    //                    item = returned_item;
    //                    backoff.snooze();  // wait for consumers to make space
    //                }
    //            }
    //        }
    //        ```
    //    The Err(returned_item) pattern is important: ArrayQueue gives you the item BACK
    //    on failure, so you don't lose it. This is a zero-cost retry pattern.
    //
    // 4. Spawn `num_consumers` consumer threads (same as Exercise 3a, using Backoff).
    //
    // 5. Join all threads, verify completeness (same as Exercise 3a).
    //
    // 6. Print results and verify.
    //
    // Key insight: With capacity=16 and 4 producers each sending 500 items, the queue
    // will be full MOST of the time. Producers spend significant time in the retry loop.
    // This is intentional — it demonstrates that backpressure works and no items are lost
    // even under extreme contention on a tiny buffer.
    //
    // Performance note: ArrayQueue's fixed array layout means the head and tail of the
    // queue are always in the same cache lines. For small queues that fit in L1 cache,
    // this gives excellent performance. But beware: the head and tail indices themselves
    // can cause false sharing between producers (writing tail) and consumers (writing head).
    // crossbeam's internal implementation uses CachePadded on these indices.

    todo!("Exercise 3b: Implement bounded MPMC with ArrayQueue and retry-with-backoff")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segqueue_mpmc() {
        segqueue_mpmc();
    }

    #[test]
    fn test_arrayqueue_bounded() {
        arrayqueue_bounded();
    }
}
