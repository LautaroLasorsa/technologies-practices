//! Exercise 7: Benchmarking Lock-Free vs Mutex
//!
//! Theory says lock-free structures win under high contention and short critical
//! sections. This exercise makes that claim empirically testable.
//!
//! We compare two implementations of a concurrent FIFO queue:
//! 1. `Mutex<VecDeque<T>>` — blocking, lock-based
//! 2. `SegQueue<T>` — lock-free (crossbeam)
//!
//! Each thread performs N push-then-pop cycles. We measure total wall-clock time
//! at different thread counts (1, 2, 4, 8) to observe how each implementation
//! scales with contention.
//!
//! Expected results:
//! - At 1 thread: Mutex is likely FASTER (no contention, just one atomic CAS for lock)
//! - At 2 threads: Roughly even, or Mutex still wins
//! - At 4+ threads: SegQueue starts to pull ahead as mutex contention increases
//! - At 8+ threads: SegQueue significantly faster (mutex convoy effect)
//!
//! IMPORTANT: Run benchmarks with `cargo run --release -- 7`
//! Debug builds include bounds checks, no inlining, and no optimizations,
//! which completely distort the results. Release mode is mandatory for meaningful numbers.

use crossbeam_queue::SegQueue;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Operations per thread in the benchmark.
const OPS_PER_THREAD: u64 = 500_000;

/// Benchmark: Mutex<VecDeque> — push/pop cycles with lock-based synchronization.
///
/// Each thread acquires the mutex, pushes one item, releases the mutex,
/// then acquires again to pop. This simulates a realistic pattern where
/// the critical section is SHORT (one push or pop), which is where the
/// mutex overhead is most significant relative to the actual work.
pub fn bench_mutex_queue(num_threads: usize) -> Duration {
    // TODO(human): Implement the Mutex<VecDeque> benchmark.
    //
    // This benchmark measures the cost of mutex-based synchronization under contention.
    // The critical section is deliberately short (one push or one pop) to make the
    // locking overhead proportionally significant.
    //
    // Steps:
    //
    // 1. Create the shared queue: `let queue = Arc::new(Mutex::new(VecDeque::<u64>::new()));`
    //
    // 2. Record start time: `let start = Instant::now();`
    //
    // 3. Spawn `num_threads` threads. Each thread performs `OPS_PER_THREAD` cycles:
    //    ```
    //    for i in 0..OPS_PER_THREAD {
    //        // Push
    //        {
    //            let mut q = queue.lock().unwrap();
    //            q.push_back(i);
    //        }  // mutex released here (MutexGuard dropped)
    //
    //        // Pop
    //        {
    //            let mut q = queue.lock().unwrap();
    //            q.pop_front();
    //        }
    //    }
    //    ```
    //    The braces around each lock/unlock are CRITICAL: they ensure the MutexGuard
    //    is dropped (releasing the lock) between push and pop. Without them, one thread
    //    would hold the lock for both operations, increasing contention.
    //
    // 4. Join all threads.
    //
    // 5. Return `start.elapsed()`.
    //
    // What to observe at different thread counts:
    // - 1 thread: Fast. Mutex fast-path is just one CAS (uncontended).
    // - 2 threads: Noticeable slowdown. Mutex bounces between cores.
    // - 4+ threads: Significant slowdown. Threads queue up waiting for the mutex.
    //   Under contention, the OS may context-switch waiting threads, adding microseconds
    //   of latency per operation. This is the "convoy effect."
    //
    // Why VecDeque (not Vec):
    // - VecDeque supports efficient O(1) push_back + pop_front (FIFO)
    // - Vec requires O(n) shift for pop_front, which would dominate the benchmark
    //   and hide the lock contention effects we're trying to measure

    todo!("Exercise 7a: Implement Mutex<VecDeque> benchmark")
}

/// Benchmark: SegQueue — push/pop cycles with lock-free synchronization.
///
/// Same logical operations as bench_mutex_queue, but using crossbeam's lock-free
/// SegQueue. No mutex, no blocking — threads make progress independently.
pub fn bench_lockfree_queue(num_threads: usize) -> Duration {
    // TODO(human): Implement the SegQueue benchmark.
    //
    // This benchmark measures the throughput of lock-free push/pop under contention.
    // The operations are identical to the Mutex benchmark for a fair comparison.
    //
    // Steps:
    //
    // 1. Create the shared queue: `let queue = Arc::new(SegQueue::<u64>::new());`
    //
    // 2. Record start time.
    //
    // 3. Spawn `num_threads` threads. Each thread performs `OPS_PER_THREAD` cycles:
    //    ```
    //    for i in 0..OPS_PER_THREAD {
    //        queue.push(i);   // Never blocks, never fails
    //        queue.pop();     // Returns None if empty (we don't care about the value)
    //    }
    //    ```
    //    Note: No lock/unlock overhead. push() and pop() are independent atomic
    //    operations. Under contention, a failed CAS causes a retry (not a context switch).
    //
    // 4. Join all threads.
    //
    // 5. Return `start.elapsed()`.
    //
    // What to observe at different thread counts:
    // - 1 thread: Slightly slower than Mutex (lock-free has higher constant overhead
    //   due to linked-list traversal and CAS retries, even with no contention).
    // - 2 threads: Close to Mutex or slightly faster.
    // - 4+ threads: SegQueue scales much better. Failed CAS causes a cheap retry
    //   (a few nanoseconds), while failed mutex acquisition causes a context switch
    //   (microseconds). This difference compounds at high thread counts.
    //
    // Why SegQueue (not ArrayQueue):
    // - SegQueue is unbounded, matching VecDeque's unbounded nature for fairness.
    // - ArrayQueue would add backpressure concerns, complicating the comparison.
    // - SegQueue's segments amortize allocation cost, so per-operation allocation
    //   overhead is minimal.

    todo!("Exercise 7b: Implement SegQueue benchmark")
}

/// Run both benchmarks at multiple thread counts and print a comparison table.
pub fn run_comparison() {
    let thread_counts = [1, 2, 4, 8];

    // TODO(human): Implement the comparison runner.
    //
    // This ties together the two benchmarks and presents results in a clear table.
    // The goal is to develop empirical intuition for the lock-free vs lock-based
    // trade-off — not just "lock-free is faster" but "lock-free is faster WHEN..."
    //
    // Steps:
    //
    // 1. Print table header:
    //    ```
    //    println!("  {:>8} {:>14} {:>14} {:>10}", "Threads", "Mutex", "SegQueue", "Winner");
    //    println!("  {}", "-".repeat(50));
    //    ```
    //
    // 2. For each thread count in [1, 2, 4, 8]:
    //    a) Run bench_mutex_queue(n) → mutex_time
    //    b) Run bench_lockfree_queue(n) → lockfree_time
    //    c) Determine winner and speedup ratio
    //    d) Print row:
    //       ```
    //       let winner = if mutex_time < lockfree_time { "Mutex" } else { "SegQueue" };
    //       let ratio = mutex_time.as_nanos() as f64 / lockfree_time.as_nanos() as f64;
    //       println!(
    //           "  {:>8} {:>14?} {:>14?} {:>10} ({:.2}x)",
    //           n, mutex_time, lockfree_time, winner, ratio
    //       );
    //       ```
    //
    // 3. Print summary observations:
    //    - At what thread count did SegQueue start winning?
    //    - What's the maximum speedup observed?
    //
    // Example output (your numbers will vary):
    //   Threads          Mutex       SegQueue     Winner
    //   --------------------------------------------------
    //         1        120ms         145ms      Mutex (0.83x)
    //         2        280ms         190ms   SegQueue (1.47x)
    //         4        650ms         230ms   SegQueue (2.83x)
    //         8       1450ms         310ms   SegQueue (4.68x)
    //
    // Key insight: The crossover point depends on:
    // - CPU architecture (number of cores, cache hierarchy)
    // - Critical section length (shorter → lock-free wins sooner)
    // - Operation cost (cheap ops → lock overhead dominates sooner)
    // - OS scheduler behavior (context switch cost varies by OS)
    //
    // IMPORTANT: Always run with `--release`. In debug mode:
    // - Rust inserts overflow checks on every arithmetic operation
    // - No inlining → every function call has stack frame overhead
    // - No loop unrolling, no SIMD, no constant propagation
    // - Results will be 10-50x slower AND the relative comparison distorted
    //   (debug overhead can mask the Mutex vs SegQueue difference)

    todo!("Exercise 7c: Implement comparison runner with table output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_mutex() {
        let time = bench_mutex_queue(2);
        assert!(time.as_secs() < 30, "Benchmark took too long: {:?}", time);
    }

    #[test]
    fn test_bench_lockfree() {
        let time = bench_lockfree_queue(2);
        assert!(time.as_secs() < 30, "Benchmark took too long: {:?}", time);
    }
}
