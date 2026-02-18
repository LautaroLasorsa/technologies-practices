//! Exercise 2: CachePadded & Backoff
//!
//! This exercise demonstrates two essential low-level concurrency utilities:
//! - CachePadded: prevents false sharing between adjacent atomic variables
//! - Backoff: implements exponential backoff for spin-wait loops
//!
//! Both are critical for writing high-performance concurrent code. False sharing
//! is one of the most common hidden performance killers, and proper backoff
//! strategy can make the difference between a spin-lock that works and one that
//! brings the system to its knees under contention.

use crossbeam_utils::{Backoff, CachePadded};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Demonstrates the performance impact of false sharing and how CachePadded fixes it.
///
/// False sharing occurs when two threads modify different variables that reside
/// on the **same CPU cache line** (typically 64 bytes). Even though the threads
/// are accessing different memory locations, the CPU's cache coherence protocol
/// (MESI/MOESI) treats the entire cache line as a unit: when Thread A writes to
/// its variable, the CPU invalidates Thread B's cached copy of the entire line,
/// forcing Thread B to re-fetch it from a higher cache level or main memory.
///
/// This ping-pong invalidation can make concurrent code 10-50x slower than expected.
///
/// `CachePadded<T>` solves this by padding `T` to 64 bytes (one cache line),
/// ensuring each value occupies its own cache line. The trade-off is wasted memory
/// (e.g., an AtomicU64 is 8 bytes but CachePadded<AtomicU64> is 64 bytes), which
/// is negligible for a few hot variables.
pub fn benchmark_false_sharing() {
    let iterations: u64 = 5_000_000;

    // TODO(human): Benchmark false sharing vs CachePadded.
    //
    // This exercise makes false sharing tangible — you'll see a measurable performance
    // difference on real hardware. The effect is most dramatic on multi-socket systems
    // but is observable even on consumer CPUs with multiple cores.
    //
    // Part A: False sharing (adjacent counters on the same cache line)
    //
    // 1. Define a struct `AdjacentCounters`:
    //    ```
    //    struct AdjacentCounters {
    //        counter_a: AtomicU64,  // 8 bytes
    //        counter_b: AtomicU64,  // 8 bytes — likely on the SAME cache line as counter_a
    //    }
    //    ```
    //    Since AtomicU64 is 8 bytes and a cache line is 64 bytes, both counters fit
    //    on one cache line. When Thread A writes counter_a, the CPU invalidates Thread B's
    //    cache line containing counter_b, even though counter_b wasn't modified.
    //
    // 2. Wrap in Arc, spawn 2 threads:
    //    - Thread 1: increments counter_a `iterations` times using fetch_add(1, Relaxed)
    //    - Thread 2: increments counter_b `iterations` times using fetch_add(1, Relaxed)
    //    Relaxed ordering is fine here because we don't need happens-before relationships
    //    between the two counters — we only care about the final count of each.
    //
    // 3. Measure elapsed time with `Instant::now()` and `elapsed()`.
    //
    // Part B: No false sharing (CachePadded counters on separate cache lines)
    //
    // 4. Define a struct `PaddedCounters`:
    //    ```
    //    struct PaddedCounters {
    //        counter_a: CachePadded<AtomicU64>,  // padded to 64 bytes
    //        counter_b: CachePadded<AtomicU64>,  // on its OWN cache line
    //    }
    //    ```
    //    CachePadded ensures each AtomicU64 occupies a full 64-byte cache line.
    //    Now Thread A's writes to counter_a do NOT invalidate Thread B's cache line.
    //
    // 5. Repeat the same benchmark. Measure elapsed time.
    //
    // 6. Print both times and the speedup ratio:
    //    ```
    //    println!("  Adjacent (false sharing): {:?}", adjacent_time);
    //    println!("  CachePadded (no sharing): {:?}", padded_time);
    //    println!("  Speedup: {:.2}x", adjacent_time.as_nanos() as f64 / padded_time.as_nanos() as f64);
    //    ```
    //
    // Expected result: CachePadded version is 2-10x faster depending on your CPU.
    // On some systems with aggressive prefetching, the effect may be smaller.
    // The effect is most visible in tight loops with no other work — which is exactly
    // the scenario in real-world hot paths (atomic counters, ring buffer indices, etc.).
    //
    // Why Relaxed ordering: We use Ordering::Relaxed because each thread only writes
    // to its own counter. Relaxed provides atomicity (no torn reads/writes) but no
    // cross-thread ordering guarantees. Since the two counters are independent, no
    // ordering is needed. Using stronger orderings (Acquire/Release) would add
    // unnecessary memory fence instructions.
    //
    // Note: Use `crossbeam::scope` for spawning threads so you can borrow the Arc
    // without needing 'static. Or use std::thread::spawn with Arc::clone.

    todo!("Exercise 2a: Benchmark false sharing vs CachePadded")
}

/// Demonstrates spin-wait with exponential Backoff vs naive spinning.
///
/// When a thread needs to wait for a condition (e.g., a flag set by another thread),
/// there are three main strategies:
///
/// 1. **Busy-wait (naive spin)**: `while !flag.load() {}` — burns 100% CPU, floods
///    the memory bus with load requests, and steals execution resources from the
///    thread that needs to SET the flag. Worst option under contention.
///
/// 2. **Backoff spin**: Uses `std::hint::spin_loop()` (PAUSE instruction on x86)
///    for the first few iterations, then falls back to `thread::yield_now()`.
///    Reduces cache-line traffic and gives the OS a chance to schedule the setter thread.
///
/// 3. **Blocking wait**: `thread::park()` or condvar — zero CPU usage while waiting
///    but higher latency to wake up (requires OS scheduler involvement).
///
/// `Backoff` from crossbeam-utils implements strategy #2 with a clean API:
/// - `backoff.spin()`: Returns true while in the spinning phase (PAUSE hint)
/// - `backoff.snooze()`: Always yields (PAUSE if spinning, yield_now if past threshold)
/// - `backoff.is_completed()`: True when past the spin threshold (time to try parking or yielding)
pub fn spin_with_backoff() {
    // TODO(human): Compare naive spinning vs Backoff-assisted spinning.
    //
    // This exercise demonstrates that HOW you spin matters as much as WHETHER you spin.
    // In HFT systems, the spin strategy directly impacts tail latency.
    //
    // Setup:
    // 1. Create an `Arc<AtomicBool>` flag, initially false.
    //
    // Part A: Naive spin-wait
    //
    // 2. Clone the flag. Spawn a "setter" thread that:
    //    - Sleeps for 1ms (simulates some work before setting the flag)
    //    - Sets the flag to true: `flag.store(true, Ordering::Release)`
    //    Release ordering ensures that any writes BEFORE the store are visible
    //    to threads that observe the flag as true with Acquire ordering.
    //
    // 3. In the main thread, spin-wait naively:
    //    ```
    //    let start = Instant::now();
    //    let mut spin_count: u64 = 0;
    //    while !flag.load(Ordering::Acquire) {
    //        spin_count += 1;
    //        // No hint — hammers the memory bus
    //    }
    //    let naive_time = start.elapsed();
    //    ```
    //    Count how many iterations the spin loop executes. This number shows how
    //    much CPU time was wasted.
    //
    // 4. Join the setter thread. Print naive_time and spin_count.
    //
    // Part B: Backoff-assisted spin-wait
    //
    // 5. Reset the flag to false. Spawn another setter thread (same 1ms sleep).
    //
    // 6. In the main thread, spin-wait with Backoff:
    //    ```
    //    let start = Instant::now();
    //    let mut spin_count: u64 = 0;
    //    let backoff = Backoff::new();
    //    while !flag.load(Ordering::Acquire) {
    //        backoff.snooze();  // PAUSE hint → then yield_now
    //        spin_count += 1;
    //    }
    //    let backoff_time = start.elapsed();
    //    ```
    //    snooze() does two things:
    //    - First few calls: executes PAUSE instruction (x86) or equivalent hint,
    //      telling the CPU "I'm spin-waiting, reduce power and free execution resources"
    //    - After a threshold: calls thread::yield_now(), giving the OS scheduler
    //      a chance to run the setter thread on this core
    //
    // 7. Join the setter thread. Print backoff_time and spin_count.
    //
    // 8. Print comparison:
    //    ```
    //    println!("  Naive:   {:?} ({} spins)", naive_time, naive_spin_count);
    //    println!("  Backoff: {:?} ({} spins)", backoff_time, backoff_spin_count);
    //    ```
    //
    // Expected result:
    // - Naive spinning: millions of spin iterations, ~1ms elapsed
    // - Backoff spinning: far fewer spin iterations, ~1ms elapsed
    // - Both take ~1ms (bounded by the setter's sleep), but backoff uses
    //   dramatically fewer CPU cycles and causes less cache-line traffic.
    //
    // In real lock-free code, Backoff is used inside CAS retry loops:
    //   loop {
    //       let current = atomic.load(Acquire);
    //       let new = compute(current);
    //       match atomic.compare_exchange_weak(current, new, AcqRel, Acquire) {
    //           Ok(_) => break,
    //           Err(_) => backoff.spin(), // or snooze() for longer waits
    //       }
    //   }

    todo!("Exercise 2b: Implement spin-wait comparison with Backoff")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_false_sharing() {
        benchmark_false_sharing();
    }

    #[test]
    fn test_spin_with_backoff() {
        spin_with_backoff();
    }
}
