//! Exercise 4: Data Race Detection
//!
//! A data race occurs when two threads access the same memory location
//! concurrently, at least one access is a write, and there is no
//! synchronization (happens-before relationship) between them.
//!
//! Data races are UB in Rust (and C/C++) — not just "you might read a stale
//! value" but "the compiler can do literally anything." The compiler assumes
//! no data races exist and optimizes accordingly: it may cache values in
//! registers, reorder writes, or eliminate "redundant" reads.
//!
//! Miri detects data races using a vector clock algorithm: each thread and
//! each memory location has a vector clock, and Miri checks that every access
//! is properly ordered relative to other threads' accesses.
//!
//! Use `-Zmiri-many-seeds=0..64` to test with 64 different thread schedules,
//! increasing the chance of triggering race-dependent bugs.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Demonstrate basic atomic operations — these are race-free by definition.
pub fn demonstrate_atomic_basics() {
    let counter = AtomicU64::new(0);

    // Atomic operations are indivisible — no data race possible.
    counter.fetch_add(1, Ordering::SeqCst);
    counter.fetch_add(1, Ordering::SeqCst);

    let value = counter.load(Ordering::SeqCst);
    println!("  Atomic counter: {value}");
}

// =============================================================================
// TODO(human): Fix data race with atomics
// =============================================================================

/// Fix a data race by using atomic operations.
///
/// The buggy version (in the test below) has two threads incrementing a shared
/// non-atomic variable — this is a textbook data race.
///
/// TODO(human): Implement this function that:
///
/// 1. Creates an `Arc<AtomicU64>` initialized to 0.
///    ```rust
///    let counter = Arc::new(AtomicU64::new(0));
///    ```
///
///    `Arc` (Atomic Reference Counting) is needed to share the counter between
///    threads. `AtomicU64` provides lock-free, race-free read-modify-write
///    operations.
///
/// 2. Spawns two threads, each incrementing the counter `iterations` times
///    using `fetch_add(1, Ordering::Relaxed)`.
///
///    Memory ordering explanation:
///    - `Ordering::Relaxed` — only guarantees atomicity (no torn reads/writes),
///      but no ordering relative to other memory operations. Sufficient here
///      because we only care about the counter's final value, not about ordering
///      other data relative to the counter.
///    - `Ordering::Release` / `Ordering::Acquire` — used when atomic operations
///      need to synchronize access to OTHER (non-atomic) data.
///    - `Ordering::SeqCst` — strongest ordering, provides a single total order
///      visible to all threads. Easiest to reason about but may be slower.
///
///    For a simple counter, `Relaxed` is correct and optimal.
///
/// 3. Joins both threads and returns the final counter value.
///
/// Why atomics fix the race:
/// - `fetch_add` is a single indivisible operation at the hardware level (uses
///   CPU instructions like `lock xadd` on x86 or `ldxr/stxr` loop on ARM).
/// - No thread can read a "half-written" value or lose an increment.
/// - Miri verifies that all accesses to the AtomicU64 are properly atomic.
///
/// The result should always be exactly `2 * iterations`.
pub fn fix_race_with_atomic(iterations: u64) -> u64 {
    // TODO(human): Implement using Arc<AtomicU64> and two threads.
    // Each thread calls fetch_add(1, Ordering::Relaxed) `iterations` times.
    // Join both threads and return the final counter value.
    todo!()
}

// =============================================================================
// TODO(human): Fix data race with Mutex
// =============================================================================

/// Fix a data race by using a Mutex.
///
/// TODO(human): Implement this function that:
///
/// 1. Creates an `Arc<Mutex<u64>>` initialized to 0.
///    ```rust
///    let counter = Arc::new(Mutex::new(0_u64));
///    ```
///
/// 2. Spawns two threads, each incrementing the counter `iterations` times:
///    ```rust
///    let mut guard = counter.lock().unwrap();
///    *guard += 1;
///    // guard is dropped here, releasing the lock
///    ```
///
///    IMPORTANT: The lock must be acquired and released for EACH increment,
///    not held for the entire loop. Holding it for the entire loop would
///    serialize the threads completely (correct but defeats the purpose).
///
/// 3. Joins both threads and returns the final counter value.
///
/// Why Mutex fixes the race:
/// - `Mutex::lock()` establishes a happens-before relationship: when thread B
///   acquires the lock after thread A released it, thread B is guaranteed to
///   see all writes thread A made before releasing.
/// - The lock/unlock creates mutual exclusion: only one thread can access the
///   data at a time.
/// - Miri verifies the happens-before ordering via vector clocks.
///
/// Mutex vs Atomic trade-off:
/// - Atomics: lock-free, lower overhead, but limited to simple operations
///   (add, compare-swap, load, store). Can't protect multi-step operations.
/// - Mutex: can protect arbitrarily complex critical sections, but higher
///   overhead (OS futex or spin-lock) and risk of deadlock if not careful.
///
/// For a simple counter, atomics are preferred. For complex state updates
/// (e.g., modifying a HashMap), Mutex is the right choice.
pub fn fix_race_with_mutex(iterations: u64) -> u64 {
    // TODO(human): Implement using Arc<Mutex<u64>> and two threads.
    // Each thread locks, increments, and unlocks for each iteration.
    // Join both threads and return the final counter value.
    todo!()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Intentionally buggy test (Miri WILL flag this) ----

    /// INTENTIONAL BUG: Classic data race — two threads writing to shared memory.
    ///
    /// Miri error will say:
    ///   "Data race detected between (1) Write on thread `<unnamed>` and
    ///    (2) Write on thread `<unnamed>` at alloc123..."
    ///
    /// Why this is UB (not just "maybe wrong"):
    /// - The Rust memory model (based on C++11) says concurrent non-atomic
    ///   access where at least one is a write is UNDEFINED BEHAVIOR.
    /// - The compiler assumes no data races and may:
    ///   a) Cache the value in a register (never re-reading from memory)
    ///   b) Reorder writes across the "unsynchronized boundary"
    ///   c) Optimize away "redundant" reads (if it assumes single-threaded access)
    /// - On x86, this might "work" because x86 has a strong memory model (TSO).
    ///   On ARM or RISC-V, you'd see incorrect values due to weak ordering.
    ///
    /// Note: This test uses `UnsafeCell` to create shared mutable state without
    /// synchronization — the exact anti-pattern Miri catches.
    #[test]
    #[ignore] // Remove #[ignore] to see Miri flag this as a data race
    fn ex4_buggy_data_race() {
        use std::cell::UnsafeCell;

        // UnsafeCell allows mutation through shared references — but we're
        // responsible for synchronization. Here we deliberately omit it.
        struct UnsafeCounter(UnsafeCell<u64>);
        unsafe impl Sync for UnsafeCounter {}

        let counter = Arc::new(UnsafeCounter(UnsafeCell::new(0)));

        let c1 = Arc::clone(&counter);
        let t1 = std::thread::spawn(move || {
            for _ in 0..100 {
                // UB: writing without synchronization!
                unsafe { *c1.0.get() += 1; }
            }
        });

        let c2 = Arc::clone(&counter);
        let t2 = std::thread::spawn(move || {
            for _ in 0..100 {
                // UB: concurrent write without synchronization!
                unsafe { *c2.0.get() += 1; }
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();
    }

    // ---- Fix verification tests ----

    #[test]
    fn ex4_fix_race_with_atomic_small() {
        let iterations = if cfg!(miri) { 100 } else { 10_000 };
        let result = fix_race_with_atomic(iterations);
        assert_eq!(result, 2 * iterations);
    }

    #[test]
    fn ex4_fix_race_with_mutex_small() {
        let iterations = if cfg!(miri) { 100 } else { 10_000 };
        let result = fix_race_with_mutex(iterations);
        assert_eq!(result, 2 * iterations);
    }

    #[test]
    fn ex4_atomic_ordering_demo() {
        // Demonstrate that different orderings work for a simple counter.
        let counter = AtomicU64::new(0);

        // All of these are correct for a simple counter:
        counter.fetch_add(1, Ordering::Relaxed);
        counter.fetch_add(1, Ordering::Release);
        counter.fetch_add(1, Ordering::SeqCst);

        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }
}
