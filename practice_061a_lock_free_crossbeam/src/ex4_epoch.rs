//! Exercise 4: Epoch-Based Reclamation Fundamentals
//!
//! This is the conceptual core of the practice. Epoch-based reclamation (EBR) is
//! what makes crossbeam's lock-free data structures safe without a garbage collector.
//!
//! The problem: in a lock-free data structure, when Thread A removes a node,
//! Thread B might still be reading it (holding a pointer from a previous load).
//! Thread A cannot free the node immediately — it would create a use-after-free.
//! But Thread A also can't wait for Thread B (that would be blocking, not lock-free).
//!
//! EBR solution: Thread A marks the node as "garbage" tagged with the current epoch.
//! The node is only freed when the global epoch has advanced far enough that no
//! thread could possibly still hold a reference to it (all threads have "moved on").
//!
//! The crossbeam-epoch API:
//! - `pin()` → Guard: "I'm about to access the data structure. Don't free anything
//!   I might be looking at." Internally: marks thread as active, records current epoch.
//! - `guard.defer_destroy(shared)`: "This node is no longer in the data structure,
//!   but defer freeing it until it's safe." Internally: adds to epoch-stamped garbage bag.
//! - `drop(guard)`: "I'm done accessing. Safe to advance the epoch." Internally:
//!   marks thread as inactive, attempts epoch advancement, possibly frees old garbage.
//!
//! The three-epoch system ensures that garbage from 2 epochs ago is safe to free,
//! because all active threads have advanced past that epoch.

use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};
use crossbeam_utils::Backoff;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// Demonstrates the atomic swap pattern with epoch-based reclamation.
///
/// This is the fundamental pattern for updating a shared value in a lock-free structure:
/// 1. Pin the epoch (protect yourself from concurrent frees)
/// 2. Load the current value (get a Shared pointer valid for the Guard's lifetime)
/// 3. Create a new Owned value (heap-allocate the replacement)
/// 4. Swap atomically (CAS or store)
/// 5. Defer destruction of the old value (can't free immediately — other threads may have it)
/// 6. Drop the guard (unpin, allow epoch advancement)
pub fn atomic_swap_with_epoch() {
    // TODO(human): Implement concurrent atomic swaps using crossbeam-epoch.
    //
    // This exercise teaches the core epoch API: Atomic, Owned, Shared, Guard, pin(),
    // and defer_destroy(). Every lock-free data structure in crossbeam uses this pattern.
    //
    // Setup:
    //
    // 1. Create an Atomic<String> initialized with an Owned value:
    //    ```
    //    let shared_data = Arc::new(Atomic::new("initial".to_string()));
    //    ```
    //    Atomic<T> is like AtomicPtr<T> but integrated with the epoch GC.
    //    Atomic::new(value) wraps the value in an Owned (heap-allocated), then stores it.
    //
    // 2. Spawn 4 threads, each performing 100 swap operations:
    //
    //    For each swap:
    //    a) Pin the epoch:
    //       ```
    //       let guard = epoch::pin();
    //       ```
    //       This declares: "I'm accessing epoch-managed data. Don't free anything
    //       that was reachable when I pinned." The Guard's lifetime constrains all
    //       Shared pointers loaded during this critical section.
    //
    //    b) Load the current value:
    //       ```
    //       let current: Shared<'_, String> = shared_data.load(Ordering::Acquire, &guard);
    //       ```
    //       Acquire ordering ensures we see all writes that happened before the
    //       store that published this pointer. The returned Shared<'_, String> is
    //       valid for the lifetime of `guard` — Rust's borrow checker enforces this.
    //       If you try to use `current` after dropping `guard`, it won't compile.
    //
    //    c) Read the old value (for logging):
    //       ```
    //       if !current.is_null() {
    //           let old_str: &String = unsafe { current.deref() };
    //           // ... use old_str ...
    //       }
    //       ```
    //       deref() is unsafe because the compiler can't prove the pointer is valid.
    //       But we KNOW it is because: (1) we loaded it from an Atomic that was
    //       initialized with a valid Owned, and (2) we're pinned, so no concurrent
    //       thread can free it during our access.
    //
    //    d) Create a new Owned value and swap it in:
    //       ```
    //       let new_val = Owned::new(format!("thread-{}-iter-{}", thread_id, i));
    //       let old = shared_data.swap(new_val, Ordering::AcqRel, &guard);
    //       ```
    //       swap() atomically replaces the pointer and returns the old Shared.
    //       AcqRel ordering: Release semantics on the store (publishes our new value),
    //       Acquire semantics on the load (we see the old value fully).
    //
    //    e) Defer destruction of the old value:
    //       ```
    //       if !old.is_null() {
    //           unsafe { guard.defer_destroy(old); }
    //       }
    //       ```
    //       defer_destroy() is unsafe because we're asserting: "this pointer is no
    //       longer reachable from the data structure, and I'm the one who removed it."
    //       The epoch system will free the memory once it's safe (all threads have
    //       advanced past this epoch).
    //
    //       WITHOUT defer_destroy(): memory leak — the old value is never freed.
    //       WITH immediate drop instead of defer: use-after-free — another thread
    //       might be reading the old value right now.
    //       defer_destroy() is the ONLY correct approach for lock-free removal.
    //
    //    f) The guard is dropped at end of scope (or explicitly with `drop(guard)`).
    //       This unpins the thread, allowing epoch advancement.
    //
    // 3. After all threads complete, pin one more time and load the final value:
    //    ```
    //    let guard = epoch::pin();
    //    let final_val = shared_data.load(Ordering::Acquire, &guard);
    //    if !final_val.is_null() {
    //        println!("  Final value: {}", unsafe { final_val.deref() });
    //    }
    //    ```
    //
    // 4. Clean up: swap in a null and defer_destroy the last value, then drop the Atomic.
    //    Or use `unsafe { drop(shared_data.into_owned()) }` to take ownership of
    //    the final value.
    //
    // Key insight: The epoch system's power is that pinning is CHEAP (one atomic load
    // + one atomic store of the thread-local epoch). Compare with hazard pointers which
    // require one atomic store PER pointer accessed. For data structures where you load
    // many pointers per operation (trees, skip lists), EBR is much cheaper.

    todo!("Exercise 4a: Implement concurrent atomic swaps with epoch-based reclamation")
}

/// Build a lock-free counter using CAS (compare_exchange) retry loop.
///
/// This exercise strips away the data structure complexity and focuses on the
/// fundamental lock-free programming pattern: the CAS retry loop.
///
/// Pattern:
///   loop {
///       let current = load();
///       let desired = compute(current);
///       if CAS(current, desired) succeeds → done
///       else → another thread won the race, retry with new current
///   }
///
/// This is "optimistic concurrency": you assume no conflict, do the work, and
/// only retry if you discover (via CAS failure) that someone else modified the
/// value concurrently. Under low contention, CAS succeeds on the first try.
/// Under high contention, multiple retries may be needed — but at least one
/// thread always makes progress (lock-free guarantee).
pub fn concurrent_atomic_counter() {
    let num_threads = 8;
    let increments_per_thread = 10_000;

    // TODO(human): Implement a lock-free counter using Atomic<u64> and compare_exchange.
    //
    // NOTE: In practice, you'd use AtomicU64::fetch_add() for a simple counter.
    // This exercise uses the general CAS pattern to teach the retry loop that you'll
    // need for more complex lock-free operations (Treiber stack push/pop, etc.).
    //
    // Steps:
    //
    // 1. Create a shared counter:
    //    ```
    //    let counter = Arc::new(Atomic::<u64>::null());
    //    // Initialize with 0:
    //    counter.store(Owned::new(0u64), Ordering::Release);
    //    ```
    //    We use Atomic<u64> (crossbeam-epoch's atomic pointer to a heap-allocated u64)
    //    rather than AtomicU64 to practice the epoch API. In real code, AtomicU64
    //    would be correct and faster for a simple counter.
    //
    // 2. Spawn `num_threads` threads. Each thread increments the counter
    //    `increments_per_thread` times using this CAS loop:
    //    ```
    //    let guard = epoch::pin();
    //    let backoff = Backoff::new();
    //    loop {
    //        // Load current value
    //        let current_shared = counter.load(Ordering::Acquire, &guard);
    //        let current_val: u64 = unsafe { *current_shared.deref() };
    //
    //        // Compute desired value
    //        let new_val = current_val + 1;
    //        let new_owned = Owned::new(new_val);
    //
    //        // Attempt CAS: replace current pointer with new pointer
    //        match counter.compare_exchange(
    //            current_shared,     // expected: the pointer we loaded
    //            new_owned,          // desired: our new allocation
    //            Ordering::AcqRel,   // success ordering: publish new value
    //            Ordering::Acquire,  // failure ordering: re-read current value
    //            &guard,
    //        ) {
    //            Ok(old_shared) => {
    //                // CAS succeeded — we won the race!
    //                // Defer destruction of the old value
    //                unsafe { guard.defer_destroy(old_shared); }
    //                backoff.reset();
    //                break;  // move to next increment
    //            }
    //            Err(err) => {
    //                // CAS failed — another thread modified the counter.
    //                // err.new is our Owned allocation, returned to us (not lost).
    //                // err.current is the new current pointer.
    //                // We must retry with the updated value.
    //                // The Owned in err.new is dropped here (or we could reuse it).
    //                backoff.spin();
    //                continue;  // retry the CAS loop
    //            }
    //        }
    //    }
    //    drop(guard);  // unpin after each increment (allows GC to proceed)
    //    ```
    //
    //    IMPORTANT: Pin and unpin inside the increment loop (not outside all increments).
    //    If you pin once and stay pinned for 10,000 increments, you prevent epoch
    //    advancement for the entire duration → garbage accumulates → memory bloat.
    //    Pin for the shortest duration possible.
    //
    //    IMPORTANT: compare_exchange returns the Owned allocation back on failure
    //    (in err.new). This means the allocation is NOT leaked on CAS failure — it's
    //    dropped. In this exercise, we allocate a new Owned on each retry, which is
    //    wasteful. The Treiber stack (Exercise 5) shows how to reuse the Owned across
    //    retries by destructuring the error.
    //
    // 3. After all threads complete, read the final counter value:
    //    ```
    //    let guard = epoch::pin();
    //    let final_shared = counter.load(Ordering::Acquire, &guard);
    //    let final_val = unsafe { *final_shared.deref() };
    //    ```
    //
    // 4. Verify: `final_val == num_threads * increments_per_thread`
    //    Print: "  Final counter: {} (expected: {})"
    //
    // 5. Clean up the final value to avoid memory leaks:
    //    ```
    //    let guard = epoch::pin();
    //    let last = counter.swap(Shared::null(), Ordering::AcqRel, &guard);
    //    if !last.is_null() {
    //        unsafe { guard.defer_destroy(last); }
    //    }
    //    ```
    //
    // Key insight: The CAS retry loop is the fundamental building block of ALL
    // lock-free algorithms. Treiber stack push/pop, Michael-Scott queue enqueue/dequeue,
    // lock-free hash maps — they all follow this pattern:
    //   1. Read current state
    //   2. Compute desired state
    //   3. CAS(current → desired)
    //   4. On failure: retry

    todo!("Exercise 4b: Implement lock-free counter with CAS retry loop")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_swap() {
        atomic_swap_with_epoch();
    }

    #[test]
    fn test_concurrent_counter() {
        concurrent_atomic_counter();
    }
}
