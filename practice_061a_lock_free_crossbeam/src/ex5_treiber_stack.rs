//! Exercise 5: Treiber Stack — A Classic Lock-Free Data Structure
//!
//! The Treiber stack (R. Kent Treiber, 1986) is the simplest non-trivial lock-free
//! data structure. It's a singly-linked list where both push and pop operate on the
//! head (top) pointer using CAS.
//!
//! Structure:
//!   head → [Node(val=3, next)] → [Node(val=2, next)] → [Node(val=1, next)] → null
//!
//! Push algorithm:
//!   1. Allocate new node
//!   2. Set new_node.next = current head
//!   3. CAS(head, current_head, new_node)
//!   4. If CAS fails (another thread pushed/popped), goto step 2 (re-read head)
//!
//! Pop algorithm:
//!   1. Read current head
//!   2. If null → empty stack, return None
//!   3. Read head.next (the node that will become the new head)
//!   4. CAS(head, current_head, head.next)
//!   5. If CAS fails, goto step 1 (re-read head)
//!   6. Defer destruction of the popped node (epoch GC!)
//!   7. Return the popped value
//!
//! Why this works:
//! - CAS on the head pointer serializes concurrent push/pop operations
//! - If two threads push simultaneously, one wins the CAS and the other retries
//!   with the updated head — no data is lost
//! - Epoch-based reclamation ensures popped nodes aren't freed while other threads
//!   might be reading them
//!
//! Without epoch GC (the ABA problem):
//! - Thread A pops: reads head=X, head.next=Y, about to CAS(head, X, Y)
//! - Thread B pops X, pops Y, pushes Z, pushes a NEW node at address X (reuse!)
//! - Thread A's CAS(head, X, Y) succeeds (head still points to address X)
//! - But Y is no longer in the stack! The stack is corrupted.
//! - Epoch GC prevents this: X won't be freed (and thus can't be reused) while
//!   Thread A is still pinned and holding a reference to it.

use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};
use crossbeam_utils::Backoff;
use std::sync::atomic::Ordering;
use std::sync::Arc;

/// A node in the Treiber stack's linked list.
///
/// Each node holds a value and an atomic pointer to the next node.
/// The `next` pointer is `Atomic<Node<T>>` because pop() needs to read it
/// while other threads may be modifying the head.
struct Node<T> {
    value: T,
    next: Atomic<Node<T>>,
}

/// A lock-free stack using the Treiber algorithm with epoch-based reclamation.
///
/// The stack is parameterized over `T: Send` — values must be sendable between threads.
/// The `head` is an `Atomic<Node<T>>`, which starts as null (empty stack).
///
/// Thread safety:
/// - `push(&self, value: T)`: Takes `&self` (not `&mut self`), so multiple threads
///   can push concurrently without exclusive access. Lock-free via CAS.
/// - `pop(&self) -> Option<T>`: Also `&self`. Lock-free via CAS + epoch GC.
pub struct TreiberStack<T> {
    head: Atomic<Node<T>>,
}

impl<T> TreiberStack<T> {
    /// Creates a new empty stack.
    pub fn new() -> Self {
        TreiberStack {
            head: Atomic::null(),
        }
    }

    /// Pushes a value onto the top of the stack.
    ///
    /// This operation is lock-free: it uses a CAS retry loop to atomically update
    /// the head pointer. No mutex, no blocking — if CAS fails because another thread
    /// modified the head, we simply retry with the new head value.
    pub fn push(&self, value: T) {
        // TODO(human): Implement lock-free push.
        //
        // The push algorithm:
        //
        // 1. Allocate a new node on the heap:
        //    ```
        //    let mut new_node = Owned::new(Node {
        //        value,
        //        next: Atomic::null(),  // will be set to current head in the loop
        //    });
        //    ```
        //    Owned<T> is like Box<T> — it's a heap-allocated value that we own exclusively.
        //    It hasn't been published to the data structure yet, so no other thread can see it.
        //
        // 2. Pin the epoch:
        //    ```
        //    let guard = epoch::pin();
        //    ```
        //
        // 3. Enter a CAS retry loop:
        //    ```
        //    let backoff = Backoff::new();
        //    loop {
        //        // Load the current head pointer
        //        let head = self.head.load(Ordering::Relaxed, &guard);
        //
        //        // Set our new node's next pointer to the current head.
        //        // This makes our new node point to the rest of the stack.
        //        // We can safely write to new_node because we still own it (Owned).
        //        new_node.next.store(head, Ordering::Relaxed);
        //
        //        // Attempt to CAS the head from `head` to `new_node`.
        //        // If successful: head now points to our new node → push complete.
        //        // If failed: another thread changed head → we retry with new head.
        //        match self.head.compare_exchange_weak(
        //            head,        // expected: head we just loaded
        //            new_node,    // desired: our new node (Owned → converted to Shared)
        //            Ordering::Release,  // success: publish our new node to other threads
        //            Ordering::Relaxed,  // failure: we'll just retry
        //            &guard,
        //        ) {
        //            Ok(_) => return,  // Push succeeded!
        //            Err(err) => {
        //                // CAS failed. err.new gives us back our Owned node (not lost!).
        //                // This is critical: we REUSE the same allocation on retry,
        //                // avoiding unnecessary alloc/dealloc churn.
        //                new_node = err.new;
        //                backoff.spin();  // reduce contention
        //            }
        //        }
        //    }
        //    ```
        //
        // Why Relaxed on loads and `store(head, Relaxed)`:
        // - The head load uses Relaxed because we'll verify correctness via CAS anyway.
        //   If we load a stale value, CAS will fail and we'll retry.
        // - The next pointer store uses Relaxed because the new node isn't visible yet.
        //   The Release on CAS success ensures our next pointer write is visible when
        //   another thread loads the head with Acquire.
        //
        // Why compare_exchange_WEAK (not compare_exchange):
        // - compare_exchange_weak may spuriously fail (return Err even when the value
        //   matches). This is fine in a loop — we'll just retry.
        // - On ARM and other architectures with LL/SC (load-linked/store-conditional),
        //   weak is cheaper because it doesn't need to prevent spurious failures.
        // - Rule of thumb: use _weak in CAS loops, use _strong for single-shot CAS.
        //
        // Why the Owned is returned on CAS failure:
        // - compare_exchange takes ownership of the Owned (it might store it).
        // - If CAS fails, the Owned is returned in Err.new — we can reuse it.
        // - If CAS succeeds, the Owned is consumed (converted to Shared in the Atomic).
        //   We no longer own it — the data structure does.

        todo!("Exercise 5a: Implement TreiberStack::push()")
    }

    /// Pops the top value from the stack, or returns None if empty.
    ///
    /// This operation is lock-free: it uses a CAS retry loop to atomically update
    /// the head pointer. Epoch-based reclamation ensures the popped node is not freed
    /// until all threads that might hold a reference have unpinned.
    pub fn pop(&self) -> Option<T>
    where
        T: Clone, // Clone needed to extract the value before deferring destruction
    {
        // TODO(human): Implement lock-free pop with epoch-based reclamation.
        //
        // The pop algorithm:
        //
        // 1. Pin the epoch:
        //    ```
        //    let guard = epoch::pin();
        //    ```
        //
        // 2. Enter a CAS retry loop:
        //    ```
        //    let backoff = Backoff::new();
        //    loop {
        //        // Load the current head
        //        let head = self.head.load(Ordering::Acquire, &guard);
        //
        //        // Check for empty stack
        //        if head.is_null() {
        //            return None;  // Stack is empty
        //        }
        //
        //        // Read the head node's value and next pointer.
        //        // SAFETY: head is not null (checked above) and is valid because
        //        // we're pinned — the epoch system guarantees it won't be freed.
        //        let head_ref = unsafe { head.deref() };
        //        let next = head_ref.next.load(Ordering::Relaxed, &guard);
        //
        //        // Attempt to CAS head from current to next (skip the head node).
        //        // If successful: head now points to the second node → pop complete.
        //        // If failed: another thread modified the stack → retry.
        //        match self.head.compare_exchange_weak(
        //            head,    // expected: the head we loaded
        //            next,    // desired: the node after head (Shared, not Owned!)
        //            Ordering::Release,  // success: publish the new head
        //            Ordering::Relaxed,  // failure: retry
        //            &guard,
        //        ) {
        //            Ok(_) => {
        //                // Pop succeeded! We've unlinked `head` from the stack.
        //
        //                // Extract the value BEFORE deferring destruction.
        //                // Once defer_destroy is called, we must not access the node.
        //                let value = unsafe { head.deref().value.clone() };
        //
        //                // Defer destruction of the popped node.
        //                // We can't free it now — other threads may have loaded `head`
        //                // (via a stale read) and might be about to deref it.
        //                // The epoch system will free it once all pinned threads
        //                // that could have seen `head` have unpinned.
        //                unsafe { guard.defer_destroy(head); }
        //
        //                return Some(value);
        //            }
        //            Err(_) => {
        //                // CAS failed — another push/pop changed the head.
        //                // For pop, the Err doesn't return an Owned (we passed a Shared).
        //                backoff.spin();
        //            }
        //        }
        //    }
        //    ```
        //
        // Why we need Clone:
        // - We must extract the value BEFORE calling defer_destroy, because after that
        //   the node might be freed at any time (once we unpin).
        // - We can't move the value out of the node because the node is behind a
        //   shared reference (Shared → &Node). Moving requires ownership.
        // - Clone is the safe alternative. In a more optimized implementation, you
        //   might use ManuallyDrop or ptr::read to avoid the clone, but that requires
        //   very careful unsafe reasoning.
        //
        // Why Acquire on the head load:
        // - We need to see the full node contents (value, next pointer) that were
        //   written by the thread that pushed this node with Release ordering.
        // - Acquire-Release forms a "happens-before" relationship: the push's Release
        //   store guarantees that the node's fields are fully initialized before
        //   our Acquire load sees the pointer.
        //
        // What would go wrong WITHOUT epoch pinning:
        // - Thread A loads head = X, reads X.next = Y
        // - Thread B pops X, immediately frees X's memory
        // - Thread A tries to CAS(head, X, Y) — X is already freed! Use-after-free.
        // - With epoch pinning: Thread A is pinned, so X won't be freed until A unpins.

        todo!("Exercise 5b: Implement TreiberStack::pop()")
    }

    /// Returns true if the stack is empty.
    ///
    /// Note: This is a snapshot — the stack may become non-empty immediately after
    /// this returns true (another thread pushes). In lock-free programming, "is empty"
    /// is only meaningful as a hint, not a guarantee.
    pub fn is_empty(&self) -> bool {
        let guard = epoch::pin();
        self.head.load(Ordering::Relaxed, &guard).is_null()
    }
}

// SAFETY: TreiberStack is safe to share across threads if T is Send.
// The Atomic<Node<T>> uses atomic operations for all mutations.
unsafe impl<T: Send> Send for TreiberStack<T> {}
unsafe impl<T: Send> Sync for TreiberStack<T> {}

impl<T> Drop for TreiberStack<T> {
    /// Clean up remaining nodes when the stack is dropped.
    ///
    /// Since the stack owns all remaining nodes, and no other thread can access
    /// the stack after it's dropped (Rust's ownership system guarantees this),
    /// we can safely walk the list and free each node.
    fn drop(&mut self) {
        // We have &mut self, so no other thread has access.
        // We can use an unprotected load (no need to pin).
        unsafe {
            let guard = epoch::unprotected();
            let mut current = self.head.load(Ordering::Relaxed, guard);
            while !current.is_null() {
                let next = current.deref().next.load(Ordering::Relaxed, guard);
                // Convert Shared → Owned → drop it
                drop(current.into_owned());
                current = next;
            }
        }
    }
}

/// Runs the Treiber stack demo: basic operations + concurrent stress test.
pub fn run_treiber_demo() {
    basic_operations();
    stress_test_treiber();
}

/// Basic single-threaded test of push/pop.
fn basic_operations() {
    let stack = TreiberStack::new();

    // Push 1, 2, 3
    stack.push(1u64);
    stack.push(2);
    stack.push(3);

    // Pop should return 3, 2, 1 (LIFO)
    assert_eq!(stack.pop(), Some(3));
    assert_eq!(stack.pop(), Some(2));
    assert_eq!(stack.pop(), Some(1));
    assert_eq!(stack.pop(), None);

    println!("  Basic push/pop: OK (LIFO order verified)");
}

/// Concurrent stress test for the Treiber stack.
fn stress_test_treiber() {
    let num_threads = 8;
    let ops_per_thread = 5_000;

    // TODO(human): Implement a concurrent stress test for TreiberStack.
    //
    // The goal: verify that the Treiber stack is correct under high contention.
    // "Correct" means: every pushed value is popped exactly once, no values lost,
    // no values duplicated, no panics or crashes.
    //
    // Steps:
    //
    // 1. Create a shared TreiberStack<u64>: `let stack = Arc::new(TreiberStack::new());`
    //
    // 2. Phase 1 — Push: Spawn `num_threads` threads. Each thread pushes
    //    `ops_per_thread` unique values:
    //    ```
    //    for i in 0..ops_per_thread {
    //        let value = (thread_id * ops_per_thread + i) as u64;
    //        stack.push(value);
    //    }
    //    ```
    //    Total items pushed: num_threads * ops_per_thread
    //
    // 3. Phase 2 — Pop: Spawn `num_threads` threads. Each thread pops until it
    //    gets `ops_per_thread` values (or the stack is empty):
    //    - Collect popped values into a per-thread Vec
    //    - Use Backoff when pop() returns None (stack temporarily empty)
    //    - Return the collected Vec
    //
    //    Alternatively, use a simpler approach: pop in a single thread until empty,
    //    then verify count and uniqueness.
    //
    // 4. Collect ALL popped values from all threads. Verify:
    //    - Total count == num_threads * ops_per_thread
    //    - All values unique (insert into HashSet, check size)
    //    - All expected values present (the set should equal {0, 1, ..., total-1})
    //
    // 5. Print: "  Stress test: {} pushes, {} pops, all values accounted for: {}"
    //
    // Key insight: If your TreiberStack implementation has a bug (wrong ordering,
    // missing defer_destroy, incorrect CAS), this stress test will likely catch it
    // through: assertion failures, double-frees (SIGABRT), use-after-free (SIGSEGV),
    // or lost/duplicated values.

    todo!("Exercise 5c: Implement stress test for TreiberStack")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        basic_operations();
    }

    #[test]
    fn test_stress_test() {
        stress_test_treiber();
    }
}
