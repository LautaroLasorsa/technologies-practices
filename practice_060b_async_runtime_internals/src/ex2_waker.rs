//! Exercise 2: Building a Waker
//!
//! The Waker is the mechanism that connects futures to executors. When a future
//! returns `Pending`, it stores the waker. When the underlying resource is ready,
//! something calls `waker.wake()`, which tells the executor to re-poll the task.
//!
//! This exercise teaches both the low-level (`RawWaker`) and high-level (`Wake` trait)
//! approaches to building wakers, then uses them to manually poll a future.
//!
//! Key concepts:
//! - `Waker` is a type-erased handle (like a trait object) backed by a vtable
//! - `RawWaker` + `RawWakerVTable` is the low-level unsafe API: 4 function pointers
//!   (clone, wake, wake_by_ref, drop) operating on a `*const ()` data pointer
//! - `Wake` trait (std::task::Wake) is the safe high-level API: implement wake()
//!   on an Arc<YourType> and convert it to a Waker
//! - The Waker design avoids requiring Arc — embedded systems can use raw pointers,
//!   indexes into arrays, or any scheme that fits their constraints

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Wake, Waker};

// We'll use the Countdown from Exercise 1 to test our wakers.
use crate::ex1_manual_future::Countdown;

// ─── Exercise 2a: RawWaker with Flag ────────────────────────────────

/// Build a Waker from scratch using the low-level `RawWaker` API.
///
/// The waker sets an `AtomicBool` flag to `true` when `wake()` is called.
/// This is the most primitive useful waker: the executor can check the flag
/// to know whether any future requested a re-poll.
///
/// # TODO(human): Implement the four vtable functions and the constructor
///
/// You need to implement:
///
/// 1. `flag_waker_clone(ptr: *const ()) -> RawWaker`
///    - The executor may clone the waker (e.g., to store it in multiple places).
///    - `ptr` points to an `Arc<AtomicBool>`. You need to:
///      a) Reconstruct the Arc from the raw pointer: `Arc::from_raw(ptr as *const AtomicBool)`
///      b) Clone the Arc (incrementing the reference count): `arc.clone()`
///      c) Forget the ORIGINAL arc (so we don't decrement the refcount): `std::mem::forget(arc)`
///      d) Convert the clone to a raw pointer: `Arc::into_raw(cloned) as *const ()`
///      e) Return `RawWaker::new(new_ptr, &FLAG_VTABLE)`
///
///    Why `from_raw` + `forget`? Because `from_raw` takes ownership (will drop when the Arc
///    goes out of scope). But we don't OWN this Arc — we're just borrowing the ptr to clone it.
///    So we `forget` the original to avoid a double-free.
///
/// 2. `flag_waker_wake(ptr: *const ())`
///    - Called when the future says "I'm ready to be polled again".
///    - Reconstruct the Arc: `Arc::from_raw(ptr as *const AtomicBool)`
///    - Set the flag: `arc.store(true, Ordering::SeqCst)`
///    - The Arc is dropped here (consumed), decrementing the refcount. This is correct:
///      `wake()` consumes the Waker, so we should consume the Arc too.
///
/// 3. `flag_waker_wake_by_ref(ptr: *const ())`
///    - Like `wake()` but does NOT consume the waker (it may be called again).
///    - Reconstruct the Arc, set the flag, then FORGET the Arc (don't drop it).
///    - This is the "borrow" version — we read the ptr but don't take ownership.
///
/// 4. `flag_waker_drop(ptr: *const ())`
///    - Called when the Waker is dropped (reference count cleanup).
///    - Reconstruct the Arc: `Arc::from_raw(ptr as *const AtomicBool)`
///    - Let it drop naturally, decrementing the refcount.
///
/// # Why is this unsafe?
///
/// You're manually managing Arc reference counts through raw pointers. If you:
/// - Forget to `forget()` in clone → double-free when both the original and clone drop
/// - Forget to `forget()` in wake_by_ref → use-after-free on subsequent wake calls
/// - Forget to reconstruct in drop → memory leak (Arc refcount never reaches 0)
///
/// This is why the `Wake` trait (Exercise 2b) exists — it handles all this automatically.

/// The vtable for our flag-based waker.
/// TODO(human): This is pre-filled — you implement the four functions it references.
static FLAG_VTABLE: RawWakerVTable = RawWakerVTable::new(
    flag_waker_clone,
    flag_waker_wake,
    flag_waker_wake_by_ref,
    flag_waker_drop,
);

unsafe fn flag_waker_clone(ptr: *const ()) -> RawWaker {
    // TODO(human): Clone the Arc<AtomicBool> behind the pointer.
    //
    // Steps:
    //   1. let arc = Arc::from_raw(ptr as *const AtomicBool);
    //   2. let cloned = arc.clone();
    //   3. std::mem::forget(arc);   // don't drop the original!
    //   4. let new_ptr = Arc::into_raw(cloned) as *const ();
    //   5. return RawWaker::new(new_ptr, &FLAG_VTABLE);
    //
    // Why forget? Arc::from_raw takes ownership. If we let `arc` drop here,
    // the refcount would decrement, potentially freeing the AtomicBool while
    // the original owner still holds a pointer to it. We only wanted to clone,
    // not consume the original.
    todo!("Exercise 2a: Implement flag_waker_clone")
}

unsafe fn flag_waker_wake(ptr: *const ()) {
    // TODO(human): Set the flag to true, consuming the Arc.
    //
    // Steps:
    //   1. let arc = Arc::from_raw(ptr as *const AtomicBool);
    //   2. arc.store(true, Ordering::SeqCst);
    //   // arc drops here — this is intentional. wake() consumes the Waker.
    //
    // The difference from wake_by_ref: here we DO let the Arc drop,
    // because wake() takes ownership of the Waker (it's a move, not a borrow).
    todo!("Exercise 2a: Implement flag_waker_wake")
}

unsafe fn flag_waker_wake_by_ref(ptr: *const ()) {
    // TODO(human): Set the flag to true WITHOUT consuming the Arc.
    //
    // Steps:
    //   1. let arc = Arc::from_raw(ptr as *const AtomicBool);
    //   2. arc.store(true, Ordering::SeqCst);
    //   3. std::mem::forget(arc);   // don't drop — we're borrowing, not consuming
    //
    // wake_by_ref is called when the executor calls waker.wake_by_ref()
    // (keeping the Waker alive for future use). We must NOT decrement the refcount.
    todo!("Exercise 2a: Implement flag_waker_wake_by_ref")
}

unsafe fn flag_waker_drop(ptr: *const ()) {
    // TODO(human): Drop the Arc, decrementing the refcount.
    //
    // Steps:
    //   1. let _arc = Arc::from_raw(ptr as *const AtomicBool);
    //   // _arc drops here, decrementing refcount. If refcount reaches 0,
    //   // the AtomicBool is freed.
    //
    // This is called when the Waker (or a clone of it) is dropped.
    // It's the cleanup counterpart to clone.
    todo!("Exercise 2a: Implement flag_waker_drop")
}

/// Create a Waker backed by an `Arc<AtomicBool>` flag.
///
/// When `wake()` is called, the flag is set to `true`.
/// The caller can check the flag to know whether re-polling is needed.
pub fn create_flag_waker(flag: &Arc<AtomicBool>) -> Waker {
    // Convert the Arc to a raw pointer for the RawWaker.
    // Arc::into_raw increments the refcount (the raw pointer "owns" one count).
    let ptr = Arc::into_raw(Arc::clone(flag)) as *const ();
    let raw = RawWaker::new(ptr, &FLAG_VTABLE);
    // SAFETY: The vtable functions correctly manage the Arc refcount.
    // clone increments, wake/drop decrement, wake_by_ref leaves it unchanged.
    unsafe { Waker::from_raw(raw) }
}

// ─── Exercise 2b: Wake Trait (Safe API) ─────────────────────────────

/// A task-like struct that tracks whether it has been woken.
///
/// This demonstrates the `Wake` trait — the safe alternative to `RawWaker`.
/// Instead of manually managing vtables and raw pointers, you implement
/// `Wake` on a type and the standard library handles the plumbing.
///
/// # TODO(human): Implement the `Wake` trait for `TaskFlag`
///
/// The `Wake` trait has one required method:
///
/// ```ignore
/// fn wake(self: Arc<Self>) {
///     // Called when waker.wake() is invoked. `self` is consumed.
/// }
/// ```
///
/// And one optional method (has a default implementation):
///
/// ```ignore
/// fn wake_by_ref(self: &Arc<Self>) {
///     // Called when waker.wake_by_ref() is invoked. `self` is borrowed.
///     // Default: clones the Arc and calls wake(). Override for efficiency.
/// }
/// ```
///
/// # What to implement
///
/// In `wake()`: set `self.woken` to `true` using `store(true, Ordering::SeqCst)`.
/// Optionally override `wake_by_ref()` to do the same without consuming.
///
/// # Why this is better than RawWaker
///
/// - No unsafe code needed.
/// - No manual refcount management.
/// - No vtable construction.
/// - The compiler ensures correctness.
/// - Use RawWaker only for no_std / embedded where Arc is unavailable.
pub struct TaskFlag {
    pub woken: AtomicBool,
}

impl TaskFlag {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            woken: AtomicBool::new(false),
        })
    }

    pub fn is_woken(&self) -> bool {
        self.woken.load(Ordering::SeqCst)
    }

    pub fn reset(&self) {
        self.woken.store(false, Ordering::SeqCst);
    }
}

impl Wake for TaskFlag {
    /// Called when `waker.wake()` is invoked (consumes the waker).
    ///
    /// # TODO(human): Set `self.woken` to true.
    ///
    /// Use `self.woken.store(true, Ordering::SeqCst)`.
    ///
    /// In a real executor, this method would push the task back onto the
    /// ready queue. Here we just set a flag for demonstration.
    fn wake(self: Arc<Self>) {
        // TODO(human): Set woken flag to true.
        //
        // This is the entire implementation — one line:
        //   self.woken.store(true, Ordering::SeqCst);
        //
        // In production, this would do something like:
        //   ready_queue.push(task_id);
        //   unpark_worker_thread();
        todo!("Exercise 2b: Implement Wake::wake for TaskFlag")
    }
}

// ─── Exercise 2c: Manual Polling with Custom Waker ──────────────────

/// Manually poll a `Countdown` future using the flag waker from Exercise 2a.
///
/// # TODO(human): Implement the polling loop
///
/// This function demonstrates the executor's inner loop:
///
/// 1. Create an `Arc<AtomicBool>` flag.
/// 2. Create a `Waker` from it using `create_flag_waker(&flag)`.
/// 3. Create a `Context` from the waker: `Context::from_waker(&waker)`.
/// 4. In a loop:
///    a) Reset the flag to `false`.
///    b) Poll the future: `Pin::new(&mut future).poll(&mut cx)`.
///    c) If `Ready(output)` — return the output.
///    d) If `Pending` — check that `flag` is `true` (the future called `wake_by_ref`).
///       If the flag is false, panic: the future violated the contract!
///    e) Continue the loop (re-poll).
///
/// # What this teaches
///
/// This is a **correct** executor loop for a single future. It verifies the
/// waker contract: every `Pending` must be accompanied by a `wake()` call.
/// Production executors skip this check for performance, but it's invaluable
/// for debugging.
pub fn poll_with_flag_waker(mut future: impl Future<Output = ()> + Unpin) {
    // TODO(human): Implement the manual polling loop with flag-based waker.
    //
    // let flag = Arc::new(AtomicBool::new(false));
    // let waker = create_flag_waker(&flag);
    // let mut cx = Context::from_waker(&waker);
    //
    // loop {
    //     flag.store(false, Ordering::SeqCst);
    //     match Pin::new(&mut future).poll(&mut cx) {
    //         Poll::Ready(()) => { println!("  Future completed!"); return; }
    //         Poll::Pending => {
    //             assert!(flag.load(Ordering::SeqCst), "Future returned Pending without waking!");
    //             println!("  Pending (waker was called, re-polling...)");
    //         }
    //     }
    // }
    todo!("Exercise 2c: Implement manual polling loop with flag waker")
}

/// Manually poll a `Countdown` using the `Wake` trait waker from Exercise 2b.
///
/// # TODO(human): Same logic as above, but using `TaskFlag` and `Wake`
///
/// The difference is how you create the Waker:
/// - Instead of `create_flag_waker()`, use `Waker::from(task_flag.clone())`
///   where `task_flag: Arc<TaskFlag>`.
/// - To check if woken: `task_flag.is_woken()`.
/// - To reset: `task_flag.reset()`.
///
/// This demonstrates that the `Wake` trait produces the same result as
/// the manual `RawWaker` approach — with zero unsafe code.
pub fn poll_with_wake_trait(mut future: impl Future<Output = ()> + Unpin) {
    // TODO(human): Implement the manual polling loop with Wake-trait waker.
    //
    // let task_flag = TaskFlag::new();
    // let waker = Waker::from(task_flag.clone());
    // let mut cx = Context::from_waker(&waker);
    //
    // loop {
    //     task_flag.reset();
    //     match Pin::new(&mut future).poll(&mut cx) {
    //         Poll::Ready(()) => { println!("  Future completed!"); return; }
    //         Poll::Pending => {
    //             assert!(task_flag.is_woken(), "Future returned Pending without waking!");
    //             println!("  Pending (waker was called, re-polling...)");
    //         }
    //     }
    // }
    todo!("Exercise 2c: Implement manual polling loop with Wake trait waker")
}

// ─── Demo runner ────────────────────────────────────────────────────

pub fn run() {
    println!("--- 2a/2c: Polling Countdown(3) with RawWaker flag ---");
    poll_with_flag_waker(Countdown::new(3));
    println!();

    println!("--- 2b/2c: Polling Countdown(3) with Wake trait ---");
    poll_with_wake_trait(Countdown::new(3));
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_waker_sets_flag_on_wake() {
        let flag = Arc::new(AtomicBool::new(false));
        let waker = create_flag_waker(&flag);
        assert!(!flag.load(Ordering::SeqCst));
        waker.wake_by_ref();
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn flag_waker_clone_is_independent() {
        let flag = Arc::new(AtomicBool::new(false));
        let waker = create_flag_waker(&flag);
        let cloned = waker.clone();
        drop(waker);
        // cloned still works after original is dropped
        cloned.wake_by_ref();
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn wake_trait_sets_flag() {
        let task = TaskFlag::new();
        let waker = Waker::from(task.clone());
        assert!(!task.is_woken());
        waker.wake_by_ref();
        assert!(task.is_woken());
    }

    #[test]
    fn poll_with_flag_waker_completes() {
        poll_with_flag_waker(Countdown::new(5));
    }

    #[test]
    fn poll_with_wake_trait_completes() {
        poll_with_wake_trait(Countdown::new(5));
    }
}
