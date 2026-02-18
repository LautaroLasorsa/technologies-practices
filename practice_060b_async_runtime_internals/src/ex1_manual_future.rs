//! Exercise 1: Manual Future Implementation
//!
//! This exercise teaches the most fundamental concept in async Rust: the `Future` trait.
//! You will implement `Future` by hand — no `async`/`await` sugar — to understand
//! exactly what the compiler generates when you write `async fn`.
//!
//! Key concepts:
//! - `Future::poll()` is called by the executor to drive the future forward
//! - `Poll::Ready(value)` means the future is complete
//! - `Poll::Pending` means "not done yet, I'll wake you when I have progress"
//! - The future MUST call `cx.waker().wake_by_ref()` before returning `Pending`,
//!   otherwise the executor will never re-poll it and the task hangs forever
//!
//! Think of `poll()` like a state machine step function: each call advances
//! the state by one step, until the machine reaches a terminal state.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

// ─── Exercise 1a: Countdown Future ──────────────────────────────────

/// A future that counts down from `remaining` to 0.
///
/// Each call to `poll` decrements the counter by 1 and returns `Pending`.
/// When the counter reaches 0, `poll` returns `Ready(())`.
///
/// This is the simplest possible future that takes multiple polls to complete.
/// It demonstrates:
/// - State stored inside the future struct (the counter)
/// - Progressive completion (multiple `Pending` before `Ready`)
/// - The absolute requirement to wake before returning `Pending`
pub struct Countdown {
    /// How many more polls before this future completes.
    /// Starts at the initial count and decrements each poll.
    pub remaining: u32,
}

impl Countdown {
    pub fn new(count: u32) -> Self {
        Self { remaining: count }
    }
}

impl Future for Countdown {
    type Output = ();

    /// Poll the countdown future.
    ///
    /// # TODO(human): Implement the poll logic
    ///
    /// The logic is straightforward:
    ///
    /// 1. Check if `self.remaining` is 0.
    ///    - If yes: return `Poll::Ready(())` — the future is complete.
    ///    - If no: decrement `self.remaining` by 1, then:
    ///      a) Call `cx.waker().wake_by_ref()` to tell the executor to re-poll us.
    ///      b) Return `Poll::Pending`.
    ///
    /// # Why `wake_by_ref()` is CRITICAL
    ///
    /// When a future returns `Pending`, the executor removes it from the run queue.
    /// The ONLY way it gets re-added is if someone calls `waker.wake()`. For a
    /// countdown, we know we'll always be ready on the next poll (there's no real
    /// I/O to wait for), so we wake immediately. In real futures (e.g., reading
    /// from a socket), the waker is stored and called later when the OS signals
    /// data is available.
    ///
    /// # What would break
    ///
    /// If you return `Pending` WITHOUT calling `wake_by_ref()`:
    /// - The executor thinks "this task is waiting for something external"
    /// - Nothing external ever calls `wake()`
    /// - The task hangs forever — a silent deadlock
    /// - This is the #1 bug in hand-written futures
    ///
    /// # Accessing `self.remaining` through Pin
    ///
    /// The method signature is `self: Pin<&mut Self>`. To access fields, you use
    /// `self.get_mut()` to get `&mut Self` (this is safe because `Countdown` is
    /// `Unpin` — it has no self-referential fields). For `Unpin` types, `Pin`
    /// has no effect and you can freely access mutable fields.
    ///
    /// Alternatively, since `Countdown` is `Unpin`, you can access fields directly
    /// through the `Pin` — Rust auto-derefs `Pin<&mut T>` to `&mut T` for `Unpin` types.
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // TODO(human): Implement the countdown poll logic.
        //
        // Step 1: Get a mutable reference to self (use self.get_mut() or direct field access)
        // Step 2: If remaining == 0, return Poll::Ready(())
        // Step 3: Otherwise, decrement remaining, wake the waker, return Poll::Pending
        //
        // Hint: println! the remaining count to see the future progressing when tested.
        // Example: println!("  Countdown: {} remaining", this.remaining);
        todo!("Exercise 1a: Implement Countdown::poll")
    }
}

// ─── Exercise 1b: Ready Future ──────────────────────────────────────

/// A future that is immediately ready with a value.
///
/// This is the simplest possible future — it completes on the first poll.
/// It's the hand-written equivalent of `async { value }` or `std::future::ready(value)`.
///
/// Why build this? Because it demonstrates that `Future` is just a trait, and
/// `Ready` is how simple values get lifted into the async world. Every time you
/// write `async { 42 }`, the compiler generates something like this.
pub struct Ready<T> {
    /// The value to return. Wrapped in `Option` so we can `take()` it out
    /// on the first poll without requiring `Clone`. After the first poll,
    /// this is `None` — polling a completed future should panic.
    value: Option<T>,
}

impl<T> Ready<T> {
    pub fn new(value: T) -> Self {
        Self {
            value: Some(value),
        }
    }
}

impl<T> Future for Ready<T> {
    type Output = T;

    /// Poll the ready future.
    ///
    /// # TODO(human): Implement the poll logic
    ///
    /// This is even simpler than Countdown:
    ///
    /// 1. Take the value out of `self.value` using `.take()` (which replaces it with `None`).
    /// 2. If `Some(v)` — return `Poll::Ready(v)`.
    /// 3. If `None` — the future was already polled to completion. Panic with a message
    ///    like "Ready polled after completion". This mirrors the contract: once a future
    ///    returns `Ready`, it must never be polled again.
    ///
    /// # Why Option + take()?
    ///
    /// `poll` takes `Pin<&mut Self>`, not `self` — it cannot consume the struct.
    /// So we use `Option::take()` to move the value out without consuming self.
    /// After take(), the Option is None, acting as a "consumed" flag.
    /// This pattern appears everywhere in async Rust (e.g., `tokio::sync::oneshot`).
    ///
    /// # No waker needed
    ///
    /// Since this future always returns `Ready` on the first poll, we never
    /// return `Pending`, so we never need to call `wake()`. The waker protocol
    /// only matters when you return `Pending`.
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        // TODO(human): Take the value from self.value and return it as Ready,
        // or panic if the future was already completed.
        //
        // Hint: self.get_mut().value.take() gives you Option<T>
        // Then match on Some(v) => Poll::Ready(v), None => panic!(...)
        todo!("Exercise 1b: Implement Ready::poll")
    }
}

// ─── Exercise 1c: Sequencing two futures (Then combinator) ──────────

/// A future that runs `first`, then feeds its output into a closure `f`
/// to produce a `second` future, and runs that too.
///
/// This is the hand-written equivalent of:
/// ```ignore
/// let result = first.await;
/// let second = f(result);
/// second.await
/// ```
///
/// It demonstrates how `.await` chains become nested state machines.
/// The compiler does this automatically for every `.await` — `Then` makes it explicit.
///
/// # State machine
///
/// The enum captures the two states of this combined future:
/// - `Running first`: we're polling `first`, and `f` is waiting to be called.
/// - `Running second`: `first` completed, `f` was called, now polling `second`.
///
/// This mirrors exactly what the compiler generates for sequential `.await`s.
pub enum Then<Fut1, Fut2, F> {
    /// Currently polling `first`. Stores `f` to produce `second` later.
    First { first: Fut1, f: Option<F> },
    /// `first` completed; now polling `second`.
    Second { second: Fut2 },
    /// Terminal state — future already completed.
    Done,
}

impl<Fut1, Fut2, F> Then<Fut1, Fut2, F>
where
    Fut1: Future,
    Fut2: Future,
    F: FnOnce(Fut1::Output) -> Fut2,
{
    pub fn new(first: Fut1, f: F) -> Self {
        Then::First {
            first,
            f: Some(f),
        }
    }
}

impl<Fut1, Fut2, F> Future for Then<Fut1, Fut2, F>
where
    Fut1: Future + Unpin,
    Fut2: Future + Unpin,
    F: FnOnce(Fut1::Output) -> Fut2,
{
    type Output = Fut2::Output;

    /// Poll the Then combinator.
    ///
    /// # TODO(human): Implement the state machine poll logic
    ///
    /// This is a two-state machine:
    ///
    /// **State: First { first, f }**
    /// 1. Poll `first` (you'll need to pin it — since Fut1: Unpin, use `Pin::new(&mut first)`).
    /// 2. If `first` returns `Ready(output)`:
    ///    a) Take `f` out of the Option (`.take().unwrap()`).
    ///    b) Call `f(output)` to produce the second future.
    ///    c) Transition to `Second { second }` by assigning `*self = Then::Second { second }`.
    ///    d) Call `cx.waker().wake_by_ref()` and return `Pending` (to re-enter poll in the new state).
    ///       Alternatively, you can immediately poll `second` in the same call (tail-poll optimization).
    /// 3. If `first` returns `Pending` — just return `Pending` (the waker is already registered by `first`).
    ///
    /// **State: Second { second }**
    /// 1. Poll `second` (pin it the same way).
    /// 2. If `Ready(value)` — set `*self = Then::Done` and return `Ready(value)`.
    /// 3. If `Pending` — return `Pending`.
    ///
    /// **State: Done**
    /// - Panic: "Then polled after completion".
    ///
    /// # Key insight: state transitions via `*self = ...`
    ///
    /// Because `poll` takes `Pin<&mut Self>` (and our type is `Unpin`), we can
    /// reassign `*self` to transition between states. This is exactly what the
    /// compiler does for `async fn` — each `.await` point is a state transition.
    ///
    /// # Why `f` is in an Option
    ///
    /// `FnOnce` can only be called once, but `poll` may be called multiple times
    /// in the `First` state. We wrap `f` in `Option` so we can `.take()` it out
    /// exactly once when `first` completes. This is the standard pattern for
    /// "consume once inside a repeatedly-called method."
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // TODO(human): Implement the Then state machine.
        //
        // Pattern match on self.get_mut():
        //   Then::First { first, f } => poll first, if ready: take f, produce second, transition
        //   Then::Second { second } => poll second, if ready: transition to Done, return value
        //   Then::Done => panic!("Then polled after completion")
        //
        // Pinning hint for Unpin futures: Pin::new(&mut *first).poll(cx)
        todo!("Exercise 1c: Implement Then::poll state machine")
    }
}

// ─── Demo runner ────────────────────────────────────────────────────

/// Runs Exercise 1 demonstrations.
///
/// Since we don't have an executor yet (that's Exercise 3), we use
/// a simple `block_on` helper that manually polls futures in a loop.
/// This is intentionally primitive — you'll build a proper executor later.
pub fn run() {
    println!("--- 1a: Countdown Future ---");
    println!("Polling Countdown(3) to completion...");
    block_on_simple(Countdown::new(3));
    println!("Countdown complete!\n");

    println!("--- 1b: Ready Future ---");
    let value = block_on_simple(Ready::new(42));
    println!("Ready returned: {}\n", value);

    println!("--- 1c: Then Combinator ---");
    let future = Then::new(Ready::new(10), |x| Ready::new(x * 2));
    let result = block_on_simple(future);
    println!("Then(Ready(10), |x| Ready(x*2)) = {}\n", result);

    println!("--- 1c: Then with Countdown ---");
    let future = Then::new(Countdown::new(2), |()| Ready::new("done after countdown"));
    let result = block_on_simple(future);
    println!("Then(Countdown(2), ...) = {}", result);
}

/// Minimal block_on: polls a future in a busy loop using a no-op waker.
///
/// This is the simplest possible "executor". It works for futures that
/// always call `wake_by_ref()` before returning `Pending` (like our Countdown).
/// It does NOT work for I/O futures that need a real reactor to call `wake()`.
///
/// You'll build a proper executor in Exercise 3.
fn block_on_simple<F: Future + Unpin>(mut future: F) -> F::Output {
    use std::task::{RawWaker, RawWakerVTable, Waker};

    // A no-op waker — does nothing on wake(). This is fine for busy-polling.
    // In Exercise 2, you'll build a real waker that does useful work.
    fn no_op(_: *const ()) {}
    fn clone(ptr: *const ()) -> RawWaker {
        RawWaker::new(ptr, &VTABLE)
    }
    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, no_op, no_op, no_op);

    let raw = RawWaker::new(std::ptr::null(), &VTABLE);
    let waker = unsafe { Waker::from_raw(raw) };
    let mut cx = Context::from_waker(&waker);

    loop {
        match Pin::new(&mut future).poll(&mut cx) {
            Poll::Ready(output) => return output,
            Poll::Pending => {
                // Busy-wait: immediately re-poll. This is wasteful but simple.
                // A real executor would park the thread and wait for wake().
                continue;
            }
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn countdown_zero_is_immediately_ready() {
        let result = block_on_simple(Countdown::new(0));
        assert_eq!(result, ());
    }

    #[test]
    fn countdown_three_completes() {
        let result = block_on_simple(Countdown::new(3));
        assert_eq!(result, ());
    }

    #[test]
    fn ready_returns_value() {
        assert_eq!(block_on_simple(Ready::new(99)), 99);
        assert_eq!(block_on_simple(Ready::new("hello")), "hello");
    }

    #[test]
    fn then_chains_two_futures() {
        let f = Then::new(Ready::new(5), |x| Ready::new(x + 10));
        assert_eq!(block_on_simple(f), 15);
    }

    #[test]
    fn then_with_countdown() {
        let f = Then::new(Countdown::new(5), |()| Ready::new(42));
        assert_eq!(block_on_simple(f), 42);
    }
}
