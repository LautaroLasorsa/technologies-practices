//! Exercise 7: Capstone — Async Channel from Scratch
//!
//! This capstone ties together everything from the previous exercises:
//! - Manual `Future` implementation (Exercise 1)
//! - Waker management (Exercise 2)
//! - Understanding the executor loop (Exercise 3)
//! - Pin and safe interior mutability (Exercise 4)
//!
//! You'll build a simple single-producer, single-consumer (SPSC) async channel.
//! This is a simplified version of `tokio::sync::oneshot` — a channel that
//! delivers exactly one value from a sender to a receiver.
//!
//! Architecture:
//! ```text
//! ┌──────────┐       Arc<Mutex<SharedState>>       ┌──────────┐
//! │  Sender  │ ────────────┐ ┌───────────────────> │ Receiver │
//! │          │             │ │                      │ (Future) │
//! │ send(v)  │───> value ──┼─┘   ┌── waker ──────> │ poll()   │
//! │          │             │     │                  │          │
//! └──────────┘             └─────┘                  └──────────┘
//! ```
//!
//! Key concepts:
//! - Shared state between sender and receiver via `Arc<Mutex<...>>`
//! - The receiver implements `Future` — when polled, it checks for a value
//! - If no value yet, the receiver stores its `Waker` in the shared state
//! - When the sender sends a value, it stores it and calls `waker.wake()`
//! - This is the fundamental pattern behind ALL async channels, mutexes, and signals

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

// ─── Shared state ───────────────────────────────────────────────────

/// The state shared between `Sender` and `Receiver`.
///
/// This is the core data structure. Both sides hold an `Arc<Mutex<Inner<T>>>`.
///
/// # Fields
///
/// - `value`: `Option<T>` — `None` until the sender sends a value, then `Some(v)`.
/// - `waker`: `Option<Waker>` — stored by the receiver when it's polled and finds
///   no value. The sender calls `.wake()` on this when it sends a value.
/// - `sender_dropped`: `bool` — set to `true` when the sender is dropped without
///   sending. The receiver uses this to detect that no value will ever arrive.
struct Inner<T> {
    value: Option<T>,
    waker: Option<Waker>,
    sender_dropped: bool,
}

// ─── Channel constructor ────────────────────────────────────────────

/// Create a new oneshot channel, returning `(Sender<T>, Receiver<T>)`.
///
/// # TODO(human): Implement the channel constructor
///
/// Steps:
///
/// 1. Create the shared state:
///    ```ignore
///    let inner = Arc::new(Mutex::new(Inner {
///        value: None,
///        waker: None,
///        sender_dropped: false,
///    }));
///    ```
///
/// 2. Create `Sender` and `Receiver`, each holding a clone of the `Arc`:
///    ```ignore
///    let sender = Sender { inner: inner.clone() };
///    let receiver = Receiver { inner };
///    (sender, receiver)
///    ```
///
/// # Why Arc<Mutex<...>>?
///
/// Sender and receiver may live on different tasks (different threads in a
/// multi-thread runtime). `Arc` provides shared ownership across threads.
/// `Mutex` provides interior mutability with exclusive access.
///
/// For a oneshot channel, `Mutex` contention is minimal (at most 2 accesses:
/// receiver stores waker, sender stores value). For high-throughput channels,
/// lock-free designs are preferred.
pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    // TODO(human): Create the shared state and return (Sender, Receiver).
    //
    // let inner = Arc::new(Mutex::new(Inner {
    //     value: None,
    //     waker: None,
    //     sender_dropped: false,
    // }));
    // (Sender { inner: inner.clone() }, Receiver { inner })
    todo!("Exercise 7: Implement channel() constructor")
}

// ─── Sender ─────────────────────────────────────────────────────────

/// The sending half of the oneshot channel.
///
/// Calling `send(value)` stores the value in the shared state and wakes
/// the receiver (if it's waiting).
pub struct Sender<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

impl<T> Sender<T> {
    /// Send a value through the channel.
    ///
    /// # TODO(human): Implement send
    ///
    /// Steps:
    ///
    /// 1. Lock the shared state: `let mut guard = self.inner.lock().unwrap();`
    /// 2. Store the value: `guard.value = Some(value);`
    /// 3. Take the waker (if any): `if let Some(waker) = guard.waker.take() { ... }`
    /// 4. Drop the lock BEFORE calling wake (to avoid holding the lock during wake,
    ///    which could cause a deadlock if wake tries to poll the receiver synchronously):
    ///    ```ignore
    ///    let waker = guard.waker.take();
    ///    drop(guard);
    ///    if let Some(w) = waker { w.wake(); }
    ///    ```
    ///
    /// # Why drop the lock before wake?
    ///
    /// `waker.wake()` causes the executor to re-poll the receiver. If the executor
    /// is single-threaded and runs the receiver synchronously (inline), the receiver
    /// will try to lock the mutex. If we still hold the lock, DEADLOCK.
    ///
    /// This is a common pattern in async Rust: always drop locks before calling wake().
    ///
    /// # Return value
    ///
    /// Returns `Ok(())` on success. Returns `Err(value)` if the receiver has been
    /// dropped (the value has nowhere to go). Check `Arc::strong_count(&self.inner) == 1`
    /// to detect this (only the sender's Arc remains).
    pub fn send(self, value: T) -> Result<(), T> {
        // TODO(human): Implement Sender::send.
        //
        // let mut guard = self.inner.lock().unwrap();
        // guard.value = Some(value);
        // let waker = guard.waker.take();
        // drop(guard);
        // if let Some(w) = waker {
        //     w.wake();
        // }
        // Ok(())
        //
        // Bonus: check if receiver is dropped before sending:
        //   if Arc::strong_count(&self.inner) == 1 { return Err(value); }
        todo!("Exercise 7: Implement Sender::send")
    }
}

impl<T> Drop for Sender<T> {
    /// When the sender is dropped without sending, mark the channel as closed.
    ///
    /// # TODO(human): Implement the drop logic
    ///
    /// Steps:
    ///
    /// 1. Lock the shared state.
    /// 2. Set `sender_dropped = true`.
    /// 3. Take the waker and drop the lock.
    /// 4. If there was a waker, call `.wake()` — this wakes the receiver so it
    ///    can observe that the sender is gone and return an error.
    ///
    /// # Why wake on drop?
    ///
    /// If the receiver is awaiting a value and the sender is dropped, the receiver
    /// is stuck forever (no one will send a value). By waking the receiver, we let
    /// it discover `sender_dropped == true` and return an error.
    ///
    /// This is the same pattern as `tokio::sync::oneshot::Sender::drop`.
    fn drop(&mut self) {
        // TODO(human): Mark sender_dropped and wake the receiver.
        //
        // let mut guard = self.inner.lock().unwrap();
        // guard.sender_dropped = true;
        // let waker = guard.waker.take();
        // drop(guard);
        // if let Some(w) = waker {
        //     w.wake();
        // }
        //
        // Note: we use todo!() wrapped in a block that checks a condition
        // to avoid the todo!() panicking during normal test cleanup.
        // For the scaffold, we'll use a simple flag check.
        if Arc::strong_count(&self.inner) > 0 {
            // TODO(human): Replace this block with the actual implementation above.
            // The todo!() is not placed here directly because Drop is called implicitly
            // and would panic during tests of other exercises. Instead, implement the
            // logic described in the comments above.
            let mut guard = self.inner.lock().unwrap();
            guard.sender_dropped = true;
            let waker = guard.waker.take();
            drop(guard);
            if let Some(w) = waker {
                w.wake();
            }
        }
    }
}

// ─── Receiver ───────────────────────────────────────────────────────

/// The receiving half of the oneshot channel.
///
/// `Receiver` implements `Future`, so you `.await` it to get the value.
/// If no value has been sent yet, the future returns `Pending` and stores
/// its waker. When the sender sends a value, it calls the waker, causing
/// the executor to re-poll the receiver.
pub struct Receiver<T> {
    inner: Arc<Mutex<Inner<T>>>,
}

/// Error returned when the sender was dropped without sending a value.
#[derive(Debug, PartialEq)]
pub struct RecvError;

impl std::fmt::Display for RecvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "sender dropped without sending a value")
    }
}

impl<T> Future for Receiver<T> {
    type Output = Result<T, RecvError>;

    /// Poll the receiver.
    ///
    /// # TODO(human): Implement the receiver poll logic
    ///
    /// Steps:
    ///
    /// 1. Lock the shared state: `let mut guard = self.inner.lock().unwrap();`
    ///
    /// 2. Check if a value is available: `if let Some(value) = guard.value.take()`
    ///    - Return `Poll::Ready(Ok(value))` — the channel delivered successfully.
    ///
    /// 3. Check if the sender was dropped: `if guard.sender_dropped`
    ///    - Return `Poll::Ready(Err(RecvError))` — no value will ever arrive.
    ///
    /// 4. No value yet, sender still alive — store the waker for later:
    ///    ```ignore
    ///    guard.waker = Some(cx.waker().clone());
    ///    ```
    ///    Then return `Poll::Pending`.
    ///
    /// # Why clone the waker?
    ///
    /// The waker from `cx.waker()` is borrowed — it lives only for this poll call.
    /// We need to store it so the sender can call `.wake()` later (possibly on a
    /// different thread, long after this poll returns). Cloning gives us an owned
    /// copy that outlives the poll call.
    ///
    /// # Why check value BEFORE sender_dropped?
    ///
    /// The sender might send a value and THEN get dropped (this is normal — `send()`
    /// consumes the sender). If we checked `sender_dropped` first, we'd miss the value.
    /// Order matters:
    /// 1. Check value (might have been sent just before the sender dropped)
    /// 2. Check sender_dropped (no value, and no sender — permanent error)
    /// 3. Store waker (no value yet, but sender is alive — will be woken later)
    ///
    /// # What would break
    ///
    /// If you forget to store the waker (skip step 4):
    /// - The receiver returns Pending, but nobody can wake it
    /// - When the sender sends a value, it finds `waker: None`
    /// - The receiver task hangs forever — the #1 bug in hand-written futures
    ///
    /// If you don't clone the waker (just store a reference):
    /// - Won't compile — the waker is borrowed from `cx` which doesn't live long enough
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // TODO(human): Implement Receiver::poll.
        //
        // let mut guard = self.inner.lock().unwrap();
        //
        // if let Some(value) = guard.value.take() {
        //     return Poll::Ready(Ok(value));
        // }
        //
        // if guard.sender_dropped {
        //     return Poll::Ready(Err(RecvError));
        // }
        //
        // guard.waker = Some(cx.waker().clone());
        // Poll::Pending
        todo!("Exercise 7: Implement Receiver::poll")
    }
}

// ─── Demo runner ────────────────────────────────────────────────────

pub fn run() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    println!("--- 7a: Basic send/receive ---");
    rt.block_on(async {
        let (tx, rx) = channel::<i32>();

        // Spawn a task that sends a value after a short delay
        tokio::spawn(async move {
            tokio::task::yield_now().await; // Let the receiver start waiting first
            println!("  Sender: sending 42");
            tx.send(42).unwrap();
        });

        println!("  Receiver: waiting for value...");
        match rx.await {
            Ok(val) => println!("  Receiver: got {}!", val),
            Err(e) => println!("  Receiver: error: {}", e),
        }
    });

    println!();
    println!("--- 7b: Sender dropped without sending ---");
    rt.block_on(async {
        let (tx, rx) = channel::<String>();

        tokio::spawn(async move {
            tokio::task::yield_now().await;
            println!("  Sender: dropping without sending");
            drop(tx);
        });

        println!("  Receiver: waiting for value...");
        match rx.await {
            Ok(val) => println!("  Receiver: got {:?}!", val),
            Err(_) => println!("  Receiver: sender dropped — RecvError (expected)"),
        }
    });

    println!();
    println!("--- 7c: Immediate send (value ready before first poll) ---");
    rt.block_on(async {
        let (tx, rx) = channel::<&str>();
        tx.send("hello").unwrap(); // Send before receiver is ever polled
        match rx.await {
            Ok(val) => println!("  Receiver: got {:?} (was already sent)", val),
            Err(e) => println!("  Receiver: error: {}", e),
        }
    });
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn send_then_receive() {
        let (tx, rx) = channel();
        tx.send(42).unwrap();
        assert_eq!(rx.await, Ok(42));
    }

    #[tokio::test]
    async fn receive_before_send() {
        let (tx, rx) = channel();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            tx.send(99).unwrap();
        });
        assert_eq!(rx.await, Ok(99));
    }

    #[tokio::test]
    async fn sender_dropped() {
        let (tx, rx) = channel::<i32>();
        drop(tx);
        assert_eq!(rx.await, Err(RecvError));
    }

    #[tokio::test]
    async fn sender_dropped_after_yield() {
        let (tx, rx) = channel::<i32>();
        tokio::spawn(async move {
            tokio::task::yield_now().await;
            drop(tx);
        });
        assert_eq!(rx.await, Err(RecvError));
    }

    #[tokio::test]
    async fn send_string() {
        let (tx, rx) = channel();
        tx.send(String::from("hello async")).unwrap();
        assert_eq!(rx.await, Ok(String::from("hello async")));
    }
}
