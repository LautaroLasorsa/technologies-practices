//! Exercise 6: Cooperative Scheduling & Budget
//!
//! Tokio uses cooperative multitasking: tasks run until they yield (return `Pending`
//! at an `.await` point). A task that never yields starves all other tasks on the
//! same worker thread.
//!
//! To mitigate this, Tokio has a **cooperative budget**: each task gets ~128 operations
//! per scheduling tick. Each `.await` on a Tokio resource (channel, mutex, timer, I/O)
//! decrements the budget. When the budget runs out, Tokio resources return `Pending`
//! even if they have data, forcing the task to yield.
//!
//! This exercise demonstrates:
//! 1. The budget system in action (greedy task gets throttled)
//! 2. `tokio::task::unconstrained()` to bypass the budget
//! 3. `tokio::task::coop::consume_budget()` for manual yield points
//!
//! Key concepts:
//! - Cooperative scheduling = tasks must yield voluntarily (no preemption)
//! - Budget prevents monopolization: each `.await` costs one budget unit
//! - `unconstrained` disables the budget for a future (dangerous!)
//! - `consume_budget` lets you add yield points in compute-heavy loops
//! - `tokio::task::yield_now()` yields unconditionally (always returns Pending once)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ─── Exercise 6a: Observing the cooperative budget ──────────────────

/// Demonstrate that Tokio's budget forces yielding in hot loops.
///
/// # TODO(human): Show the budget system in action
///
/// Steps:
///
/// 1. Create a multi_thread runtime with 1 worker thread (to make starvation obvious).
/// 2. Create a `tokio::sync::mpsc::channel(1000)` with a large buffer.
/// 3. Spawn a "producer" task that sends 500 messages in a tight loop:
///    ```ignore
///    for i in 0..500 {
///        tx.send(i).await.unwrap(); // Each .await costs one budget unit
///    }
///    ```
/// 4. Spawn a "fairness probe" task that records when it first runs:
///    ```ignore
///    let start = Instant::now();
///    tokio::task::yield_now().await; // Give the producer a head start
///    let first_run = Instant::now();
///    println!("  Probe first ran after {:?}", first_run - start);
///    ```
/// 5. Run both tasks and observe: the probe runs DURING the producer's loop,
///    not after it finishes. This proves the budget forced the producer to yield.
///
/// # What to observe
///
/// Without the budget system, the producer would monopolize the worker thread for
/// all 500 sends. The probe would only run after the producer finishes.
///
/// WITH the budget: after ~128 channel sends, Tokio forces the producer to yield,
/// allowing the probe to run. You'll see the probe runs after just a few milliseconds,
/// not after the full producer loop.
///
/// # The budget mechanism in detail
///
/// Each Tokio resource (channel, mutex, timer, I/O) checks the per-task budget
/// before returning Ready. If the budget is exhausted, the resource returns Pending
/// even though it could succeed. This forces the task to yield back to the executor.
/// The executor resets the budget when it next picks up the task.
pub fn demo_coop_budget() {
    // TODO(human): Build a 1-worker runtime and demonstrate the budget system.
    //
    // let rt = tokio::runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     let (tx, mut rx) = tokio::sync::mpsc::channel::<u32>(1000);
    //     let counter = Arc::new(AtomicU64::new(0));
    //     let counter_clone = counter.clone();
    //
    //     // Producer: sends many messages in a tight loop
    //     let producer = tokio::spawn(async move {
    //         for i in 0..500u32 {
    //             tx.send(i).await.unwrap();
    //             counter_clone.fetch_add(1, Ordering::Relaxed);
    //         }
    //         println!("  Producer: sent 500 messages");
    //     });
    //
    //     // Probe: checks when it first gets scheduled
    //     let probe = tokio::spawn(async move {
    //         tokio::task::yield_now().await; // Let producer start first
    //         let sent_so_far = counter.load(Ordering::Relaxed);
    //         println!("  Probe ran! Producer had sent {} messages so far", sent_so_far);
    //         if sent_so_far < 500 {
    //             println!("  >>> Budget forced producer to yield before finishing!");
    //         } else {
    //             println!("  Producer finished before probe ran (unexpected with budget)");
    //         }
    //
    //         // Drain the channel
    //         while rx.recv().await.is_some() {}
    //     });
    //
    //     let _ = tokio::join!(producer, probe);
    // });
    todo!("Exercise 6a: Demonstrate cooperative budget in action")
}

// ─── Exercise 6b: unconstrained — bypassing the budget ──────────────

/// Demonstrate `tokio::task::unconstrained()` — opting out of the budget.
///
/// # TODO(human): Show how unconstrained causes starvation
///
/// Steps:
///
/// 1. Same setup as 6a: 1-worker runtime, mpsc channel, producer + probe.
/// 2. But this time, wrap the producer's future in `tokio::task::unconstrained()`:
///    ```ignore
///    let producer = tokio::spawn(tokio::task::unconstrained(async move {
///        for i in 0..500u32 {
///            tx.send(i).await.unwrap();
///        }
///    }));
///    ```
/// 3. Observe: the probe does NOT run until the producer finishes all 500 sends.
///    The budget system has been disabled, so the producer monopolizes the thread.
///
/// # What to observe
///
/// With `unconstrained`, the producer's channel sends never get throttled.
/// The probe only runs after the producer finishes. On a single-worker runtime,
/// this means complete starvation of all other tasks.
///
/// # When to use unconstrained
///
/// Almost never. The only legitimate use case is latency-critical code where
/// you KNOW the task will complete quickly and yielding would add unacceptable
/// overhead (e.g., a hot path in a trading system processing a single message).
/// Even then, you must carefully reason about starvation.
///
/// # Warning
///
/// `unconstrained` in production can cause:
/// - Tail latency spikes (other tasks starved)
/// - Watchdog timeouts (health checks can't run)
/// - Deadlocks (if the unconstrained task waits on a channel that another starved task feeds)
pub fn demo_unconstrained() {
    // TODO(human): Show starvation caused by unconstrained.
    //
    // let rt = tokio::runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     let (tx, mut rx) = tokio::sync::mpsc::channel::<u32>(1000);
    //     let counter = Arc::new(AtomicU64::new(0));
    //     let counter_clone = counter.clone();
    //
    //     // Unconstrained producer — budget is DISABLED
    //     let producer = tokio::spawn(tokio::task::unconstrained(async move {
    //         for i in 0..500u32 {
    //             tx.send(i).await.unwrap();
    //             counter_clone.fetch_add(1, Ordering::Relaxed);
    //         }
    //         println!("  Producer (unconstrained): sent 500 messages");
    //     }));
    //
    //     let probe = tokio::spawn(async move {
    //         tokio::task::yield_now().await;
    //         let sent_so_far = counter.load(Ordering::Relaxed);
    //         println!("  Probe ran! Producer had sent {} messages so far", sent_so_far);
    //         if sent_so_far >= 500 {
    //             println!("  >>> Producer finished before probe! Budget was bypassed.");
    //         }
    //         while rx.recv().await.is_some() {}
    //     });
    //
    //     let _ = tokio::join!(producer, probe);
    // });
    todo!("Exercise 6b: Demonstrate starvation with unconstrained")
}

// ─── Exercise 6c: consume_budget — manual yield points ──────────────

/// Demonstrate `tokio::task::consume_budget()` for inserting yield points
/// in compute-heavy code that doesn't use Tokio resources.
///
/// # TODO(human): Show how consume_budget enables cooperative scheduling in CPU loops
///
/// Steps:
///
/// 1. Create a 1-worker runtime.
/// 2. Spawn a "compute" task that does a CPU-heavy loop (e.g., fibonacci or sum):
///    ```ignore
///    let mut total = 0u64;
///    for i in 0..1_000_000 {
///        total = total.wrapping_add(i);
///        if i % 1024 == 0 {
///            tokio::task::consume_budget().await;
///        }
///    }
///    ```
///    `consume_budget()` decrements the budget by 1. After enough calls, the budget
///    is exhausted and the next `consume_budget().await` returns `Pending`, yielding.
///
/// 3. Spawn a "monitor" task that prints a timestamp periodically.
/// 4. Observe: the monitor runs interspersed with the compute task, proving that
///    `consume_budget` enabled fair scheduling despite the CPU-heavy work.
///
/// 5. Compare with the same loop WITHOUT `consume_budget` — the monitor is starved.
///
/// # When to use consume_budget
///
/// Use it in long-running computation loops that don't naturally hit `.await` points.
/// Examples: processing large data structures, serialization, compression.
/// It's much lighter than `tokio::task::yield_now()` (which ALWAYS yields),
/// because `consume_budget` only yields when the budget is actually exhausted.
///
/// Rule of thumb: insert `consume_budget().await` every ~1000 iterations in hot loops.
pub fn demo_consume_budget() {
    // TODO(human): Show consume_budget enabling fair scheduling in CPU loops.
    //
    // println!("  --- Without consume_budget (starves monitor) ---");
    // let rt = tokio::runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     let done = Arc::new(AtomicU64::new(0));
    //     let done2 = done.clone();
    //
    //     let compute = tokio::spawn(async move {
    //         let mut total = 0u64;
    //         for i in 0..2_000_000u64 {
    //             total = total.wrapping_add(i);
    //             // NO yield point — monopolizes the worker
    //         }
    //         done2.store(1, Ordering::Release);
    //         println!("  Compute done (no budget): total = {}", total);
    //     });
    //
    //     let monitor = tokio::spawn(async move {
    //         let start = Instant::now();
    //         tokio::task::yield_now().await;
    //         println!("  Monitor ran after {:?} (compute done = {})",
    //             start.elapsed(), done.load(Ordering::Acquire));
    //     });
    //
    //     let _ = tokio::join!(compute, monitor);
    // });
    //
    // println!("\n  --- With consume_budget (fair scheduling) ---");
    // let rt2 = tokio::runtime::Builder::new_multi_thread()
    //     .worker_threads(1)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt2.block_on(async {
    //     let done = Arc::new(AtomicU64::new(0));
    //     let done2 = done.clone();
    //
    //     let compute = tokio::spawn(async move {
    //         let mut total = 0u64;
    //         for i in 0..2_000_000u64 {
    //             total = total.wrapping_add(i);
    //             if i % 1024 == 0 {
    //                 tokio::task::consume_budget().await; // <-- yield point
    //             }
    //         }
    //         done2.store(1, Ordering::Release);
    //         println!("  Compute done (with budget): total = {}", total);
    //     });
    //
    //     let monitor = tokio::spawn(async move {
    //         let start = Instant::now();
    //         tokio::task::yield_now().await;
    //         println!("  Monitor ran after {:?} (compute done = {})",
    //             start.elapsed(), done.load(Ordering::Acquire));
    //     });
    //
    //     let _ = tokio::join!(compute, monitor);
    // });
    todo!("Exercise 6c: Demonstrate consume_budget for CPU-heavy loops")
}

// ─── Demo runner ────────────────────────────────────────────────────

pub fn run() {
    println!("--- 6a: Cooperative budget in action ---\n");
    demo_coop_budget();
    println!();

    println!("--- 6b: unconstrained bypasses the budget ---\n");
    demo_unconstrained();
    println!();

    println!("--- 6c: consume_budget for CPU-heavy code ---\n");
    demo_consume_budget();
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coop_budget_demo_runs() {
        demo_coop_budget();
    }

    #[test]
    fn unconstrained_demo_runs() {
        demo_unconstrained();
    }

    #[test]
    fn consume_budget_demo_runs() {
        demo_consume_budget();
    }
}
