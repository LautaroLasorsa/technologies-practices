//! Exercise 5: Tokio Runtime Configuration
//!
//! Now we shift from building things from scratch to exploring the production
//! runtime. Tokio provides two runtime flavors:
//!
//! - `current_thread`: single-threaded, no work-stealing, all tasks on one thread.
//!   Use for: tests, simple CLI tools, applications where simplicity > throughput.
//!
//! - `multi_thread`: N worker threads with work-stealing scheduler.
//!   Use for: production servers, anything that needs to scale across CPU cores.
//!
//! This exercise explores how tasks are scheduled across threads and how
//! work-stealing distributes load automatically.
//!
//! Key concepts:
//! - `tokio::runtime::Builder` configures the runtime before starting it
//! - `worker_threads(N)` sets the number of OS threads in the multi-thread pool
//! - `thread_name("...")` gives worker threads recognizable names (visible in debugger)
//! - `tokio::spawn` puts a task on the global queue (any worker can steal it)
//! - Tasks can migrate between workers — they are not pinned to a thread

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::runtime;

// ─── Exercise 5a: Current-thread runtime ────────────────────────────

/// Build and run a `current_thread` runtime.
///
/// # TODO(human): Create and configure a single-threaded Tokio runtime
///
/// Steps:
///
/// 1. Use `runtime::Builder::new_current_thread()` to create a builder.
/// 2. Call `.enable_all()` to enable I/O and timer drivers (needed for `tokio::time::sleep`).
/// 3. Call `.build().unwrap()` to construct the runtime.
/// 4. Use `rt.block_on(async { ... })` to run an async block.
/// 5. Inside the async block, spawn 4 tasks that each:
///    a) Print their task number and `thread::current().id()`.
///    b) Call `tokio::time::sleep(Duration::from_millis(10)).await` to simulate I/O.
///    c) Print their task number and thread ID again after sleeping.
/// 6. Await all task handles with `tokio::try_join!` or just `.await` each.
///
/// # What to observe
///
/// All tasks print THE SAME thread ID — because `current_thread` runs everything
/// on the thread that called `block_on`. There is no thread pool, no work-stealing.
/// This is the simplest runtime: great for tests and single-threaded programs.
///
/// # Why enable_all()?
///
/// Without `enable_all()`, `tokio::time::sleep` panics because the timer driver
/// is not enabled. Each Tokio feature (timers, I/O, signal handling) requires
/// its driver to be enabled on the runtime. `enable_all()` is the convenient
/// "turn everything on" option.
pub fn demo_current_thread() {
    // TODO(human): Build a current_thread runtime and observe task execution.
    //
    // let rt = runtime::Builder::new_current_thread()
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     println!("  Main task on thread: {:?}", thread::current().id());
    //
    //     let mut handles = Vec::new();
    //     for i in 0..4 {
    //         handles.push(tokio::spawn(async move {
    //             println!("  Task {} START on thread: {:?}", i, thread::current().id());
    //             tokio::time::sleep(Duration::from_millis(10)).await;
    //             println!("  Task {} END   on thread: {:?}", i, thread::current().id());
    //         }));
    //     }
    //
    //     for h in handles {
    //         h.await.unwrap();
    //     }
    //
    //     println!("  All tasks ran on the SAME thread (current_thread runtime).");
    // });
    todo!("Exercise 5a: Build and use a current_thread Tokio runtime")
}

// ─── Exercise 5b: Multi-thread runtime with work-stealing ───────────

/// Build and run a `multi_thread` runtime, observing task migration.
///
/// # TODO(human): Create a multi-thread runtime and observe work-stealing
///
/// Steps:
///
/// 1. Use `runtime::Builder::new_multi_thread()` to create a builder.
/// 2. Set `.worker_threads(4)` — four OS worker threads.
/// 3. Set `.thread_name("worker")` — makes threads identifiable.
/// 4. Call `.enable_all()` and `.build().unwrap()`.
/// 5. Use `rt.block_on(async { ... })`.
/// 6. Inside, spawn 8 tasks (more than workers to demonstrate distribution).
///    Each task should:
///    a) Print task number and thread name (`thread::current().name()`).
///    b) Do some "work" (a short sleep or spin loop).
///    c) Print task number and thread name again.
/// 7. Collect and await all handles.
///
/// # What to observe
///
/// - Tasks appear on DIFFERENT threads (named "worker").
/// - Some tasks may start on one thread and resume on a different thread
///   after an `.await` point (task migration due to work-stealing).
/// - Not all 4 workers are necessarily used if tasks are short — the
///   scheduler is lazy and reuses the current worker when possible.
///
/// # Understanding thread::current().name()
///
/// When you set `.thread_name("worker")`, Tokio names threads "worker"
/// (or "worker-0", "worker-1" etc. depending on Tokio version).
/// The main thread retains its default name. This is invaluable for
/// debugging: in production logs, you can see which worker ran each task.
pub fn demo_multi_thread() {
    // TODO(human): Build a multi_thread runtime and observe task migration.
    //
    // let rt = runtime::Builder::new_multi_thread()
    //     .worker_threads(4)
    //     .thread_name("worker")
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     println!("  Main task on thread: {:?} ({:?})",
    //         thread::current().id(),
    //         thread::current().name());
    //
    //     let mut handles = Vec::new();
    //     for i in 0..8 {
    //         handles.push(tokio::spawn(async move {
    //             let start_thread = format!("{:?}", thread::current().id());
    //             println!("  Task {} START on {:?} ({:?})",
    //                 i, thread::current().id(), thread::current().name());
    //
    //             // Simulate I/O — this is where task migration can happen
    //             tokio::time::sleep(Duration::from_millis(10)).await;
    //
    //             let end_thread = format!("{:?}", thread::current().id());
    //             println!("  Task {} END   on {:?} ({:?})",
    //                 i, thread::current().id(), thread::current().name());
    //
    //             if start_thread != end_thread {
    //                 println!("  >>> Task {} MIGRATED from {} to {}!", i, start_thread, end_thread);
    //             }
    //         }));
    //     }
    //
    //     for h in handles {
    //         h.await.unwrap();
    //     }
    // });
    todo!("Exercise 5b: Build multi_thread runtime and observe work-stealing")
}

// ─── Exercise 5c: Measuring spawn_blocking vs spawn ─────────────────

/// Demonstrate the difference between `tokio::spawn` and `tokio::spawn_blocking`.
///
/// # TODO(human): Show why blocking operations must use spawn_blocking
///
/// Steps:
///
/// 1. Create a multi_thread runtime with 2 worker threads.
/// 2. Spawn 2 tasks with `tokio::spawn` that each call `std::thread::sleep(1s)`.
///    This BLOCKS the worker threads — no other async tasks can run.
/// 3. Spawn a 3rd async task that just prints "hello". Observe that it's delayed
///    until the blocking tasks finish (because both workers are blocked).
/// 4. Now repeat with `tokio::task::spawn_blocking` for the blocking work.
///    Observe that the 3rd task runs immediately because worker threads are free.
///
/// # What to observe
///
/// With `tokio::spawn` + blocking: the async "hello" task is delayed ~1 second.
/// With `spawn_blocking`: the async "hello" task runs immediately.
///
/// # Why this matters
///
/// `tokio::spawn` tasks run on the worker thread pool. If they block (CPU-heavy work,
/// synchronous I/O, `std::thread::sleep`), they steal a worker thread from all async tasks.
/// `spawn_blocking` runs on a SEPARATE thread pool designed for blocking operations.
/// It returns a `JoinHandle<T>` that can be `.await`ed from async code.
///
/// Rule of thumb: if it touches the filesystem, does CPU-heavy computation, or calls
/// a synchronous library, use `spawn_blocking`.
pub fn demo_spawn_blocking() {
    // TODO(human): Demonstrate spawn_blocking vs blocking in async.
    //
    // println!("  --- Blocking inside tokio::spawn (BAD) ---");
    // let rt = runtime::Builder::new_multi_thread()
    //     .worker_threads(2)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt.block_on(async {
    //     let start = Instant::now();
    //
    //     // Block both worker threads
    //     let h1 = tokio::spawn(async {
    //         std::thread::sleep(Duration::from_millis(500)); // BLOCKS worker!
    //     });
    //     let h2 = tokio::spawn(async {
    //         std::thread::sleep(Duration::from_millis(500)); // BLOCKS worker!
    //     });
    //
    //     // This task can't run until a worker is free
    //     let h3 = tokio::spawn(async {
    //         println!("  'hello' task ran");
    //     });
    //
    //     let _ = tokio::join!(h1, h2, h3);
    //     println!("  Elapsed (blocking in spawn): {:?}", start.elapsed());
    // });
    //
    // println!("\n  --- Using spawn_blocking (GOOD) ---");
    // let rt2 = runtime::Builder::new_multi_thread()
    //     .worker_threads(2)
    //     .enable_all()
    //     .build()
    //     .unwrap();
    //
    // rt2.block_on(async {
    //     let start = Instant::now();
    //
    //     // Blocking work on separate thread pool
    //     let h1 = tokio::task::spawn_blocking(|| {
    //         std::thread::sleep(Duration::from_millis(500));
    //     });
    //     let h2 = tokio::task::spawn_blocking(|| {
    //         std::thread::sleep(Duration::from_millis(500));
    //     });
    //
    //     // This task runs immediately — worker threads are free
    //     let h3 = tokio::spawn(async {
    //         println!("  'hello' task ran");
    //     });
    //
    //     let _ = tokio::join!(h1, h2, h3);
    //     println!("  Elapsed (spawn_blocking): {:?}", start.elapsed());
    // });
    todo!("Exercise 5c: Demonstrate spawn_blocking vs blocking in async")
}

// ─── Demo runner ────────────────────────────────────────────────────

pub fn run() {
    println!("--- 5a: Current-thread runtime (all tasks on one thread) ---\n");
    demo_current_thread();
    println!();

    println!("--- 5b: Multi-thread runtime (work-stealing) ---\n");
    demo_multi_thread();
    println!();

    println!("--- 5c: spawn_blocking vs blocking in async ---\n");
    demo_spawn_blocking();
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_thread_runs() {
        demo_current_thread();
    }

    #[test]
    fn multi_thread_runs() {
        demo_multi_thread();
    }

    #[test]
    fn spawn_blocking_runs() {
        demo_spawn_blocking();
    }
}
