//! Exercise 3: Mini Single-Threaded Executor
//!
//! This is the heart of the practice: building a working async runtime from scratch.
//!
//! Every async runtime — Tokio, async-std, smol — is fundamentally this same loop:
//!
//! ```text
//! loop {
//!     task = ready_queue.pop();
//!     waker = make_waker(task);
//!     match task.future.poll(waker) {
//!         Ready(val)  => task.complete(val),
//!         Pending     => { /* waker will re-enqueue when ready */ }
//!     }
//! }
//! ```
//!
//! Tokio adds work-stealing, I/O drivers, timers, and cooperative scheduling on top,
//! but the core algorithm is identical to what you'll build here.
//!
//! Key concepts:
//! - Tasks wrap futures and live on the heap (Box<dyn Future>)
//! - The ready queue holds task IDs (or references) of tasks ready to be polled
//! - The Waker's job is to push the task ID back onto the ready queue
//! - An executor without pending I/O or timers exits when the ready queue is empty

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

// ─── Task representation ────────────────────────────────────────────

/// A task is a heap-allocated future that the executor can poll.
///
/// We use `Pin<Box<dyn Future<Output = ()>>>` because:
/// - `Box` puts the future on the heap (required for type erasure with `dyn`)
/// - `Pin` guarantees the future won't be moved (required by `Future::poll`)
/// - `dyn Future<Output = ()>` allows the executor to hold futures of different types
/// - `Output = ()` simplifies the executor (tasks don't return values; use channels for that)
struct Task {
    /// The future this task is driving to completion.
    future: Pin<Box<dyn Future<Output = ()>>>,

    /// Unique identifier for this task. Used by the waker to re-enqueue it.
    id: usize,
}

// ─── Exercise 3a: Task Waker ────────────────────────────────────────

/// A waker that pushes a task ID onto a shared ready queue.
///
/// When a future returns `Pending` and later calls `waker.wake()`, this
/// struct pushes the task's ID onto the `ready_queue`, signaling the
/// executor to re-poll that task.
///
/// # Why Arc<Mutex<VecDeque>>?
///
/// The ready queue is shared between:
/// - The executor loop (which pops from it)
/// - All wakers (which push onto it)
///
/// In a single-threaded executor, we could use `Rc<RefCell<...>>` instead.
/// We use `Arc<Mutex<...>>` to be compatible with the `Wake` trait, which
/// requires `Send + Sync` (the trait is designed to work across threads).
struct TaskWaker {
    /// The ID of the task this waker belongs to.
    task_id: usize,

    /// Shared ready queue — waking pushes `task_id` onto this queue.
    ready_queue: Arc<Mutex<VecDeque<usize>>>,
}

/// # TODO(human): Implement the `Wake` trait for `TaskWaker`
///
/// The `Wake` trait requires:
///
/// ```ignore
/// fn wake(self: Arc<Self>) {
///     // Push self.task_id onto self.ready_queue
/// }
/// ```
///
/// Implementation:
/// 1. Lock the ready queue: `self.ready_queue.lock().unwrap()`
/// 2. Push the task ID: `.push_back(self.task_id)`
///
/// That's it! When the executor sees this ID in the queue, it knows
/// to poll the corresponding task again.
///
/// # What would break
///
/// If wake() doesn't push the ID onto the queue:
/// - The task that returned Pending will never be polled again
/// - The executor thinks it's done (empty queue) and exits
/// - Silent task loss — one of the hardest bugs to debug
///
/// If wake() pushes the ID multiple times (e.g., duplicate wake calls):
/// - The task gets polled more than necessary — wasteful but not incorrect
/// - Production executors use atomic flags to deduplicate wake calls
impl Wake for TaskWaker {
    fn wake(self: Arc<Self>) {
        // TODO(human): Push self.task_id onto self.ready_queue.
        //
        // self.ready_queue.lock().unwrap().push_back(self.task_id);
        //
        // This is the fundamental action of a waker: "schedule this task
        // for re-polling." Everything else in the executor revolves around
        // this single enqueue operation.
        todo!("Exercise 3a: Implement Wake for TaskWaker")
    }
}

// ─── Exercise 3b: The Executor ──────────────────────────────────────

/// A minimal single-threaded async executor.
///
/// This executor manages a collection of tasks and a ready queue.
/// It polls ready tasks until all tasks are complete.
///
/// # Architecture
///
/// ```text
/// ┌──────────────────────────────────────────────┐
/// │                 MiniExecutor                  │
/// │                                              │
/// │  tasks: HashMap<id, Task>                    │
/// │    ┌────┐  ┌────┐  ┌────┐                   │
/// │    │ T0 │  │ T1 │  │ T2 │  ...              │
/// │    └────┘  └────┘  └────┘                    │
/// │                                              │
/// │  ready_queue: VecDeque<id>                   │
/// │    ┌─────────────────────┐                   │
/// │    │ 0 │ 2 │ ... │      │  ← wakers push    │
/// │    └─────────────────────┘    IDs here       │
/// │         ↑ executor pops                      │
/// └──────────────────────────────────────────────┘
/// ```
pub struct MiniExecutor {
    /// All tasks, indexed by ID.
    tasks: Vec<Option<Task>>,

    /// Queue of task IDs that are ready to be polled.
    /// Shared with wakers via Arc<Mutex<...>>.
    ready_queue: Arc<Mutex<VecDeque<usize>>>,

    /// Next task ID to assign.
    next_id: usize,
}

impl MiniExecutor {
    pub fn new() -> Self {
        Self {
            tasks: Vec::new(),
            ready_queue: Arc::new(Mutex::new(VecDeque::new())),
            next_id: 0,
        }
    }

    /// Spawn a new task onto the executor.
    ///
    /// # TODO(human): Implement task spawning
    ///
    /// Steps:
    /// 1. Assign an ID to the task: `let id = self.next_id; self.next_id += 1;`
    /// 2. Create a `Task` struct wrapping the future: `Task { future: Box::pin(future), id }`
    /// 3. Store the task: grow `self.tasks` if needed, then `self.tasks[id] = Some(task)`.
    ///    (Hint: while `self.tasks.len() <= id { self.tasks.push(None); }`)
    /// 4. Add the task ID to the ready queue so it gets polled on the first iteration.
    /// 5. Print a message like `"  Spawned task {id}"`.
    ///
    /// # Why Box::pin?
    ///
    /// `Future::poll` requires `Pin<&mut Self>`. By storing the future in a `Pin<Box<...>>`,
    /// we guarantee it stays at the same memory address. `Box::pin(future)` does this in
    /// one step: allocates on the heap and pins the allocation.
    ///
    /// # Why add to ready_queue immediately?
    ///
    /// A newly spawned task must be polled at least once — it might be immediately ready,
    /// or it might register wakers for I/O. If we don't enqueue it, it will never run.
    pub fn spawn(&mut self, future: impl Future<Output = ()> + 'static) {
        // TODO(human): Implement task spawning.
        //
        // let id = self.next_id;
        // self.next_id += 1;
        // let task = Task { future: Box::pin(future), id };
        // while self.tasks.len() <= id { self.tasks.push(None); }
        // self.tasks[id] = Some(task);
        // self.ready_queue.lock().unwrap().push_back(id);
        // println!("  Spawned task {}", id);
        todo!("Exercise 3b: Implement MiniExecutor::spawn")
    }

    /// Run the executor until all tasks complete.
    ///
    /// # TODO(human): Implement the executor event loop
    ///
    /// The loop structure:
    ///
    /// ```text
    /// loop {
    ///     // 1. Pop a task ID from the ready queue
    ///     // 2. If no task is ready:
    ///     //    a) If all tasks are done → break (executor complete)
    ///     //    b) If tasks remain but none are ready → panic! (deadlock: no waker will fire)
    ///     // 3. Look up the task by ID; skip if already completed (None)
    ///     // 4. Create a Waker for this task using TaskWaker + Arc
    ///     // 5. Create a Context from the waker
    ///     // 6. Poll the task's future
    ///     // 7. If Ready → remove the task (set slot to None)
    ///     // 8. If Pending → put the task back (it's already there; waker will re-enqueue)
    /// }
    /// ```
    ///
    /// # Creating the Waker
    ///
    /// ```ignore
    /// let task_waker = Arc::new(TaskWaker {
    ///     task_id: id,
    ///     ready_queue: self.ready_queue.clone(),
    /// });
    /// let waker = Waker::from(task_waker);
    /// let mut cx = Context::from_waker(&waker);
    /// ```
    ///
    /// # Polling the task
    ///
    /// You need to temporarily take the task out of the slot to get a mutable reference:
    /// ```ignore
    /// let mut task = self.tasks[id].take().unwrap();
    /// match task.future.as_mut().poll(&mut cx) {
    ///     Poll::Ready(()) => { /* don't put task back — it's done */ }
    ///     Poll::Pending => { self.tasks[id] = Some(task); /* put it back */ }
    /// }
    /// ```
    ///
    /// Why `take()` then put back? Because `poll` needs `Pin<&mut ...>`, which requires
    /// exclusive access. We can't get `&mut task.future` while `task` is inside the Vec
    /// through a shared index. Taking it out gives us full ownership temporarily.
    ///
    /// # Detecting deadlock
    ///
    /// If the ready queue is empty but tasks remain, we have a deadlock:
    /// some future returned `Pending` without calling `wake()`. A production
    /// executor would block waiting for I/O events; our mini executor has no I/O,
    /// so an empty queue with live tasks means a bug.
    pub fn run(&mut self) {
        // TODO(human): Implement the executor event loop.
        //
        // loop {
        //     let task_id = {
        //         let mut queue = self.ready_queue.lock().unwrap();
        //         queue.pop_front()
        //     };
        //
        //     let Some(id) = task_id else {
        //         // Queue is empty — check if all tasks are done
        //         let remaining = self.tasks.iter().filter(|t| t.is_some()).count();
        //         if remaining == 0 {
        //             println!("  All tasks completed!");
        //             break;
        //         } else {
        //             panic!("Deadlock! {} tasks remain but ready queue is empty", remaining);
        //         }
        //     };
        //
        //     // Skip if task was already completed (stale wake)
        //     if self.tasks.get(id).and_then(|t| t.as_ref()).is_none() {
        //         continue;
        //     }
        //
        //     // Create waker and context
        //     let task_waker = Arc::new(TaskWaker {
        //         task_id: id,
        //         ready_queue: self.ready_queue.clone(),
        //     });
        //     let waker = Waker::from(task_waker);
        //     let mut cx = Context::from_waker(&waker);
        //
        //     // Take task out, poll, put back if pending
        //     let mut task = self.tasks[id].take().unwrap();
        //     match task.future.as_mut().poll(&mut cx) {
        //         Poll::Ready(()) => {
        //             println!("  Task {} completed", id);
        //         }
        //         Poll::Pending => {
        //             self.tasks[id] = Some(task);
        //         }
        //     }
        // }
        todo!("Exercise 3b: Implement MiniExecutor::run event loop")
    }
}

// ─── Exercise 3c: Testing the executor ──────────────────────────────

/// A simple async function to test the executor.
///
/// This uses the `async` keyword — the compiler transforms it into a state machine
/// (an anonymous type that implements `Future`). Our mini executor polls this
/// compiler-generated state machine just like it polls our hand-written futures.
async fn print_numbers(label: &'static str, count: u32) {
    for i in 0..count {
        println!("  [{}] step {}", label, i);
        // yield_now() returns Pending once, then Ready on the next poll.
        // This simulates an async operation and lets other tasks run.
        yield_now().await;
    }
    println!("  [{}] done", label);
}

/// A future that yields once (returns Pending) then completes.
///
/// This is a simplified version of `tokio::task::yield_now()`.
/// It demonstrates cooperative scheduling: by yielding, a task voluntarily
/// gives other tasks a chance to run.
struct YieldNow {
    yielded: bool,
}

fn yield_now() -> YieldNow {
    YieldNow { yielded: false }
}

impl Future for YieldNow {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        let this = self.get_mut();
        if this.yielded {
            Poll::Ready(())
        } else {
            this.yielded = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

// ─── Demo runner ────────────────────────────────────────────────────

pub fn run() {
    println!("--- 3a/3b: Running MiniExecutor with 3 concurrent tasks ---\n");

    let mut executor = MiniExecutor::new();

    // Spawn three tasks. They'll interleave on our single-threaded executor
    // because each yields between steps.
    executor.spawn(print_numbers("alpha", 3));
    executor.spawn(print_numbers("beta", 2));
    executor.spawn(print_numbers("gamma", 4));

    println!();
    executor.run();

    println!("\n--- 3c: Observe interleaving ---");
    println!("Notice how alpha, beta, and gamma steps are interleaved.");
    println!("This is cooperative multitasking: each task yields voluntarily,");
    println!("letting the executor poll the next ready task.");
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn executor_runs_single_task() {
        let mut executor = MiniExecutor::new();
        executor.spawn(async { println!("single task ran"); });
        executor.run();
    }

    #[test]
    fn executor_runs_multiple_tasks() {
        let mut executor = MiniExecutor::new();
        executor.spawn(print_numbers("a", 2));
        executor.spawn(print_numbers("b", 3));
        executor.run();
    }

    #[test]
    fn executor_handles_immediate_ready() {
        let mut executor = MiniExecutor::new();
        // An async block with no .await is immediately ready on first poll.
        executor.spawn(async {});
        executor.spawn(async {});
        executor.spawn(async {});
        executor.run();
    }
}
