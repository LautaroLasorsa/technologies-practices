# Practice 060b: Async Runtime Internals — Future, Waker & Tokio Architecture

## Technologies

- **tokio** (full features) — Production async runtime: work-stealing scheduler, I/O driver, timers
- **futures** — Utility traits and combinators for the `Future` ecosystem
- **pin-project-lite** — Lightweight macro for safe pinned field projections

## Stack

- Rust (cargo, edition 2021)

## Theoretical Context

### What Problem Does Async Solve?

Servers handling thousands of concurrent connections cannot afford one OS thread per connection. Threads are expensive: each consumes ~8 MB of stack, and context-switching between thousands of them wastes CPU time on scheduling overhead. Async I/O solves this by multiplexing many logical tasks onto a small number of OS threads. When a task is waiting for I/O, it *yields* the thread so other tasks can run. The result: a handful of threads can service tens of thousands of concurrent operations.

Rust's async model is unique among mainstream languages because it is **zero-cost**: async functions compile to state machines with no heap allocation for the future itself (unless you `Box::pin` it), no garbage collector, and no hidden runtime. The programmer pays only for what they use.

Sources: [Asynchronous Programming in Rust](https://rust-lang.github.io/async-book/), [Tokio Tutorial](https://tokio.rs/tokio/tutorial)

### The `Future` Trait

The core abstraction is `std::future::Future`:

```rust
pub trait Future {
    type Output;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}
```

A future represents a value that may not be available yet. The `poll` method attempts to resolve it:

- **`Poll::Ready(value)`** — the future is complete; return the final value.
- **`Poll::Pending`** — the future is not ready; the caller (executor) should try again later.

Critical contract: when `poll` returns `Pending`, the future **must** have arranged for the `Waker` (obtained from `cx.waker()`) to be called when progress is possible. Forgetting to wake results in a task that hangs forever — it will never be polled again.

The `self: Pin<&mut Self>` receiver is explained below in the Pin section.

Source: [std::future::Future](https://doc.rust-lang.org/std/future/trait.Future.html)

### How `async`/`.await` Desugars to State Machines

When the compiler sees:

```rust
async fn fetch_and_process(url: &str) -> Result<Data> {
    let response = http_get(url).await;    // await point 1
    let parsed = parse(response).await;    // await point 2
    Ok(parsed)
}
```

It generates a hidden enum (state machine) roughly equivalent to:

```rust
enum FetchAndProcess<'a> {
    /// Initial state: about to call http_get
    State0 { url: &'a str },
    /// Waiting on http_get future
    State1 { url: &'a str, fut: HttpGetFuture<'a> },
    /// Waiting on parse future
    State2 { response: Response, fut: ParseFuture },
    /// Terminal state (already resolved)
    Complete,
}
```

Each `.await` becomes a state transition. The compiler generates a `poll` implementation that:
1. Matches on the current state.
2. Polls the inner future for that state.
3. If `Ready`, extracts the value, transitions to the next state, and continues.
4. If `Pending`, stores the inner future back and returns `Pending`.

This is why async Rust has zero-cost overhead — no heap allocation, no runtime closure capture. The state machine is a flat enum whose size is the max of all states (like a C union), stored inline wherever the future lives.

Sources: [Comprehensive Rust — State Machine](https://google.github.io/comprehensive-rust/concurrency/async/state-machine.html), [EventHelix — Async/Await to Assembly](https://www.eventhelix.com/rust/rust-to-assembly-async-await/), [Tyler Mandry — Optimizing Await](https://tmandry.gitlab.io/blog/posts/optimizing-await-1/)

### The Waker and Context Protocol

When `poll` returns `Pending`, who re-polls the future? The **Waker**.

```rust
pub struct Context<'a> {
    waker: &'a Waker,
    // ...
}
```

`Context` provides `cx.waker()` — a handle that the future clones and stores. When the underlying resource becomes ready (e.g., a socket has data), the resource calls `waker.wake()`, which tells the executor: "re-schedule this task for polling."

Under the hood, `Waker` is built from `RawWaker` + `RawWakerVTable` — a manually-constructed vtable with four function pointers: `clone`, `wake`, `wake_by_ref`, and `drop`. This low-level design avoids requiring `Arc` or any specific allocation strategy, making it suitable for embedded systems and custom executors.

For convenience, the standard library provides the `Wake` trait (stable since Rust 1.51, usable with `Arc<impl Wake>`), which generates the `RawWaker`/vtable automatically:

```rust
impl Wake for MyTask {
    fn wake(self: Arc<Self>) {
        // push task back onto the ready queue
    }
}
```

Sources: [Async Book — Task Wakeups with Waker](https://rust-lang.github.io/async-book/02_execution/03_wakeups.html), [std::task::Wake](https://doc.rust-lang.org/stable/std/task/trait.Wake.html), [RusPiro — Context and Waker](https://ruspiro.github.io/ruspiro-async-book/03-03-context-and-waker.html)

### Pin and Self-Referential Futures

#### The Problem

Consider an async function with a local variable and a reference to it across an `.await`:

```rust
async fn example() {
    let data = vec![1, 2, 3];
    let slice = &data[..];       // borrows `data`
    some_io().await;              // <-- state machine stores both `data` AND `slice`
    println!("{:?}", slice);
}
```

The compiler-generated state machine stores both `data` and `slice` as fields. But `slice` is a pointer into `data`. If the entire future were moved to a different memory address (e.g., by `std::mem::swap`), `slice` would become a dangling pointer — **undefined behavior**.

#### The Solution: Pin

`Pin<P>` is a wrapper around a pointer type `P` (like `&mut T` or `Box<T>`) that enforces a contract: **once pinned, the value will not be moved until it is dropped**. The `poll` method takes `self: Pin<&mut Self>` precisely for this reason — it guarantees that the future's memory location is stable between polls.

Types that do NOT contain self-references implement `Unpin` (an auto-trait). For `Unpin` types, `Pin` has no effect — you can move them freely. Most "leaf" types (integers, strings, plain structs) are `Unpin`. The compiler-generated state machines for async functions are `!Unpin` because they may contain self-references.

#### pin-project-lite

When you write a custom `Future` combinator struct that wraps an inner future, you need to "project" the pinned reference into the struct's fields. The `pin_project!` macro from `pin-project-lite` generates safe projection methods:

```rust
use pin_project_lite::pin_project;

pin_project! {
    struct Timeout<F> {
        #[pin]
        future: F,          // pinned: inner future must not move
        deadline: Instant,  // not pinned: can be moved freely
    }
}
```

Sources: [Cloudflare — Pin and Unpin in Rust](https://blog.cloudflare.com/pin-and-unpin-in-rust/), [std::pin](https://doc.rust-lang.org/std/pin/index.html), [pin-project-lite docs](https://docs.rs/pin-project-lite)

### The Executor's Job

An executor is the event loop that drives futures to completion. Its core loop:

1. Pick a task from the ready queue.
2. Create a `Waker` for that task.
3. Call `future.poll(cx)`.
4. If `Ready` — task is done; remove it.
5. If `Pending` — the future has stored the waker; executor parks the task until `waker.wake()` is called.
6. When woken, push the task back onto the ready queue.
7. Repeat.

A minimal single-threaded executor is ~50 lines of Rust. Understanding this loop is the key to understanding everything else — Tokio is just a highly optimized version of this pattern.

Source: [Async Book — Build an Executor](https://rust-lang.github.io/async-book/02_execution/04_executor.html)

### Tokio Architecture

Tokio is a production-grade async runtime comprising three major components:

#### 1. Work-Stealing Scheduler

Tokio's multi-thread runtime spawns N worker threads (default: number of CPU cores). Each worker has:

- **A local run queue** (LIFO slot + fixed-size deque) — newly spawned tasks go here first.
- **Access to a global (injector) queue** — overflow from local queues, or tasks spawned from non-worker threads.

When a worker's local queue is empty, it **steals half the tasks** from a sibling worker's queue. This ensures load balancing without a central lock on every task schedule.

The LIFO slot is a special optimization: the most recently spawned/woken task is placed in a single-element slot and executed immediately. This improves cache locality (the spawning task's data is hot). To prevent starvation, the LIFO slot is disabled after three consecutive uses without processing from the main deque.

Workers check the global queue every `global_queue_interval` ticks (dynamically tuned to ~10ms by default) to avoid starvation of globally-injected tasks.

Sources: [Making the Tokio scheduler 10x faster](https://tokio.rs/blog/2019-10-scheduler), [Tokio runtime docs](https://docs.rs/tokio/latest/tokio/runtime/index.html), [Rust Magazine — How Tokio Schedules Tasks](https://rustmagazine.org/issue-4/how-tokio-schedule-tasks/)

#### 2. Cooperative Scheduling (Budget)

Since Tokio uses cooperative multitasking (no preemption), a task that loops forever in `poll` would starve all other tasks on that worker thread. To mitigate this, Tokio assigns each task a **budget of 128 operations per scheduling tick**. Each `.await` on a Tokio resource (channel recv, socket read, mutex lock) decrements the budget. Once exhausted, all Tokio resources return `Pending` even if they have data, forcing the task to yield.

`tokio::task::unconstrained()` opts a future out of the budget system — it will never be forcibly yielded. Use this only for latency-critical paths where you accept the starvation risk.

`tokio::task::coop::consume_budget().await` lets you manually consume one budget unit in compute-heavy loops, inserting voluntary yield points without touching Tokio resources.

Source: [Reducing tail latencies with automatic cooperative task yielding](https://tokio.rs/blog/2020-04-preemption), [tokio::task::coop](https://docs.rs/tokio/latest/tokio/task/coop/index.html)

#### 3. I/O Driver (Reactor)

Tokio's I/O driver is built on [mio](https://github.com/tokio-rs/mio), which wraps the OS event notification system:

| OS | Mechanism | Model |
|----|-----------|-------|
| Linux | `epoll` | Readiness-based |
| macOS/BSD | `kqueue` | Readiness-based |
| Windows | IOCP | Completion-based |

When you `.await` a `TcpStream::read()`, Tokio:
1. Attempts a non-blocking read. If it would block, registers interest with mio.
2. Returns `Pending`, storing the task's waker.
3. The I/O driver thread calls `mio::Poll::poll()` (which calls `epoll_wait`/`kevent`/IOCP).
4. When the OS signals readiness, the driver wakes the task's waker.
5. The scheduler re-polls the task, which now succeeds.

Source: [Caffeinated Bitstream — Tokio Internals](https://cafbit.com/post/tokio_internals/), [mio on GitHub](https://github.com/tokio-rs/mio)

### `tokio::spawn` vs OS Threads

| Aspect | `tokio::spawn` | `std::thread::spawn` |
|--------|----------------|----------------------|
| Weight | ~few hundred bytes per task | ~8 MB stack per thread |
| Creation cost | Nanoseconds | Microseconds (syscall) |
| Scheduling | Cooperative (runtime) | Preemptive (OS) |
| Cancellation | Drop `JoinHandle` detaches; `abort()` cancels | No clean cancellation (must signal) |
| I/O model | Non-blocking, event-driven | Blocking (or manual non-blocking) |
| Use case | I/O-bound concurrency | CPU-bound parallelism |

**Task cancellation in Tokio**: calling `handle.abort()` causes the next `.await` point to return a cancellation error, and the future is dropped. This means destructors run, but code between two `.await` points is NOT interrupted — async cancellation is cooperative, not preemptive.

Sources: [tokio::task](https://docs.rs/tokio/latest/tokio/task/), [Tokio — Spawning](https://tokio.rs/tokio/tutorial/spawning), [Cybernetist — Tokio Task Cancellation Patterns](https://cybernetist.com/2024/04/19/rust-tokio-task-cancellation-patterns/)

### Common Async Anti-Patterns

1. **Blocking in async context**: Calling `std::thread::sleep()`, `std::sync::Mutex::lock()`, or CPU-heavy computation inside an async task blocks the worker thread, starving all other tasks on it. Use `tokio::time::sleep()`, `tokio::sync::Mutex`, or `tokio::task::spawn_blocking()`.

2. **Holding locks across `.await`**: If you hold a `std::sync::MutexGuard` across an `.await`, the lock is held while the task is suspended — other tasks (possibly on the same thread) trying to acquire it will deadlock. Solution: drop the guard before `.await`, or use `tokio::sync::Mutex`.

3. **Unnecessary spawning**: Not every async operation needs `tokio::spawn`. Spawning adds overhead (task allocation, scheduling). Use `join!` or `select!` for concurrent operations within a single task. Spawn only when you need independent cancellation or parallelism across threads.

4. **Forgetting to wake**: In a manual `Future` implementation, returning `Pending` without arranging for `waker.wake()` causes the task to sleep forever.

Sources: [The Hidden Bottleneck: Blocking in Async Rust](https://cong-or.xyz/blocking-async-rust), [Tokio — Shared State](https://tokio.rs/tokio/tutorial/shared-state), [How to Use async Rust Without Blocking the Runtime](https://oneuptime.com/blog/post/2026-01-07-rust-async-without-blocking/view)

## Description

This practice takes you under the hood of Rust's async machinery. You will:

1. Implement `Future` by hand for a custom type (no async/await sugar).
2. Build a working `Waker` using both the low-level `RawWaker` API and the `Wake` trait.
3. Write a minimal single-threaded executor that polls futures to completion.
4. Use `pin-project-lite` to build a future combinator with pinned fields.
5. Explore Tokio's runtime configuration and observe work-stealing in action.
6. Demonstrate cooperative scheduling budget behavior and starvation scenarios.
7. Capstone: build an async channel from scratch using the `Future`/`Waker` protocol.

## Instructions

### Exercise 1: Manual Future Implementation

**What you learn**: The `Future` trait, `Poll::Ready` vs `Poll::Pending`, and how `poll` drives computation.

Implement `Future` for a `Countdown` struct that counts down from N to 0. Each call to `poll` decrements the counter and returns `Pending` until the counter reaches 0, at which point it returns `Ready(())`. This teaches the most fundamental concept: a future is just a struct with a `poll` method.

**Critical detail**: Even this simple future must call `cx.waker().wake_by_ref()` before returning `Pending`. Without this, no executor will ever re-poll the future, and it hangs forever.

### Exercise 2: Building a Waker

**What you learn**: How `Waker`, `RawWaker`, and `RawWakerVTable` work under the hood.

Build a `Waker` from scratch using two approaches:
- **Low-level**: Construct a `RawWakerVTable` with clone/wake/wake_by_ref/drop functions, and create a `Waker` from `RawWaker`. This is how embedded executors work.
- **High-level**: Implement the `Wake` trait on a struct and use `Arc::into()` to create a `Waker`.

Then use your waker to manually poll the `Countdown` future from Exercise 1.

### Exercise 3: Mini Single-Threaded Executor

**What you learn**: The executor event loop — the core of every async runtime.

Build a `MiniExecutor` that:
1. Accepts futures and wraps them in tasks.
2. Maintains a ready queue (e.g., `VecDeque`).
3. Polls each ready task, using a waker that re-enqueues the task when `wake()` is called.
4. Runs until all tasks complete.

This is a working (toy) async runtime in ~80 lines. Every production runtime (Tokio, async-std, smol) is an optimized version of this loop.

### Exercise 4: Pin and Future Combinators

**What you learn**: Why `Pin` exists, how to use `pin-project-lite`, and how to compose futures.

Build a `Timeout` combinator that wraps any `Future` and a `Duration`. If the inner future completes before the duration, return its value. If the timer fires first, return an error. Use `pin_project!` to safely project pinned fields.

This exercise demonstrates that real-world async code is built by composing future combinators — and `Pin` is the mechanism that makes this safe.

### Exercise 5: Tokio Runtime Configuration

**What you learn**: How to configure Tokio's scheduler and observe work-stealing.

Explore Tokio's `Builder` API to create:
- A `current_thread` runtime (single-threaded, no work-stealing).
- A `multi_thread` runtime with explicit worker count.

Spawn tasks that log their thread ID and observe how tasks migrate between worker threads under the multi-thread scheduler. Compare latency and throughput between the two configurations.

### Exercise 6: Cooperative Scheduling & Budget

**What you learn**: How Tokio prevents task starvation and what `unconstrained` does.

Build a scenario where:
1. A "greedy" task performs many Tokio channel operations in a tight loop.
2. A "fair" task tries to run concurrently.
3. Observe that the greedy task yields automatically due to budget exhaustion.
4. Wrap the greedy task in `tokio::task::unconstrained()` and observe starvation.

Also demonstrate `tokio::task::coop::consume_budget()` for manual yield points in compute-heavy code.

### Exercise 7: Capstone — Async Channel from Scratch

**What you learn**: Putting it all together — `Future`, `Waker`, `Pin`, `Context` in a real data structure.

Build a simple single-producer, single-consumer async channel:
- `Sender::send(value)` stores the value and wakes the receiver.
- `Receiver` implements `Future` — it returns `Pending` if no value is available (storing the waker), and `Ready(value)` when a value arrives.
- Use `Arc<Mutex<SharedState>>` for the shared state between sender and receiver.

This capstone ties together everything: manual `Future` implementation, proper waker management, and safe interior mutability.

## Motivation

- **Debugging async code**: When a task hangs, understanding the waker protocol tells you exactly where to look — some future returned `Pending` without registering a waker.
- **Performance tuning**: Knowing about cooperative scheduling, work-stealing, and budget helps you diagnose latency spikes and starvation in production Tokio services.
- **Writing custom futures**: Libraries like `tower`, `hyper`, and `tonic` require writing manual `Future` implementations. Understanding `Pin`, `poll`, and projection is prerequisite.
- **Avoiding anti-patterns**: Blocking in async, holding locks across `.await`, and unnecessary spawning are the top three Tokio performance killers. Understanding *why* they are harmful (at the executor level) prevents them.
- **Interview depth**: "How does async Rust work under the hood?" is a common senior Rust interview question. This practice gives you the complete answer.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Build | `cargo build` | Compile all exercises (will panic at runtime on `todo!()` but compiles) |
| Check | `cargo check` | Type-check without generating binaries |
| Run all | `cargo run` | Run all exercises sequentially |
| Run one | `cargo run -- 3` | Run only exercise 3 (mini executor) |
| Run (release) | `cargo run --release` | Run with optimizations (relevant for Exercise 5/6 timing) |
| Test | `cargo test` | Run all unit tests |
| Test one | `cargo test ex7` | Run tests for a specific exercise module |
| Clippy | `cargo clippy` | Lint the code for common mistakes |

## Notes

*(To be filled during practice.)*
