//! Async Runtime Internals — Future, Waker & Tokio Architecture
//!
//! This program runs 7 exercises that progressively teach how Rust's async
//! machinery works under the hood:
//!
//! 1. Manual Future implementation (Countdown that polls to completion)
//! 2. Building a Waker from scratch (RawWaker + Wake trait)
//! 3. Mini single-threaded executor (the core event loop)
//! 4. Pin and future combinators (Timeout via pin-project-lite)
//! 5. Tokio runtime configuration (current_thread vs multi_thread, work-stealing)
//! 6. Cooperative scheduling & budget (starvation, unconstrained, consume_budget)
//! 7. Capstone: async channel from scratch (Future + Waker in a real data structure)
//!
//! Usage:
//!   cargo run          # Run all exercises
//!   cargo run -- 3     # Run only exercise 3

mod ex1_manual_future;
mod ex2_waker;
mod ex3_mini_executor;
mod ex4_pin_combinator;
mod ex5_tokio_runtime;
mod ex6_coop_scheduling;
mod ex7_async_channel;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let exercise_filter: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    if let Some(n) = exercise_filter {
        println!("Running exercise {} only\n", n);
    } else {
        println!("Running all exercises\n");
    }

    // Exercises 1-4 run without Tokio (they build the concepts from scratch).
    // Exercises 5-7 require Tokio and run inside a runtime.
    let exercises: Vec<(u32, &str, fn())> = vec![
        (1, "Manual Future Implementation", ex1_manual_future::run),
        (2, "Building a Waker", ex2_waker::run),
        (3, "Mini Single-Threaded Executor", ex3_mini_executor::run),
        (4, "Pin and Future Combinators", ex4_pin_combinator::run),
        (5, "Tokio Runtime Configuration", ex5_tokio_runtime::run),
        (6, "Cooperative Scheduling & Budget", ex6_coop_scheduling::run),
        (7, "Capstone: Async Channel from Scratch", ex7_async_channel::run),
    ];

    for (num, name, run_fn) in &exercises {
        if exercise_filter.is_some_and(|f| f != *num) {
            continue;
        }

        println!("{}", "=".repeat(64));
        println!("  Exercise {}: {}", num, name);
        println!("{}", "=".repeat(64));
        println!();

        run_fn();

        println!();
    }

    println!("Done.");
}
