//! Practice 061a: Lock-Free Rust — crossbeam & Epoch-Based Reclamation
//!
//! This program runs 7 exercises exploring crossbeam's lock-free primitives:
//!
//! 1. crossbeam-channel & select! macro
//! 2. CachePadded & Backoff (false sharing, spin-wait)
//! 3. Lock-free queues (SegQueue, ArrayQueue)
//! 4. Epoch-based reclamation fundamentals
//! 5. Treiber stack implementation
//! 6. Concurrent SkipMap
//! 7. Benchmarking lock-free vs Mutex
//!
//! Run all exercises: `cargo run`
//! Run one exercise: `cargo run -- <number>`

mod ex1_channels;
mod ex2_utils;
mod ex3_queues;
mod ex4_epoch;
mod ex5_treiber_stack;
mod ex6_skipmap;
mod ex7_benchmark;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exercise: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    println!("=== Practice 061a: Lock-Free Rust — crossbeam & Epoch-Based Reclamation ===");
    println!();

    match exercise {
        Some(1) => run_exercise_1(),
        Some(2) => run_exercise_2(),
        Some(3) => run_exercise_3(),
        Some(4) => run_exercise_4(),
        Some(5) => run_exercise_5(),
        Some(6) => run_exercise_6(),
        Some(7) => run_exercise_7(),
        Some(n) => eprintln!("Unknown exercise: {}. Valid range: 1-7", n),
        None => run_all(),
    }
}

fn run_all() {
    run_exercise_1();
    run_exercise_2();
    run_exercise_3();
    run_exercise_4();
    run_exercise_5();
    run_exercise_6();
    run_exercise_7();

    println!();
    println!("{}", "=".repeat(60));
    println!("All exercises completed!");
}

fn run_exercise_1() {
    println!("--- Exercise 1: crossbeam-channel & select! ---\n");

    println!("[1a] Fan-out / fan-in with bounded channel:");
    ex1_channels::fan_out_fan_in();
    println!();

    println!("[1b] select! macro with timeout:");
    ex1_channels::select_timeout();
    println!();
}

fn run_exercise_2() {
    println!("--- Exercise 2: CachePadded & Backoff ---\n");

    println!("[2a] False sharing benchmark:");
    ex2_utils::benchmark_false_sharing();
    println!();

    println!("[2b] Spin-wait with Backoff:");
    ex2_utils::spin_with_backoff();
    println!();
}

fn run_exercise_3() {
    println!("--- Exercise 3: Lock-Free Queues ---\n");

    println!("[3a] SegQueue MPMC:");
    ex3_queues::segqueue_mpmc();
    println!();

    println!("[3b] ArrayQueue bounded:");
    ex3_queues::arrayqueue_bounded();
    println!();
}

fn run_exercise_4() {
    println!("--- Exercise 4: Epoch-Based Reclamation Fundamentals ---\n");

    println!("[4a] Atomic swap with epoch:");
    ex4_epoch::atomic_swap_with_epoch();
    println!();

    println!("[4b] Concurrent atomic counter:");
    ex4_epoch::concurrent_atomic_counter();
    println!();
}

fn run_exercise_5() {
    println!("--- Exercise 5: Treiber Stack ---\n");

    println!("[5] Treiber stack push/pop + stress test:");
    ex5_treiber_stack::run_treiber_demo();
    println!();
}

fn run_exercise_6() {
    println!("--- Exercise 6: Concurrent SkipMap ---\n");

    println!("[6a] Concurrent ordered insert:");
    ex6_skipmap::concurrent_ordered_insert();
    println!();

    println!("[6b] Range queries under contention:");
    ex6_skipmap::range_queries_under_contention();
    println!();
}

fn run_exercise_7() {
    println!("--- Exercise 7: Benchmarking Lock-Free vs Mutex ---\n");

    println!("[7] Lock-free vs Mutex comparison:");
    ex7_benchmark::run_comparison();
    println!();
}
