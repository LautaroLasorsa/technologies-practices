//! Practice 063a: Profiling & Flamegraphs
//!
//! This program runs 6 exercises exploring profiling tools and techniques:
//!
//! 1. Tool setup & baseline timing harness
//! 2. Deliberately slow workload with multiple bottlenecks
//! 3. Flamegraph generation & interpretation
//! 4. Heap profiling with DHAT
//! 5. Assembly inspection with cargo-show-asm
//! 6. Guided optimization with before/after comparison
//!
//! Run all exercises:  `cargo run --release`
//! Run one exercise:   `cargo run --release -- <number>`
//!
//! For DHAT heap profiling: `cargo run --release --features dhat-heap -- 4`
//!
//! IMPORTANT: Always use --release for profiling. Debug builds are orders of
//! magnitude slower and produce misleading profiles.

// When the dhat-heap feature is enabled, replace the global allocator with
// DHAT's tracking allocator. This wraps every malloc/free to record allocation
// site, size, and lifetime. When the Profiler is dropped (end of main), it
// writes the data to dhat-heap.json.
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

mod ex1_setup;
mod ex2_workload;
mod ex3_flamegraph;
mod ex4_dhat;
mod ex5_assembly;
mod ex6_optimize;

fn main() {
    // Initialize DHAT profiler if the feature is enabled.
    // The profiler MUST be created before any allocations you want to track,
    // and it writes its output when dropped (at the end of main).
    // The underscore prefix keeps Rust from warning about an unused variable,
    // but the variable IS used — its Drop impl writes the JSON file.
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args: Vec<String> = std::env::args().collect();
    let exercise: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    println!("=== Practice 063a: Profiling & Flamegraphs ===");
    println!();

    match exercise {
        Some(1) => run_exercise_1(),
        Some(2) => run_exercise_2(),
        Some(3) => run_exercise_3(),
        Some(4) => run_exercise_4(),
        Some(5) => run_exercise_5(),
        Some(6) => run_exercise_6(),
        Some(n) => eprintln!("Unknown exercise: {}. Valid range: 1-6", n),
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

    println!();
    println!("{}", "=".repeat(60));
    println!("All exercises completed!");
}

fn run_exercise_1() {
    println!("--- Exercise 1: Tool Setup & Baseline ---\n");

    println!("[1a] Checking profiling tools:");
    ex1_setup::install_tools();
    println!();

    println!("[1b] Timing harness demo:");
    ex1_setup::demo_bench_fn();
    println!();
}

fn run_exercise_2() {
    println!("--- Exercise 2: Deliberately Slow Workload ---\n");

    println!("[2] Running workload with known bottlenecks:");
    ex2_workload::run_workload();
    println!();
}

fn run_exercise_3() {
    println!("--- Exercise 3: Flamegraph Generation & Reading ---\n");

    println!("[3a] Configure workload for profiling:");
    ex3_flamegraph::configure_workload_for_profiling();
    println!();

    println!("[3b] Flamegraph analysis:");
    ex3_flamegraph::analyze_flamegraph();
    println!();
}

fn run_exercise_4() {
    println!("--- Exercise 4: Heap Profiling with DHAT ---\n");

    println!("[4a] Allocation storm:");
    ex4_dhat::allocation_storm();
    println!();

    println!("[4b] DHAT output analysis:");
    ex4_dhat::analyze_dhat_output();
    println!();
}

fn run_exercise_5() {
    println!("--- Exercise 5: Assembly Inspection ---\n");

    println!("[5a] sum_slice:");
    ex5_assembly::demo_sum_slice();
    println!();

    println!("[5b] sum_with_bounds_checks:");
    ex5_assembly::demo_sum_with_bounds_checks();
    println!();

    println!("[5c] dot_product:");
    ex5_assembly::demo_dot_product();
    println!();
}

fn run_exercise_6() {
    println!("--- Exercise 6: Guided Optimization ---\n");

    println!("[6] Before/after comparison:");
    ex6_optimize::run_comparison();
    println!();
}
