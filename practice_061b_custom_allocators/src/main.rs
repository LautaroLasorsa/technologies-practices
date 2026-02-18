//! Practice 061b: Custom Allocators — GlobalAlloc, jemalloc & Arena Patterns
//!
//! This program runs 6 exercises exploring Rust's allocation model:
//!
//! 1. Building a tracking allocator (GlobalAlloc wrapper)
//! 2. jemalloc as global allocator (Linux/WSL) / simulated on Windows
//! 3. bumpalo arena basics
//! 4. bumpalo collections (Vec, String)
//! 5. Phase-based arena pipeline
//! 6. Comparative benchmark: System vs jemalloc vs bump
//!
//! Run all exercises: `cargo run`
//! Run one exercise: `cargo run -- <number>`

mod ex1_tracking_allocator;
mod ex2_jemalloc;
mod ex3_arena_basics;
mod ex4_arena_collections;
mod ex5_arena_pipeline;
mod ex6_benchmark;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exercise: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    println!("=== Practice 061b: Custom Allocators — GlobalAlloc, jemalloc & Arena Patterns ===");
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
    println!("--- Exercise 1: Tracking Allocator ---\n");

    println!("[1a] Tracking allocator GlobalAlloc implementation:");
    ex1_tracking_allocator::demo_tracking_allocator();
    println!();

    println!("[1b] Measuring Vec allocations:");
    ex1_tracking_allocator::measure_vec_allocations();
    println!();

    println!("[1c] Measuring String allocations:");
    ex1_tracking_allocator::measure_string_allocations();
    println!();
}

fn run_exercise_2() {
    println!("--- Exercise 2: jemalloc as Global Allocator ---\n");

    println!("[2a] Multi-threaded allocation benchmark:");
    ex2_jemalloc::multi_threaded_allocation_benchmark();
    println!();

    println!("[2b] Fragmentation stress test:");
    ex2_jemalloc::fragmentation_stress_test();
    println!();
}

fn run_exercise_3() {
    println!("--- Exercise 3: bumpalo Arena Basics ---\n");

    println!("[3a] Arena lifecycle (create, alloc, use, reset):");
    ex3_arena_basics::arena_lifecycle();
    println!();

    println!("[3b] Arena vs heap benchmark:");
    ex3_arena_basics::arena_vs_heap_benchmark();
    println!();
}

fn run_exercise_4() {
    println!("--- Exercise 4: bumpalo Collections ---\n");

    println!("[4a] Arena Vec operations:");
    ex4_arena_collections::arena_vec_operations();
    println!();

    println!("[4b] Arena String building:");
    ex4_arena_collections::arena_string_building();
    println!();

    println!("[4c] Arena nested structures:");
    ex4_arena_collections::arena_nested_structures();
    println!();
}

fn run_exercise_5() {
    println!("--- Exercise 5: Phase-Based Arena Pipeline ---\n");

    println!("[5a] Three-phase arena pipeline:");
    ex5_arena_pipeline::run_arena_pipeline();
    println!();

    println!("[5b] Memory measurement across phases:");
    ex5_arena_pipeline::measure_arena_pipeline_memory();
    println!();
}

fn run_exercise_6() {
    println!("--- Exercise 6: Comparative Benchmark ---\n");

    println!("[6] System vs jemalloc vs bump comparison:");
    ex6_benchmark::run_comparison();
    println!();
}
