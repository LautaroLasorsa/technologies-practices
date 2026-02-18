//! Unsafe Rust & FFI Practice — Raw Pointers, bindgen & repr(C)
//!
//! This program runs 7 exercises that progressively teach unsafe Rust and FFI:
//!
//! 1. Raw pointer basics (create, cast, dereference, arithmetic)
//! 2. Unsafe functions & safe wrappers (the core abstraction pattern)
//! 3. Calling libc from Rust (strlen, qsort via the libc crate)
//! 4. Calling a custom C library (hand-written extern "C" declarations)
//! 5. repr(C) structs & complex data across FFI boundary
//! 6. bindgen auto-generated bindings (compare with hand-written)
//! 7. Capstone: bidirectional FFI (Rust→C and C→Rust)
//!
//! Usage:
//!   cargo run          # Run all exercises
//!   cargo run -- 3     # Run only exercise 3

mod ex1_raw_pointers;
mod ex2_unsafe_functions;
mod ex3_libc_calls;
mod ex4_manual_ffi;
mod ex5_repr_c_structs;
mod ex6_bindgen;
mod ex7_capstone;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let exercise_filter: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    if let Some(n) = exercise_filter {
        println!("Running exercise {} only\n", n);
    } else {
        println!("Running all exercises\n");
    }

    let exercises: Vec<(u32, &str, fn())> = vec![
        (1, "Raw Pointer Fundamentals", ex1_raw_pointers::run),
        (2, "Unsafe Functions & Safe Wrappers", ex2_unsafe_functions::run),
        (3, "Calling libc from Rust", ex3_libc_calls::run),
        (4, "Custom C Library via Manual FFI", ex4_manual_ffi::run),
        (5, "repr(C) Structs & Complex Data", ex5_repr_c_structs::run),
        (6, "bindgen Auto-Generated Bindings", ex6_bindgen::run),
        (7, "Capstone: Bidirectional FFI", ex7_capstone::run),
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
