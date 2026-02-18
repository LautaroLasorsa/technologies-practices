//! Practice 063b: Memory Safety Verification — Miri & Sanitizers
//!
//! This program runs 7 exercises exploring Miri and sanitizer-based UB detection:
//!
//! 1. Miri basics — safe code baseline & `cfg(miri)`
//! 2. Use-after-free & dangling pointers
//! 3. Stacked Borrows violations
//! 4. Data race detection
//! 5. Tree Borrows & aliasing exploration
//! 6. Writing Miri-clean unsafe code
//! 7. Capstone — fix the bugs
//!
//! Most exercises live in test modules (`cargo miri test exN`).
//! The main binary provides a summary and runs safe demonstrations.
//!
//! Run all tests under Miri: `cargo miri test`
//! Run natively: `cargo test`
//! Run binary under Miri: `cargo miri run`

mod ex1_miri_basics;
mod ex2_dangling_pointers;
mod ex3_stacked_borrows;
mod ex4_data_races;
mod ex5_tree_borrows;
mod ex6_miri_clean_unsafe;
mod ex7_capstone_fix_bugs;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exercise: Option<u32> = args.get(1).and_then(|s| s.parse().ok());

    println!("=== Practice 063b: Memory Safety Verification — Miri & Sanitizers ===");

    if cfg!(miri) {
        println!("[Running under Miri — execution will be slower but UB-checked]");
    } else {
        println!("[Running natively — no UB checking. Use `cargo miri run` for Miri.]");
    }
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
    println!("--- Exercise 1: Miri Basics — Safe Code Baseline ---\n");
    ex1_miri_basics::demonstrate_safe_ops();
    println!();
}

fn run_exercise_2() {
    println!("--- Exercise 2: Use-After-Free & Dangling Pointers ---\n");
    println!("  (Buggy code is in test modules — run `cargo miri test ex2`)");
    ex2_dangling_pointers::demonstrate_valid_raw_ptr();
    println!();
}

fn run_exercise_3() {
    println!("--- Exercise 3: Stacked Borrows Violations ---\n");
    println!("  (Buggy code is in test modules — run `cargo miri test ex3`)");
    ex3_stacked_borrows::demonstrate_split_at_mut_concept();
    println!();
}

fn run_exercise_4() {
    println!("--- Exercise 4: Data Race Detection ---\n");
    println!("  (Race conditions are in test modules — run `cargo miri test ex4`)");
    ex4_data_races::demonstrate_atomic_basics();
    println!();
}

fn run_exercise_5() {
    println!("--- Exercise 5: Tree Borrows & Aliasing Exploration ---\n");
    println!("  Run with default:      cargo miri test ex5");
    println!("  Run with Tree Borrows: MIRIFLAGS=\"-Zmiri-tree-borrows\" cargo miri test ex5");
    println!();
}

fn run_exercise_6() {
    println!("--- Exercise 6: Writing Miri-Clean Unsafe Code ---\n");
    println!("  Run with strict mode:");
    println!("  MIRIFLAGS=\"-Zmiri-strict-provenance -Zmiri-symbolic-alignment-check\" cargo miri test ex6");
    println!();
}

fn run_exercise_7() {
    println!("--- Exercise 7: Capstone — Fix the Bugs ---\n");
    println!("  Run: cargo miri test ex7");
    println!("  Fix all 5 bugs until Miri is happy.");
    println!();
}
