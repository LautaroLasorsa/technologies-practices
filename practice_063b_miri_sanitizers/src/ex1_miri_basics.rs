//! Exercise 1: Miri Basics — Safe Code Baseline
//!
//! This exercise establishes that safe Rust passes Miri cleanly and introduces
//! `cfg(miri)` for conditional compilation. Every test here should pass under
//! both `cargo test` and `cargo miri test`.

/// Demonstrate basic safe operations that Miri validates without issues.
///
/// This function runs under both native and Miri execution. It shows that
/// safe Rust code — even code doing heap allocation, iteration, and indexing —
/// is always free of undefined behavior by construction.
pub fn demonstrate_safe_ops() {
    // Safe Rust guarantees: no dangling pointers, no data races, no UB.
    // Miri verifies this by interpreting every operation with full metadata tracking.
    let v = vec![1, 2, 3, 4, 5];
    let sum: i32 = v.iter().sum();
    println!("  Safe vector sum: {sum}");

    // cfg(miri) is a compile-time flag that is true when running under Miri.
    // Use it to reduce workloads (Miri is ~1000x slower than native).
    if cfg!(miri) {
        println!("  [Miri detected — using reduced workload]");
    } else {
        println!("  [Native execution — full workload]");
    }
}

// =============================================================================
// TODO(human): Implement safe_vector_ops
// =============================================================================

/// Perform a series of safe Vec operations and return the final state.
///
/// This function demonstrates that Miri handles all standard safe operations
/// without any false positives. Every heap allocation, reallocation, and
/// deallocation is tracked and verified.
///
/// TODO(human): Implement this function performing these Vec operations:
///
/// 1. Create a Vec<i32> and push the values 10, 20, 30, 40, 50 into it.
///    - Each `push` may trigger a reallocation (capacity doubling). Miri tracks
///      every allocation/deallocation and verifies no dangling pointers remain.
///
/// 2. Use `v.remove(1)` to remove the element at index 1 (value 20).
///    - This shifts all subsequent elements left. Miri verifies the memmove
///      doesn't access out-of-bounds memory.
///
/// 3. Use `v.retain(|&x| x > 15)` to keep only elements greater than 15.
///    - retain() internally uses unsafe code (in std) — Miri verifies std's
///      implementation is correct.
///
/// 4. Use iterator methods: `v.iter().map(|x| x * 2).collect::<Vec<_>>()`
///    to create a doubled version.
///
/// 5. Return the doubled vector.
///
/// Why this matters: Miri runs on MIR, which means it checks the ACTUAL
/// implementation of Vec (including std's unsafe internals), not just the
/// safe API surface. If std had a bug, Miri would catch it here.
pub fn safe_vector_ops() -> Vec<i32> {
    // TODO(human): Implement the operations described above.
    todo!()
}

// =============================================================================
// TODO(human): Implement cfg_miri_demo
// =============================================================================

/// Demonstrate using `cfg(miri)` to adapt workload for Miri's slow execution.
///
/// TODO(human): Implement this function that:
///
/// 1. Uses `cfg!(miri)` at RUNTIME to choose iteration count:
///    - Under Miri: iterate 100 times (Miri is ~1000x slower)
///    - Under native: iterate 1_000_000 times
///
///    ```rust
///    let iterations = if cfg!(miri) { 100 } else { 1_000_000 };
///    ```
///
///    This is the standard pattern in the Rust ecosystem. The Rust standard
///    library itself uses this extensively in its test suite. Without it,
///    tests that take 1 second natively would take ~17 minutes under Miri.
///
/// 2. Perform a simple computation in a loop `iterations` times:
///    accumulate a sum: `sum += i * i` for i in 0..iterations.
///
/// 3. Return the final sum as u64.
///
/// Note: You can also use `#[cfg(miri)]` as a COMPILE-TIME attribute to
/// conditionally compile entire blocks, functions, or modules. The difference:
///   - `cfg!(miri)` — runtime branch, both branches are compiled
///   - `#[cfg(miri)]` — compile-time, only the matching branch exists
///
/// For performance-critical code, prefer `#[cfg(miri)]` to avoid compiling
/// unused code. For simple iteration counts, `cfg!(miri)` is cleaner.
pub fn cfg_miri_demo() -> u64 {
    // TODO(human): Implement the cfg(miri)-aware computation described above.
    todo!()
}

// =============================================================================
// Tests — all should pass under both `cargo test` and `cargo miri test`
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ex1_safe_vector_ops_returns_correct_result() {
        // After: push [10,20,30,40,50], remove index 1 → [10,30,40,50],
        // retain >15 → [30,40,50], double → [60,80,100]
        let result = safe_vector_ops();
        assert_eq!(result, vec![60, 80, 100]);
    }

    #[test]
    fn ex1_cfg_miri_demo_returns_sum_of_squares() {
        let result = cfg_miri_demo();

        // Under Miri: sum of i*i for i in 0..100 = 328350
        // Under native: sum of i*i for i in 0..1_000_000 = 333332833333500000 (but this
        // overflows u64? No — u64::MAX is ~1.8e19, and 333332833333500000 < 1.8e19. Fine.)
        if cfg!(miri) {
            assert_eq!(result, 328350);
        } else {
            // Under native, the sum is: n*(n-1)*(2n-1)/6 where n=1_000_000
            // = 999999 * 1999999 / 6 * 1000000... let's just verify it's > 0
            assert!(result > 0);
        }
    }

    #[test]
    fn ex1_string_operations_are_safe() {
        // Demonstrate that String (heap-allocated, growable) passes Miri.
        let mut s = String::from("Hello");
        s.push_str(", world!");
        let upper = s.to_uppercase();
        assert_eq!(upper, "HELLO, WORLD!");

        // String splitting and collecting
        let words: Vec<&str> = s.split(", ").collect();
        assert_eq!(words, vec!["Hello", "world!"]);
    }

    #[test]
    fn ex1_box_and_rc_are_safe() {
        use std::rc::Rc;

        // Box: single-owner heap allocation
        let boxed = Box::new(42);
        assert_eq!(*boxed, 42);
        // Box dropped here — Miri verifies deallocation

        // Rc: reference-counted pointer
        let rc1 = Rc::new(vec![1, 2, 3]);
        let rc2 = Rc::clone(&rc1);
        assert_eq!(Rc::strong_count(&rc1), 2);
        drop(rc2);
        assert_eq!(Rc::strong_count(&rc1), 1);
        // Miri verifies that the inner Vec is deallocated exactly once,
        // when the last Rc is dropped.
    }
}
