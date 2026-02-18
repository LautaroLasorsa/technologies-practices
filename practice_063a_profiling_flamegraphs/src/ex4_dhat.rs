//! Exercise 4: Heap Profiling with DHAT
//!
//! CPU profiling tells you WHERE time is spent. Heap profiling tells you WHERE
//! memory is allocated and how it flows through your program.
//!
//! DHAT (Dynamic Heap Analysis Tool) tracks every heap allocation:
//! - WHERE: Which function/line allocated the memory (full backtrace)
//! - HOW MUCH: Total bytes allocated, peak bytes live at once
//! - HOW LONG: Allocation lifetime (short-lived = potential optimization target)
//! - HOW OFTEN: Number of allocations at each call site
//!
//! Run this exercise with: `cargo run --release --features dhat-heap -- 4`
//! Then open the resulting `dhat-heap.json` at:
//! https://nnethercote.github.io/dh_view/dh_view.html

/// Generate a large number of short-lived heap allocations.
///
/// This function creates a workload designed to produce a dramatic DHAT report.
/// Each iteration of the inner loop allocates and immediately frees memory,
/// creating "allocation churn" — a pattern that is invisible in CPU profiling
/// but shows up clearly in heap profiling.
///
/// In the DHAT viewer, look for:
/// - **"Total (bytes)"**: The call site with the highest total bytes allocated.
///   This is cumulative — even if each allocation is small, doing it millions
///   of times adds up.
/// - **"Total (blocks)"**: Number of distinct allocations. A high block count
///   with low average size indicates many small allocations — a pattern that
///   stresses the allocator.
/// - **"At t-gmax (bytes)"**: Bytes live at the point of peak heap usage. This
///   tells you the high-water mark of memory consumption.
/// - **"Reads" and "Writes"**: How many times the allocated memory was accessed.
///   A high allocation count with low read/write count means you are allocating
///   memory and barely using it — a strong signal for optimization.
pub fn allocation_storm() {
    // TODO(human): Create a function with thousands of short-lived allocations.
    //
    // This exercise teaches you to recognize allocation-heavy patterns and use
    // DHAT to quantify them. The allocation_heavy() function from Exercise 2
    // is one example, but this exercise creates an even more dramatic case
    // specifically designed for DHAT analysis.
    //
    // Steps:
    //
    // 1. Define the iteration count:
    //    ```
    //    let iterations = 100_000;
    //    ```
    //
    // 2. Implement an "allocation storm" pattern — a loop where each iteration
    //    creates temporary heap allocations that are immediately dropped:
    //
    //    Pattern A: String formatting in a loop
    //    ```
    //    let mut results = Vec::new();
    //    for i in 0..iterations {
    //        // Each format! allocates a new String on the heap
    //        let key = format!("key-{:08}", i);
    //        let value = format!("value-{:08}", i * 7);
    //        // Creating a tuple of Strings, then converting to a single String
    //        let combined = format!("{}={}", key, value);
    //        results.push(combined);
    //        // `key` and `value` are dropped here — 2 allocations freed per iteration
    //    }
    //    ```
    //    This creates ~300,000 allocations: 3 per iteration (key, value, combined),
    //    of which 2 are freed each iteration and 1 (combined) is kept.
    //
    //    Pattern B: Vec-of-Vec construction
    //    ```
    //    let mut matrix: Vec<Vec<u64>> = Vec::new();
    //    for i in 0..iterations {
    //        let row: Vec<u64> = (0..10).map(|j| i as u64 * 10 + j).collect();
    //        matrix.push(row);
    //    }
    //    ```
    //    This creates `iterations` inner Vec allocations. Each inner Vec allocates
    //    a separate heap buffer.
    //
    //    Choose one pattern (or combine both) to generate a clear DHAT signal.
    //
    // 3. Use `std::hint::black_box(&results)` (or `&matrix`) to prevent the
    //    compiler from optimizing away the allocations.
    //
    // 4. Print statistics:
    //    ```
    //    println!("  Created {} allocation-heavy items", results.len());
    //    println!("  Run with --features dhat-heap to see DHAT profile");
    //    ```
    //
    // After running with `cargo run --release --features dhat-heap -- 4`:
    // - A file `dhat-heap.json` will be created in the current directory
    // - Open https://nnethercote.github.io/dh_view/dh_view.html
    // - Load the JSON file
    // - Sort by "Total (bytes)" to find the dominant allocation site
    // - Click on the entry to see the full backtrace
    //
    // IMPORTANT: DHAT slows down the program significantly (10-50x on Windows
    // because backtrace collection is slow). Be patient. If it's too slow,
    // reduce `iterations` to 10,000.
    //
    // Why this matters: In production Rust code, excessive allocations are one
    // of the most common performance issues. DHAT makes them visible and
    // quantifiable, turning "I think we allocate too much" into "line 47 of
    // parser.rs allocates 2.3GB total across 1.2M calls, with average lifetime
    // of 3 microseconds — this is a clear buffer-reuse opportunity."

    todo!("Exercise 4a: Create an allocation storm for DHAT profiling")
}

/// Document your DHAT analysis after examining the viewer output.
///
/// The DHAT viewer (https://nnethercote.github.io/dh_view/dh_view.html) shows
/// a tree of allocation sites with several metrics:
///
/// - **Total (bytes)**: Cumulative bytes allocated at this site across the
///   entire run. Sorting by this finds the most allocation-heavy code.
///
/// - **Total (blocks)**: Number of individual allocations. High block count
///   with small average size → allocator overhead dominates.
///
/// - **At t-gmax (bytes)**: Bytes live at peak heap usage. This is the
///   "high-water mark" — the moment when your program used the most memory.
///   Useful for finding memory-hungry data structures.
///
/// - **At t-end (bytes)**: Bytes still allocated when the program exits.
///   Non-zero here suggests memory leaks (in Rust, usually `Box::leak`,
///   `std::mem::forget`, or cyclic `Arc` references).
///
/// - **Reads** and **Writes**: How many times the allocated memory was
///   actually accessed. Low access count relative to allocation size means
///   the memory was barely used — a waste.
pub fn analyze_dhat_output() {
    // TODO(human): Print your DHAT analysis after examining the viewer.
    //
    // After running with --features dhat-heap and loading the JSON in the viewer:
    //
    // Steps:
    //
    // 1. Report the top allocation site:
    //    ```
    //    println!("  Top allocation site: <function name>:<line>");
    //    println!("  Total bytes allocated: <N> MB");
    //    println!("  Total blocks (allocations): <N>");
    //    println!("  Average allocation size: <N> bytes");
    //    ```
    //
    // 2. Report the peak heap usage:
    //    ```
    //    println!("  Peak heap usage (t-gmax): <N> MB");
    //    println!("  This occurred at: <describe when in the program lifecycle>");
    //    ```
    //
    // 3. Identify short-lived allocations:
    //    ```
    //    println!("  Short-lived allocations: <N> blocks freed within the same function");
    //    println!("  These are optimization targets — can be replaced with buffer reuse");
    //    ```
    //
    // 4. Propose the optimization:
    //    ```
    //    println!("  Proposed fix: <describe the buffer-reuse or arena strategy>");
    //    println!("  Expected reduction: <estimate how many allocations would be eliminated>");
    //    ```
    //
    // Why document in code? The DHAT JSON is ephemeral — it's regenerated each run.
    // Your analysis captures the insights permanently. In a real project, you would
    // put this in a PR description or performance report.

    todo!("Exercise 4b: Document your DHAT analysis")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocation_storm_runs() {
        allocation_storm();
    }

    #[test]
    fn test_analyze_dhat_output_runs() {
        analyze_dhat_output();
    }
}
