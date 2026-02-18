//! Exercise 3: Flamegraph Generation & Reading
//!
//! This exercise guides you through the complete flamegraph workflow:
//! 1. Tune workload parameters for meaningful profiling
//! 2. Generate a flamegraph with cargo-flamegraph
//! 3. Open and read the SVG
//! 4. Identify the dominant bottleneck
//!
//! The flamegraph is generated from the COMMAND LINE, not from code.
//! This module provides support functions for configuring the workload
//! and recording your analysis.

use crate::ex2_workload;

/// Configure workload parameters for profiling.
///
/// When profiling, the workload must run long enough for the sampling profiler
/// to collect a statistically meaningful number of samples. At a typical
/// sampling rate of 1000 Hz (1000 samples/second):
///
/// - 1 second  → 1,000 samples  — barely enough for a rough flamegraph
/// - 5 seconds → 5,000 samples  — adequate for identifying major bottlenecks
/// - 30 seconds → 30,000 samples — excellent for detailed analysis
///
/// Too short: the flamegraph will be noisy, with functions appearing/disappearing
/// between runs due to statistical variance. Too long: you waste time waiting.
///
/// For this exercise, aim for 5-15 seconds of execution time.
///
/// After implementing this function:
/// 1. Run it once to verify the timing: `cargo run --release -- 3`
/// 2. Generate the flamegraph (from an ADMINISTRATOR terminal):
///    `cargo flamegraph --release -o before.svg -- 2`
/// 3. Open `before.svg` in a browser — it's an interactive SVG
/// 4. Identify which of the three bottleneck functions is widest
pub fn configure_workload_for_profiling() {
    // TODO(human): Choose a workload size that runs for 5-15 seconds and run it.
    //
    // This teaches the practical skill of tuning profiling runs. Getting the
    // duration right is more important than it seems:
    // - Too short: statistical noise dominates, you might optimize the wrong thing
    // - Too long: you waste time on every profile-edit-profile cycle
    //
    // Steps:
    //
    // 1. Start with the DEFAULT_SIZE from ex2_workload (15,000).
    //    Run `ex2_workload::run_workload()` and check the total time.
    //
    // 2. If the total time is much less than 5 seconds, increase the size.
    //    If much more than 15 seconds, decrease it.
    //    Remember: quadratic_search is O(n^2), so doubling n quadruples its time.
    //
    // 3. A good approach: define a `profiling_size` constant and pass it to
    //    `ex2_workload::run_workload_with_size(profiling_size)`.
    //
    // 4. Print the chosen size and expected runtime:
    //    ```
    //    let profiling_size = 20_000; // Adjust based on your machine
    //    println!("  Profiling workload size: {}", profiling_size);
    //    println!("  Target: 5-15 seconds total runtime");
    //    ex2_workload::run_workload_with_size(profiling_size);
    //    ```
    //
    // 5. Once you have the right size, use it when generating the flamegraph:
    //    The binary argument `-- 3` will run this function, OR you can modify
    //    `run_workload_with_size` calls to use your tuned size.
    //
    // Tip: The quadratic search dominates at large sizes. If it takes >30s and
    // the other functions take <1s, you might want a smaller size so all three
    // bottlenecks are visible in the flamegraph (otherwise the flamegraph shows
    // only quadratic_search).
    //
    // A balanced approach: use different sizes for each function:
    //   - quadratic_search: smaller size (e.g., 10,000) so it takes ~3-5s
    //   - allocation_heavy: larger size (e.g., 500,000) so it takes ~2-3s
    //   - cache_unfriendly: large size (e.g., 2,000,000) so it takes ~2-3s
    // This way, all three are visible in the flamegraph.

    todo!("Exercise 3a: Configure workload size for meaningful profiling")
}

/// Record your flamegraph analysis.
///
/// After generating and examining the flamegraph SVG, implement this function
/// to document what you found. This is the analysis step — the most important
/// part of profiling. A flamegraph is useless if you cannot interpret it.
///
/// How to read the flamegraph:
///
/// 1. **Find the widest bars at the TOP of the graph**. These are the "leaf"
///    functions — the ones actually executing when samples were taken. Width
///    = proportion of total CPU time.
///
/// 2. **Follow the stack downward** from a wide leaf to see the call path.
///    If `quadratic_search` is the leaf and `run_workload_with_size` is below
///    it, you know that run_workload_with_size's time is dominated by
///    quadratic_search.
///
/// 3. **Look for unexpected widths**. If a function you thought was fast turns
///    out to be wide, that's the profiling discovery — it contradicts your mental
///    model and reveals a real bottleneck.
///
/// 4. **Click to zoom**. Click any bar to zoom in, making it full-width so you
///    can see the relative proportions of its children.
///
/// 5. **Note the self-time vs total-time distinction**:
///    - "Self time" = samples where this function is the leaf (executing its own code)
///    - "Total time" = samples where this function appears anywhere in the stack
///    A function with high total-time but low self-time is an orchestrator —
///    it delegates work. A function with high self-time is doing the actual work.
pub fn analyze_flamegraph() {
    // TODO(human): Print your flamegraph analysis.
    //
    // After generating and examining the flamegraph (cargo flamegraph --release),
    // fill in this function with your observations.
    //
    // Steps:
    //
    // 1. Report the dominant function:
    //    ```
    //    println!("  Dominant function: <name>");
    //    println!("  Estimated CPU share: ~<N>%");
    //    ```
    //    This should be `quadratic_search` if you used a large enough size,
    //    because O(n^2) grows fastest.
    //
    // 2. Report the second and third largest consumers:
    //    ```
    //    println!("  Second: <name> (~<N>%)");
    //    println!("  Third: <name> (~<N>%)");
    //    ```
    //
    // 3. For each bottleneck, state the type of problem:
    //    - quadratic_search: "Algorithmic complexity — O(n^2) nested loop"
    //    - allocation_heavy: "Allocation overhead — many short-lived heap objects"
    //    - cache_unfriendly: "Memory access pattern — random access causes cache misses"
    //
    // 4. State which optimization you would apply to each:
    //    - quadratic_search → HashSet for O(n) lookup
    //    - allocation_heavy → Reuse a buffer instead of allocating per iteration
    //    - cache_unfriendly → Sequential access instead of random
    //
    // 5. Note any SURPRISES — things that did not match your expectations.
    //    Did the flamegraph show time in unexpected places?
    //    (e.g., time in `alloc::alloc`, `_RNvNtCs...::drop`, `core::ptr::drop_in_place`)
    //    These are allocation/deallocation functions called by allocation_heavy.
    //
    // Why document the analysis in code? The analysis is more valuable than the
    // flamegraph SVG. The SVG is a snapshot; the analysis captures your understanding
    // and becomes a reference for future profiling sessions.

    todo!("Exercise 3b: Document your flamegraph analysis")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_configure_runs_without_panic() {
        // Just verify it executes — timing is machine-dependent
        configure_workload_for_profiling();
    }

    #[test]
    fn test_analyze_runs_without_panic() {
        analyze_flamegraph();
    }
}
