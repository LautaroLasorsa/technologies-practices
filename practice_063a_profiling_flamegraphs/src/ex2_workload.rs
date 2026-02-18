//! Exercise 2: The Deliberately Slow Workload
//!
//! This module contains a program with three intentional performance bottlenecks
//! of different types. You will use profiling tools to identify and characterize
//! each one before optimizing them in Exercise 6.
//!
//! The three bottleneck types:
//! 1. **Algorithmic** — O(n^2) where O(n) exists (dominates CPU flamegraph)
//! 2. **Allocation** — Excessive heap allocations in a hot loop (visible in DHAT)
//! 3. **Cache** — Random memory access pattern causes cache misses (subtle in flamegraph)

use crate::ex1_setup::bench_fn;

/// The default workload size. Tuned so the total workload runs in ~5-15 seconds
/// in release mode — long enough for meaningful profiling, short enough to iterate.
pub const DEFAULT_SIZE: usize = 15_000;

/// Bottleneck 1: Quadratic duplicate detection.
///
/// Given a vector of integers, find all values that appear more than once.
/// The WRONG way: for each element, scan the entire vector to check for duplicates.
/// This is O(n^2) and becomes the dominant bottleneck for large n.
///
/// In a flamegraph, this function will appear as a very wide bar because it
/// consumes the most CPU time. The nested loop means the inner iterations
/// show up as a plateau at the top of the flame — the CPU is spending most
/// of its time in the comparison logic.
///
/// The RIGHT way (Exercise 6): use a HashSet for O(n) duplicate detection.
/// After optimization, this function should virtually disappear from the flamegraph.
pub fn quadratic_search(data: &[u64]) -> Vec<u64> {
    // TODO(human): Implement O(n^2) duplicate detection using nested loops.
    //
    // This is the classic "wrong algorithm" bottleneck. It is deliberately bad —
    // the goal is to create a clear signal in the flamegraph so you can practice
    // identifying it.
    //
    // Steps:
    //
    // 1. Create an empty `Vec<u64>` to hold duplicates found.
    //
    // 2. For each element `data[i]` (outer loop, i from 0 to data.len()):
    //    a. For each element `data[j]` (inner loop, j from i+1 to data.len()):
    //       - If `data[i] == data[j]`, add `data[i]` to the duplicates vector
    //         (if not already present — check with `.contains()`)
    //       - The `.contains()` on the result vector adds ANOTHER linear scan,
    //         making this even worse than pure O(n^2)
    //
    // 3. Return the duplicates vector.
    //
    // Expected behavior: For DEFAULT_SIZE=15000, this should take several seconds
    // in release mode. The nested loop performs ~15000^2/2 = ~112M comparisons.
    //
    // Why this specific bottleneck? Algorithmic complexity bugs are the most
    // impactful performance issues. A 10x constant-factor optimization cannot
    // save an O(n^2) algorithm from being slower than an O(n) one for large n.
    // Profiling makes these bugs immediately visible — the function is simply
    // WIDE in the flamegraph, consuming a disproportionate share of samples.
    //
    // After implementing, run: `cargo run --release -- 2`
    // You should see this function taking multiple seconds.

    todo!("Exercise 2a: Implement O(n^2) duplicate search")
}

/// Bottleneck 2: Allocation-heavy string processing.
///
/// Processes a list of numbers by converting each to a formatted string,
/// performing string manipulation, and collecting the results. The WRONG way:
/// allocate a new String on every iteration.
///
/// In a flamegraph, this may not look dramatically wide because individual
/// allocations are fast. But in a DHAT profile, this function will stand out
/// as the dominant allocation site — thousands of short-lived Strings that are
/// created, used briefly, and immediately dropped.
///
/// The allocation overhead comes from:
/// - `malloc` system call for each String (OS kernel overhead)
/// - Initializing the String's buffer
/// - `free` system call when the String is dropped
/// - Cache pollution: each new allocation may land on a different cache line,
///   evicting useful data from L1/L2 cache
///
/// The RIGHT way (Exercise 6): pre-allocate a String buffer and reuse it
/// with `.clear()` each iteration.
pub fn allocation_heavy(data: &[u64]) -> Vec<String> {
    // TODO(human): Implement allocation-heavy string processing.
    //
    // This bottleneck demonstrates a pattern that is extremely common in
    // real-world Rust code — and in Python/Java code ported to Rust. The
    // programmer creates temporary allocations inside a loop without realizing
    // the cumulative cost.
    //
    // Steps:
    //
    // 1. Create an empty `Vec<String>` to collect results.
    //    Pre-allocate with `Vec::with_capacity(data.len())` — this is the
    //    OUTER allocation, which is fine (done once).
    //
    // 2. For each value in `data`:
    //    a. Create a new formatted String: `let s = format!("item-{:010}", value);`
    //       This allocates a new String on the heap EVERY iteration.
    //
    //    b. Perform some string manipulation that creates MORE allocations:
    //       - `let upper = s.to_uppercase();`  // Another allocation
    //       - `let reversed: String = upper.chars().rev().collect();`  // Yet another
    //       - `let trimmed = reversed.trim().to_string();`  // And another
    //
    //    c. Push the final result: `results.push(trimmed);`
    //       The intermediate Strings (s, upper, reversed) are dropped here,
    //       triggering 3 deallocations per iteration.
    //
    // 3. Return the results vector.
    //
    // Expected behavior: For DEFAULT_SIZE=15000, this should create ~60,000
    // heap allocations (4 per iteration). In a DHAT profile, you will see this
    // function's allocation sites clustered at the top of the "total bytes" ranking.
    //
    // Why this specific bottleneck? Allocation overhead is one of the most common
    // performance issues in Rust, especially for developers coming from GC'd languages
    // where allocations feel "free." In Rust, each allocation is an explicit malloc()
    // call to the system allocator. DHAT makes these invisible costs visible.
    //
    // After implementing, run with DHAT:
    //   cargo run --release --features dhat-heap -- 2

    todo!("Exercise 2b: Implement allocation-heavy string processing")
}

/// Bottleneck 3: Cache-unfriendly random access.
///
/// Traverses a large array in a random access pattern instead of sequentially.
/// This causes frequent cache misses: each access may need to fetch data from
/// main memory (~100ns) instead of L1 cache (~1ns).
///
/// In a flamegraph, this function appears as a single wide bar — but its width
/// is disproportionate to the number of instructions executed. The CPU is actually
/// idle much of the time, stalled waiting for memory. This is a MEMORY-BOUND
/// bottleneck, not a CPU-BOUND one. The flamegraph alone cannot distinguish
/// between the two — you need hardware performance counters (not covered in this
/// practice) or the `perf stat` command to see the cache miss rate.
///
/// The key lesson: flamegraphs show WHERE time is spent, but not WHY. A function
/// might be slow because of bad algorithms (Exercise 2a), excessive allocations
/// (Exercise 2b), or memory access patterns (this exercise). You need different
/// tools for each diagnosis.
pub fn cache_unfriendly(size: usize) -> u64 {
    // TODO(human): Implement a cache-unfriendly access pattern.
    //
    // This bottleneck is the subtlest of the three. Unlike quadratic_search()
    // which does more WORK, this function does the SAME amount of work as a
    // sequential traversal but much SLOWER because of cache misses.
    //
    // Steps:
    //
    // 1. Create a large data array: `let data: Vec<u64> = (0..size as u64).collect();`
    //    Make size large enough to exceed L2 cache (typically 256KB-1MB).
    //    For u64 (8 bytes each), 1M elements = 8MB — well beyond any cache level.
    //
    // 2. Create a random index permutation using the rand crate:
    //    ```
    //    use rand::seq::SliceRandom;
    //    use rand::SeedableRng;
    //    let mut rng = rand::rngs::StdRng::seed_from_u64(42); // Deterministic for reproducibility
    //    let mut indices: Vec<usize> = (0..size).collect();
    //    indices.shuffle(&mut rng);
    //    ```
    //    Using a seeded RNG makes the access pattern reproducible across runs,
    //    which is important for before/after comparisons.
    //
    // 3. Traverse the data array using the random index order:
    //    ```
    //    let mut sum: u64 = 0;
    //    for &idx in &indices {
    //        sum = sum.wrapping_add(data[idx]);
    //    }
    //    ```
    //    Each access jumps to a random location in the 8MB array. Because the
    //    CPU prefetcher cannot predict the next address, almost every access is
    //    a cache miss, hitting L3 cache or main memory.
    //
    // 4. Return `sum` to prevent the compiler from optimizing away the traversal.
    //
    // Expected behavior: This function will be ~5-10x slower than a sequential
    // sum of the same data, despite performing the exact same number of additions.
    // The difference is entirely due to cache performance.
    //
    // After optimization (Exercise 6): Simply traversing `data` sequentially
    // (in order of memory layout, not random index order) will show a dramatic
    // speedup because the CPU prefetcher can fetch cache lines ahead of time.
    //
    // Why this specific bottleneck? Cache performance is the most underappreciated
    // factor in modern software performance. CPUs are ~100x faster than main memory.
    // An algorithm that is O(n) but cache-unfriendly can be slower than an algorithm
    // that does 2-3x more work but accesses memory sequentially.

    todo!("Exercise 2c: Implement cache-unfriendly random access pattern")
}

/// Orchestrator: runs all three bottleneck functions and reports timing.
///
/// This is pre-built — it calls the three bottleneck functions and uses
/// `bench_fn` to time each one. The output gives you a baseline to compare
/// against after optimization in Exercise 6.
pub fn run_workload() {
    run_workload_with_size(DEFAULT_SIZE);
}

/// Runs the workload with a configurable size parameter.
///
/// Used by Exercise 3 to adjust the workload for profiling (need longer runs
/// for meaningful flamegraphs) and Exercise 6 for before/after comparison.
pub fn run_workload_with_size(size: usize) {
    println!("  Workload size: {}", size);
    println!();

    // Generate test data with some duplicates for quadratic_search to find
    let data: Vec<u64> = (0..size as u64)
        .map(|i| i % (size as u64 / 2)) // ~50% duplicates
        .collect();

    let (_, dupes) = bench_fn("quadratic_search", || quadratic_search(&data));
    println!("  Found {} duplicates", dupes.len());

    let (_, strings) = bench_fn("allocation_heavy", || allocation_heavy(&data));
    println!("  Processed {} strings", strings.len());

    // Use a larger size for cache test since it needs to exceed cache size
    let cache_size = size.max(500_000);
    let (_, sum) = bench_fn("cache_unfriendly", || cache_unfriendly(cache_size));
    println!("  Sum: {} (prevents dead-code elimination)", sum);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quadratic_search_finds_duplicates() {
        let data = vec![1, 2, 3, 2, 4, 3, 5];
        let dupes = quadratic_search(&data);
        assert!(dupes.contains(&2), "Should find 2 as duplicate");
        assert!(dupes.contains(&3), "Should find 3 as duplicate");
        assert!(!dupes.contains(&1), "1 is not a duplicate");
    }

    #[test]
    fn test_quadratic_search_no_duplicates() {
        let data = vec![1, 2, 3, 4, 5];
        let dupes = quadratic_search(&data);
        assert!(dupes.is_empty(), "No duplicates expected");
    }

    #[test]
    fn test_allocation_heavy_processes_all() {
        let data = vec![1, 2, 3];
        let result = allocation_heavy(&data);
        assert_eq!(result.len(), 3, "Should produce one output per input");
    }

    #[test]
    fn test_cache_unfriendly_deterministic() {
        let sum1 = cache_unfriendly(1000);
        let sum2 = cache_unfriendly(1000);
        assert_eq!(sum1, sum2, "Same seed should produce same result");
    }
}
