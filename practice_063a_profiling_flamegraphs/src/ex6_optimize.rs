//! Exercise 6: Guided Optimization — Before/After Comparison
//!
//! This is the payoff exercise: you apply targeted optimizations to each of the
//! three bottleneck functions from Exercise 2, guided by the profiling evidence
//! from Exercises 3-5. Then you measure the improvement.
//!
//! The profiling workflow cycle:
//! 1. Profile → identify bottleneck → hypothesize cause → optimize → re-measure
//!
//! Critical discipline: change ONE thing at a time. If you optimize all three
//! functions simultaneously, you cannot attribute the improvement to any specific
//! change. Worse, one optimization might mask the failure of another.
//!
//! After implementing all optimizations, generate a new flamegraph:
//!   cargo flamegraph --release -o after.svg -- 6
//! Compare `before.svg` (from Exercise 3) with `after.svg` to see the visual
//! difference. The bottleneck functions should shrink dramatically or disappear.

use crate::ex1_setup::bench_fn;
use crate::ex2_workload;

/// Optimized duplicate detection using a HashSet.
///
/// The original `quadratic_search()` is O(n^2) because it compares every pair
/// of elements. A HashSet provides O(1) average-case lookup, making the entire
/// algorithm O(n).
///
/// Expected speedup: for n=15,000, the original does ~112M comparisons. The
/// optimized version does ~15,000 hash lookups. That is a ~7,500x reduction
/// in work — you should see a speedup of at least 100x (limited by hash
/// computation overhead).
///
/// In the new flamegraph, this function should be a thin sliver instead of
/// the dominant bar.
pub fn optimized_search(data: &[u64]) -> Vec<u64> {
    // TODO(human): Implement O(n) duplicate detection with HashSet.
    //
    // This is the most impactful optimization in this practice. Algorithmic
    // improvements always beat constant-factor optimizations.
    //
    // Steps:
    //
    // 1. Create two HashSets:
    //    ```
    //    use std::collections::HashSet;
    //    let mut seen = HashSet::with_capacity(data.len());
    //    let mut duplicates = HashSet::new();
    //    ```
    //    `with_capacity` pre-allocates the hash table to avoid rehashing during
    //    insertion. This is a minor optimization but good practice.
    //
    // 2. Single pass through the data:
    //    ```
    //    for &value in data {
    //        if !seen.insert(value) {
    //            // insert() returns false if the value was already present
    //            duplicates.insert(value);
    //        }
    //    }
    //    ```
    //    `HashSet::insert()` returns a bool: `true` if the value was new,
    //    `false` if it was already in the set. This is the idiomatic Rust way
    //    to check-and-insert in one operation.
    //
    // 3. Convert to Vec and return:
    //    ```
    //    duplicates.into_iter().collect()
    //    ```
    //
    // After implementing, measure:
    //   cargo run --release -- 6
    // Compare the time with the original from Exercise 2.
    //
    // Then re-profile:
    //   cargo flamegraph --release -o after_search.svg -- 6
    // This function should virtually disappear from the flamegraph.
    //
    // Why HashSet? The hash table provides O(1) expected-time lookup and insertion.
    // The hash function (SipHash in Rust's default implementation) computes a
    // fixed-cost hash regardless of the number of elements in the set.
    // Trade-off: uses more memory (the hash table overhead), but for this use case
    // the memory cost is negligible compared to the O(n^2) → O(n) time improvement.

    todo!("Exercise 6a: Implement O(n) duplicate search with HashSet")
}

/// Optimized string processing with buffer reuse.
///
/// The original `allocation_heavy()` creates 3-4 new String allocations per
/// iteration and immediately drops them. The optimized version pre-allocates
/// a single buffer and reuses it with `.clear()`.
///
/// Expected improvement: allocation count drops from ~4*n to ~n (only the
/// final result Strings need allocation). Wall-clock speedup depends on the
/// allocator and platform — typically 2-5x.
///
/// In a DHAT re-profile, the allocation count should drop dramatically.
pub fn optimized_allocation(data: &[u64]) -> Vec<String> {
    // TODO(human): Implement allocation-efficient string processing.
    //
    // The key insight: `String::clear()` resets the string's length to 0
    // WITHOUT deallocating its buffer. On the next `write!` or `push_str`,
    // it reuses the existing allocation. This turns N allocations into 1
    // allocation + (N-1) no-cost resets.
    //
    // Steps:
    //
    // 1. Pre-allocate the result vector and a reusable buffer:
    //    ```
    //    let mut results = Vec::with_capacity(data.len());
    //    let mut buffer = String::with_capacity(64); // Pre-allocate once
    //    ```
    //
    // 2. Process each value by reusing the buffer:
    //    ```
    //    use std::fmt::Write;
    //
    //    for &value in data {
    //        buffer.clear(); // Reset length to 0, keep allocation
    //
    //        // Write the formatted string into the existing buffer
    //        write!(buffer, "item-{:010}", value).unwrap();
    //
    //        // Perform string manipulation IN PLACE where possible:
    //        // For uppercase: we need to create the result anyway (different chars)
    //        // But we can avoid the intermediate reversed String
    //        let upper = buffer.to_uppercase();
    //        let reversed: String = upper.chars().rev().collect();
    //
    //        results.push(reversed);
    //    }
    //    ```
    //
    //    Note: We cannot completely eliminate allocations because each result
    //    String in the Vec is a separate heap allocation. But we eliminated
    //    the intermediate `format!()` allocation by reusing `buffer`.
    //
    //    For even better performance, consider:
    //    - Using `make_ascii_uppercase()` on a `Vec<u8>` instead of `to_uppercase()`
    //    - Reversing bytes directly if the string is known to be ASCII
    //    - Using a fixed-size stack buffer (`[u8; 64]`) instead of a String
    //
    // 3. Return results.
    //
    // After implementing, compare:
    //   a) Wall-clock time vs original (Exercise 2): cargo run --release -- 6
    //   b) DHAT allocation count: cargo run --release --features dhat-heap -- 6
    //      Load the new dhat-heap.json and compare total blocks with Exercise 4.
    //
    // Why this optimization works: The system allocator (e.g., jemalloc, Windows Heap)
    // has per-allocation overhead: metadata bookkeeping, thread-local caching, and
    // potential system calls. By reusing a buffer, we pay this cost once instead of
    // N times. Additionally, the reused buffer stays hot in L1 cache, whereas
    // freshly allocated memory may not be in cache at all.

    todo!("Exercise 6b: Implement allocation-efficient string processing")
}

/// Optimized sequential access replacing random access.
///
/// The original `cache_unfriendly()` accesses a large array in random order,
/// causing frequent cache misses. The optimized version accesses the same data
/// sequentially, which is cache-friendly because:
///
/// 1. **Spatial locality**: Accessing element `i` brings the entire cache line
///    (64 bytes = 8 u64s) into L1 cache. Sequential access uses all 8 values
///    before the cache line is evicted. Random access wastes 7 of the 8 values.
///
/// 2. **Hardware prefetching**: The CPU detects sequential access patterns and
///    prefetches the next few cache lines BEFORE they are needed. This hides
///    memory latency almost entirely for sequential access.
///
/// Expected speedup: 3-10x depending on array size relative to cache hierarchy.
/// For an 8MB array (1M u64s): random access hits main memory (~100ns) on most
/// accesses, while sequential access hits L1 cache (~1ns) after the prefetcher
/// fills the pipeline.
pub fn optimized_cache(size: usize) -> u64 {
    // TODO(human): Implement the same computation with sequential access.
    //
    // Steps:
    //
    // 1. Create the same data array:
    //    ```
    //    let data: Vec<u64> = (0..size as u64).collect();
    //    ```
    //
    // 2. Sum the data SEQUENTIALLY instead of through random indices:
    //    ```
    //    let mut sum: u64 = 0;
    //    for &value in &data {
    //        sum = sum.wrapping_add(value);
    //    }
    //    sum
    //    ```
    //    Or even more idiomatic: `data.iter().fold(0u64, |acc, &x| acc.wrapping_add(x))`
    //
    //    Note: The sum may differ from cache_unfriendly() because the elements
    //    are summed in a different order. For u64 with wrapping_add, the result
    //    should be the same regardless of order (wrapping addition is commutative
    //    and associative). But if using floating-point, the results would differ
    //    due to rounding.
    //
    // 3. Return the sum.
    //
    // After implementing, compare:
    //   cargo run --release -- 6
    // The sequential version should be dramatically faster.
    //
    // For extra credit: try a STRIDED access pattern (every 8th element) to see
    // the intermediate case. Strided access within a cache line is still fast,
    // but strided access that skips cache lines loses prefetching benefits.
    //
    // Why this matters: Memory access patterns are THE dominant performance factor
    // in modern systems. The CPU is 100x faster than main memory. Any algorithm
    // that does not respect cache hierarchy is leaving 10-100x performance on the
    // table. This is why column-oriented databases (Parquet, DuckDB) outperform
    // row-oriented databases for analytics — same algorithm, better access pattern.

    todo!("Exercise 6c: Implement cache-friendly sequential access")
}

/// Run all benchmarks and print a comparison table.
///
/// This is the deliverable: a side-by-side comparison showing the speedup
/// from each optimization. The table format makes it easy to see which
/// optimization had the most impact.
pub fn run_comparison() {
    // TODO(human): Run original and optimized versions, print comparison.
    //
    // Steps:
    //
    // 1. Define the workload size (use DEFAULT_SIZE from ex2_workload):
    //    ```
    //    let size = ex2_workload::DEFAULT_SIZE;
    //    ```
    //
    // 2. Generate test data (same as run_workload):
    //    ```
    //    let data: Vec<u64> = (0..size as u64)
    //        .map(|i| i % (size as u64 / 2))
    //        .collect();
    //    ```
    //
    // 3. Benchmark original versions:
    //    ```
    //    let (t_search_old, _) = bench_fn("quadratic_search (original)", || {
    //        ex2_workload::quadratic_search(&data)
    //    });
    //    let (t_alloc_old, _) = bench_fn("allocation_heavy (original)", || {
    //        ex2_workload::allocation_heavy(&data)
    //    });
    //    let cache_size = size.max(500_000);
    //    let (t_cache_old, _) = bench_fn("cache_unfriendly (original)", || {
    //        ex2_workload::cache_unfriendly(cache_size)
    //    });
    //    ```
    //
    // 4. Benchmark optimized versions:
    //    ```
    //    let (t_search_new, _) = bench_fn("optimized_search", || {
    //        optimized_search(&data)
    //    });
    //    let (t_alloc_new, _) = bench_fn("optimized_allocation", || {
    //        optimized_allocation(&data)
    //    });
    //    let (t_cache_new, _) = bench_fn("optimized_cache", || {
    //        optimized_cache(cache_size)
    //    });
    //    ```
    //
    // 5. Print a comparison table:
    //    ```
    //    println!();
    //    println!("  {:<30} {:>12} {:>12} {:>10}", "Function", "Before (ms)", "After (ms)", "Speedup");
    //    println!("  {}", "-".repeat(66));
    //
    //    let speedup = |old: Duration, new: Duration| -> f64 {
    //        old.as_secs_f64() / new.as_secs_f64()
    //    };
    //
    //    println!("  {:<30} {:>12.1} {:>12.1} {:>9.1}x",
    //        "quadratic → HashSet",
    //        t_search_old.as_secs_f64() * 1000.0,
    //        t_search_new.as_secs_f64() * 1000.0,
    //        speedup(t_search_old, t_search_new));
    //
    //    // ... similar for allocation and cache ...
    //    ```
    //
    // 6. Print a total speedup:
    //    ```
    //    let total_old = t_search_old + t_alloc_old + t_cache_old;
    //    let total_new = t_search_new + t_alloc_new + t_cache_new;
    //    println!("  {:<30} {:>12.1} {:>12.1} {:>9.1}x",
    //        "TOTAL",
    //        total_old.as_secs_f64() * 1000.0,
    //        total_new.as_secs_f64() * 1000.0,
    //        speedup(total_old, total_new));
    //    ```
    //
    // Expected output (approximate, varies by machine):
    //    Function                       Before (ms)  After (ms)   Speedup
    //    ------------------------------------------------------------------
    //    quadratic → HashSet              5000.0          2.0    2500.0x
    //    allocation → buffer reuse          50.0         20.0       2.5x
    //    random → sequential access         15.0          3.0       5.0x
    //    TOTAL                            5065.0         25.0     202.6x
    //
    // The algorithmic fix dominates the total speedup. This is the most
    // important lesson: algorithm choice > constant-factor optimization.
    // But the allocation and cache optimizations matter when the algorithm
    // is already optimal and you need to squeeze out the remaining performance.

    todo!("Exercise 6d: Run comparison benchmarks and print results table")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimized_search_finds_duplicates() {
        let data = vec![1, 2, 3, 2, 4, 3, 5];
        let mut dupes = optimized_search(&data);
        dupes.sort();
        assert!(dupes.contains(&2));
        assert!(dupes.contains(&3));
        assert!(!dupes.contains(&1));
    }

    #[test]
    fn test_optimized_search_matches_original() {
        let data: Vec<u64> = (0..100).map(|i| i % 50).collect();
        let mut original = ex2_workload::quadratic_search(&data);
        let mut optimized = optimized_search(&data);
        original.sort();
        optimized.sort();
        assert_eq!(original, optimized, "Optimized should find same duplicates");
    }

    #[test]
    fn test_optimized_allocation_same_count() {
        let data = vec![1, 2, 3, 4, 5];
        let original = ex2_workload::allocation_heavy(&data);
        let optimized = optimized_allocation(&data);
        assert_eq!(original.len(), optimized.len());
    }

    #[test]
    fn test_optimized_cache_same_result() {
        let size = 1000;
        let original = ex2_workload::cache_unfriendly(size);
        let optimized = optimized_cache(size);
        // Both sum the same elements, just in different order.
        // With wrapping_add, the result is the same.
        assert_eq!(original, optimized, "Sequential sum should match random sum");
    }
}
