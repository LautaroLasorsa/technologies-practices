//! Exercise 2: jemalloc as Global Allocator
//!
//! jemalloc is an alternative heap allocator optimized for multi-threaded, long-running
//! server workloads. Its key advantages over the system allocator:
//!
//! 1. **Thread-local caches (tcache):** Each thread gets a private cache of recently
//!    freed blocks organized by size class. Most allocations are served from the tcache
//!    without any locking — just a pointer read and increment. Only when the tcache is
//!    empty does the thread fall back to shared arenas (with lightweight locking).
//!
//! 2. **Size classes and slabs:** jemalloc divides allocations into ~40 size classes
//!    (8, 16, 32, 48, 64, ..., up to 14 KB). Each size class has dedicated slab pages.
//!    This eliminates external fragmentation within a class and keeps internal
//!    fragmentation below ~25% (next class is at most 25% larger than request).
//!
//! 3. **Extent-based large allocation:** Large allocations (> 14 KB) use a separate
//!    extent management system that can split and merge regions without affecting the
//!    small-allocation slab pool.
//!
//! 4. **Decay-based purging:** Instead of immediately returning freed pages to the OS
//!    (expensive madvise/VirtualFree calls), jemalloc retains them for a configurable
//!    decay period. Rapid alloc/free cycles reuse memory without syscalls.
//!
//! **Platform note:** tikv-jemallocator does not compile on Windows MSVC. This exercise
//! uses conditional compilation to include jemalloc only on supported platforms.
//! On Windows, the benchmarks run with the System allocator and print explanatory
//! notes about where jemalloc would make a difference.

#[allow(unused_imports)]
use std::collections::HashMap;
#[allow(unused_imports)]
use std::time::Instant;

// On non-MSVC targets, import jemalloc. On MSVC, this module still compiles
// but the jemalloc-specific code is gated behind cfg.
//
// NOTE: The actual #[global_allocator] for this binary is the TrackingAllocator
// from Exercise 1 (only one global allocator per binary). In a real project,
// you would choose ONE allocator. This exercise demonstrates the *concept*
// and *benchmarking approach* — to actually use jemalloc as global allocator,
// you would replace the TrackingAllocator in ex1 with Jemalloc.

/// Multi-threaded allocation benchmark.
///
/// This benchmark measures the total time for N threads to each perform thousands
/// of allocations. Under the System allocator, threads contend for the allocator's
/// internal locks. Under jemalloc, most allocations hit the thread-local cache,
/// eliminating contention.
///
/// Expected results (on Linux with jemalloc):
/// - At 1 thread: jemalloc ~ System (tcache has negligible overhead)
/// - At 4+ threads: jemalloc significantly faster (no lock contention)
/// - At 8+ threads: System allocator's mutex becomes the bottleneck; jemalloc scales linearly
pub fn multi_threaded_allocation_benchmark() {
    // TODO(human): Implement a multi-threaded allocation benchmark.
    //
    // This exercise teaches you to measure allocator performance under concurrency —
    // the primary scenario where jemalloc shines. The system allocator uses a global
    // mutex (or a small number of arenas) to protect its free lists. When many threads
    // allocate simultaneously, they serialize on this mutex. jemalloc's thread-local
    // caches avoid this entirely for the common case.
    //
    // Steps:
    //
    // 1. Define a `bench_allocations(num_threads: usize, iters_per_thread: usize)` function
    //    that spawns `num_threads` threads, each performing `iters_per_thread` iterations of:
    //
    //    a) Allocate a Vec<u8> with random capacity between 16 and 4096 bytes
    //       (use `rand::thread_rng()` and `rand::Rng::gen_range(16..4096)`)
    //    b) Allocate a String by formatting: `format!("item_{}", i)`
    //    c) Allocate a small HashMap: `HashMap::from([(i, i * 2), (i + 1, i * 3)])`
    //    d) Read from all three to prevent the optimizer from eliding the allocations
    //       (e.g., `std::hint::black_box(&v)` or sum lengths)
    //    e) Let all three drop (deallocation happens here)
    //
    //    Measure wall-clock time for all threads to finish using `Instant::now()`.
    //
    // 2. Run the benchmark at thread counts: 1, 2, 4, 8.
    //    For each, use 10_000 iterations per thread.
    //    Print a table:
    //
    //      Threads | Time (ms) | Allocs/sec
    //      --------|-----------|----------
    //      1       | ...       | ...
    //      2       | ...       | ...
    //      4       | ...       | ...
    //      8       | ...       | ...
    //
    //    Total allocs per run = num_threads * iters_per_thread * 3 (Vec + String + HashMap).
    //    Allocs/sec = total_allocs / elapsed_seconds.
    //
    // 3. On Windows (cfg target_env = "msvc"), print a note explaining:
    //    "Running with System allocator. On Linux with jemalloc, you would see:
    //     - Similar performance at 1 thread
    //     - 2-4x better throughput at 8 threads due to tcache eliminating lock contention
    //     - More consistent latency (no mutex wait spikes)"
    //
    //    On non-MSVC targets, print a note that the binary's global allocator is
    //    the TrackingAllocator (which wraps System), so these results reflect
    //    System allocator performance. To measure jemalloc, rebuild with jemalloc
    //    as the #[global_allocator].
    //
    // Key insight: The benefit of jemalloc is NOT that individual allocations are faster.
    // It's that allocations SCALE with thread count. System allocator throughput DECREASES
    // as threads increase (contention). jemalloc throughput INCREASES linearly (no contention).

    todo!("Exercise 2a: Implement multi-threaded allocation benchmark")
}

/// Fragmentation stress test.
///
/// Long-running servers allocate and free objects of varying sizes continuously.
/// Over time, this creates "holes" in the heap — free memory that cannot be reused
/// because it is fragmented into pieces too small for new allocations. This is
/// external fragmentation.
///
/// jemalloc's size-class system mitigates this: because each size class uses dedicated
/// slab pages, fragmentation within a class is impossible (all slots are the same size).
/// Cross-class fragmentation is limited to the ~25% gap between adjacent size classes.
pub fn fragmentation_stress_test() {
    // TODO(human): Implement a fragmentation stress test.
    //
    // This test simulates a long-running server that continuously allocates and frees
    // objects of random sizes. After many cycles, measure the "overhead" — the difference
    // between the bytes we actually need and the bytes the allocator has reserved from the OS.
    //
    // Steps:
    //
    // 1. Import `AllocSnapshot` from `crate::ex1_tracking_allocator`.
    //    (This gives us allocation counts. For true RSS measurement, you would use
    //    platform-specific APIs — but allocation byte tracking is a good proxy.)
    //
    // 2. Define a "server simulation" function:
    //
    //    let mut live_objects: Vec<Vec<u8>> = Vec::new();
    //    let num_cycles = 50_000;
    //
    //    For each cycle:
    //    a) Decide randomly: allocate (60% chance) or free (40% chance).
    //       The 60/40 split means live objects grow slowly over time, like a real server
    //       with session objects, cached data, and buffered responses.
    //
    //    b) If allocate: create a `Vec<u8>` with random size between 32 and 8192 bytes.
    //       Fill it with dummy data (e.g., `vec![0xAB; size]`) to force actual page mapping.
    //       Push onto live_objects.
    //
    //    c) If free (and live_objects is non-empty): remove a random element from live_objects.
    //       This simulates objects being freed in arbitrary order (not LIFO), which is
    //       the worst case for fragmentation — it creates holes throughout the heap.
    //
    // 3. After all cycles, take an AllocSnapshot and compute:
    //    - Total bytes still allocated (alloc_bytes - dealloc_bytes) = "live bytes"
    //    - The actual memory used by live_objects: sum of vec.len() for each live vec
    //    - "Overhead" = live bytes from allocator - actual needed bytes
    //    - "Fragmentation ratio" = overhead / actual needed bytes
    //
    //    Print these metrics. Under jemalloc, the fragmentation ratio is typically
    //    5-15%. Under the system allocator, it can be 20-40% after many cycles.
    //
    // 4. Print a summary explaining:
    //    - Why random-order frees cause fragmentation (creates non-contiguous holes)
    //    - How jemalloc's size classes limit fragmentation (same-size slots, no splitting)
    //    - Why this matters for servers: high fragmentation = high RSS = OOM risk
    //
    // Practical note: In production, you can inspect jemalloc's fragmentation stats
    // via `MALLOC_CONF=stats_print:true` or the `jemalloc_ctl` crate. These provide
    // detailed breakdowns by size class, arena, and thread cache.

    todo!("Exercise 2b: Implement fragmentation stress test")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_threaded_benchmark() {
        multi_threaded_allocation_benchmark();
    }

    #[test]
    fn test_fragmentation_stress() {
        fragmentation_stress_test();
    }
}
