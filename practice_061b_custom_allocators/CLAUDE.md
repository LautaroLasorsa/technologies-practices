# Practice 061b: Custom Allocators — GlobalAlloc, jemalloc & Arena Patterns

## Technologies

- **std::alloc** — Rust's allocation interface: `GlobalAlloc` trait, `Layout`, `#[global_allocator]`, `System` allocator
- **tikv-jemallocator** (0.6) — Drop-in jemalloc global allocator for Rust (Linux/macOS only)
- **bumpalo** (3.16) — Fast bump/arena allocator with `collections` support (Vec, String)
- **std::sync::atomic** — Atomic counters for allocation tracking without locks

## Stack

- Rust (cargo, edition 2021)

## Theoretical Context

### How Heap Allocation Works

Every time Rust code calls `Box::new()`, `Vec::push()`, or `String::from()`, it requests memory from a **heap allocator**. Understanding what happens beneath those calls is essential for writing high-performance Rust — especially in servers, batch processors, and latency-sensitive systems.

**The system allocator** (glibc's `malloc` on Linux, Windows' `HeapAlloc`, macOS's `libmalloc`) manages a pool of virtual memory pages obtained from the OS via `mmap`/`VirtualAlloc`. When you allocate:

1. The allocator searches its **free lists** — data structures tracking previously freed blocks organized by size.
2. If a suitable block is found, it is split or returned as-is. If not, the allocator requests new pages from the OS.
3. The allocator writes **metadata** (block size, free-list pointers) adjacent to or near the returned pointer.
4. Deallocation returns the block to the free list. The allocator may **coalesce** adjacent free blocks to reduce fragmentation.

This is general-purpose and correct, but it has costs:

- **Fragmentation**: After many alloc/free cycles of varying sizes, the heap becomes a patchwork of used and free blocks. A 64-byte allocation might fail even though 1 MB is free — just not in one contiguous piece. There are two kinds: **external fragmentation** (free memory scattered in small non-contiguous chunks) and **internal fragmentation** (allocated blocks are larger than requested due to alignment and size-class rounding).
- **Contention**: In multi-threaded programs, the free lists must be protected by locks. Under high concurrency, threads contend for these locks, serializing allocation.
- **Syscall overhead**: Requesting new pages from the OS (`mmap`, `sbrk`) is expensive — hundreds of nanoseconds vs. tens for a free-list hit.
- **Cache pollution**: Allocations scattered across the address space lead to poor cache locality.

### Rust's GlobalAlloc Abstraction

Rust decouples allocation from the allocator implementation through the [`GlobalAlloc`](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html) trait:

```rust
pub unsafe trait GlobalAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8;
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout);

    // Provided defaults (can be overridden for efficiency):
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 { ... }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 { ... }
}
```

`Layout` encapsulates **size** and **alignment** — the two pieces of information every allocation needs. The `#[global_allocator]` attribute registers a `static` implementing `GlobalAlloc` as the process-wide allocator. All `Box`, `Vec`, `String`, `HashMap`, etc. route through it.

By default, Rust uses `std::alloc::System` — a thin wrapper around the platform's C allocator. But you can replace it globally with a single line:

```rust
#[global_allocator]
static GLOBAL: MyAllocator = MyAllocator;
```

**Key safety constraints:**
- `alloc` returns null on failure (not a panic — panicking in an allocator is undefined behavior).
- `dealloc` must receive the exact `Layout` that was used for the original `alloc`.
- Implementations must only use `core` primitives — no `std::sync::Mutex` (it may allocate internally, causing infinite recursion).

### jemalloc: Why It Outperforms the System Allocator

[jemalloc](https://jemalloc.net/) (originally developed for FreeBSD) is designed for multi-threaded, long-running server workloads. Its architecture addresses the three main costs of general-purpose allocators:

**Thread-local caches (tcache):** Each thread gets a private cache of recently freed blocks, organized by size class. Allocations that hit the tcache require **zero locking** — just a pointer bump within the thread's cache. Only when the tcache is exhausted does the thread fall back to a shared arena. This eliminates contention for the vast majority of allocations.

**Size classes and slabs:** jemalloc divides allocations into ~40 size classes (8, 16, 32, 48, 64, 80, ..., up to 14 KB for "small" allocations). Each size class has dedicated **slabs** — contiguous runs of same-sized slots. This eliminates external fragmentation within a size class (every slot is the same size) and reduces internal fragmentation (the next larger size class is at most ~25% larger than the request).

**Extent-based management:** Large allocations use "extents" — contiguous virtual memory regions managed by a separate subsystem. Extents can be split, merged, and recycled without affecting the slab allocator. This separation prevents large allocations from fragmenting the small-allocation pool.

**Decay-based purging:** Instead of immediately returning freed pages to the OS (expensive `madvise` calls), jemalloc uses a time-based decay: freed pages are retained for a configurable period, so rapid alloc/free cycles reuse memory without syscalls.

**When to use jemalloc:**
- Long-running servers (web servers, databases, message brokers) where fragmentation accumulates over hours/days
- Multi-threaded workloads with high allocation rates (the tcache eliminates lock contention)
- Programs where the system allocator shows high RSS growth over time

**When NOT to use jemalloc:**
- Short-lived CLI tools (startup cost > benefit)
- Embedded or WASM targets (jemalloc is large)
- Windows MSVC targets (jemalloc has no official Windows support)

The [tikv-jemallocator](https://github.com/tikv/jemallocator) crate provides a `Jemalloc` struct implementing `GlobalAlloc`. It is a one-line swap:

```rust
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
```

### Arena/Bump Allocation: O(1) Alloc, Batch Free

A **bump allocator** (also called a linear or arena allocator) is the simplest possible allocation strategy:

1. Pre-allocate a large contiguous block of memory (the "arena").
2. Maintain a single pointer ("bump pointer") to the next free byte.
3. To allocate N bytes: align the bump pointer, return it, advance it by N. **One pointer addition — O(1).**
4. Individual deallocation is **not supported**. When the entire batch of work is done, reset the bump pointer to the start — freeing everything at once in O(1).

```
Arena memory layout:
[AAAA][BB][CCCCCC][DDD][.............free space.............]
^                      ^                                    ^
start                  bump pointer                         end
```

**Why this matters:**

- **Speed**: No free lists, no coalescing, no metadata per allocation. Just `ptr += size`. A bump allocator is typically 5-20x faster than `malloc` for small allocations.
- **Cache locality**: Objects allocated sequentially are physically adjacent in memory. When you iterate over them, CPU cache prefetching works perfectly — each cache line contains useful data, not free-list pointers.
- **Zero fragmentation**: There is no fragmentation because there is no free list. Every byte between `start` and `bump_pointer` is in use.
- **Batch lifetime**: Perfect for phase-based processing where you allocate many objects, process them, and then discard everything. Examples: compiler passes (parse → allocate AST → analyze → drop AST), game frame processing, request handling, batch ETL transforms.

**The trade-off**: You cannot free individual objects. If you allocate 1000 objects in an arena and need to free object #500, you cannot — you must wait until the entire arena is reset. This makes arenas unsuitable for general-purpose allocation but ideal for **scoped** or **phase-based** workloads.

[bumpalo](https://docs.rs/bumpalo/latest/bumpalo/) is Rust's premier arena allocator. It provides:
- `Bump::new()` — create a new arena
- `bump.alloc(value)` — allocate a single value, returns `&mut T`
- `bump.alloc_str(s)` — allocate a string slice
- `bump.alloc_slice_copy(slice)` — allocate a copy of a slice
- `bumpalo::collections::Vec` — a `Vec` that allocates from the bump arena
- `bumpalo::collections::String` — a `String` that allocates from the bump arena
- `bump.reset()` — free everything at once (O(1))

Bumpalo's `Vec` and `String` have the same API as `std::vec::Vec` and `std::string::String` but allocate from the arena. Rust's lifetime system ensures they cannot outlive the arena — a compile-time guarantee that prevents use-after-free.

### Tracking Allocators: Instrumenting Your Allocations

A **tracking allocator** wraps another allocator (typically `System`) and records allocation metrics: count, total bytes, peak usage. This is invaluable for:
- Profiling: "How many allocations does this function make?"
- Optimization: "Where are the hot allocation paths?"
- Regression testing: "Did this refactor increase allocations?"

The pattern uses `AtomicU64` counters (no locking needed) and delegates actual allocation to the inner allocator:

```rust
struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Relaxed);
        System.alloc(layout)  // delegate to system
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_COUNT.fetch_add(1, Relaxed);
        System.dealloc(ptr, layout)
    }
}
```

`Relaxed` ordering suffices because we only need atomicity (no torn reads), not ordering guarantees relative to other memory operations. The counters are approximate under concurrency but exact for single-threaded measurement.

### Choosing the Right Allocator

| Scenario | Allocator | Why |
|----------|-----------|-----|
| General purpose | `System` | Default, no dependencies, well-tested |
| Long-running server | jemalloc | Lower fragmentation, tcache eliminates contention |
| Batch/phase processing | Bump (bumpalo) | O(1) alloc, perfect locality, batch free |
| Profiling/debugging | Tracking wrapper | Count allocs, find hot paths, detect regressions |
| Embedded / `no_std` | Custom fixed-pool | Known memory budget, no OS dependency |

### The Unstable `allocator_api`

Rust's nightly channel has an unstable feature `allocator_api` that parameterizes collections by allocator:

```rust
// Nightly only:
let bump = Bump::new();
let mut v = Vec::new_in(&bump);  // Vec<T, &Bump> — allocates in the arena
```

This is the long-term vision: every collection type accepts an optional allocator parameter. Until stabilization, bumpalo provides its own `Vec<'bump, T>` and `String<'bump>` types as a stable workaround. The `allocator-api2` crate backports this trait to stable Rust, and bumpalo supports it via the `allocator-api2` feature flag.

## Description

Build a series of exercises exploring Rust's allocation model from the ground up: implement a tracking allocator that counts every heap allocation, benchmark jemalloc against the system allocator, learn arena allocation with bumpalo, use bumpalo's collection types, design a phase-based processing pipeline with arena-per-phase, and compare all three strategies under a realistic workload.

### What you'll learn

1. **Tracking allocator** — Implement `GlobalAlloc` to instrument every allocation with atomic counters
2. **jemalloc integration** — Drop-in global allocator replacement and benchmarking (Linux/WSL only, simulated on Windows)
3. **bumpalo basics** — Arena lifecycle: create, allocate, use, reset
4. **bumpalo collections** — `Vec<'bump, T>` and `String<'bump>` for arena-backed dynamic containers
5. **Phase-based arena pipeline** — Arena-per-phase pattern for batch processing with minimal allocator overhead
6. **Comparative benchmark** — System vs jemalloc vs bump under a realistic mixed workload

## Instructions

### Exercise 1: Building a Tracking Allocator (~20 min)

Open `src/ex1_tracking_allocator.rs`. This exercise teaches the `GlobalAlloc` trait by building a wrapper that counts allocations.

Understanding what your program allocates is the first step toward optimizing it. Most Rust developers never think about allocation because `Vec`, `String`, and `HashMap` handle it transparently. But every `push()`, `format!()`, and `collect()` may trigger a heap allocation — and in hot paths, those add up. A tracking allocator makes the invisible visible.

1. **TODO(human): Implement `GlobalAlloc` for `TrackingAllocator`** — The struct wraps `System` and uses atomic counters (`ALLOC_COUNT`, `DEALLOC_COUNT`, `ALLOC_BYTES`, `DEALLOC_BYTES`) to track every allocation and deallocation. Implement `alloc`, `dealloc`, `alloc_zeroed`, and `realloc`. Each method must update the counters and delegate to `System`.

2. **TODO(human): Write `measure_vec_allocations()`** — Create Vecs of different sizes and measure how many allocations occur. This reveals Vec's growth strategy (doubling capacity) and the difference between `Vec::new()` + push vs `Vec::with_capacity()`.

3. **TODO(human): Write `measure_string_allocations()`** — Build strings with `format!()`, `push_str()`, and `String::with_capacity()`. Compare allocation counts. This teaches that string formatting is one of the most allocation-heavy operations in Rust.

### Exercise 2: jemalloc as Global Allocator (~15 min)

Open `src/ex2_jemalloc.rs`. This exercise demonstrates switching the global allocator to jemalloc and measuring the difference.

**Platform note:** tikv-jemallocator does not compile on Windows MSVC. This exercise uses conditional compilation (`#[cfg(not(target_env = "msvc"))]`) to include jemalloc only on supported platforms. On Windows, the exercise runs a simulation that demonstrates the *concept* of allocator switching with the System allocator and explains what jemalloc would change. For the full experience, use WSL or Linux.

1. **TODO(human): Implement `multi_threaded_allocation_benchmark()`** — Spawn N threads, each performing thousands of small allocations (Vec, String, HashMap) in a loop. Measure total wall-clock time and allocation counts. On Linux/WSL with jemalloc enabled, compare against the same benchmark with System. On Windows, run the benchmark with System and explain where jemalloc's tcache would eliminate contention.

2. **TODO(human): Implement `fragmentation_stress_test()`** — Allocate and free objects of random sizes in a loop to simulate long-running server behavior. Measure RSS (or allocated bytes via tracking allocator) after many cycles. This demonstrates the fragmentation problem that jemalloc's size-class system mitigates.

### Exercise 3: bumpalo Arena Basics (~15 min)

Open `src/ex3_arena_basics.rs`. This exercise introduces arena allocation with bumpalo.

Arena allocation is a paradigm shift from general-purpose allocation. Instead of allocating and freeing individual objects, you allocate many objects into a shared arena and free them all at once. This trades flexibility (no individual frees) for speed (O(1) allocation, perfect cache locality).

1. **TODO(human): Implement `arena_lifecycle()`** — Create a `Bump` arena, allocate various types (integers, structs, strings, slices), observe that all allocations are contiguous in memory (print pointer addresses), and reset the arena. This teaches the fundamental arena lifecycle: create -> allocate -> use -> reset.

2. **TODO(human): Implement `arena_vs_heap_benchmark()`** — Allocate 100,000 small objects via `Box::new()` (heap) and via `bump.alloc()` (arena). Time both. The arena should be dramatically faster because each allocation is just a pointer bump — no free-list traversal, no metadata, no fragmentation bookkeeping.

### Exercise 4: bumpalo Collections (~15 min)

Open `src/ex4_arena_collections.rs`. This exercise uses bumpalo's `Vec` and `String` types — dynamic collections backed by arena allocation.

Standard `Vec<T>` and `String` allocate from the global allocator. When they grow (capacity doubling), the old buffer is freed and a new larger one is allocated. With bumpalo's collections, the old buffer is simply abandoned in the arena (no free), and the new buffer is bump-allocated. This is faster but uses more memory within the arena — a classic space-time trade-off.

1. **TODO(human): Implement `arena_vec_operations()`** — Create a `bumpalo::collections::Vec`, push elements, extend from slices, and demonstrate `extend_from_slice_copy` (the optimized path for Copy types). Compare allocation counts (using the tracking allocator) between `std::vec::Vec` and `bumpalo::collections::Vec`.

2. **TODO(human): Implement `arena_string_building()`** — Build strings in a bumpalo arena using `bumpalo::collections::String` and `bumpalo::format!`. This teaches how arena-backed strings avoid the allocation overhead of repeated `format!()` calls.

3. **TODO(human): Implement `arena_nested_structures()`** — Allocate a tree-like structure (nodes referencing other nodes) entirely within a bump arena. This demonstrates how arena allocation simplifies ownership: all nodes share the arena's lifetime, so no `Rc`/`Arc` needed for shared references within the same phase.

### Exercise 5: Phase-Based Arena Pipeline (~20 min)

Open `src/ex5_arena_pipeline.rs`. This is the main design exercise — building a multi-phase data processing pipeline where each phase gets its own arena.

The arena-per-phase pattern is used in compilers (parse phase arena, type-check phase arena, codegen phase arena), game engines (per-frame arenas), and batch data processing. Each phase allocates freely into its arena, and when the phase completes, the entire arena is dropped — freeing all memory in one operation with no per-object destructor overhead.

1. **TODO(human): Implement `PhaseArena` and the three-phase pipeline** — Design a pipeline with:
   - **Parse phase**: Read raw data (simulated), allocate parsed records into a parse arena
   - **Transform phase**: Read from parse arena, transform records, allocate results into a transform arena
   - **Output phase**: Read from transform arena, produce final output
   After each phase, the previous phase's arena is dropped. This demonstrates how arenas provide natural memory management boundaries aligned with processing stages.

2. **TODO(human): Implement `measure_arena_pipeline_memory()`** — Run the pipeline and measure peak memory usage at each phase. Show that dropping a phase's arena immediately reclaims all its memory, unlike general-purpose allocation where freed memory may be retained by the allocator.

### Exercise 6: Comparative Benchmark — System vs jemalloc vs Bump (~15 min)

Open `src/ex6_benchmark.rs`. The capstone exercise: compare all three allocation strategies under a realistic workload.

1. **TODO(human): Implement `realistic_workload()`** — Define a workload that simulates a batch processing job: parse N records (allocate strings, structs), transform them (allocate new containers), and produce output. Run this workload with: (a) System allocator, (b) jemalloc (on supported platforms), (c) bumpalo arena-per-batch.

2. **TODO(human): Implement `print_comparison_table()`** — Measure and display: wall-clock time, allocation count, peak memory usage for each strategy. Format as a table. Analyze the results — which strategy wins for this workload and why?

## Motivation

- **Server performance**: The allocator is the single most impactful global optimization in a Rust server. Switching to jemalloc can reduce p99 latency by 30-50% in allocation-heavy workloads (this is why TiKV, the Rust database, maintains tikv-jemallocator).
- **Batch processing**: Arena allocation can make batch data processing 5-20x faster by eliminating per-object allocation overhead and improving cache locality.
- **Latency-sensitive systems**: In HFT and game engines, predictable allocation time matters more than throughput. Arena allocation provides O(1) deterministic allocation — no worst-case free-list search.
- **Debugging and profiling**: Building a tracking allocator teaches the `GlobalAlloc` trait deeply and provides a tool you can reuse in any Rust project to find allocation hot spots.
- **Complements 061a (Lock-Free)**: Lock-free data structures and custom allocators are the two pillars of low-latency Rust systems programming. Together they eliminate the two main sources of unpredictable latency: lock contention and allocation overhead.

## Commands

### Build & Run

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all exercises (verifies TODO stubs compile) |
| `cargo run` | Run the exercise runner — executes all exercises sequentially |
| `cargo run -- 1` | Run only Exercise 1 (tracking allocator) |
| `cargo run -- 2` | Run only Exercise 2 (jemalloc) |
| `cargo run -- 3` | Run only Exercise 3 (arena basics) |
| `cargo run -- 4` | Run only Exercise 4 (arena collections) |
| `cargo run -- 5` | Run only Exercise 5 (phase-based pipeline) |
| `cargo run -- 6` | Run only Exercise 6 (comparative benchmark) |

### Development

| Command | Description |
|---------|-------------|
| `cargo check` | Fast type-check without codegen (use while implementing) |
| `cargo test` | Run unit tests (each exercise has verification tests) |
| `cargo clippy` | Run linter for idiomatic Rust suggestions |
| `cargo build --release` | Optimized build (use for benchmarking exercises) |
| `cargo run --release -- 6` | Run comparative benchmark with optimizations (IMPORTANT: debug builds distort timings) |
| `cargo run --release -- 3` | Run arena benchmarks with optimizations |

### WSL (for jemalloc)

| Command | Description |
|---------|-------------|
| `wsl cargo build` | Build inside WSL where jemalloc compiles natively |
| `wsl cargo run --release -- 2` | Run jemalloc benchmark inside WSL for accurate results |

## References

- [GlobalAlloc trait — Rust std docs](https://doc.rust-lang.org/std/alloc/trait.GlobalAlloc.html)
- [std::alloc module — Rust std docs](https://doc.rust-lang.org/std/alloc/index.html)
- [tikv-jemallocator — GitHub](https://github.com/tikv/jemallocator)
- [bumpalo — docs.rs](https://docs.rs/bumpalo/latest/bumpalo/)
- [bumpalo — GitHub](https://github.com/fitzgen/bumpalo)
- [jemalloc — Official site](https://jemalloc.net/)
- [Heap Allocations — The Rust Performance Book](https://nnethercote.github.io/perf-book/heap-allocations.html)
- [Heap Allocation — Writing an OS in Rust](https://os.phil-opp.com/heap-allocation/)
- [Untangling Lifetimes: The Arena Allocator — Ryan Fleury](https://www.rfleury.com/p/untangling-lifetimes-the-arena-allocator)
- [Understanding jemalloc — Leapcell](https://leapcell.io/blog/understanding-jemalloc)

## Notes

*(To be filled during practice.)*
