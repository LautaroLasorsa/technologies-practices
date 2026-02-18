# Practice 063a: Profiling & Flamegraphs

## Technologies

- **cargo-flamegraph** (0.6+) — One-command CPU flamegraph generation for Rust/cargo projects (uses ETW on Windows, perf on Linux, dtrace on macOS)
- **samply** (0.13+) — Modern cross-platform sampling profiler with Firefox Profiler UI (works on Windows via ETW)
- **dhat** (0.3) — Heap profiling crate: tracks every allocation, reports where allocations happen, how long they live, peak memory usage
- **cargo-show-asm** — Inspect generated assembly, LLVM-IR, and MIR for any Rust function

## Stack

- Rust (cargo, edition 2021)

## Theoretical Context

### Why Profile? "Measure, Don't Guess"

Developers have notoriously poor intuition about where performance bottlenecks actually are. Studies consistently show that programmers guess wrong about hot spots more than 90% of the time. Without measurement, you optimize code that does not matter and miss the code that does. Profiling replaces guessing with evidence.

Donald Knuth's famous quote applies: *"Premature optimization is the root of all evil"* — but the full quote continues *"...yet we should not pass up our opportunities in that critical 3%."* Profiling tells you where that critical 3% is.

### Two Families of Profilers: Sampling vs Instrumentation

There are two fundamentally different approaches to profiling:

**Sampling profilers** (what we use in this practice) periodically interrupt the program (e.g., 1000 times/sec) and record the current call stack. The key insight: if a function appears in 30% of the samples, it consumes roughly 30% of CPU time. This is statistically accurate for hot code and has very low overhead (typically 2-5%) because the profiler does not touch the program's execution path — it only observes.

| Property | Sampling | Instrumentation |
|----------|----------|-----------------|
| **Mechanism** | Periodic interrupts record call stack | Every function entry/exit is hooked with timing code |
| **Overhead** | Low (2-5%) | High (10-100x slowdown) |
| **Accuracy** | Statistical (more samples = more precise) | Exact call counts and durations |
| **Hot path bias** | Naturally weighted — hot code appears more | Equal weight to hot and cold code |
| **Cold path visibility** | May miss rarely-called code | Sees everything, even code called once |
| **Production safe** | Yes, commonly used in production | Rarely — overhead too high |
| **Examples** | perf, samply, cargo-flamegraph, Instruments | callgrind (Valgrind), gprof, tracing/spans |

For performance optimization, sampling profilers are almost always the right choice: they find the hot spots with minimal distortion.

### How Stack Sampling Works

When the profiler fires (e.g., on a timer interrupt), the OS or profiler:

1. **Pauses the target thread** (briefly, microseconds)
2. **Walks the call stack** — reads frame pointers or unwind tables to reconstruct the chain of function calls from the current instruction pointer back to `main()`
3. **Records the stack trace** — a sequence like `main → process_data → parse_line → regex::find`
4. **Resumes the thread**

After collecting thousands of samples, each unique stack trace has a count. Aggregating these counts produces a profile showing which functions (and which call paths to those functions) consumed the most CPU time.

**Debug symbols** are critical: without them, the profiler only sees memory addresses (`0x7ff6a3b41020`). With debug info, it can map addresses to function names, file names, and line numbers. This is why we set `debug = true` in the release profile — it embeds DWARF/PDB debug info without disabling optimizations.

### Flamegraphs: Visualizing Stack Samples

Invented by Brendan Gregg at Netflix, flamegraphs are the standard visualization for stack sample data. Here is how to read one:

```
                    [regex::find]         <- leaf (on-CPU function)
              [parse_line_______________] <- caller
        [process_data___________________]
  [main_________________________________] <- root
```

**Anatomy:**
- **Y-axis** = call stack depth. Bottom is the entry point (`main`), top is the function that was actually executing when the sample was taken. Each box is one stack frame.
- **X-axis** = sample population, NOT time. The x-axis spans all collected samples. Left-to-right ordering is **alphabetical** (to maximize merging of identical frames), not chronological.
- **Width** = proportion of total samples that include this function in their stack. A function that is wide at the bottom means many call paths go through it. A function that is wide at the top means it was frequently the function actually on-CPU.
- **Color** = typically random warm colors (the "flame" aesthetic). In some tools, color encodes language, module, or hot/cold status.

**Reading strategy:**
1. Look for **wide plateaus at the top** — these are functions that spend a lot of CPU time executing their own code (not calling other functions). These are your primary optimization targets.
2. Look for **wide bars lower in the stack** — these represent functions called very frequently or from many paths, even if they delegate work to children.
3. **Narrow towers** are rarely interesting — they represent deep call chains that consume little total time.
4. **Click to zoom** — interactive SVG flamegraphs let you click a frame to zoom in, making it the full width so you can see its children's relative proportions.

### cargo-flamegraph: One-Command Profiling

`cargo flamegraph` is a cargo subcommand that:
1. Builds your project in release mode (with debug info)
2. Runs it under the platform's native sampling profiler (perf on Linux, dtrace on macOS, ETW on Windows)
3. Collects stack samples
4. Generates an interactive SVG flamegraph

On **Windows**, it uses Event Tracing for Windows (ETW) by default, which requires running as Administrator. No additional tools need to be installed — ETW is built into Windows. The tool produces `flamegraph.svg` in the current directory, which you open in a browser.

Key flags:
- `--release` — profile release build (default)
- `--dev` — profile debug build (usually too slow to be useful)
- `--bin <name>` — profile a specific binary in a workspace
- `-o <file>` — output to a specific file instead of `flamegraph.svg`
- `--open` — open the SVG in the browser after generation
- `-- <args>` — pass arguments to the profiled program

### samply: Firefox Profiler UI

samply is a newer sampling profiler that produces output viewable in the Firefox Profiler web UI (`https://profiler.firefox.com`). It provides:

- **Timeline view**: See CPU usage over time (not just aggregated)
- **Call tree**: Hierarchical breakdown of time per function
- **Flamegraph**: Same as cargo-flamegraph but interactive in-browser
- **Source view**: Click a function to see annotated source code with per-line timing
- **Marker support**: Custom markers for events

On Windows, samply uses ETW and requires Administrator privileges. After recording, it starts a local web server and opens the Firefox Profiler UI in your browser.

### Heap Profiling with DHAT

CPU profiling tells you where time is spent. **Heap profiling** tells you where memory is allocated. The `dhat` crate provides:

- **Total allocations**: How many `malloc`/`Box::new`/`Vec::push` calls happened and where
- **Peak memory**: The maximum heap usage at any point during execution
- **Allocation lifetimes**: How long each allocation lived before being freed (short-lived allocations suggest opportunities for stack allocation or arena allocation)
- **Hot allocation sites**: Which lines of code are responsible for the most allocations

DHAT works by replacing the global allocator with a tracking wrapper (`dhat::Alloc`). Every allocation and deallocation is recorded with a backtrace. On program exit, the data is written to a JSON file (`dhat-heap.json`) viewable in the [DHAT viewer](https://nnethercote.github.io/dh_view/dh_view.html).

**Setup pattern:**
```rust
#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    // ... application code ...
}  // Profiler dropped here → writes dhat-heap.json
```

The feature flag pattern (`#[cfg(feature = "dhat-heap")]`) means DHAT adds zero overhead when not enabled. You only activate it when profiling: `cargo run --release --features dhat-heap`.

**Important**: DHAT significantly slows down the program (especially on Windows where backtrace collection is slow). Always profile with `--release` and be patient — it can be 10-50x slower than normal execution.

### Release Builds with Debug Info

By default, Rust release builds strip debug symbols. For profiling, we need both optimizations AND debug info:

```toml
[profile.release]
debug = true    # Full debug info (DWARF on Linux/macOS, PDB on Windows)
```

This does NOT disable optimizations. The binary will be larger (debug info is stored separately on Windows in a `.pdb` file) but runs at the same speed. The debug info allows profilers to map instruction addresses back to function names, file names, and line numbers.

You can also use `debug = 1` for line tables only (smaller, but no variable info) or `debug = 2` (same as `true`, full info). For profiling, `debug = true` is recommended.

### cargo-show-asm: Viewing Generated Assembly

Sometimes profiling reveals a hot function, and you need to understand what the compiler actually generated. `cargo-show-asm` lets you inspect the assembly for any public function:

```bash
cargo asm --lib my_crate::hot_function    # x86-64 assembly (Intel syntax)
cargo asm --lib --llvm my_crate::fn_name  # LLVM-IR (before machine codegen)
cargo asm --lib --mir my_crate::fn_name   # Rust MIR (before LLVM)
cargo asm --lib --rust my_crate::fn_name  # Assembly interleaved with Rust source
```

This is like a local Compiler Explorer (godbolt.org) — you see exactly what the compiler emits, including whether SIMD instructions were auto-vectorized, whether bounds checks were elided, and whether small functions were inlined.

**Tip**: Mark functions with `#[inline(never)]` to prevent them from being inlined away when inspecting their assembly.

### The Profiling Workflow

A disciplined profiling workflow follows this cycle:

1. **Establish a baseline**: Run the program, measure wall-clock time (`std::time::Instant`)
2. **Profile**: Generate a flamegraph or profiler capture
3. **Identify the hotspot**: Find the widest plateau at the top of the flamegraph
4. **Hypothesize**: Form a theory about why that code is slow
5. **Optimize**: Apply a targeted change (algorithm, data structure, allocation pattern)
6. **Measure again**: Re-profile to confirm improvement and check for regression elsewhere
7. **Repeat**: Go to step 2 until satisfied

Never skip step 6. Optimizations that seem correct can backfire — e.g., pre-allocating a huge buffer to avoid small allocations may cause cache misses that cost more than the allocations saved.

## Description

Build a deliberately inefficient Rust program with multiple distinct performance bottlenecks (CPU-bound, allocation-heavy, cache-unfriendly). Use cargo-flamegraph and samply to identify CPU hotspots, DHAT to find allocation-heavy code, and cargo-show-asm to inspect generated assembly. Then optimize each bottleneck guided by profiling data, comparing before and after measurements to quantify the improvement.

### What you'll learn

1. **Tool installation** — Setting up cargo-flamegraph, samply, cargo-show-asm
2. **Release+debug builds** — Why and how to keep debug info in optimized builds
3. **Flamegraph generation** — Running cargo-flamegraph, reading the SVG output
4. **samply profiling** — Recording a profile, navigating the Firefox Profiler UI
5. **Heap profiling** — Using DHAT to find allocation-heavy code
6. **Assembly inspection** — Viewing compiler output for hot functions
7. **Optimization cycle** — Before/after measurement, guided by profiling evidence

## Instructions

### Exercise 1: Tool Setup & Baseline (~10 min)

Open `src/ex1_setup.rs`. This exercise establishes the baseline: installing profiling tools and creating a timing harness.

Before you can profile, you need the tools installed. This exercise also creates a reusable `bench_fn` timing utility and validates that your release build has debug symbols enabled (without which profilers produce useless output).

1. **TODO(human): `install_tools()`** — Write a function that prints the cargo install commands for each tool and checks if they are available on PATH. This teaches the tool ecosystem and verifies your environment is ready.

2. **TODO(human): `bench_fn()`** — Create a generic timing harness that runs a closure, measures wall-clock time with `Instant::now()` and `elapsed()`, and prints the result. This is your baseline measurement tool for the entire practice.

### Exercise 2: The Deliberately Slow Workload (~15 min)

Open `src/ex2_workload.rs`. This exercise creates a program with known, distinct bottlenecks that you will later find with profiling tools.

A good profiling target needs multiple bottlenecks of different types so you can practice identifying each one. The workload has three intentional problems: (1) an O(n^2) algorithm where O(n log n) exists, (2) excessive heap allocations in a hot loop, and (3) a cache-unfriendly memory access pattern. Each bottleneck will show up differently in profiling output.

1. **TODO(human): `quadratic_search()`** — Implement a function that finds duplicates in a vector using nested loops (O(n^2)) instead of a HashSet (O(n)). This is the classic "wrong algorithm" bottleneck — profiling will show this function dominating the flamegraph width.

2. **TODO(human): `allocation_heavy()`** — Implement a function that processes data by creating a new `String` or `Vec` allocation on every iteration of a hot loop, instead of reusing a buffer. This bottleneck shows up in DHAT as thousands of short-lived allocations at the same call site.

3. **TODO(human): `cache_unfriendly()`** — Implement a function that accesses a large array in a strided or random pattern instead of sequentially. This bottleneck is subtle in flamegraphs (the function itself is the hot spot, but the cause is memory latency, not instruction count) and teaches the difference between CPU-bound and memory-bound bottlenecks.

4. **Pre-built: `run_workload()`** — Orchestrator that calls all three bottleneck functions and reports timing. This is already implemented.

### Exercise 3: Flamegraph Generation & Reading (~20 min)

Open `src/ex3_flamegraph.rs`. This is the core profiling exercise — generating a flamegraph and interpreting what it shows.

This exercise guides you through the complete flamegraph workflow: build with debug info, run under the profiler, open the SVG, and identify which of the three bottleneck functions from Exercise 2 dominates. You will learn to read flamegraph width, understand the call stack hierarchy, and identify the leaf functions where CPU time is actually spent.

1. **TODO(human): `configure_workload_for_profiling()`** — Set up the workload parameters (input sizes) so that the program runs long enough to collect meaningful samples (at least 5-10 seconds). Too short and the profiler collects too few samples for a useful flamegraph; too long and you waste time. This teaches the practical skill of tuning profiling runs.

2. **TODO(human): `analyze_flamegraph()`** — After generating the flamegraph with cargo-flamegraph (run from the command line, not from code), implement a function that prints your analysis: which function was widest, what percentage of samples it contained (estimate from the SVG), and what optimization you would apply. This teaches the critical skill of translating profiling data into actionable insights.

### Exercise 4: Heap Profiling with DHAT (~20 min)

Open `src/ex4_dhat.rs`. This exercise profiles heap allocation behavior to find allocation-heavy code.

CPU profiling shows where time is spent, but allocation profiling shows where memory pressure comes from. Short-lived allocations in hot loops cause: (1) allocator overhead from frequent malloc/free calls, (2) cache pollution as newly allocated memory displaces hot data from cache, and (3) GC pressure in languages with garbage collection (not Rust, but relevant for FFI interop). DHAT finds these patterns.

1. **TODO(human): `allocation_storm()`** — Create a function that deliberately generates thousands of small, short-lived heap allocations (e.g., creating Strings in a loop, or building Vec<Vec<u8>> with inner vecs). This will produce a dramatic DHAT report showing one allocation site responsible for most of the program's allocations.

2. **TODO(human): `analyze_dhat_output()`** — After running with `--features dhat-heap` and loading the JSON in the DHAT viewer, implement a function that prints your analysis: total allocations, peak heap size, the dominant allocation site, and the proposed fix (e.g., pre-allocate, reuse buffer, use stack allocation). This teaches reading the DHAT viewer's "Invocation Tree" and "Sort Metrics".

### Exercise 5: Assembly Inspection with cargo-show-asm (~15 min)

Open `src/ex5_assembly.rs`. This exercise teaches you to read compiler output for hot functions.

When profiling identifies a hot function and you have optimized the algorithm, the next question is: "Is the compiler generating efficient code?" cargo-show-asm answers this by showing the actual machine instructions. You will look for: bounds check elimination (or failure), auto-vectorization (SIMD instructions like `vaddps`, `vmulps`), and unnecessary copies or moves.

1. **TODO(human): `sum_slice()`** — Implement a simple function that sums a `&[f64]` slice. Mark it `#[inline(never)]` so it appears in assembly output. After implementing, inspect the assembly with cargo-show-asm and look for whether the compiler auto-vectorized it (uses SSE/AVX instructions like `addpd`/`vaddpd` instead of scalar `addsd`).

2. **TODO(human): `sum_with_bounds_checks()`** — Implement the same sum but using indexing (`slice[i]`) instead of iterators. Compare the assembly output — indexing generates bounds checks (conditional branches) that iterators avoid. This teaches why idiomatic Rust (iterators) often generates better code than C-style loops.

3. **TODO(human): `dot_product()`** — Implement a dot product of two slices. Inspect whether the compiler fuses the multiply and add into FMA instructions (`vfmadd`). This teaches how to verify auto-vectorization and spot missed optimization opportunities.

### Exercise 6: Guided Optimization (~20 min)

Open `src/ex6_optimize.rs`. This exercise closes the loop: apply optimizations guided by profiling evidence and measure the improvement.

This is where the practice pays off. You have identified the bottlenecks (Exercises 3-4), understood the compiler output (Exercise 5), and now you apply targeted fixes. The key discipline is: change ONE thing at a time, re-measure, and verify the improvement matches your expectation. If it does not, your mental model is wrong and you need to investigate further.

1. **TODO(human): `optimized_search()`** — Replace the O(n^2) duplicate search with a HashSet-based O(n) approach. Measure speedup vs the original. This should show a dramatic improvement that is clearly visible in a new flamegraph (the function disappears or becomes a thin sliver).

2. **TODO(human): `optimized_allocation()`** — Replace the allocation-heavy loop with a buffer-reuse pattern (pre-allocate a `String` or `Vec` and `.clear()` + reuse it each iteration). Measure speedup and re-run DHAT to verify the allocation count dropped.

3. **TODO(human): `optimized_cache()`** — Replace the cache-unfriendly access pattern with sequential access. Measure speedup. This optimization often shows a smaller improvement in flamegraphs (the function was always the same number of instructions) but a large improvement in wall-clock time (fewer cache misses).

4. **TODO(human): `run_comparison()`** — Run all original and optimized versions, print a comparison table showing before/after wall-clock times and speedup ratios. This is the deliverable: evidence-based optimization results.

## Motivation

- **"Measure, don't guess" is the #1 performance engineering principle**: Every senior engineer needs profiling skills. Without them, optimization efforts are random and often counterproductive.
- **Flamegraphs are the industry standard**: Used at Netflix, Google, Meta, and throughout the systems programming community. Reading a flamegraph is a core competency for performance work.
- **Heap profiling prevents death by a thousand allocations**: Allocation overhead is one of the most common performance issues in Rust (despite no GC), especially when coming from Python/Java where allocations feel "free".
- **Bridges all other performance practices**: Profiling is the prerequisite for SIMD (Practice 062), lock-free optimization (061a), and any future work on latency-critical systems.
- **Windows-native**: Both cargo-flamegraph and samply work on Windows via ETW, making this practice fully executable on the user's platform.

## Commands

### Tool Installation

| Command | Description |
|---------|-------------|
| `cargo install flamegraph` | Install cargo-flamegraph (CPU flamegraph generator) |
| `cargo install samply` | Install samply (sampling profiler with Firefox Profiler UI) |
| `cargo install cargo-show-asm` | Install cargo-show-asm (view assembly/LLVM-IR/MIR for any function) |

### Build

| Command | Description |
|---------|-------------|
| `cargo check` | Fast type-check without codegen (use while implementing TODO exercises) |
| `cargo build --release` | Optimized build with debug info (see `[profile.release] debug = true` in Cargo.toml) |
| `cargo build --release --features dhat-heap` | Build with DHAT heap profiling enabled |

### Run

| Command | Description |
|---------|-------------|
| `cargo run --release` | Run all exercises with optimizations (baseline timing) |
| `cargo run --release -- 1` | Run only Exercise 1 (setup & baseline) |
| `cargo run --release -- 2` | Run only Exercise 2 (workload with bottlenecks) |
| `cargo run --release -- 3` | Run only Exercise 3 (flamegraph analysis) |
| `cargo run --release -- 4` | Run only Exercise 4 (DHAT heap profiling) |
| `cargo run --release -- 5` | Run only Exercise 5 (assembly inspection) |
| `cargo run --release -- 6` | Run only Exercise 6 (optimization & comparison) |

### Profiling — cargo-flamegraph (run as Administrator)

| Command | Description |
|---------|-------------|
| `cargo flamegraph --release -- 2` | Generate flamegraph of Exercise 2 workload (produces `flamegraph.svg`) |
| `cargo flamegraph --release -o before.svg -- 2` | Save flamegraph as `before.svg` for comparison |
| `cargo flamegraph --release -o after.svg -- 6` | Generate flamegraph of optimized workload for comparison |

### Profiling — samply (run as Administrator)

| Command | Description |
|---------|-------------|
| `cargo build --release` | Build first (samply profiles a binary, not a cargo project) |
| `samply record target/release/profiling-flamegraphs.exe 2` | Record profile of Exercise 2 (opens Firefox Profiler in browser) |
| `samply record target/release/profiling-flamegraphs.exe 6` | Record profile of optimized Exercise 6 for comparison |

### Profiling — DHAT Heap Profiling

| Command | Description |
|---------|-------------|
| `cargo run --release --features dhat-heap -- 4` | Run Exercise 4 with DHAT profiling (produces `dhat-heap.json`) |
| Open `https://nnethercote.github.io/dh_view/dh_view.html` | Load `dhat-heap.json` in the DHAT viewer web app |

### Assembly Inspection — cargo-show-asm

| Command | Description |
|---------|-------------|
| `cargo asm --lib` | List all available public functions to inspect |
| `cargo asm --lib profiling_flamegraphs::ex5_assembly::sum_slice` | Show assembly for `sum_slice()` |
| `cargo asm --lib --rust profiling_flamegraphs::ex5_assembly::sum_slice` | Show assembly interleaved with Rust source |
| `cargo asm --lib --llvm profiling_flamegraphs::ex5_assembly::sum_slice` | Show LLVM-IR for `sum_slice()` |
| `cargo asm --lib profiling_flamegraphs::ex5_assembly::dot_product` | Show assembly for `dot_product()` |

### Development

| Command | Description |
|---------|-------------|
| `cargo clippy` | Run linter for idiomatic Rust suggestions |
| `cargo test` | Run unit tests for exercises |

## References

- [Brendan Gregg — Flame Graphs](https://www.brendangregg.com/flamegraphs.html) — Original flamegraph concept, reading guide, and variations
- [Brendan Gregg — CPU Flame Graphs](https://www.brendangregg.com/FlameGraphs/cpuflamegraphs.html) — Detailed guide to CPU flamegraph interpretation
- [cargo-flamegraph GitHub](https://github.com/flamegraph-rs/flamegraph) — cargo-flamegraph source, README, and Windows ETW support docs
- [samply GitHub](https://github.com/mstange/samply) — samply source and cross-platform setup instructions
- [dhat crate docs.rs](https://docs.rs/dhat/latest/dhat/) — Full API docs for heap profiling and ad hoc profiling
- [dhat-rs GitHub](https://github.com/nnethercote/dhat-rs) — DHAT for Rust, examples, and testing capabilities
- [DHAT Viewer](https://nnethercote.github.io/dh_view/dh_view.html) — Web-based viewer for dhat-heap.json files
- [cargo-show-asm GitHub](https://github.com/pacak/cargo-show-asm) — Assembly/LLVM-IR/MIR viewer for Rust functions
- [The Rust Performance Book — Profiling](https://nnethercote.github.io/perf-book/profiling.html) — Overview of profiling tools for Rust
- [The Rust Performance Book — Heap Allocations](https://nnethercote.github.io/perf-book/heap-allocations.html) — Strategies for reducing heap allocations

## Notes

*(To be filled during practice.)*
