# Practice 020a: HFT Low-Latency C++ Patterns

## Technologies

- **C++17** -- `alignas`, `std::atomic`, `if constexpr`, CRTP, `std::variant`, `std::visit`, fold expressions
- **Windows API** -- `__rdtsc()`, `QueryPerformanceCounter`, `VirtualAlloc` (huge pages)
- **CMake 3.16+** -- MSVC build with `/O2`, `/arch:AVX2` for Release

## Stack

- C++17 (MSVC via VS 2022)
- No external dependencies -- pure standard library + OS APIs

## Description

Build the **core data structures and patterns** used in real high-frequency trading systems. No finance domain knowledge required -- this is about **systems programming**: memory layout, lock-free concurrency, allocation-free hot paths, compile-time dispatch, and nanosecond-precision timing.

Every exercise includes benchmarks so you can *see* the performance difference, not just read about it.

### What you'll learn

1. **Cache-friendly data structures** -- Why struct layout matters more than algorithmic complexity at HFT scale
2. **Lock-free SPSC ring buffer** -- THE fundamental IPC primitive in trading systems
3. **Memory pools & arena allocators** -- Why `new`/`delete` are banned on the hot path
4. **Compile-time dispatch** -- CRTP, `if constexpr`, and why vtables are too slow
5. **TSC-based timing** -- Sub-nanosecond precision timing using CPU cycle counters
6. **Integrated pipeline** -- Combine everything into a realistic market-data-to-order pipeline

### Key numbers to internalize

| Operation | Latency |
|-----------|---------|
| L1 cache hit | ~1 ns (4 cycles) |
| L2 cache hit | ~4 ns (12 cycles) |
| L3 cache hit | ~12 ns (40 cycles) |
| Main memory (DRAM) | ~60-100 ns |
| Cache line | 64 bytes |
| Context switch | ~1-10 us |
| `malloc`/`new` (cold) | ~50-100 ns |
| Virtual function call | ~2-5 ns (indirect branch + possible cache miss) |
| `std::chrono::steady_clock` | ~15-25 ns |
| `__rdtsc()` | ~5-10 ns |
| **HFT tick-to-trade budget** | **~500 ns - 5 us** |

## Instructions

### Phase 1: Cache-Friendly Data Structures (~20 min)

1. **Concepts:** Cache lines (64B), `alignas`, struct padding, AoS vs SoA, hot/cold splitting
2. **Exercise 1:** Implement `OrderUpdate` with `alignas(64)` and `static_assert(sizeof == 64)`
3. **Exercise 2:** Implement hot/cold data split -- separate frequently-accessed from rarely-accessed fields
4. **Exercise 3:** Benchmark sequential vs random access to demonstrate cache miss cost
5. Key question: When would you choose SoA over AoS? (Hint: think about which fields you iterate over)

### Phase 2: Lock-Free SPSC Ring Buffer (~25 min)

1. **Concepts:** `std::atomic`, memory ordering (acquire/release vs seq_cst), false sharing, power-of-2 masking
2. **Exercise 1:** Implement `try_push` and `try_pop` with correct memory ordering
3. **Exercise 2:** Add cache-line padding between head and tail to eliminate false sharing
4. **Exercise 3:** Benchmark throughput (M ops/sec) -- compare padded vs unpadded, seq_cst vs acq/rel
5. Key question: Why is `memory_order_acquire` on load and `memory_order_release` on store sufficient? What ordering guarantee do they provide?

### Phase 3: Memory Pools & Allocation-Free Hot Path (~20 min)

1. **Concepts:** Object pools, arena/bump allocators, placement new, deterministic latency
2. **Exercise 1:** Implement a fixed-size object pool (`allocate` / `deallocate` returning raw pointers)
3. **Exercise 2:** Implement a bump arena allocator for temporary per-message allocations
4. **Exercise 3:** Benchmark pool vs `new`/`delete` -- measure P50/P99/max latency over 10M iterations
5. Key question: Why does `new`/`delete` cause latency *jitter* even when the average is fast?

### Phase 4: Compile-Time Dispatch & Branch Elimination (~15 min)

1. **Concepts:** CRTP (static polymorphism), `if constexpr`, `[[likely]]`/`[[unlikely]]`, `std::variant` + `std::visit`
2. **Exercise 1:** Implement CRTP-based message handler vs virtual function version
3. **Exercise 2:** Use `if constexpr` for zero-cost order-type dispatch
4. **Exercise 3:** Benchmark: virtual vs CRTP vs `std::visit` -- measure per-call overhead
5. Key question: What does the compiler generate differently for CRTP vs virtual? (Hint: inlining)

### Phase 5: Timestamps & Low-Latency Timing (~15 min)

1. **Concepts:** RDTSC, TSC calibration, `QueryPerformanceCounter`, latency histograms
2. **Exercise 1:** Implement a TSC reader using `__rdtsc()` intrinsic
3. **Exercise 2:** Calibrate TSC frequency (cycles-to-nanoseconds conversion)
4. **Exercise 3:** Compare timing methods: `__rdtsc()` vs `steady_clock` vs `QueryPerformanceCounter`
5. **Exercise 4:** Implement a latency histogram with sub-microsecond buckets
6. Key question: Why can't you just use `std::chrono` everywhere? When does 20ns of timing overhead matter?

### Phase 6: Putting It Together -- Mini Hot-Path Pipeline (~20 min)

1. **Concepts:** End-to-end latency, pipeline architecture, zero-allocation message flow
2. **Exercise:** Build market data -> signal -> order pipeline combining all phases:
   - Market data arrives on SPSC queue (Phase 2)
   - Signal generator processes it allocation-free (Phase 3 pool)
   - Order dispatched via CRTP handler (Phase 4)
   - Pipeline timed with TSC (Phase 5)
   - Latency histogram collected per message
3. **Target:** Sub-microsecond processing per message
4. Key question: Where are the remaining bottlenecks? What would a real HFT firm optimize next?

## Motivation

- **HFT is the pinnacle of C++ systems programming** -- These patterns (lock-free queues, memory pools, cache-aware layout) apply to any latency-sensitive system: game engines, audio processing, embedded, network stacks.
- **Bridges CP and production** -- CP teaches algorithmic thinking; HFT teaches *systems* thinking. Same language (C++17), completely different optimization axis (memory hierarchy vs asymptotic complexity).
- **Interview-relevant** -- Jane Street, Citadel, Two Sigma, HRT, Optiver all ask about these patterns. Even non-HFT systems roles (Google, Meta infra) value lock-free data structure knowledge.
- **Complements practice 012a** -- That covered C++17 *language* features. This covers C++17 *systems* patterns. Together they show full-stack C++ proficiency.
- **Rust parallel** -- `crossbeam::channel` (SPSC), `typed-arena`, `bumpalo`, zero-cost abstractions via monomorphization -- these are Rust equivalents you already know conceptually.

## References

- [What Every Programmer Should Know About Memory (Drepper)](https://people.freebsd.org/~lstewart/articles/cpumemory.pdf) -- The definitive guide to memory hierarchy
- [Lock-Free Programming (Preshing on Programming)](https://preshing.com/20120612/an-introduction-to-lock-free-programming/) -- Best introductory series on lock-free
- [Memory Ordering at Compile Time (Preshing)](https://preshing.com/20120625/memory-ordering-at-compile-time/) -- Why `acquire`/`release` suffice for SPSC
- [CppCon 2017: Carl Cook "When a Microsecond Is an Eternity"](https://www.youtube.com/watch?v=NH1Tta7purM) -- HFT-specific C++ patterns
- [CppCon 2019: Fedor Pikus "The C++ Memory Model"](https://www.youtube.com/watch?v=A8eCGOqgvH4) -- Deep dive on atomics and ordering
- [Mechanical Sympathy (Martin Thompson)](https://mechanical-sympathy.blogspot.com/) -- Cache-aware programming
- [cppreference: std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic) -- Memory ordering reference
- [cppreference: alignas](https://en.cppreference.com/w/cpp/language/alignas) -- Alignment specifier

## Commands

### Setup & Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Release"` | Configure CMake (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases"` | Build all 6 phases (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Debug"` | Configure CMake (Debug) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Debug --target all_phases"` | Build all 6 phases (Debug) |
| `build.bat` | Build all targets (Release) via helper script |
| `build.bat debug` | Build all targets (Debug) via helper script |
| `build.bat clean` | Remove the build directory |
| `gen_compile_commands.bat` | Generate `compile_commands.json` for clangd (Ninja + MSVC) |

### Build Individual Phases

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_cache"` | Build Phase 1 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_spsc"` | Build Phase 2 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_memory"` | Build Phase 3 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_dispatch"` | Build Phase 4 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_timing"` | Build Phase 5 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_pipeline"` | Build Phase 6 only |
| `build.bat phase1` | Build Phase 1 via helper script |
| `build.bat phase2` | Build Phase 2 via helper script |
| `build.bat phase3` | Build Phase 3 via helper script |
| `build.bat phase4` | Build Phase 4 via helper script |
| `build.bat phase5` | Build Phase 5 via helper script |
| `build.bat phase6` | Build Phase 6 via helper script |

### Run Phases

| Command | Description |
|---------|-------------|
| `build\Release\phase1_cache.exe` | Run Phase 1: Cache-Friendly Data Structures |
| `build\Release\phase2_spsc.exe` | Run Phase 2: Lock-Free SPSC Ring Buffer |
| `build\Release\phase3_memory.exe` | Run Phase 3: Memory Pools & Arena Allocator |
| `build\Release\phase4_dispatch.exe` | Run Phase 4: Compile-Time Dispatch |
| `build\Release\phase5_timing.exe` | Run Phase 5: Timestamps & Low-Latency Timing |
| `build\Release\phase6_pipeline.exe` | Run Phase 6: Integrated Hot-Path Pipeline |

## State

`not-started`
