# Practice 022: Concurrent Data Structures -- moodycamel::ConcurrentQueue

## Technologies

- **moodycamel::ConcurrentQueue** -- Industrial-strength lock-free MPMC (Multi-Producer Multi-Consumer) queue
- **moodycamel::BlockingConcurrentQueue** -- Blocking variant with wait/signal semantics
- **C++17** -- `<thread>`, `<atomic>`, `<chrono>`, structured bindings, `if constexpr`
- **CMake 3.16+** -- FetchContent for header-only dependency management

## Stack

- C++17 (MSVC via VS 2022)
- moodycamel/concurrentqueue v1.0.4 (fetched via CMake FetchContent)

## Theoretical Context

### What Lock-Free MPMC Queues Are

A **Multi-Producer Multi-Consumer (MPMC) queue** allows multiple threads to enqueue and dequeue concurrently without requiring a global mutex. This contrasts with Single-Producer Single-Consumer (SPSC) queues (covered in practice 020a), which are simpler and faster but only support one producer and one consumer.

Lock-free data structures guarantee **system-wide progress**: at least one thread always makes progress, even if others are paused by the OS scheduler. They achieve this via **atomic Compare-And-Swap (CAS)** operations—hardware instructions that atomically read, compare, and conditionally write a memory location. On x86/x64, CAS compiles to `LOCK CMPXCHG`, which takes ~10-30 ns under low contention and ~30-100 ns under high contention.

**moodycamel::ConcurrentQueue** is the most widely-used lock-free MPMC queue in production C++. It's used in game engines (Unreal Engine), databases, real-time audio, and high-performance servers. Its key innovations: (1) per-producer sub-queues reduce contention, (2) bulk operations amortize overhead, (3) explicit tokens provide thread-local fast paths.

### How Lock-Free Queues Work Internally

Traditional mutex-based queues serialize all operations through a single lock. Under contention (multiple threads competing), mutex overhead explodes: lock acquisition becomes a bottleneck, and every thread spins waiting its turn.

**Lock-free MPMC design (simplified):**
1. **Array of slots** (circular buffer), each with an atomic sequence number
2. **Head and tail indices** (atomics tracking next enqueue/dequeue position)
3. **Enqueue:**
   - Load tail, CAS-increment it to claim a slot
   - If another thread won the CAS, retry
   - Write data to claimed slot, update sequence number to mark it ready
4. **Dequeue:**
   - Load head, CAS-increment it to claim a slot
   - Spin-wait until sequence number shows data is ready
   - Read data, mark slot empty

Key insight: Threads compete only on the tail/head indices (two atomic variables), not on every data element. False CAS retries are cheap (~20-50 ns retry loop) compared to mutex contention (~50-200 ns blocked time).

**moodycamel's sub-queue optimization:** Each producer gets a dedicated sub-queue (via `ProducerToken`). Enqueue fast-path doesn't CAS at all—it's sequential writes to the thread's private sub-queue. Consumers multiplex across all sub-queues. This reduces the MPMC problem to "multiple SPSC-like channels merged into one MPMC view," achieving near-SPSC performance.

**Bulk operations:** `enqueue_bulk` and `try_dequeue_bulk` amortize per-operation overhead. Instead of N separate CAS operations, perform one CAS to reserve N slots, then write/read them sequentially. Throughput can improve 3-10× for batch workloads.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Lock-free** | Concurrent data structure where at least one thread always makes progress. No mutexes/locks. Uses atomic CAS operations. |
| **CAS (Compare-And-Swap)** | Atomic instruction: `if (*ptr == expected) { *ptr = desired; return true; } else return false;`. Foundation of lock-free algorithms. |
| **ABA problem** | CAS sees `*ptr == A`, assumes no change, but actually `A → B → A` occurred. Solved via tagged pointers or epoch-based reclamation. |
| **MPMC (Multi-Producer Multi-Consumer)** | Multiple threads can enqueue and dequeue concurrently. Harder than SPSC (covered in 020a) due to contention. |
| **ProducerToken** | Thread-local handle that provides a dedicated sub-queue, eliminating CAS contention on enqueue fast-path. |
| **ConsumerToken** | Thread-local handle that caches consumer-side metadata (queue iteration state), reducing CAS overhead. |
| **Bulk operations** | `enqueue_bulk(items, count)` reserves N slots with one CAS, writes sequentially. Amortizes per-item overhead. |
| **Blocking queue** | `BlockingConcurrentQueue` variant that blocks (sleeps) waiting threads instead of spin-waiting. Trades latency for CPU efficiency. |
| **False sharing** | Two threads writing to different variables in the same cache line cause cache invalidation. Padding to separate cache lines avoids this. |
| **Memory ordering** | `memory_order_acquire/release/seq_cst` control visibility of writes across threads. Relaxed ordering is faster but requires careful reasoning. |

### Ecosystem Context and Trade-offs

Lock-free MPMC queues optimize for **high contention** and **sustained throughput**. Under low contention, a mutex-based queue can be competitive or even faster due to simpler code paths. Lock-free shines when 4+ threads are hammering the queue simultaneously.

**Trade-offs:**
- **Complexity** → Lock-free algorithms are notoriously hard to get right. ABA problems, memory ordering bugs, and race conditions are subtle. moodycamel abstracts this, but understanding it aids debugging.
- **Bounded capacity** → Most lock-free queues (including moodycamel's default) dynamically allocate when full, which adds latency. Pre-sizing mitigates this but wastes memory.
- **Priority inversion** → Lock-free doesn't mean "fair." A thread can starve if it repeatedly loses CAS races. Real-time systems often prefer mutexes with priority inheritance.
- **Memory reclamation** → When a node is dequeued, when is it safe to free? Lock-free requires hazard pointers or epoch-based reclamation. moodycamel handles this internally.

**Alternatives:**
- **Mutex + std::queue** → Simple, correct, good enough for low contention (<4 threads)
- **SPSC ring buffer (020a)** → Fastest option if topology allows single producer/consumer
- **Disruptor (LMAX)** → Java framework with similar ideas (ring buffer, pre-allocation, no locks)
- **Crossbeam (Rust)** → MPMC `channel::unbounded()`, similar design to moodycamel

**When to use lock-free MPMC:**
- Fan-out/fan-in pipelines: one dispatcher distributing tasks to N workers
- Multi-stage processing: each stage has multiple worker threads
- Producer-consumer with bursty traffic (bulk operations shine here)
- Systems where P99 latency matters more than P50 (mutexes have worse tail latency)

**When NOT to use:**
- Low contention (1-2 threads): mutex is simpler
- Need strict FIFO across all producers: lock-free MPMC doesn't guarantee global ordering
- Ultra-low-latency hot path: SPSC is faster (5-15 ns vs 15-40 ns)

moodycamel's sweet spot is **general-purpose server software**: web servers, databases, game engines, data pipelines. It's not HFT-hot-path fast (that's SPSC territory), but it's 3-10× faster than mutex queues under real-world contention.

## Description

Master **concurrent queue patterns** using moodycamel::ConcurrentQueue -- the most widely-used lock-free MPMC queue in production C++. While practice 020a covered SPSC (single-producer single-consumer) ring buffers for HFT hot paths, this practice covers the **general-purpose MPMC** case: multiple threads producing and consuming concurrently without locks.

### What you'll learn

1. **MPMC vs SPSC tradeoffs** -- Why MPMC is harder and when you need it vs SPSC
2. **Lock-free internals** -- CAS (compare-and-swap) underneath, why it beats mutex under contention
3. **Producer/Consumer tokens** -- Thread-local fast paths that reduce contention to near-zero
4. **Bulk operations** -- Amortized per-item overhead for batch processing
5. **Blocking queues** -- When to trade latency for CPU efficiency
6. **Real-world patterns** -- Fan-out, fan-in, multi-stage pipelines, work stealing

### Key numbers to internalize

| Operation | Typical Latency |
|-----------|----------------|
| `std::mutex` lock/unlock (uncontended) | ~15-25 ns |
| `std::mutex` lock/unlock (4 threads contending) | ~50-200 ns |
| moodycamel enqueue/dequeue (no tokens) | ~15-40 ns |
| moodycamel enqueue/dequeue (with tokens) | ~8-20 ns |
| moodycamel bulk enqueue (per item, batch=64) | ~3-8 ns |
| SPSC ring buffer (from 020a) | ~5-15 ns |
| CAS retry under contention | ~10-50 ns per retry |

## Instructions

### Phase 1: ConcurrentQueue Basics (~15 min)

1. **Concepts:** MPMC vs SPSC vs MPSC tradeoffs, why lock-free matters, CAS underneath
2. **Exercise 1:** Create a `ConcurrentQueue<int>`, enqueue 100 values, dequeue and verify
3. **Exercise 2:** Use `try_enqueue` and `try_dequeue` -- understand bool return values
4. **Exercise 3:** Pre-sized queue with `ConcurrentQueue<T>(initial_size)` -- why pre-sizing avoids runtime allocation
5. Key insight: Unlike `std::queue + mutex`, this queue allows concurrent push/pop without locking

### Phase 2: Multi-Threaded Producer/Consumer (~20 min)

1. **Concepts:** Spawning producer/consumer threads, coordinating shutdown, measuring throughput
2. **Exercise 1:** N producers each enqueue M items, 1 consumer dequeues all -- verify total == N*M
3. **Exercise 2:** N producers, N consumers -- each consumer counts its items, verify global total
4. **Exercise 3:** Throughput benchmark: ops/sec for ratios 1:1, 4:1, 1:4, 4:4
5. **Exercise 4:** Compare with `std::queue<T>` + `std::mutex` -- measure the speedup
6. Key insight: moodycamel shines under contention -- more threads = bigger advantage over mutex

### Phase 3: Producer/Consumer Tokens (~20 min)

1. **Concepts:** Thread-local fast paths via `ProducerToken` / `ConsumerToken`, skip CAS on fast path
2. **Exercise 1:** Refactor Phase 2 producers to use `ProducerToken` -- one token per thread
3. **Exercise 2:** Refactor consumers to use `ConsumerToken`
4. **Exercise 3:** Benchmark with vs without tokens -- expected 2-3x speedup under contention
5. Key insight: Tokens give each thread a dedicated sub-queue, reducing contention to near-zero. Like per-thread SPSC channels merging into a shared MPMC view.

### Phase 4: Bulk Operations (~15 min)

1. **Concepts:** `enqueue_bulk` / `try_dequeue_bulk` for batch processing, amortized overhead
2. **Exercise 1:** Enqueue 1000 items one-by-one vs `enqueue_bulk` -- benchmark both
3. **Exercise 2:** Dequeue in batches of 64 using `try_dequeue_bulk` -- measure throughput
4. **Exercise 3:** Batched pipeline: producer enqueues bulk, consumer dequeues bulk, measure end-to-end
5. Key insight: Bulk ops amortize per-item overhead (memory barriers, CAS retries). In HFT, batch market data updates for this reason.

### Phase 5: BlockingConcurrentQueue (~20 min)

1. **Concepts:** `wait_dequeue` / `wait_dequeue_timed` -- blocks until item available
2. **Exercise 1:** Producer-consumer with blocking dequeue -- no busy-wait spin needed
3. **Exercise 2:** Timed wait: `wait_dequeue_timed(item, milliseconds(100))` -- handle timeout
4. **Exercise 3:** Graceful shutdown: poison pill / sentinel value to signal consumers to stop
5. **Exercise 4:** CPU usage comparison: busy-wait `try_dequeue` loop vs `wait_dequeue`
6. Key insight: Use `BlockingConcurrentQueue` when latency isn't ultra-critical but you want low CPU. Use `ConcurrentQueue` + spin for minimum latency (HFT hot path).

### Phase 6: Real-World Patterns (~25 min)

1. **Concepts:** Common concurrent patterns built on MPMC queues
2. **Exercise 1:** **Fan-out**: One dispatcher distributes tasks to N worker queues
3. **Exercise 2:** **Fan-in / aggregation**: N producers push events into one queue, one aggregator consumes and merges
4. **Exercise 3:** **Pipeline stages**: Chain 3 queues (parse -> process -> output), each stage a thread pool, measure throughput and per-stage latency
5. **Exercise 4:** **Comparison matrix**: Benchmark all queue types for the pipeline:
   - `std::queue + mutex`
   - `moodycamel::ConcurrentQueue` (no tokens)
   - `moodycamel::ConcurrentQueue` (with tokens)
   - `moodycamel::BlockingConcurrentQueue`
6. Key insight: Most real systems are pipelines. The queue between stages determines system throughput.

## Motivation

- **Production concurrency gap**: CP is single-threaded. Production systems use multiple cores via concurrent queues -- this is the #1 pattern in server software, game engines, data pipelines.
- **moodycamel is industry standard**: Used in game engines (Unreal), databases, and high-performance servers. Understanding it signals real concurrency experience.
- **Bridges 020a SPSC to general MPMC**: Practice 020a taught SPSC for the HFT hot path (one producer, one consumer). Real systems also need MPMC for fan-out, fan-in, and work distribution.
- **Rust parallel**: `crossbeam::channel` (MPMC bounded/unbounded), `flume` (MPMC), `tokio::sync::mpsc` -- moodycamel is the C++ equivalent you'd reach for in production.
- **Interview-relevant**: Lock-free data structures, producer-consumer patterns, and throughput benchmarking are core systems interview topics at Google, Meta, trading firms.

## References

- [moodycamel::ConcurrentQueue (GitHub)](https://github.com/cameron314/concurrentqueue) -- Source, API docs, design blog
- [moodycamel blog: Lock-free queue design](https://moodycamel.com/blog/2014/a-fast-general-purpose-lock-free-queue-for-c++.htm) -- Design rationale and benchmarks
- [moodycamel blog: Detailed design](https://moodycamel.com/blog/2014/detailed-design-of-a-lock-free-queue.htm) -- Internal sub-queue architecture
- [CppCon 2015: Fedor Pikus "Live Lock-Free or Deadlock"](https://www.youtube.com/watch?v=lVBvHbJsg5Y) -- Lock-free fundamentals
- [Preshing: Lock-Free Programming](https://preshing.com/20120612/an-introduction-to-lock-free-programming/) -- CAS, memory ordering
- [cppreference: std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic) -- Memory ordering reference
- [crossbeam (Rust)](https://docs.rs/crossbeam/latest/crossbeam/) -- Rust MPMC equivalent

## Commands

### Setup & Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Release"` | Configure CMake (Release); first run fetches moodycamel/concurrentqueue via FetchContent |
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
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_basics"` | Build Phase 1 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_multithreaded"` | Build Phase 2 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_tokens"` | Build Phase 3 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_bulk"` | Build Phase 4 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_blocking"` | Build Phase 5 only |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_patterns"` | Build Phase 6 only |

### Run Phases

| Command | Description |
|---------|-------------|
| `build\Release\phase1_basics.exe` | Run Phase 1: ConcurrentQueue Basics |
| `build\Release\phase2_multithreaded.exe` | Run Phase 2: Multi-Threaded Producer/Consumer |
| `build\Release\phase3_tokens.exe` | Run Phase 3: Producer/Consumer Tokens |
| `build\Release\phase4_bulk.exe` | Run Phase 4: Bulk Operations |
| `build\Release\phase5_blocking.exe` | Run Phase 5: BlockingConcurrentQueue |
| `build\Release\phase6_patterns.exe` | Run Phase 6: Real-World Patterns (Fan-out, Fan-in, Pipeline) |

## State

`not-started`
