# Practice 061a: Lock-Free Rust — crossbeam & Epoch-Based Reclamation

## Technologies

- **crossbeam** (0.8) — Umbrella crate re-exporting the crossbeam ecosystem
- **crossbeam-epoch** — Epoch-based garbage collection for lock-free data structures
- **crossbeam-queue** — Lock-free concurrent queues (SegQueue, ArrayQueue)
- **crossbeam-channel** — Multi-producer multi-consumer channels with `select!`
- **crossbeam-skiplist** — Lock-free concurrent sorted map / set (SkipMap, SkipSet)
- **crossbeam-utils** — Low-level utilities: CachePadded, Backoff, scoped threads

## Stack

- Rust (cargo, edition 2021)

## Theoretical Context

### Lock-Free vs Lock-Based vs Wait-Free

Concurrent data structures are classified by their **progress guarantees**:

| Guarantee | Definition | Example |
|-----------|-----------|---------|
| **Blocking (lock-based)** | A thread holding a lock can prevent all others from making progress. If it is suspended by the OS, all waiters stall. | `Mutex<T>`, `RwLock<T>` |
| **Lock-free** | At least one thread makes progress in a finite number of steps, even if other threads are suspended or delayed. No thread can block the system forever. | Treiber stack, Michael-Scott queue, crossbeam SegQueue |
| **Wait-free** | Every thread makes progress in a bounded number of steps, regardless of other threads. The strongest guarantee but hardest to achieve. | `fetch_add` on `AtomicU64`, some specialized counters |

Lock-based code is correct but vulnerable to **priority inversion** (high-priority thread blocked by low-priority lock holder), **convoying** (many threads queue behind one slow critical section), and **deadlock** (circular lock dependencies). Lock-free algorithms eliminate these problems by using atomic operations — primarily **Compare-And-Swap (CAS)** — instead of locks.

The trade-off: lock-free algorithms are significantly harder to design correctly and require careful memory management, but they provide better worst-case latency and throughput under high contention.

### Compare-And-Swap (CAS)

CAS is the fundamental building block of lock-free algorithms. It atomically performs:

```
CAS(address, expected, desired):
    if *address == expected:
        *address = desired
        return Ok(expected)
    else:
        return Err(current_value)
```

In Rust, this is `AtomicPtr::compare_exchange(current, new, success_ordering, failure_ordering)`. The key insight is that CAS can fail — when another thread modified the value between our read and our CAS attempt. Lock-free algorithms handle this with a **retry loop**: read the current state, compute the desired new state, attempt CAS, and if it fails (another thread won the race), re-read and retry.

This retry pattern is called an **optimistic concurrency** approach — you assume no conflict and only retry on collision, as opposed to pessimistic locking which blocks preemptively.

### The ABA Problem

The ABA problem is the most subtle bug in lock-free programming. Consider a lock-free stack:

1. Thread A reads top-of-stack pointer → node X (value A)
2. Thread A is preempted by the OS
3. Thread B pops X (value A), pops Y, pushes Z, pushes a NEW node at the same memory address as X (value A again — memory was freed and reused)
4. Thread A resumes, does CAS(top, X, ...) — CAS **succeeds** because the pointer equals X, but the stack structure is now corrupted because the nodes between X and the new top are gone

CAS only checks pointer equality, not whether the world changed underneath. The pointer looks the same (address A), but the node and the list it pointed into are completely different.

Solutions to ABA:
- **Tagged pointers**: Pack a monotonic counter into the pointer's unused bits. Each CAS increments the tag, so even if the address is reused, the tag differs. Limited by available bits.
- **Hazard pointers**: Each thread publishes which pointers it is currently examining. Memory can only be reclaimed when no thread's hazard list contains it. Per-pointer tracking with O(threads * hazard_count) overhead.
- **Epoch-based reclamation (EBR)**: Batch memory reclamation using global epoch advancement. Memory removed in epoch E is only freed when all threads have advanced past E. This is what crossbeam uses.

### Epoch-Based Reclamation (EBR) — How crossbeam-epoch Works

EBR solves the memory reclamation problem for lock-free data structures with minimal overhead. The key idea: instead of tracking individual pointers (like hazard pointers), track **time intervals** during which threads are active.

**The three-epoch system:**

The global state consists of:
- A **global epoch counter** that cycles through three values: 0, 1, 2
- A **per-thread record** containing: (1) an "active" flag, and (2) the epoch the thread last observed

**How it works step by step:**

1. **Pinning** (`epoch::pin()` → `Guard`): When a thread wants to access a lock-free data structure, it calls `pin()`. This marks the thread as "active" and records the current global epoch. The returned `Guard` keeps the thread pinned. When the `Guard` is dropped, the thread is unpinned (marked inactive).

2. **Reading safely**: While pinned, the thread can load `Atomic<T>` pointers and get `Shared<'guard, T>` references. The lifetime `'guard` is tied to the `Guard`, ensuring the thread cannot use the pointer after unpinning. Rust's borrow checker enforces this statically.

3. **Removing nodes**: When a thread removes a node from a data structure (e.g., popping from a stack), it cannot immediately free the memory — other pinned threads may still hold `Shared` pointers to it. Instead, it calls `guard.defer_destroy(shared_ptr)`, which adds the node to a **garbage bag** stamped with the current epoch.

4. **Epoch advancement**: Periodically (typically during `pin()` calls), a thread attempts to advance the global epoch. Advancement succeeds only if **all active threads** have their local epoch equal to the current global epoch — meaning they've all "caught up." When the epoch advances from E to E+1, garbage from epoch E-1 (two epochs ago) becomes safe to free, because no active thread could possibly hold a reference from two epochs ago.

5. **Why three epochs?** During the transition from epoch E to E+1, some threads may be in epoch E and others in E+1. So garbage from epoch E is NOT safe yet. But garbage from epoch E-1 IS safe, because all active threads are in E or E+1, and they pinned AFTER E-1 ended. Three epochs provide this two-epoch buffer.

**crossbeam-epoch API types:**

| Type | Analogy | Purpose |
|------|---------|---------|
| `Atomic<T>` | `AtomicPtr<T>` | Atomic pointer stored in the data structure. Supports `load`, `store`, `compare_exchange`. |
| `Owned<T>` | `Box<T>` | Uniquely-owned heap allocation, not yet shared. Created with `Owned::new(value)`. |
| `Shared<'g, T>` | `&'g T` | A pointer loaded from an `Atomic`, guaranteed valid for the lifetime of the `Guard`. Cannot outlive the guard. |
| `Guard` | Scope/RAII token | Returned by `pin()`. Keeps the thread pinned. Provides `defer_destroy()` for deferred cleanup. |

**Typical lock-free operation pattern:**
```
let guard = epoch::pin();                          // pin thread
let current = atomic.load(Ordering::Acquire, &guard); // load → Shared
// ... compute new value ...
let new = Owned::new(new_value);                   // allocate new node
match atomic.compare_exchange(current, new, ..., &guard) {
    Ok(old) => unsafe { guard.defer_destroy(old); } // defer free of old node
    Err(_) => { /* retry */ }
}
drop(guard);                                        // unpin thread
```

### Memory Orderings in Lock-Free Code

Atomic operations require specifying **memory ordering** — how reads and writes are visible across threads:

| Ordering | Guarantee | Typical Use |
|----------|-----------|-------------|
| `Relaxed` | No ordering guarantees, only atomicity | Counters, statistics |
| `Acquire` | All subsequent reads/writes happen AFTER this load | Loading a pointer before traversing |
| `Release` | All prior reads/writes happen BEFORE this store | Publishing a new node |
| `AcqRel` | Acquire on load + Release on store | Read-modify-write (CAS) |
| `SeqCst` | Total global ordering of all SeqCst operations | When you need a single total order (rare, expensive) |

For most lock-free data structures: use `Acquire` on loads, `Release` on stores, and `AcqRel` or `(Release, Relaxed)` on `compare_exchange`.

### False Sharing and CachePadded

Modern CPUs transfer memory in **cache lines** (typically 64 bytes). When two threads modify different variables that happen to reside on the same cache line, the CPU's coherence protocol bounces the cache line back and forth between cores — this is **false sharing**. It doesn't cause correctness issues but destroys performance.

`CachePadded<T>` from crossbeam-utils pads `T` to 64 bytes, ensuring each value occupies its own cache line. Critical for concurrent counters, atomic flags, and any per-thread state that is written frequently.

### Backoff Strategy

When a CAS operation fails (contention), naive spinning wastes CPU cycles and increases cache-line traffic. `Backoff` from crossbeam-utils implements **exponential backoff**: the first few spins use `spin_loop()` hints (telling the CPU "I'm waiting"), and after a threshold, it yields to the OS scheduler with `thread::yield_now()`. This reduces contention and improves overall throughput.

### crossbeam Ecosystem Overview

| Crate | Purpose |
|-------|---------|
| **crossbeam-epoch** | Epoch-based GC for building your own lock-free structures |
| **crossbeam-queue** | Ready-made lock-free MPMC queues: `SegQueue` (unbounded, segmented), `ArrayQueue` (bounded, pre-allocated) |
| **crossbeam-channel** | Feature-rich MPMC channels: bounded, unbounded, `select!` macro for multiplexing, zero-capacity rendezvous channels |
| **crossbeam-skiplist** | Lock-free `SkipMap<K,V>` and `SkipSet<K>` — concurrent sorted containers using epoch-based GC internally |
| **crossbeam-utils** | `CachePadded`, `Backoff`, `WaitGroup`, scoped threads (`scope()`), `AtomicCell` |

### When Lock-Free Beats Locks (and Vice Versa)

**Lock-free wins when:**
- High contention with many threads competing for the same structure
- Real-time or low-latency requirements (no worst-case mutex stalls)
- Short critical sections (the CAS retry cost is low)
- Reader-heavy workloads (no lock acquisition overhead for reads)

**Locks win when:**
- Low contention (mutex fast-path is just one atomic + branch)
- Complex multi-step operations (hard to express as a single CAS)
- Simpler code and easier reasoning about correctness
- Operations that take a long time (disk I/O, network) — CAS retry would waste CPU

## Description

Build a series of concurrent programs exploring crossbeam's lock-free primitives, from channel-based communication through lock-free queues to implementing a custom Treiber stack using epoch-based reclamation. The capstone exercise benchmarks lock-free structures against `Mutex`-based equivalents under varying contention levels to develop intuition for when each approach wins.

### What you'll learn

1. **crossbeam-channel** — Bounded/unbounded MPMC channels and the `select!` macro
2. **crossbeam-utils** — CachePadded to avoid false sharing, Backoff for spin loops, scoped threads
3. **crossbeam-queue** — SegQueue (unbounded) and ArrayQueue (bounded) lock-free queues
4. **crossbeam-epoch** — Pinning, Guard, Atomic, Owned, Shared, deferred destruction
5. **Treiber stack** — Implement a classic lock-free stack from scratch using epoch-based GC
6. **crossbeam-skiplist** — Concurrent sorted map for ordered concurrent access
7. **Benchmarking** — Empirical comparison of lock-free vs lock-based under contention

## Instructions

### Exercise 1: crossbeam-channel & select! (~15 min)

Open `src/ex1_channels.rs`. This exercise teaches crossbeam's channel types and the `select!` macro for multiplexing.

crossbeam-channel is a significant upgrade over `std::sync::mpsc`: it supports multiple consumers (MPMC), bounded channels with backpressure, zero-capacity rendezvous channels for synchronous handoff, and a `select!` macro for waiting on multiple channels simultaneously — similar to Go's `select` statement.

1. **TODO(human): `fan_out_fan_in()`** — Create a bounded channel, spawn N producer threads that send work items, and M consumer threads that process them. Use scoped threads (`crossbeam::scope`) so you can borrow local variables. This teaches the fundamental producer-consumer pattern with backpressure.

2. **TODO(human): `select_timeout()`** — Use the `select!` macro to multiplex between two channels with a timeout. This teaches non-blocking channel operations and the select pattern for handling multiple event sources.

### Exercise 2: CachePadded & Backoff (~15 min)

Open `src/ex2_utils.rs`. This exercise demonstrates the performance impact of false sharing and how to mitigate it.

False sharing is one of the most common hidden performance killers in concurrent code. Two threads incrementing adjacent atomic counters can be 10-50x slower than expected because the CPU's cache coherence protocol bounces the cache line between cores on every write. CachePadded eliminates this by padding each value to a full cache line (64 bytes).

1. **TODO(human): `benchmark_false_sharing()`** — Create a struct with two adjacent `AtomicU64` counters and have two threads increment them in a tight loop. Then repeat with `CachePadded<AtomicU64>`. Compare the elapsed times. This makes false sharing tangible — you'll see a dramatic difference.

2. **TODO(human): `spin_with_backoff()`** — Implement a spin-wait loop using `Backoff` that waits for an `AtomicBool` flag to be set by another thread. Compare naive spinning (`while !flag.load(Relaxed) {}`) vs `Backoff`-assisted spinning. This teaches how exponential backoff reduces contention on the cache line.

### Exercise 3: Lock-Free Queues (~15 min)

Open `src/ex3_queues.rs`. This exercise compares crossbeam's two lock-free queue implementations.

`SegQueue` is an unbounded queue backed by linked segments — it grows dynamically and never blocks on push. `ArrayQueue` is a bounded queue backed by a pre-allocated array — push fails if full, but it has lower overhead per operation due to no allocation. Both are MPMC (multiple producers, multiple consumers) and lock-free.

1. **TODO(human): `segqueue_mpmc()`** — Use `SegQueue` with multiple producer and consumer threads. Producers push numbered items; consumers pop and collect them. Verify that every item is consumed exactly once (no lost or duplicated items). This teaches the basic MPMC queue pattern and demonstrates SegQueue's unbounded nature.

2. **TODO(human): `arrayqueue_bounded()`** — Use `ArrayQueue` with a small capacity. Producers must handle the `Err` case when the queue is full (retry with backoff). This teaches bounded queue backpressure and the difference between blocking and non-blocking queue APIs.

### Exercise 4: Epoch-Based Reclamation Fundamentals (~20 min)

Open `src/ex4_epoch.rs`. This is the core exercise — understanding how crossbeam-epoch enables safe lock-free memory management.

The epoch system solves a fundamental problem: in a lock-free data structure, when you remove a node, you cannot free it immediately because other threads may still hold pointers to it. crossbeam-epoch's `pin()` / `Guard` / `defer_destroy()` API provides a safe, efficient solution. Understanding this exercise is essential before implementing the Treiber stack.

1. **TODO(human): `atomic_swap_with_epoch()`** — Create an `Atomic<String>` pointer. Pin the epoch, load the current value as `Shared`, swap in a new `Owned` value, and defer destruction of the old value. Do this from multiple threads to see the epoch system in action. This teaches the core pin → load → CAS → defer_destroy lifecycle.

2. **TODO(human): `concurrent_atomic_counter()`** — Build a simple lock-free counter using `Atomic<u64>` with `compare_exchange` in a retry loop. Multiple threads increment the counter concurrently. Verify the final count is correct. This teaches the CAS retry loop — the fundamental lock-free programming pattern.

### Exercise 5: Treiber Stack (~25 min)

Open `src/ex5_treiber_stack.rs`. This is the main implementation exercise — building a classic lock-free data structure from scratch using crossbeam-epoch.

The [Treiber stack](https://en.wikipedia.org/wiki/Treiber_stack) (R. Kent Treiber, 1986) is the simplest non-trivial lock-free data structure. It is a singly-linked list where push and pop both operate on the head pointer using CAS. Push allocates a new node pointing to the current head and CAS-swaps it in. Pop reads the current head, follows its next pointer, and CAS-swaps the next node to become the new head. If CAS fails (another thread modified the head), retry.

Without epoch-based reclamation, a popped node cannot be safely freed — another thread's `pop()` might be reading the same node's `next` pointer at that very moment. crossbeam-epoch's `Guard` ensures that deferred destruction only happens when no thread can possibly hold a reference.

1. **TODO(human): `TreiberStack::push()`** — Allocate a new node with `Owned::new()`, load the current head, set the new node's next pointer to the current head, and CAS the head to point to the new node. Retry on CAS failure.

2. **TODO(human): `TreiberStack::pop()`** — Pin the epoch, load the current head, read the head's value and next pointer, CAS the head to the next node, defer destruction of the old head, and return the value. Handle the empty stack case (head is null).

3. **TODO(human): `stress_test_treiber()`** — Spawn many threads that push and pop concurrently. Verify that all pushed values are eventually popped and no values are lost or duplicated.

### Exercise 6: Concurrent SkipMap (~15 min)

Open `src/ex6_skipmap.rs`. This exercise uses crossbeam-skiplist's `SkipMap` — a lock-free sorted map.

A skip list is a probabilistic data structure that provides O(log n) search, insert, and delete — like a balanced BST but easier to make concurrent. crossbeam's `SkipMap` uses epoch-based reclamation internally, so inserts and removals are lock-free. Unlike `BTreeMap` wrapped in a `RwLock`, `SkipMap` allows concurrent reads AND writes without any locking.

1. **TODO(human): `concurrent_ordered_insert()`** — Spawn multiple threads that insert key-value pairs into a shared `SkipMap`. After all threads finish, iterate the map and verify the entries are sorted. This teaches how SkipMap supports concurrent mutation via `&self` (not `&mut self`).

2. **TODO(human): `range_queries_under_contention()`** — While writer threads continuously insert and remove entries, reader threads perform range queries (`map.range(start..end)`). Verify that readers always see a consistent snapshot (no torn reads). This teaches the entry-based API and how epoch-based GC makes concurrent range iteration safe.

### Exercise 7: Benchmarking Lock-Free vs Mutex (~15 min)

Open `src/ex7_benchmark.rs`. The capstone exercise: empirical evidence for when lock-free beats locks.

Theory says lock-free structures win under high contention and short critical sections. This exercise makes that claim testable. You will implement the same logical operations with both a `Mutex<VecDeque>` and a `SegQueue`, then measure throughput at different thread counts (1, 2, 4, 8, 16). The results often surprise — at low contention, Mutex can be faster due to lower constant overhead.

1. **TODO(human): `bench_mutex_queue()`** — Implement a producer-consumer benchmark using `Arc<Mutex<VecDeque<u64>>>`. Each thread does N push+pop cycles. Measure total elapsed time.

2. **TODO(human): `bench_lockfree_queue()`** — Same benchmark but using `Arc<SegQueue<u64>>`. Measure total elapsed time.

3. **TODO(human): `run_comparison()`** — Run both benchmarks at thread counts 1, 2, 4, 8 and print a comparison table. Analyze where the crossover point is — at what thread count does lock-free start winning?

## Motivation

- **HFT & low-latency systems**: Lock-free structures are essential in trading systems where microsecond-level tail latency matters and mutex stalls are unacceptable
- **High-throughput concurrent systems**: Message queues, work-stealing schedulers, and connection pools all benefit from lock-free designs
- **Complements Practice 060a (Unsafe Rust)**: Lock-free programming is one of the primary domains where unsafe Rust is necessary and justified
- **crossbeam is foundational**: Used internally by tokio, rayon, and many Rust ecosystem crates — understanding it deepens understanding of the entire async/parallel Rust stack
- **Bridges C++ and Rust concurrency**: Maps directly to concepts from Practice 022 (moodycamel, lock-free C++) but with Rust's ownership guarantees preventing data races at compile time

## Commands

### Build & Run

| Command | Description |
|---------|-------------|
| `cargo build` | Compile all exercises (verifies TODO stubs compile) |
| `cargo run` | Run the exercise runner — executes all exercises sequentially |
| `cargo run -- 1` | Run only Exercise 1 (channels) |
| `cargo run -- 2` | Run only Exercise 2 (CachePadded & Backoff) |
| `cargo run -- 3` | Run only Exercise 3 (lock-free queues) |
| `cargo run -- 4` | Run only Exercise 4 (epoch-based reclamation) |
| `cargo run -- 5` | Run only Exercise 5 (Treiber stack) |
| `cargo run -- 6` | Run only Exercise 6 (SkipMap) |
| `cargo run -- 7` | Run only Exercise 7 (benchmarking) |

### Development

| Command | Description |
|---------|-------------|
| `cargo check` | Fast type-check without codegen (use while implementing) |
| `cargo test` | Run unit tests (each exercise has verification tests) |
| `cargo clippy` | Run linter for idiomatic Rust suggestions |
| `cargo build --release` | Optimized build (use for Exercise 7 benchmarking — debug builds distort timings) |
| `cargo run --release -- 7` | Run benchmarks with optimizations (IMPORTANT: debug builds give misleading results) |

## References

- [crossbeam GitHub Repository](https://github.com/crossbeam-rs/crossbeam)
- [crossbeam-epoch docs.rs](https://docs.rs/crossbeam-epoch/latest/crossbeam_epoch/)
- [crossbeam-channel docs.rs](https://docs.rs/crossbeam-channel/latest/crossbeam_channel/)
- [crossbeam-queue docs.rs](https://docs.rs/crossbeam-queue/latest/crossbeam_queue/)
- [crossbeam-skiplist docs.rs](https://docs.rs/crossbeam-skiplist/latest/crossbeam_skiplist/)
- [crossbeam-utils docs.rs](https://docs.rs/crossbeam-utils/latest/crossbeam_utils/)
- [Aaron Turon — Lock-freedom without garbage collection](https://aturon.github.io/blog/2015/08/27/epoch/) — Original blog post explaining crossbeam's epoch design
- [Code and Bitters — Learning Rust: crossbeam::epoch](https://codeandbitters.com/learning-rust-crossbeam-epoch/) — Practical tutorial
- [Treiber Stack (Wikipedia)](https://en.wikipedia.org/wiki/Treiber_stack) — The classic lock-free stack algorithm
- [ABA Problem (Wikipedia)](https://en.wikipedia.org/wiki/ABA_problem) — Detailed explanation of the ABA problem

## Notes

*(To be filled during practice.)*
