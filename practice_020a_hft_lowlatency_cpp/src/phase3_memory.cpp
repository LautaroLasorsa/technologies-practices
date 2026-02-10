// =============================================================================
// Phase 3: Memory Pools & Allocation-Free Hot Path
// =============================================================================
//
// HFT CONTEXT:
//   In a trading system, the hot path (market data -> signal -> order) must
//   NEVER call malloc/new/delete. Here's why:
//
//   1. LATENCY: malloc takes 50-100ns on average. But it can spike to
//      MICROSECONDS when it needs to request memory from the OS (mmap/sbrk).
//      Your tick-to-trade budget is ~500ns. One bad malloc = blown budget.
//
//   2. JITTER: malloc's latency is unpredictable. P50 might be 30ns, but
//      P99.9 is 5000ns. HFT cares about WORST CASE, not average case.
//      Memory pools give you O(1) constant-time allocation with zero jitter.
//
//   3. FRAGMENTATION: After hours of trading with millions of allocations,
//      the heap fragments. malloc starts searching free lists, TLB misses
//      increase, performance degrades over the trading day. Pools don't fragment.
//
//   The solution: pre-allocate ALL memory at startup. On the hot path, use:
//   - Object pools (fixed-size, typed allocations)
//   - Arena/bump allocators (variable-size, short-lived allocations)
//   - Stack-allocated buffers (when the max size is known at compile time)
//
// RUST EQUIVALENT:
//   - Object pool: typed-arena, slab crate
//   - Arena: bumpalo crate
//   - Rust's ownership model makes this easier -- the borrow checker prevents
//     use-after-free bugs that are common with manual pool management in C++.
// =============================================================================

#include "hft_common.h"
#include <array>
#include <cassert>
#include <memory>
#include <new>

// =============================================================================
// Exercise 1: Fixed-Size Object Pool
// =============================================================================
//
// GOAL: Implement a pool that pre-allocates N objects and hands them out
//       in O(1) time using a free list.
//
// DESIGN:
//   - Storage: aligned array of N slots, each sizeof(T) bytes
//   - Free list: singly-linked list through the unused slots themselves
//     (when a slot is free, its first 8 bytes store a pointer to the next free slot)
//   - allocate(): pop from free list head -> O(1)
//   - deallocate(): push to free list head -> O(1)
//   - No syscalls, no fragmentation, deterministic latency
//
// MEMORY LAYOUT (4 slots, 2 allocated):
//   Slot 0: [====DATA====]  (allocated, in use)
//   Slot 1: [->Slot 3    ]  (free, stores pointer to next free slot)
//   Slot 2: [====DATA====]  (allocated, in use)
//   Slot 3: [->nullptr   ]  (free, end of free list)
//   free_head_ -> Slot 1
//
// WARNING: This pool is NOT thread-safe. In HFT, each thread owns its own pool.
//          Thread-safe pools add ~10-20ns overhead per operation (atomic CAS).
//          Not worth it when you can just partition pools per thread.
// =============================================================================

template <typename T, size_t Capacity>
class ObjectPool {
public:
    ObjectPool() {
        // TODO(human): Initialize the free list.
        //
        // STEPS:
        //   1. For each slot i from 0 to Capacity-2:
        //      Interpret the slot's memory as a pointer (FreeNode*)
        //      and set it to point to slot i+1.
        //   2. The last slot's next pointer = nullptr (end of list).
        //   3. free_head_ = pointer to slot 0.
        //
        // HINT: Use reinterpret_cast<FreeNode*>(slot_ptr(i)) to treat
        //       raw memory as a linked list node.
        //
        // WHY THIS WORKS: Free slots aren't holding a T object, so we can
        //   reuse their memory to store a pointer. This is "intrusive" free list --
        //   the list metadata lives inside the unused slots themselves.
        //   Minimum requirement: sizeof(T) >= sizeof(void*) (8 bytes on x64).
        static_assert(sizeof(T) >= sizeof(void*),
            "Object size must be >= pointer size for intrusive free list");

        free_head_ = nullptr;  // placeholder
        allocated_count_ = 0;
    }

    // -------------------------------------------------------------------------
    // allocate: Return a pointer to an uninitialized slot, or nullptr if full.
    //
    // TODO(human): Implement.
    //   1. If free_head_ == nullptr, return nullptr (pool exhausted)
    //   2. Save free_head_ as the result
    //   3. Advance free_head_ to the next free node
    //   4. ++allocated_count_
    //   5. Return the pointer (as T*)
    //
    // NOTE: This returns raw memory. The caller must construct the object
    //       using placement new: new (ptr) T(args...).
    //       Or use the typed allocate(args...) helper below.
    // -------------------------------------------------------------------------
    T* allocate() {
        // --- TODO(human) ---
        return nullptr;  // placeholder
    }

    // Typed allocation with forwarded constructor arguments
    template <typename... Args>
    T* construct(Args&&... args) {
        T* ptr = allocate();
        if (ptr) {
            new (ptr) T(std::forward<Args>(args)...);
        }
        return ptr;
    }

    // -------------------------------------------------------------------------
    // deallocate: Return a slot to the pool.
    //
    // TODO(human): Implement.
    //   1. Call ptr->~T() to destroy the object (invoke destructor)
    //   2. Reinterpret ptr as FreeNode*
    //   3. Set its next pointer to current free_head_
    //   4. Update free_head_ to point to this slot
    //   5. --allocated_count_
    //
    // WARNING: Calling deallocate on a pointer not from this pool = undefined behavior.
    //          In production, you'd add a bounds check: assert(ptr >= storage_ && ptr < storage_ + Capacity)
    // -------------------------------------------------------------------------
    void deallocate(T* ptr) {
        // --- TODO(human) ---
        (void)ptr;  // placeholder
    }

    size_t allocated() const { return allocated_count_; }
    size_t available() const { return Capacity - allocated_count_; }
    static constexpr size_t capacity() { return Capacity; }

private:
    // Intrusive free list node -- overlays the T storage when slot is free
    struct FreeNode {
        FreeNode* next;
    };

    // Helper: get pointer to the i-th slot
    void* slot_ptr(size_t i) {
        return &storage_[i * sizeof(T)];
    }

    // Raw storage for Capacity objects of type T, cache-line aligned
    alignas(alignof(T) > CACHE_LINE_SIZE ? alignof(T) : CACHE_LINE_SIZE)
        char storage_[Capacity * sizeof(T)];

    FreeNode* free_head_;
    size_t allocated_count_;
};

// =============================================================================
// Exercise 2: Arena (Bump) Allocator
// =============================================================================
//
// GOAL: Implement the simplest possible allocator -- a bump pointer that
//       allocates by incrementing a pointer, and frees everything at once.
//
// DESIGN:
//   - Pre-allocate a large buffer (e.g., 1 MB)
//   - allocate(n): advance pointer by n bytes, return old pointer -> O(1)
//   - No individual deallocation -- only reset() frees everything
//   - Perfect for per-message temporary allocations in HFT:
//     process message -> allocate temp data -> compute -> reset arena
//
// WHY IT'S FAST:
//   - One pointer increment per allocation (1 cycle)
//   - No free list, no metadata, no fragmentation
//   - Everything is contiguous -> excellent cache locality
//   - reset() is O(1) -- just move the pointer back to the start
//
// LIMITATION:
//   - Can't free individual allocations (only bulk reset)
//   - Must know maximum usage upfront
//   - Objects with non-trivial destructors need manual destruction before reset
//
// RUST EQUIVALENT: bumpalo::Bump
//   let arena = Bump::new();
//   let x = arena.alloc(42);        // bump allocate
//   let s = arena.alloc_str("hi");  // bump allocate
//   drop(arena);                    // frees everything
// =============================================================================

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity)
        : capacity_(capacity), offset_(0)
    {
        // Allocate backing buffer (this is the ONE heap allocation, done at startup)
        buffer_ = std::make_unique<char[]>(capacity);
    }

    // -------------------------------------------------------------------------
    // allocate: Bump the pointer by `size` bytes, aligned to `alignment`.
    //
    // TODO(human): Implement.
    //   1. Align offset_ up to the requested alignment:
    //      aligned_offset = (offset_ + alignment - 1) & ~(alignment - 1)
    //   2. Check if aligned_offset + size > capacity_ -> return nullptr (OOM)
    //   3. Save aligned_offset as the result pointer
    //   4. Update offset_ = aligned_offset + size
    //   5. Return buffer_.get() + result_offset
    //
    // WHY ALIGNMENT MATTERS:
    //   Misaligned access on x86 works but is slower (~2x for crossing cache line).
    //   Misaligned atomics are undefined behavior on some architectures.
    //   Always align to at least alignof(T) -- we default to 8 (natural alignment).
    // -------------------------------------------------------------------------
    void* allocate(size_t size, size_t alignment = 8) {
        // --- TODO(human) ---
        (void)size;
        (void)alignment;
        return nullptr;  // placeholder
    }

    // Typed allocation helper
    template <typename T, typename... Args>
    T* construct(Args&&... args) {
        void* mem = allocate(sizeof(T), alignof(T));
        if (!mem) return nullptr;
        return new (mem) T(std::forward<Args>(args)...);
    }

    // Reset the arena -- O(1), frees ALL allocations at once
    void reset() {
        offset_ = 0;
        // NOTE: This does NOT call destructors! If you stored objects with
        // non-trivial destructors, you must destroy them before reset.
    }

    size_t used() const { return offset_; }
    size_t remaining() const { return capacity_ - offset_; }
    size_t capacity() const { return capacity_; }

private:
    std::unique_ptr<char[]> buffer_;
    size_t capacity_;
    size_t offset_;
};

// =============================================================================
// Exercise 3: Benchmark -- Pool vs new/delete
// =============================================================================
//
// TODO(human): Benchmark allocation latency.
//
// TEST SETUP:
//   1. Define a small struct (e.g., 64-byte Order with a few fields)
//   2. Run 10M iterations of: allocate -> use -> deallocate
//   3. Measure per-iteration latency with rdtsc
//   4. Collect into a vector, compute LatencyStats
//
// THREE VARIANTS:
//   a) new Order() / delete order  (heap)
//   b) pool.allocate() / pool.deallocate()  (object pool)
//   c) arena.allocate() / arena.reset() per batch  (arena, reset every 1000)
//
// EXPECTED RESULTS:
//   new/delete:  P50 ~30-50ns, P99 ~100-500ns, Max ~1000-5000ns (jitter!)
//   Pool:        P50 ~5-10ns,  P99 ~10-15ns,   Max ~20-50ns (consistent)
//   Arena:       P50 ~2-5ns,   P99 ~5-10ns,    Max ~10-20ns (fastest)
//
// KEY INSIGHT: The *average* of new/delete might look OK. But look at P99 and Max.
//   That's where malloc hits slow paths (free list search, mmap, page faults).
//   In HFT, you're judged on P99, not P50. Pools win on P99 by 10-100x.
// =============================================================================

struct alignas(CACHE_LINE_SIZE) BenchOrder {
    uint64_t order_id{};
    int64_t price{};
    uint32_t quantity{};
    uint32_t instrument_id{};
    uint8_t side{};
    uint8_t status{};
    char padding[38]{};
};

HFT_ASSERT_CACHE_LINES(BenchOrder, 1);

void benchmark_allocators() {
    std::cout << "\n=== Exercise 3: Allocation Benchmark ===\n\n";

    constexpr size_t NUM_ITERS = 10'000'000;

    std::cout << "Struct size: " << sizeof(BenchOrder) << " bytes\n";
    std::cout << "Iterations: " << NUM_ITERS << "\n";

    // --- Benchmark new/delete ---
    {
        std::cout << "\n--- new/delete ---\n";
        // TODO(human): Measure per-iteration latency of new + delete.
        //
        // for (size_t i = 0; i < NUM_ITERS; ++i) {
        //     uint64_t start = rdtsc();
        //     auto* order = new BenchOrder();
        //     order->order_id = i;
        //     order->price = 15000 + (i % 100);
        //     do_not_optimize(*order);
        //     delete order;
        //     uint64_t end = rdtsc();
        //     latencies.push_back(tsc_to_ns(end - start, tsc_freq));
        // }
        std::cout << "[Placeholder] Benchmark new/delete.\n";
    }

    // --- Benchmark object pool ---
    {
        std::cout << "\n--- Object Pool ---\n";
        // TODO(human): Same benchmark but using ObjectPool<BenchOrder, N>.
        //
        // ObjectPool<BenchOrder, 1024> pool;  // small pool, allocate+dealloc in loop
        // for (size_t i = 0; i < NUM_ITERS; ++i) {
        //     uint64_t start = rdtsc();
        //     auto* order = pool.construct(/* ... */);
        //     order->order_id = i;
        //     do_not_optimize(*order);
        //     pool.deallocate(order);
        //     uint64_t end = rdtsc();
        //     ...
        // }
        std::cout << "[Placeholder] Benchmark object pool.\n";
    }

    // --- Benchmark arena ---
    {
        std::cout << "\n--- Arena Allocator ---\n";
        // TODO(human): Benchmark arena allocation with periodic reset.
        //
        // ArenaAllocator arena(1024 * 1024);  // 1 MB
        // constexpr size_t BATCH = 1000;
        // for (size_t i = 0; i < NUM_ITERS; ++i) {
        //     uint64_t start = rdtsc();
        //     auto* order = arena.construct<BenchOrder>();
        //     order->order_id = i;
        //     do_not_optimize(*order);
        //     uint64_t end = rdtsc();
        //     if ((i + 1) % BATCH == 0) arena.reset();
        //     ...
        // }
        std::cout << "[Placeholder] Benchmark arena allocator.\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 3: Memory Pools & Arena Allocator\n";
    std::cout << "========================================\n";

    // Quick sanity check for the pool
    {
        std::cout << "\n--- Object Pool sanity check ---\n";
        ObjectPool<BenchOrder, 4> pool;
        std::cout << "Capacity: " << pool.capacity()
                  << ", Available: " << pool.available() << "\n";

        auto* o1 = pool.allocate();
        auto* o2 = pool.allocate();
        std::cout << "After 2 allocations: available = " << pool.available() << "\n";
        std::cout << "o1 = " << (o1 ? "valid" : "null")
                  << ", o2 = " << (o2 ? "valid" : "null") << "\n";

        // NOTE: allocate() returns nullptr until you implement it.
        // After implementation, it should return valid pointers.

        if (o1) pool.deallocate(o1);
        if (o2) pool.deallocate(o2);
        std::cout << "After deallocation: available = " << pool.available() << "\n";
    }

    // Quick sanity check for the arena
    {
        std::cout << "\n--- Arena sanity check ---\n";
        ArenaAllocator arena(4096);
        std::cout << "Capacity: " << arena.capacity()
                  << ", Used: " << arena.used() << "\n";

        auto* p1 = arena.allocate(32);
        auto* p2 = arena.allocate(64);
        std::cout << "After allocations: used = " << arena.used()
                  << ", remaining = " << arena.remaining() << "\n";
        std::cout << "p1 = " << (p1 ? "valid" : "null")
                  << ", p2 = " << (p2 ? "valid" : "null") << "\n";

        arena.reset();
        std::cout << "After reset: used = " << arena.used() << "\n";

        // NOTE: allocate returns nullptr until you implement it.
    }

    benchmark_allocators();

    std::cout << "\n========================================\n";
    std::cout << "Phase 3 complete.\n";
    std::cout << "========================================\n";

    return 0;
}
