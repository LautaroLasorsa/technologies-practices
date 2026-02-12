// =============================================================================
// Phase 1: Cache-Friendly Data Structures
// =============================================================================
//
// HFT CONTEXT:
//   In high-frequency trading, you process millions of market data updates per
//   second. At 10M messages/sec, you have 100ns per message. A single L3 cache
//   miss costs ~40-60ns -- that's HALF your budget gone on one memory access.
//
//   The #1 optimization in HFT is not algorithmic -- it's making sure your data
//   fits in cache and is accessed sequentially. This phase teaches you how.
//
// KEY NUMBERS:
//   Cache line    = 64 bytes (the unit of transfer between memory and cache)
//   L1 cache      = 32-48 KB per core (~1ns access)
//   L2 cache      = 256-512 KB per core (~4ns access)
//   L3 cache      = 8-32 MB shared (~12ns access)
//   DRAM          = GBs (~60-100ns access)
//
//   Rule of thumb: sequential access = ~4 GB/s, random access = ~400 MB/s (10x!)
//
// EXERCISES:
//   1. Cache-line-aligned OrderUpdate struct
//   2. Hot/cold data splitting
//   3. Sequential vs random access benchmark
// =============================================================================

#include "hft_common.h"
#include <array>
#include <random>
#include <vector>

// =============================================================================
// Exercise 1: Cache-Line-Aligned OrderUpdate
// =============================================================================
//
// GOAL: Design a struct that fits exactly in one 64-byte cache line.
//
// WHY: When the CPU fetches one field of your struct, it loads the entire
//      cache line (64 bytes). If your struct is 65 bytes, every access loads
//      TWO cache lines. If it's 48 bytes with 16 bytes of padding, you're
//      wasting 25% of your precious L1 cache.
//
// RULES:
//   - alignas(64) forces the struct to start on a cache line boundary
//   - Pack fields to use exactly 64 bytes (or a multiple)
//   - Use static_assert to prove it at compile time
//
// HINT: Remember field ordering matters! In C++, struct fields are laid out
//       in declaration order. A sequence like: uint8_t, uint64_t, uint8_t
//       wastes bytes on padding. Group fields by alignment requirement.
//
// REFERENCE: https://en.cppreference.com/w/cpp/language/alignas
// RUST EQUIVALENT: #[repr(C, align(64))] struct OrderUpdate { ... }
// =============================================================================

// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches cache-line alignment for eliminating wasted memory transfers.
// Every memory access loads a full 64-byte cache line—structs that don't fit exactly
// waste bandwidth and cache capacity, directly reducing hot-path throughput.
//
// TODO(human): Design OrderUpdate to fit exactly in 64 bytes.
//
// Fields you need to represent a market order update:
//   - order_id    : uint64_t  (8 bytes)  -- unique order identifier
//   - price       : int64_t   (8 bytes)  -- price in fixed-point (cents * 100)
//   - quantity    : uint32_t  (4 bytes)  -- number of shares
//   - filled_qty  : uint32_t  (4 bytes)  -- how many already filled
//   - instrument  : uint32_t  (4 bytes)  -- instrument/symbol ID
//   - flags       : uint16_t  (2 bytes)  -- bit flags (buy/sell, IOC, etc.)
//   - side        : uint8_t   (1 byte)   -- 0=buy, 1=sell
//   - status      : uint8_t   (1 byte)   -- order status enum
//   - timestamp   : uint64_t  (8 bytes)  -- TSC timestamp
//   - sequence_no : uint64_t  (8 bytes)  -- exchange sequence number
//   - padding     : fill remaining bytes with reserved[] array
//
// Total so far: 8+8+4+4+4+2+1+1+8+8 = 48 bytes. You have 16 bytes for padding.
//
// STEP 1: Declare the struct with alignas(64)
// STEP 2: Order fields largest-first to minimize compiler padding
// STEP 3: Add a char reserved[16] at the end for explicit padding
// STEP 4: Add static_assert(sizeof(OrderUpdate) == 64)

struct alignas(CACHE_LINE_SIZE) OrderUpdate {
    // --- TODO(human): declare the fields here ---
    // Placeholder so it compiles:
    char placeholder[CACHE_LINE_SIZE]{};
};

HFT_ASSERT_CACHE_LINES(OrderUpdate, 1);
HFT_ASSERT_ALIGNED(OrderUpdate);

void exercise1_aligned_struct() {
    std::cout << "=== Exercise 1: Cache-Line-Aligned OrderUpdate ===\n\n";

    std::cout << "sizeof(OrderUpdate) = " << sizeof(OrderUpdate) << " bytes\n";
    std::cout << "alignof(OrderUpdate) = " << alignof(OrderUpdate) << " bytes\n";

    // Allocate an array and verify alignment
    constexpr size_t N = 8;
    std::array<OrderUpdate, N> orders{};

    for (size_t i = 0; i < N; ++i) {
        auto addr = reinterpret_cast<uintptr_t>(&orders[i]);
        bool aligned = (addr % CACHE_LINE_SIZE == 0);
        std::cout << "  orders[" << i << "] @ 0x" << std::hex << addr
                  << std::dec << " -- " << (aligned ? "ALIGNED" : "MISALIGNED") << "\n";
    }

    // TODO(human): After implementing OrderUpdate with real fields,
    // create a few orders and print their field values to verify layout.
    // Example:
    //   orders[0].order_id = 12345;
    //   orders[0].price = 15023;  // $150.23 in fixed-point
    //   orders[0].side = 0;       // buy
    //   std::cout << "Order " << orders[0].order_id << ": "
    //             << (orders[0].side == 0 ? "BUY" : "SELL")
    //             << " @ $" << orders[0].price / 100.0 << "\n";

    std::cout << "\n[Placeholder] OrderUpdate struct compiles and is aligned.\n";
    std::cout << "Replace placeholder with real fields and re-verify.\n";
}

// =============================================================================
// Exercise 2: Hot/Cold Data Splitting
// =============================================================================
//
// GOAL: Separate frequently-accessed ("hot") fields from rarely-accessed
//       ("cold") fields into separate structs.
//
// WHY: If your Order struct has 20 fields but your hot loop only reads
//      price, quantity, and side, you're loading 20 fields into cache but
//      only using 3. That wastes cache capacity by 6-7x.
//
//      Hot/cold splitting means: hot fields packed together in one struct,
//      cold fields in another. Your hot loop only touches the hot struct,
//      so you fit 3-4x more orders in L1 cache.
//
// PATTERN:
//   struct OrderHot {   // fits in 1 cache line, accessed every tick
//       price, quantity, side, instrument_id, flags
//   };
//   struct OrderCold {  // separate cache line, accessed on fill/cancel
//       order_id, client_id, account, create_time, text, ...
//   };
//   // Link them by index: hot[i] corresponds to cold[i]
//
// RUST EQUIVALENT: This is essentially SoA (Struct of Arrays) -- Rust's ECS
//   frameworks (bevy_ecs, specs) use this exact pattern for game entities.
// =============================================================================

// ── Exercise Context ──────────────────────────────────────────────────
// This exercise teaches hot/cold data splitting to maximize cache density. Separating
// frequently-accessed fields from rarely-used ones lets you fit 3-4× more "hot" entries
// in L1 cache, dramatically improving scan throughput in tight loops.
//
// TODO(human): Define OrderHot and OrderCold structs.
//
// OrderHot (accessed every tick in the matching engine):
//   - price        : int64_t  (8 bytes)
//   - quantity      : uint32_t (4 bytes)
//   - instrument_id : uint32_t (4 bytes)
//   - side          : uint8_t  (1 byte)
//   - flags         : uint8_t  (1 byte)
//   - _pad          : fill to 32 bytes (or 64 if you prefer 1 full cache line)
//   Total: ~18 bytes core data. Pad to 32 so two orders fit per cache line,
//          or pad to 64 for isolated cache line per order.
//
// OrderCold (accessed only on fill, cancel, or admin queries):
//   - order_id      : uint64_t
//   - client_id     : uint32_t
//   - account_id    : uint32_t
//   - create_time   : uint64_t
//   - modify_time   : uint64_t
//   - text          : char[32]  (free-form order text)
//
// STEP 1: Define both structs with alignas
// STEP 2: static_assert their sizes
// STEP 3: Create parallel arrays: OrderHot hot[N], OrderCold cold[N]

struct OrderHot {
    // TODO(human): hot-path fields here
    char placeholder[32]{};
};

struct OrderCold {
    // TODO(human): cold-path fields here
    char placeholder[64]{};
};

void exercise2_hot_cold_split() {
    std::cout << "\n=== Exercise 2: Hot/Cold Data Splitting ===\n\n";

    std::cout << "sizeof(OrderHot)  = " << sizeof(OrderHot)  << " bytes\n";
    std::cout << "sizeof(OrderCold) = " << sizeof(OrderCold) << " bytes\n";

    // Demonstrate: iterate over hot data only
    constexpr size_t N = 1'000'000;
    std::vector<OrderHot> hot_data(N);
    std::vector<OrderCold> cold_data(N);

    // TODO(human): Fill hot_data with sample prices/quantities.
    // Then benchmark iterating over hot_data only vs iterating over
    // a combined struct that includes all fields.
    //
    // Expected result: hot-only iteration is 2-4x faster because
    // you fit 2-4x more entries per cache line.

    std::cout << "\n[Placeholder] Hot/cold structs created (" << N << " entries).\n";
    std::cout << "Fill with data and benchmark hot-only vs combined iteration.\n";
}

// =============================================================================
// Exercise 3: Sequential vs Random Access Benchmark
// =============================================================================
//
// GOAL: Demonstrate the massive cost of cache misses by comparing sequential
//       array traversal (cache-friendly) vs random access (cache-hostile).
//
// EXPECTED RESULTS:
//   Sequential: ~0.3-0.5 ns per access (always hitting L1/L2, hardware prefetcher active)
//   Random:     ~50-100 ns per access (constant cache misses to DRAM)
//   Ratio:      100-300x slower for random access!
//
// WHY THIS MATTERS IN HFT:
//   Your order book is an array of price levels. If you access them sequentially
//   (best bid, next bid, next bid...), you get L1 hits. If you jump to random
//   price levels, every access is a cache miss. This is why HFT order books
//   use arrays, not trees -- even though trees have better big-O for insertion.
//
// CONCEPTS:
//   - Hardware prefetcher: CPU detects sequential access pattern and loads
//     the next cache line before you ask for it (zero latency!)
//   - Cache miss: data not in any cache level, must fetch from DRAM (~100 cycles)
//   - TLB miss: virtual address translation not cached (~10-20 cycles on top)
// =============================================================================

void exercise3_sequential_vs_random() {
    std::cout << "\n=== Exercise 3: Sequential vs Random Access ===\n\n";

    // ── Exercise Context ──────────────────────────────────────────────────
    // This benchmark demonstrates the 100-300× cost of cache misses by comparing sequential
    // (cache-friendly) vs random (cache-hostile) array traversal. Sequential access enables
    // hardware prefetching; random access causes every load to stall on DRAM latency.
    //
    // TODO(human): Implement this benchmark.
    //
    // STEPS:
    //   1. Allocate a large array of uint64_t (e.g., 16M entries = 128 MB)
    //      This must be larger than L3 cache to see DRAM latency.
    //
    //   2. Create a sequential index array:  idx[i] = i
    //      Create a random index array:      idx[i] = random permutation
    //
    //   3. Benchmark: sum += data[sequential_idx[i]]  (for all i)
    //      vs:        sum += data[random_idx[i]]       (for all i)
    //
    //   4. Measure with ScopedTimer or rdtsc().
    //      Print ns/access for each.
    //
    // HINTS:
    //   - Use std::shuffle with std::mt19937 for the random permutation
    //   - Use do_not_optimize(sum) to prevent the compiler from removing the loop
    //   - Use a large enough array (>= 64 MB) to defeat L3 cache
    //   - Run multiple iterations and take the median
    //
    // EXPECTED OUTPUT:
    //   Sequential: ~0.3-0.5 ns/access
    //   Random:     ~50-100 ns/access
    //   Ratio:      ~100-300x
    //
    // RUST EQUIVALENT: Same benchmark with Vec<u64>, identical results --
    //   cache behavior is a hardware property, not a language property.

    constexpr size_t N = 16 * 1024 * 1024;  // 16M entries = 128 MB

    std::cout << "Array size: " << N << " entries ("
              << (N * sizeof(uint64_t)) / (1024 * 1024) << " MB)\n";
    std::cout << "\n[Placeholder] Implement sequential and random access benchmarks.\n";
    std::cout << "Expected: random access ~100-300x slower than sequential.\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 1: Cache-Friendly Data Structures\n";
    std::cout << "========================================\n";

    exercise1_aligned_struct();
    exercise2_hot_cold_split();
    exercise3_sequential_vs_random();

    std::cout << "\n========================================\n";
    std::cout << "Phase 1 complete.\n";
    std::cout << "========================================\n";

    return 0;
}
