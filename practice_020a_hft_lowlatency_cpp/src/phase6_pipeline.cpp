// =============================================================================
// Phase 6: Putting It Together -- Mini Hot-Path Pipeline
// =============================================================================
//
// HFT CONTEXT:
//   A real HFT system has this pipeline:
//
//     NIC (network) -> Kernel bypass -> Market data parser -> Signal generator
//       -> Risk check -> Order builder -> Order sender -> NIC (network)
//
//   The "tick-to-trade" latency is the time from receiving a market data packet
//   to sending an order packet. Top firms achieve 200-800ns end-to-end.
//
//   In this exercise, you'll build a simplified version:
//
//     Market data (SPSC) -> Signal processor (pool) -> Order handler (CRTP)
//       -> Latency measurement (TSC) -> Histogram
//
//   This combines ALL concepts from Phases 1-5 into one working pipeline.
//
// ARCHITECTURE:
//
//   Thread 1 (Market Data Feed):
//     - Generates synthetic market data updates
//     - Pushes them onto SPSC queue
//     - Stamps each message with TSC timestamp
//
//   Thread 2 (Strategy / Hot Path):
//     - Pops messages from SPSC queue
//     - Allocates a signal object from the object pool (NOT new/delete)
//     - Processes the signal (simple moving average crossover)
//     - If signal fires, dispatches order via CRTP handler (NOT virtual)
//     - Records end-to-end latency in histogram
//     - Returns signal object to pool
//
//   Everything on Thread 2's hot path is:
//     [x] Lock-free (SPSC queue)
//     [x] Allocation-free (object pool)
//     [x] Virtual-free (CRTP dispatch)
//     [x] Timed with TSC (not steady_clock)
//
// RUST EQUIVALENT:
//   crossbeam SPSC channel + typed-arena + monomorphized trait impl + quanta TSC.
//   The Rust version would be nearly identical, with the borrow checker
//   preventing use-after-free on pool objects.
// =============================================================================

#include "hft_common.h"
#include <atomic>
#include <thread>
#include <optional>
#include <cstring>
#include <array>

// =============================================================================
// Message types for the pipeline
// =============================================================================

// Market data update (arrives from exchange)
struct alignas(CACHE_LINE_SIZE) MarketDataUpdate {
    uint64_t sequence_no;       // Exchange sequence number
    uint64_t tsc_timestamp;     // TSC at time of "arrival" (simulated)
    int64_t price;              // Price in fixed-point (cents * 100)
    uint32_t quantity;          // Volume
    uint32_t instrument_id;     // Which instrument
    uint8_t side;               // 0=bid, 1=ask
    uint8_t msg_type;           // 0=trade, 1=quote
    char padding[22]{};         // Pad to 64 bytes
};

HFT_ASSERT_CACHE_LINES(MarketDataUpdate, 1);

// Signal output from the strategy
struct Signal {
    int64_t target_price;
    uint32_t target_quantity;
    uint8_t direction;          // 0=buy, 1=sell
    bool should_trade;          // Does the signal fire?
};

// Order to send (output of the pipeline)
struct OutboundOrder {
    uint64_t order_id;
    int64_t price;
    uint32_t quantity;
    uint32_t instrument_id;
    uint8_t side;
};

// =============================================================================
// SPSC Queue (from Phase 2 -- copy your implementation here)
// =============================================================================
//
// TODO(human): Copy your working SPSCQueue from Phase 2, or implement it
// fresh here. The pipeline needs it for the market data feed.
//
// If you haven't completed Phase 2 yet, here's a minimal version you can
// use as a starting point (but you should go back and do Phase 2 properly).

template <typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr size_t MASK = Capacity - 1;

public:
    SPSCQueue() : head_(0), tail_(0) {}

    // TODO(human): Copy your try_push/try_pop from Phase 2
    bool try_push(const T& value) {
        (void)value;
        return false;  // placeholder
    }

    std::optional<T> try_pop() {
        return std::nullopt;  // placeholder
    }

    size_t size_approx() const {
        return static_cast<size_t>(
            tail_.load(std::memory_order_relaxed) -
            head_.load(std::memory_order_relaxed));
    }

private:
    alignas(CACHE_LINE_SIZE) T buffer_[Capacity];
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tail_;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head_;
};

// =============================================================================
// Object Pool (from Phase 3 -- simplified inline version)
// =============================================================================
//
// TODO(human): Copy your ObjectPool from Phase 3, or use this simplified version.

template <typename T, size_t Capacity>
class SimplePool {
public:
    SimplePool() : count_(0) {
        // Pre-allocate all objects
        for (size_t i = 0; i < Capacity; ++i) {
            pool_[i] = T{};
        }
    }

    // TODO(human): Implement a proper free-list pool (from Phase 3).
    // For now, this uses a simple stack-based approach as a placeholder.
    T* allocate() {
        if (count_ >= Capacity) return nullptr;
        return &pool_[count_++];
    }

    void deallocate(T* ptr) {
        (void)ptr;
        if (count_ > 0) --count_;
    }

    void reset() { count_ = 0; }

private:
    std::array<T, Capacity> pool_;
    size_t count_;
};

// =============================================================================
// CRTP Signal Processor (from Phase 4 pattern)
// =============================================================================
//
// TODO(human): Implement the signal processing logic using CRTP.
//
// The signal processor takes a MarketDataUpdate and produces a Signal.
// A simple strategy: moving average crossover.
//   - Maintain a fast MA (last 5 prices) and slow MA (last 20 prices)
//   - If fast MA crosses above slow MA -> BUY signal
//   - If fast MA crosses below slow MA -> SELL signal
//
// CRTP STRUCTURE:
//   template <typename Derived>
//   struct StrategyBase {
//       Signal process(const MarketDataUpdate& md) {
//           return static_cast<Derived*>(this)->on_market_data(md);
//       }
//   };
//
//   struct MovingAvgStrategy : StrategyBase<MovingAvgStrategy> {
//       Signal on_market_data(const MarketDataUpdate& md) {
//           // update MAs, check crossover, return Signal
//       }
//   };

template <typename Derived>
struct StrategyBase {
    Signal process(const MarketDataUpdate& md) {
        return static_cast<Derived*>(this)->on_market_data(md);
    }
};

struct MovingAvgStrategy : StrategyBase<MovingAvgStrategy> {
    // TODO(human): Implement moving average crossover strategy.
    //
    // Maintain:
    //   - double fast_ma_ (last 5 prices, simple average)
    //   - double slow_ma_ (last 20 prices, simple average)
    //   - std::array<int64_t, 20> price_history_ (circular buffer of recent prices)
    //   - size_t history_count_ = 0
    //   - size_t history_idx_ = 0
    //
    // on_market_data(md):
    //   1. Add md.price to price_history_[history_idx_]
    //   2. Advance history_idx_ = (history_idx_ + 1) % 20
    //   3. Increment history_count_ (up to 20)
    //   4. Compute fast_ma_ = average of last min(5, history_count_) prices
    //   5. Compute slow_ma_ = average of last min(20, history_count_) prices
    //   6. Signal fires if:
    //      - history_count_ >= 20 (enough data)
    //      - fast_ma_ crosses above slow_ma_ (buy) or below (sell)
    //   7. Return Signal{price, quantity=100, direction, should_trade}

    Signal on_market_data(const MarketDataUpdate& md) {
        (void)md;
        // Placeholder: never trades
        return Signal{0, 0, 0, false};
    }

    double fast_ma_ = 0.0;
    double slow_ma_ = 0.0;
    std::array<int64_t, 20> price_history_{};
    size_t history_count_ = 0;
    size_t history_idx_ = 0;
};

// =============================================================================
// CRTP Order Handler (from Phase 4 pattern)
// =============================================================================

template <typename Derived>
struct OrderHandlerBase {
    void submit(const OutboundOrder& order) {
        static_cast<Derived*>(this)->on_order(order);
    }
};

struct SimulatedOrderHandler : OrderHandlerBase<SimulatedOrderHandler> {
    void on_order(const OutboundOrder& order) {
        ++orders_sent_;
        last_order_ = order;
    }

    uint64_t orders_sent_ = 0;
    OutboundOrder last_order_{};
};

// =============================================================================
// Pipeline: Market Data -> Strategy -> Order
// =============================================================================
//
// TODO(human): Implement the full pipeline.
//
// THREAD 1 (Producer): Simulates market data feed
//   void produce_market_data(SPSCQueue<MarketDataUpdate, QUEUE_SIZE>& queue,
//                            size_t num_messages) {
//       for (size_t i = 0; i < num_messages; ++i) {
//           MarketDataUpdate md{};
//           md.sequence_no = i;
//           md.tsc_timestamp = rdtsc();           // timestamp "arrival"
//           md.price = 15000 + (i % 200) - 100;   // oscillating price
//           md.quantity = 100;
//           md.instrument_id = 1;
//           md.side = i & 1;
//           md.msg_type = 0;
//
//           while (!queue.try_push(md)) {
//               cpu_pause();
//           }
//       }
//   }
//
// THREAD 2 (Consumer / Hot Path):
//   void consume_and_process(SPSCQueue<MarketDataUpdate, QUEUE_SIZE>& queue,
//                            size_t num_messages,
//                            LatencyHistogram& histogram,
//                            uint64_t tsc_freq) {
//       MovingAvgStrategy strategy;
//       SimulatedOrderHandler order_handler;
//       SimplePool<Signal, 64> signal_pool;
//
//       size_t processed = 0;
//       while (processed < num_messages) {
//           auto maybe_md = queue.try_pop();
//           if (!maybe_md) {
//               cpu_pause();
//               continue;
//           }
//
//           // --- HOT PATH START ---
//           auto& md = *maybe_md;
//
//           // Process through strategy (CRTP, no virtual)
//           Signal sig = strategy.process(md);
//
//           // If signal fires, submit order (CRTP, no virtual)
//           if (sig.should_trade) {
//               OutboundOrder order{
//                   processed, sig.target_price, sig.target_quantity,
//                   md.instrument_id, sig.direction
//               };
//               order_handler.submit(order);
//           }
//
//           // Record end-to-end latency
//           uint64_t end_tsc = rdtsc();
//           double latency_ns = tsc_to_ns(end_tsc - md.tsc_timestamp, tsc_freq);
//           histogram.record(latency_ns);
//           // --- HOT PATH END ---
//
//           ++processed;
//       }
//
//       std::cout << "Orders sent: " << order_handler.orders_sent_ << "\n";
//   }

// Placeholder latency histogram (copy your implementation from Phase 5)
class PipelineHistogram {
public:
    PipelineHistogram(double bucket_width, size_t num_buckets)
        : bucket_width_(bucket_width), buckets_(num_buckets, 0),
          total_(0), overflow_(0) {}

    void record(double ns) {
        // TODO(human): Same as Phase 5 histogram
        (void)ns;
        ++total_;
    }

    void print() const {
        std::cout << "Histogram: " << total_ << " samples recorded\n";
        // TODO(human): Print bucket distribution
    }

    uint64_t total() const { return total_; }

private:
    double bucket_width_;
    std::vector<uint64_t> buckets_;
    uint64_t total_;
    uint64_t overflow_;
};

void run_pipeline() {
    std::cout << "\n=== Pipeline Benchmark ===\n\n";

    constexpr size_t QUEUE_SIZE = 65536;  // 2^16
    constexpr size_t NUM_MESSAGES = 1'000'000;

    // TODO(human): Wire up the full pipeline:
    //   1. Calibrate TSC
    //   2. Create SPSC queue, histogram
    //   3. Launch producer thread
    //   4. Launch consumer thread (hot path)
    //   5. Join threads
    //   6. Print histogram and stats
    //
    // EXPECTED RESULTS:
    //   - P50 latency: ~100-500 ns (includes SPSC transfer + strategy + order)
    //   - P99 latency: ~500-2000 ns
    //   - Most time spent in SPSC queue wait (producer-consumer scheduling)
    //   - Processing itself (strategy + order): ~50-200 ns
    //
    // AFTER RUNNING: Ask yourself:
    //   - Where is the bottleneck? (Usually SPSC queue + thread scheduling)
    //   - What would you optimize next? (CPU affinity, busy-polling, huge pages)
    //   - How does this compare to real HFT? (They get ~200-800ns tick-to-trade)

    SPSCQueue<MarketDataUpdate, QUEUE_SIZE> queue;
    PipelineHistogram histogram(10.0, 500);  // 10ns buckets, 0-5000ns range

    std::cout << "Queue size: " << QUEUE_SIZE << "\n";
    std::cout << "Messages: " << NUM_MESSAGES << "\n";
    std::cout << "\n[Placeholder] Implement the producer-consumer pipeline.\n";
    std::cout << "Combine SPSC (Phase 2), Pool (Phase 3), CRTP (Phase 4), TSC (Phase 5).\n";
    std::cout << "\nTarget: sub-microsecond P50 processing latency.\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 6: Mini Hot-Path Pipeline\n";
    std::cout << "========================================\n";

    // Quick component checks
    {
        std::cout << "\n--- Component verification ---\n";
        std::cout << "MarketDataUpdate: " << sizeof(MarketDataUpdate)
                  << " bytes (should be 64)\n";
        std::cout << "Signal: " << sizeof(Signal) << " bytes\n";
        std::cout << "OutboundOrder: " << sizeof(OutboundOrder) << " bytes\n";

        // Test strategy
        MovingAvgStrategy strategy;
        MarketDataUpdate md{};
        md.price = 15000;
        md.quantity = 100;
        Signal sig = strategy.process(md);
        std::cout << "Strategy test: should_trade=" << sig.should_trade
                  << " (expected: false, not enough history)\n";

        // Test order handler
        SimulatedOrderHandler handler;
        handler.submit(OutboundOrder{1, 15000, 100, 1, 0});
        std::cout << "Order handler test: orders_sent=" << handler.orders_sent_
                  << " (expected: 1)\n";
    }

    run_pipeline();

    std::cout << "\n========================================\n";
    std::cout << "Phase 6 complete.\n";
    std::cout << "========================================\n";
    std::cout << "\nCONGRATULATIONS! You've built the core components of an HFT system:\n";
    std::cout << "  [Phase 1] Cache-aligned data structures\n";
    std::cout << "  [Phase 2] Lock-free SPSC message queue\n";
    std::cout << "  [Phase 3] Allocation-free memory pools\n";
    std::cout << "  [Phase 4] Zero-overhead compile-time dispatch\n";
    std::cout << "  [Phase 5] Sub-nanosecond TSC timing\n";
    std::cout << "  [Phase 6] Integrated hot-path pipeline\n";
    std::cout << "\nNext steps for a real HFT system:\n";
    std::cout << "  - CPU affinity (pin threads to specific cores)\n";
    std::cout << "  - Huge pages (2MB pages, fewer TLB misses)\n";
    std::cout << "  - Kernel bypass (DPDK/Solarflare for network I/O)\n";
    std::cout << "  - NUMA awareness (keep memory local to the socket)\n";
    std::cout << "  - Compiler intrinsics (__builtin_expect, __builtin_prefetch)\n";

    return 0;
}
