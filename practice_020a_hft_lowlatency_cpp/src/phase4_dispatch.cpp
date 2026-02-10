// =============================================================================
// Phase 4: Compile-Time Dispatch & Branch Elimination
// =============================================================================
//
// HFT CONTEXT:
//   Virtual function calls cost ~2-5ns due to indirect branch + possible
//   icache miss. On a hot path processing 100M+ messages/sec, that's
//   200-500ms per second wasted on dispatch overhead alone.
//
//   The fix: move dispatch to COMPILE TIME. The compiler sees the concrete
//   type, inlines the method body, and generates a direct call (or no call
//   at all). Zero overhead.
//
//   Three techniques, from most to least common in HFT:
//   1. CRTP (Curiously Recurring Template Pattern) -- static polymorphism
//   2. if constexpr -- compile-time branching within templates
//   3. std::variant + std::visit -- type-safe union with compile-time dispatch
//
// KEY INSIGHT:
//   Virtual dispatch:   runtime decision -> indirect branch -> icache miss
//   CRTP:               compile-time decision -> inlined -> zero overhead
//   if constexpr:       compile-time decision -> dead code eliminated -> zero overhead
//   std::visit:         compile-time dispatch table -> usually inlined -> near-zero overhead
//
// WHY NOT VIRTUAL IN HFT:
//   1. vtable pointer = 8 bytes wasted per object (cache pollution)
//   2. Indirect call = CPU can't predict target -> pipeline stall
//   3. Virtual prevents inlining -> no optimization across call boundary
//   4. With CRTP, the compiler KNOWS the type -> full optimization
//
// RUST EQUIVALENT:
//   Virtual dispatch = dyn Trait (trait object, fat pointer)
//   CRTP = monomorphization (impl Trait -> compiler generates concrete code)
//   if constexpr = match on type with generics (monomorphized away)
//   Rust defaults to monomorphization (zero-cost). You opt INTO dynamic dispatch
//   with `dyn`. C++ defaults to virtual (dynamic). You opt OUT with CRTP.
// =============================================================================

#include "hft_common.h"
#include <variant>
#include <functional>

// =============================================================================
// Example message types (common to all exercises)
// =============================================================================

// Market data message types that arrive on the feed
struct NewOrder {
    uint64_t order_id;
    int64_t price;
    uint32_t quantity;
    uint8_t side;  // 0=buy, 1=sell
};

struct CancelOrder {
    uint64_t order_id;
};

struct ModifyOrder {
    uint64_t order_id;
    int64_t new_price;
    uint32_t new_quantity;
};

// =============================================================================
// Exercise 1: CRTP vs Virtual Dispatch
// =============================================================================
//
// GOAL: Implement the same message handler interface using:
//   a) Virtual functions (traditional OOP)
//   b) CRTP (static polymorphism)
//   Then benchmark both.
//
// CRTP PATTERN:
//   template <typename Derived>
//   struct HandlerBase {
//       void handle(const NewOrder& msg) {
//           static_cast<Derived*>(this)->on_new_order(msg);
//       }
//       void handle(const CancelOrder& msg) {
//           static_cast<Derived*>(this)->on_cancel_order(msg);
//       }
//   };
//
//   struct MyHandler : HandlerBase<MyHandler> {
//       void on_new_order(const NewOrder& msg) { /* ... */ }
//       void on_cancel_order(const CancelOrder& msg) { /* ... */ }
//   };
//
// HOW CRTP ACHIEVES ZERO COST:
//   1. HandlerBase<MyHandler> is a unique type -- the compiler knows Derived = MyHandler
//   2. static_cast<MyHandler*>(this)->on_new_order(msg) is a DIRECT call to MyHandler::on_new_order
//   3. The compiler inlines it completely -- no call instruction at all
//   4. No vtable, no indirect branch, no icache miss
// =============================================================================

// --- Virtual version (for comparison) ---

class VirtualHandler {
public:
    virtual ~VirtualHandler() = default;
    virtual void on_new_order(const NewOrder& msg) = 0;
    virtual void on_cancel_order(const CancelOrder& msg) = 0;
    virtual void on_modify_order(const ModifyOrder& msg) = 0;
};

// TODO(human): Implement a concrete VirtualHandler.
//
// class VirtualOrderBook : public VirtualHandler {
// public:
//     void on_new_order(const NewOrder& msg) override {
//         total_quantity_ += msg.quantity;
//         last_price_ = msg.price;
//     }
//     void on_cancel_order(const CancelOrder& msg) override {
//         ++cancel_count_;
//         (void)msg;
//     }
//     void on_modify_order(const ModifyOrder& msg) override {
//         total_quantity_ += msg.new_quantity;
//         last_price_ = msg.new_price;
//     }
//     uint64_t total_quantity() const { return total_quantity_; }
//     int64_t last_price() const { return last_price_; }
//     uint64_t cancel_count() const { return cancel_count_; }
// private:
//     uint64_t total_quantity_ = 0;
//     int64_t last_price_ = 0;
//     uint64_t cancel_count_ = 0;
// };

class VirtualOrderBook : public VirtualHandler {
public:
    // TODO(human): Implement the three handlers
    void on_new_order(const NewOrder& msg) override { (void)msg; }
    void on_cancel_order(const CancelOrder& msg) override { (void)msg; }
    void on_modify_order(const ModifyOrder& msg) override { (void)msg; }

    uint64_t total_quantity() const { return total_quantity_; }
    int64_t last_price() const { return last_price_; }
    uint64_t cancel_count() const { return cancel_count_; }

private:
    uint64_t total_quantity_ = 0;
    int64_t last_price_ = 0;
    uint64_t cancel_count_ = 0;
};

// --- CRTP version ---

// TODO(human): Implement the CRTP base and derived handler.
//
// STEP 1: Define CRTPHandlerBase<Derived> with handle() methods
//   that static_cast<Derived*>(this)->on_xxx(msg).
//
// STEP 2: Define CRTPOrderBook : CRTPHandlerBase<CRTPOrderBook>
//   with the same logic as VirtualOrderBook.
//
// HINT:
//   template <typename Derived>
//   struct CRTPHandlerBase {
//       void handle(const NewOrder& msg) {
//           static_cast<Derived*>(this)->on_new_order(msg);
//       }
//       // ... same for CancelOrder, ModifyOrder
//   };

template <typename Derived>
struct CRTPHandlerBase {
    void handle(const NewOrder& msg) {
        // TODO(human): static_cast to Derived and call on_new_order
        (void)msg;
    }
    void handle(const CancelOrder& msg) {
        // TODO(human): static_cast to Derived and call on_cancel_order
        (void)msg;
    }
    void handle(const ModifyOrder& msg) {
        // TODO(human): static_cast to Derived and call on_modify_order
        (void)msg;
    }
};

struct CRTPOrderBook : CRTPHandlerBase<CRTPOrderBook> {
    // TODO(human): Implement on_new_order, on_cancel_order, on_modify_order
    // with the same accumulation logic as VirtualOrderBook.

    void on_new_order(const NewOrder& msg) { (void)msg; }
    void on_cancel_order(const CancelOrder& msg) { (void)msg; }
    void on_modify_order(const ModifyOrder& msg) { (void)msg; }

    uint64_t total_quantity_ = 0;
    int64_t last_price_ = 0;
    uint64_t cancel_count_ = 0;
};

// =============================================================================
// Exercise 2: if constexpr for Order Type Dispatch
// =============================================================================
//
// GOAL: Use if constexpr to dispatch different message types at compile time
//       within a single template function.
//
// PATTERN:
//   template <typename MsgT>
//   void process_message(const MsgT& msg) {
//       if constexpr (std::is_same_v<MsgT, NewOrder>) {
//           // handle new order -- this code ONLY exists when MsgT=NewOrder
//       } else if constexpr (std::is_same_v<MsgT, CancelOrder>) {
//           // handle cancel -- this code ONLY exists when MsgT=CancelOrder
//       } else {
//           static_assert(always_false<MsgT>, "Unknown message type");
//       }
//   }
//
// WHY THIS IS ZERO COST:
//   The compiler evaluates if constexpr at compile time. The "false" branches
//   are not just not-executed -- they're not even COMPILED. No dead code,
//   no branch instruction, no branch predictor involved.
//
// COMBINED WITH [[likely]]/[[unlikely]]:
//   Within a branch, you can still hint the CPU about runtime probabilities:
//   if constexpr (std::is_same_v<MsgT, NewOrder>) {
//       if (msg.quantity > 0) [[likely]] {  // most orders have qty > 0
//           // hot path
//       }
//   }
// =============================================================================

// Helper for static_assert in else branch
template <typename> inline constexpr bool always_false_v = false;

// TODO(human): Implement process_message using if constexpr.
//
// For NewOrder: accumulate quantity, update last price
// For CancelOrder: increment cancel count
// For ModifyOrder: accumulate new quantity, update last price
//
// Return value: the "result" (e.g., the accumulated total, for benchmarking).
//
// HINT:
//   template <typename MsgT>
//   uint64_t process_message(const MsgT& msg, uint64_t& total_qty,
//                            int64_t& last_price, uint64_t& cancel_count) {
//       if constexpr (std::is_same_v<MsgT, NewOrder>) {
//           total_qty += msg.quantity;
//           last_price = msg.price;
//       } else if constexpr (...) {
//           ...
//       } else {
//           static_assert(always_false_v<MsgT>, "Unknown message type");
//       }
//       return total_qty;
//   }

template <typename MsgT>
uint64_t process_message(const MsgT& msg, uint64_t& total_qty,
                         int64_t& last_price, uint64_t& cancel_count) {
    // --- TODO(human): implement with if constexpr ---
    (void)msg;
    (void)total_qty;
    (void)last_price;
    (void)cancel_count;
    return 0;  // placeholder
}

// =============================================================================
// Exercise 3: Benchmark -- Virtual vs CRTP vs std::visit
// =============================================================================
//
// TODO(human): Implement a benchmark comparing dispatch mechanisms.
//
// SETUP:
//   - Generate 10M NewOrder messages (the most common type)
//   - Process all messages through each dispatch mechanism
//   - Measure total time, compute ns/message
//
// THREE VARIANTS:
//   a) Virtual: VirtualHandler* handler = &orderbook; handler->on_new_order(msg);
//   b) CRTP:    CRTPOrderBook book; book.handle(msg);
//   c) std::visit on std::variant<NewOrder, CancelOrder, ModifyOrder>
//
// std::visit version (for you to benchmark):
//   using Message = std::variant<NewOrder, CancelOrder, ModifyOrder>;
//   std::vector<Message> messages;
//   // fill with NewOrder messages...
//   for (auto& msg : messages) {
//       std::visit([&](auto& m) { process_message(m, ...); }, msg);
//   }
//
// EXPECTED RESULTS:
//   Virtual: ~3-5 ns/msg (indirect call + possible icache miss)
//   CRTP:    ~0.5-1 ns/msg (fully inlined, direct call)
//   visit:   ~1-2 ns/msg (jump table, usually inlined for small variant)
//
// KEY INSIGHT: The difference seems small per call, but at 100M msgs/sec,
//   virtual = 300-500ms wasted per second. CRTP = 50-100ms. Over a trading day,
//   that's the difference between first and last in the queue.
// =============================================================================

void benchmark_dispatch() {
    std::cout << "\n=== Exercise 3: Dispatch Benchmark ===\n\n";

    constexpr size_t NUM_MSGS = 10'000'000;

    // Pre-generate messages
    std::vector<NewOrder> messages(NUM_MSGS);
    for (size_t i = 0; i < NUM_MSGS; ++i) {
        messages[i] = NewOrder{i, static_cast<int64_t>(15000 + (i % 200)),
                               static_cast<uint32_t>(100 + (i % 50)), static_cast<uint8_t>(i & 1)};
    }

    // --- Virtual dispatch ---
    {
        std::cout << "--- Virtual dispatch ---\n";
        VirtualOrderBook book;
        VirtualHandler* handler = &book;  // erase concrete type -> forces virtual call

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < NUM_MSGS; ++i) {
            handler->on_new_order(messages[i]);
        }
        auto end = std::chrono::steady_clock::now();
        do_not_optimize(book.total_quantity());

        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "  Total: " << static_cast<uint64_t>(ns / 1000) << " us, "
                  << std::fixed << std::setprecision(2) << (ns / NUM_MSGS) << " ns/msg\n";
        std::cout << "  (Result: qty=" << book.total_quantity()
                  << " cancels=" << book.cancel_count() << ")\n";
    }

    // --- CRTP dispatch ---
    {
        std::cout << "\n--- CRTP dispatch ---\n";
        CRTPOrderBook book;

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < NUM_MSGS; ++i) {
            book.handle(messages[i]);
        }
        auto end = std::chrono::steady_clock::now();
        do_not_optimize(book.total_quantity_);

        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "  Total: " << static_cast<uint64_t>(ns / 1000) << " us, "
                  << std::fixed << std::setprecision(2) << (ns / NUM_MSGS) << " ns/msg\n";
        std::cout << "  (Result: qty=" << book.total_quantity_
                  << " cancels=" << book.cancel_count_ << ")\n";
    }

    // --- if constexpr dispatch ---
    {
        std::cout << "\n--- if constexpr dispatch ---\n";
        uint64_t total_qty = 0;
        int64_t last_price = 0;
        uint64_t cancel_count = 0;

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < NUM_MSGS; ++i) {
            process_message(messages[i], total_qty, last_price, cancel_count);
        }
        auto end = std::chrono::steady_clock::now();
        do_not_optimize(total_qty);

        double ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "  Total: " << static_cast<uint64_t>(ns / 1000) << " us, "
                  << std::fixed << std::setprecision(2) << (ns / NUM_MSGS) << " ns/msg\n";
        std::cout << "  (Result: qty=" << total_qty
                  << " cancels=" << cancel_count << ")\n";
    }

    // --- std::variant + std::visit ---
    {
        std::cout << "\n--- std::variant + std::visit ---\n";

        // TODO(human): Create a vector of std::variant<NewOrder, CancelOrder, ModifyOrder>,
        // fill with NewOrder messages, and benchmark std::visit dispatch.
        //
        // using Message = std::variant<NewOrder, CancelOrder, ModifyOrder>;
        // std::vector<Message> var_messages(NUM_MSGS);
        // for (size_t i = 0; i < NUM_MSGS; ++i) {
        //     var_messages[i] = messages[i];
        // }
        //
        // uint64_t total_qty = 0;
        // int64_t last_price = 0;
        // uint64_t cancel_count = 0;
        //
        // for (auto& msg : var_messages) {
        //     std::visit([&](auto& m) {
        //         process_message(m, total_qty, last_price, cancel_count);
        //     }, msg);
        // }

        std::cout << "  [Placeholder] Implement std::visit benchmark.\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "Phase 4: Compile-Time Dispatch\n";
    std::cout << "========================================\n";

    // Quick functionality check
    {
        std::cout << "\n--- Functionality check ---\n";

        CRTPOrderBook book;
        book.handle(NewOrder{1, 15000, 100, 0});
        book.handle(NewOrder{2, 15050, 200, 1});
        book.handle(CancelOrder{1});

        // NOTE: These show 0 until you implement the handlers.
        std::cout << "CRTP OrderBook: qty=" << book.total_quantity_
                  << " last_price=" << book.last_price_
                  << " cancels=" << book.cancel_count_ << "\n";
        std::cout << "(Expected after implementation: qty=300 last_price=15050 cancels=1)\n";
    }

    benchmark_dispatch();

    std::cout << "\n========================================\n";
    std::cout << "Phase 4 complete.\n";
    std::cout << "========================================\n";

    return 0;
}
