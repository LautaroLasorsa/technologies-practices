#pragma once

// =============================================================================
// feed_handler.h -- Market Data Feed Handler
//
// In real trading, you don't have direct access to the exchange's internal
// order book. Instead, the exchange broadcasts market data messages over a
// network feed, and each participant must maintain a LOCAL COPY of the book
// by applying these messages in order.
//
// TWO TYPES OF MARKET DATA:
//
//   1. INCREMENTAL UPDATES (most of the data):
//      - AddOrder: new order placed on the book
//      - ModifyOrder: quantity changed (usually reduced)
//      - CancelOrder: order removed from the book
//      - Trade: two orders matched (informational, after the fact)
//
//   2. SNAPSHOTS (periodic or on-demand):
//      - Full state of the book at a point in time
//      - Used to initialize the local book, or to recover after gaps
//
// SEQUENCE NUMBERS:
//   Every message has a monotonically increasing sequence number.
//   If you receive seq 100, 101, 103 -- you MISSED message 102.
//   A missed message means your local book is WRONG. You must:
//     1. Detect the gap
//     2. Request a snapshot
//     3. Rebuild the book from the snapshot
//     4. Resume processing incrementals from the snapshot's sequence number
//
//   Gap detection is CRITICAL. If you miss a CancelOrder message, your book
//   shows a phantom order that doesn't exist on the exchange. If you trade
//   based on that, your strategy is making decisions on stale/wrong data.
//
// REAL-WORLD PROTOCOLS:
//   - Nasdaq ITCH 5.0: Binary, ~50 bytes/message, ~1 billion messages/day
//   - CME MDP 3.0: Binary, multicast UDP, incremental + snapshots on separate channels
//   - FIX/FAST: Text/binary hybrid, older but still widely used
//   - SIP (Securities Information Processor): Consolidated feed for US equities
//
// We simulate a simplified version of the incremental update pattern.
// =============================================================================

#include <variant>
#include <vector>

#include "order_book.h"
#include "types.h"

// ── Market Data Messages ────────────────────────────────────────────────────
// Using std::variant to model the message types (sum type, like Rust's enum).
// This is exactly how 012a taught you to use variant + visit.

struct MsgAddOrder {
    SeqNum  seq_num  = 0;
    OrderId order_id = 0;
    Side    side     = Side::Buy;
    price_t price    = 0;
    qty_t   quantity = 0;
};

struct MsgModifyOrder {
    SeqNum  seq_num  = 0;
    OrderId order_id = 0;
    qty_t   new_quantity = 0;  // New remaining quantity (always <= old quantity)
};

struct MsgCancelOrder {
    SeqNum  seq_num  = 0;
    OrderId order_id = 0;
};

struct MsgTrade {
    SeqNum  seq_num       = 0;
    TradeId trade_id      = 0;
    OrderId buy_order_id  = 0;
    OrderId sell_order_id = 0;
    price_t price         = 0;
    qty_t   quantity      = 0;
};

// Snapshot of the full book state (used for recovery after gap).
struct MsgSnapshot {
    SeqNum seq_num = 0;  // Sequence number this snapshot is valid at
    std::vector<Order> orders;  // All resting orders in the book
};

// The variant type: one message is exactly one of these types.
// Pattern matching via std::visit (learned in 012a).
using MarketDataMessage = std::variant<
    MsgAddOrder,
    MsgModifyOrder,
    MsgCancelOrder,
    MsgTrade,
    MsgSnapshot
>;

// ── Feed Handler ────────────────────────────────────────────────────────────

// Maintains a local order book by processing market data messages.
// Detects sequence number gaps and triggers snapshot recovery.
class FeedHandler {
public:
    // Process a single market data message.
    //
    // TODO(human): Implement this using std::visit.
    // Hint:
    //   1. Extract the sequence number from the message (all types have seq_num)
    //   2. Check for gap: if seq_num != expected_seq_ + 1, set gap_detected_ = true
    //   3. If gap detected, don't process the message (wait for snapshot)
    //   4. If no gap, dispatch to the appropriate handler:
    //      - MsgAddOrder:    create Order and call book_.add_order()
    //      - MsgModifyOrder: cancel old order and re-add with new qty
    //                        (simplification -- real feeds modify in-place)
    //      - MsgCancelOrder: call book_.cancel_order()
    //      - MsgTrade:       record the trade (informational, book already updated)
    //      - MsgSnapshot:    rebuild the book from scratch (clear + add all orders)
    //   5. Update expected_seq_
    //
    // Use the overloaded lambda pattern from 012a:
    //   auto visitor = overloaded{
    //       [&](const MsgAddOrder& msg) { ... },
    //       [&](const MsgModifyOrder& msg) { ... },
    //       ...
    //   };
    //   std::visit(visitor, message);
    void on_message(const MarketDataMessage& message);

    // Apply a snapshot to rebuild the book (called when gap is detected).
    //
    // TODO(human): Implement this.
    // Hint:
    //   1. Clear the current book (create a fresh OrderBook)
    //   2. Add all orders from the snapshot
    //   3. Update expected_seq_ to the snapshot's sequence number
    //   4. Reset gap_detected_ to false
    //   5. Process any buffered messages with seq > snapshot.seq_num
    void apply_snapshot(const MsgSnapshot& snapshot);

    // Accessors
    const OrderBook& book() const { return book_; }
    OrderBook& book() { return book_; }
    bool has_gap() const { return gap_detected_; }
    SeqNum expected_seq() const { return expected_seq_; }
    const std::vector<Trade>& recent_trades() const { return recent_trades_; }
    size_t gap_count() const { return gap_count_; }

private:
    OrderBook book_;
    SeqNum    expected_seq_ = 0;   // Next expected sequence number
    bool      gap_detected_ = false;
    size_t    gap_count_    = 0;    // Total gaps detected (for stats)

    // Trades received via MsgTrade (for informational purposes / signal generation).
    std::vector<Trade> recent_trades_;

    // Buffer messages received during a gap (to replay after snapshot).
    std::vector<MarketDataMessage> gap_buffer_;
};

// ── Overloaded helper ───────────────────────────────────────────────────────
// Classic C++17 pattern for std::visit with multiple lambdas.
// You learned this in 012a. Included here for convenience.
template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;
