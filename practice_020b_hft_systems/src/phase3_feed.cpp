// =============================================================================
// Phase 3: Market Data Feed Handler
//
// In real trading, you don't have the exchange's internal order book.
// Instead, the exchange broadcasts market data messages (adds, cancels,
// modifies, trades) and each participant builds a LOCAL COPY of the book.
//
// You'll implement:
//   1. FeedHandler::on_message()    -- Process incremental updates
//   2. FeedHandler::apply_snapshot() -- Rebuild book from snapshot
//   3. Sequence number gap detection
//
// This simulates the pattern used by:
//   - Nasdaq ITCH 5.0 (binary, ~50 bytes/msg, ~1B messages/day)
//   - CME MDP 3.0 (multicast UDP, incremental + snapshot channels)
//   - FIX/FAST (text/binary hybrid)
//
// KEY INSIGHT: If you miss a message (network drop, late join), your
// book is WRONG. A missed CancelOrder means you see a phantom order.
// A missed AddOrder means you're missing real liquidity. Both lead
// to bad trading decisions. Gap detection and snapshot recovery are
// therefore CRITICAL infrastructure in every trading system.
// =============================================================================

#include <iostream>
#include <vector>

#include "absl/strings/str_format.h"
#include "feed_handler.h"

// ── FeedHandler implementation ──────────────────────────────────────────────

void FeedHandler::on_message(const MarketDataMessage& message) {
    // TODO(human): Process a market data message using std::visit.
    //
    // Steps:
    //   1. Extract the sequence number from the message.
    //      All message types have a seq_num field.
    //      Use std::visit with a lambda that gets seq_num from any variant:
    //        SeqNum seq = std::visit([](const auto& msg) { return msg.seq_num; }, message);
    //
    //   2. Check for gap:
    //      if (expected_seq_ > 0 && seq != expected_seq_) {
    //          gap_detected_ = true;
    //          gap_count_++;
    //          gap_buffer_.push_back(message);
    //          return;  // Don't process -- wait for snapshot
    //      }
    //
    //   3. If gap already detected, buffer the message:
    //      if (gap_detected_) { gap_buffer_.push_back(message); return; }
    //
    //   4. Process the message using std::visit with the overloaded pattern:
    //
    //      std::visit(overloaded{
    //          [&](const MsgAddOrder& msg) {
    //              Order order;
    //              order.order_id = msg.order_id;
    //              order.side = msg.side;
    //              order.price = msg.price;
    //              order.quantity = msg.quantity;
    //              order.type = OrderType::Limit;
    //              book_.add_order(order);
    //          },
    //          [&](const MsgModifyOrder& msg) {
    //              // Simplification: cancel + re-add with new quantity.
    //              // Real feeds modify in-place. Our OrderBook doesn't support
    //              // in-place modify, so this is a valid workaround.
    //              // Note: we'd need to know the order's side/price to re-add.
    //              // For simplicity, just cancel it (modify down to 0 = cancel).
    //              book_.cancel_order(msg.order_id);
    //              // TODO: For full fidelity, re-add with new qty
    //          },
    //          [&](const MsgCancelOrder& msg) {
    //              book_.cancel_order(msg.order_id);
    //          },
    //          [&](const MsgTrade& msg) {
    //              Trade t;
    //              t.trade_id = msg.trade_id;
    //              t.aggressor_id = msg.buy_order_id;
    //              t.passive_id = msg.sell_order_id;
    //              t.price = msg.price;
    //              t.quantity = msg.quantity;
    //              recent_trades_.push_back(t);
    //          },
    //          [&](const MsgSnapshot& msg) {
    //              apply_snapshot(msg);
    //          }
    //      }, message);
    //
    //   5. Update expected_seq_ = seq + 1;
    //
    // The overloaded{} pattern was covered in 012a (Practice 012a, Phase 2).

    // --- Placeholder: does nothing ---
    (void)message;

    // TODO(human): Replace with your implementation.
}

void FeedHandler::apply_snapshot(const MsgSnapshot& snapshot) {
    // TODO(human): Rebuild the book from a snapshot.
    //
    // Steps:
    //   1. Create a fresh OrderBook:
    //      book_ = OrderBook{};
    //
    //   2. Add all orders from the snapshot:
    //      for (const auto& order : snapshot.orders) {
    //          book_.add_order(order);
    //      }
    //
    //   3. Update expected_seq_ = snapshot.seq_num + 1
    //
    //   4. Reset gap state:
    //      gap_detected_ = false;
    //
    //   5. Replay buffered messages that have seq_num >= expected_seq_:
    //      auto buffered = std::move(gap_buffer_);
    //      gap_buffer_.clear();
    //      for (const auto& msg : buffered) {
    //          SeqNum seq = std::visit([](const auto& m) { return m.seq_num; }, msg);
    //          if (seq >= expected_seq_) {
    //              on_message(msg);  // Recursive call -- will process normally now
    //          }
    //      }
    //
    // Why clear and replay? The snapshot gives us state at a point in time.
    // Any messages received AFTER that point (buffered during the gap) need
    // to be applied on top of the snapshot to get the current state.

    // --- Placeholder: does nothing ---
    (void)snapshot;

    // TODO(human): Replace with your implementation.
}

// ── Test harness ────────────────────────────────────────────────────────────

namespace {

void print_book_summary(const FeedHandler& fh) {
    auto& book = fh.book();
    std::cout << absl::StrFormat("  Orders in book: %d\n", book.total_orders());
    if (auto bid = book.best_bid(); bid.has_value()) {
        std::cout << absl::StrFormat("  Best bid: %.4f x %d\n",
            price_to_double(bid->price), bid->total_qty);
    }
    if (auto ask = book.best_ask(); ask.has_value()) {
        std::cout << absl::StrFormat("  Best ask: %.4f x %d\n",
            price_to_double(ask->price), ask->total_qty);
    }
    std::cout << absl::StrFormat("  Expected seq: %u, Gap: %s, Gap count: %d\n",
        fh.expected_seq(), fh.has_gap() ? "YES" : "no", fh.gap_count());
    std::cout << absl::StrFormat("  Recent trades: %d\n", fh.recent_trades().size());
}

}  // namespace

int main() {
    std::cout << "=== Phase 3: Market Data Feed Handler ===\n\n";

    FeedHandler fh;

    // --- Simulate a stream of market data messages ---
    std::cout << "--- Sending initial order flow ---\n";

    // Seq 1: Add buy order
    fh.on_message(MsgAddOrder{1, 101, Side::Buy, price_from_double(150.10), 500});
    // Seq 2: Add buy order
    fh.on_message(MsgAddOrder{2, 102, Side::Buy, price_from_double(150.09), 300});
    // Seq 3: Add sell order
    fh.on_message(MsgAddOrder{3, 201, Side::Sell, price_from_double(150.11), 400});
    // Seq 4: Add sell order
    fh.on_message(MsgAddOrder{4, 202, Side::Sell, price_from_double(150.12), 600});

    print_book_summary(fh);

    // --- Test cancel ---
    std::cout << "\n--- Cancelling order 102 ---\n";
    fh.on_message(MsgCancelOrder{5, 102});
    print_book_summary(fh);

    // --- Test trade message ---
    std::cout << "\n--- Trade occurred ---\n";
    fh.on_message(MsgTrade{6, 1, 101, 201, price_from_double(150.11), 200});
    print_book_summary(fh);

    // --- Simulate a gap ---
    std::cout << "\n--- Simulating sequence gap (skip msg 7) ---\n";
    // Skip seq 7, send seq 8
    fh.on_message(MsgAddOrder{8, 301, Side::Buy, price_from_double(150.10), 100});

    std::cout << "After gap:\n";
    print_book_summary(fh);

    // --- Recovery via snapshot ---
    std::cout << "\n--- Sending snapshot for recovery ---\n";
    MsgSnapshot snap;
    snap.seq_num = 8;
    snap.orders.push_back(Order{101, price_from_double(150.10), 300, 300, 0, Side::Buy, OrderType::Limit});
    snap.orders.push_back(Order{201, price_from_double(150.11), 200, 200, 0, Side::Sell, OrderType::Limit});
    snap.orders.push_back(Order{202, price_from_double(150.12), 600, 600, 0, Side::Sell, OrderType::Limit});
    snap.orders.push_back(Order{301, price_from_double(150.10), 100, 100, 0, Side::Buy, OrderType::Limit});

    fh.apply_snapshot(snap);

    std::cout << "After snapshot recovery:\n";
    print_book_summary(fh);

    // --- Resume normal flow ---
    std::cout << "\n--- Resuming normal message flow ---\n";
    fh.on_message(MsgAddOrder{9, 302, Side::Sell, price_from_double(150.13), 250});
    print_book_summary(fh);

    std::cout << "\n=== Phase 3 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - std::visit + overloaded{} dispatches message types cleanly\n";
    std::cout << "  - Sequence number gaps mean your book is WRONG -- must detect immediately\n";
    std::cout << "  - Snapshot recovery: clear book, rebuild from snapshot, replay buffer\n";
    std::cout << "  - Nasdaq ITCH processes ~1 billion of these messages per day\n";

    return 0;
}
