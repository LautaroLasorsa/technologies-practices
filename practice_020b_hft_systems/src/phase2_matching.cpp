// =============================================================================
// Phase 2: Matching Engine
//
// The matching engine is the heart of every exchange. It receives incoming
// orders, matches them against resting orders in the book using price-time
// priority, and generates trade records for every fill.
//
// You'll implement:
//   1. MatchingEngine::match_against_book() -- Core matching algorithm
//   2. MatchingEngine::can_fill_completely() -- FOK check
//   3. MatchingEngine::submit_order()       -- Top-level order handling
//
// MATCHING RULES:
//   - BUY order matches against ASK side (lowest ask first)
//     Condition: incoming.price >= resting.price
//   - SELL order matches against BID side (highest bid first)
//     Condition: incoming.price <= resting.price
//   - At each price level: FIFO (oldest resting order matches first)
//   - Execution price = resting order's price (maker sets the price)
//   - Partial fills: both incoming and resting orders can be partially filled
//
// ORDER TYPES:
//   LIMIT: Match what you can, rest the remainder on the book
//   IOC:   Match what you can, cancel the remainder (don't rest)
//   FOK:   Match entirely or cancel entirely (all-or-nothing)
//
// REAL-WORLD CONTEXT:
//   CME Globex matches a futures order in < 5 microseconds.
//   Nasdaq handles 100,000+ messages/second at peak.
//   The matching algorithm is conceptually simple -- the engineering
//   challenge is making it fast enough for microsecond latencies.
// =============================================================================

#include <iostream>
#include <vector>

#include "absl/strings/str_format.h"
#include "matching_engine.h"

// ── MatchingEngine implementation ───────────────────────────────────────────

std::vector<Trade> MatchingEngine::match_against_book(Order& incoming) {
    // TODO(human): Match the incoming order against the opposite side of the book.
    //
    // Algorithm (price-time priority):
    //
    //   If incoming is BUY:
    //     auto& opposite = book_.asks();   // match against asks
    //     Iterate from begin() (lowest ask price first):
    //       while incoming.remaining_qty > 0 && !opposite.empty():
    //         auto it = opposite.begin();
    //         auto& [level_price, level] = *it;
    //         if level_price > incoming.price: break  // no more matchable prices
    //         while incoming.remaining_qty > 0 && !level.empty():
    //           auto& resting = level.front();
    //           qty_t fill_qty = std::min(incoming.remaining_qty, resting.remaining_qty);
    //           incoming.remaining_qty -= fill_qty;
    //           resting.remaining_qty  -= fill_qty;
    //           trades.push_back(Trade{next_trade_id(), incoming.order_id, resting.order_id,
    //                                  resting.price, fill_qty, incoming.side, 0});
    //           if resting.is_filled():
    //             book_.erase_order_location(resting.order_id);
    //             level.orders.pop_front();  // remove filled order from FIFO
    //         if level.empty():
    //           opposite.erase(it);  // clean up empty price level
    //
    //   If incoming is SELL: same but match against bids (rbegin, highest first)
    //     Condition: level_price >= incoming.price
    //     NOTE: iterating rbegin on std::map is tricky with erasure.
    //     Tip: convert to forward iteration pattern:
    //       while !opposite.empty():
    //         auto it = std::prev(opposite.end());  // highest bid
    //         if it->first < incoming.price: break
    //         ... match ...
    //         if level.empty(): opposite.erase(it);
    //
    // IMPORTANT: The execution price is ALWAYS resting.price (maker's price).
    // The aggressor crosses the spread and gets the maker's terms.

    // --- Placeholder: no matching, returns empty ---
    (void)incoming;
    return {};

    // TODO(human): Replace with your implementation.
}

bool MatchingEngine::can_fill_completely(const Order& order) const {
    // TODO(human): Check if a FOK order can be entirely filled.
    //
    // Walk the opposite side of the book (same logic as match_against_book)
    // but only COUNT available quantity -- don't modify anything.
    //
    // Return true if total available qty >= order.remaining_qty.
    //
    // Hint: This is a const method -- you're just reading the book.
    //   qty_t available = 0;
    //   For each matchable level on opposite side:
    //     For each order in the level:
    //       available += resting.remaining_qty;
    //       if available >= order.remaining_qty: return true;
    //   return false;

    // --- Placeholder: always returns false ---
    (void)order;
    return false;

    // TODO(human): Replace with your implementation.
}

MatchResult MatchingEngine::submit_order(Order order) {
    // TODO(human): Top-level order submission.
    //
    // Steps:
    //   1. Set order.remaining_qty = order.quantity
    //
    //   2. If order is FOK: check can_fill_completely() first
    //      - If not fillable, return {trades={}, remaining=qty, was_cancelled=true}
    //
    //   3. Call match_against_book(order) to get trades
    //
    //   4. Build the MatchResult:
    //      result.trades = trades from matching
    //      result.filled_qty = order.quantity - order.remaining_qty
    //      result.remaining_qty = order.remaining_qty
    //
    //   5. Handle remainder based on order type:
    //      - LIMIT with remaining > 0: add to book, set result.was_resting = true
    //      - IOC with remaining > 0: set result.was_cancelled = true
    //      - FOK: already handled in step 2
    //
    //   6. Return result

    // --- Placeholder: adds order to book without matching ---
    order.remaining_qty = order.quantity;
    MatchResult result;
    result.remaining_qty = order.remaining_qty;
    if (order.type == OrderType::Limit) {
        book_.add_order(order);
        result.resting = order;
        result.was_resting = true;
    } else {
        result.was_cancelled = true;
    }
    return result;

    // TODO(human): Replace with your implementation.
}

// ── Test harness ────────────────────────────────────────────────────────────

namespace {

uint64_t next_id() {
    static uint64_t id = 1;
    return id++;
}

Order make_order(Side side, double price, qty_t qty, OrderType type = OrderType::Limit) {
    Order o;
    o.order_id = next_id();
    o.side     = side;
    o.price    = price_from_double(price);
    o.quantity = qty;
    o.type     = type;
    return o;
}

void print_trades(const std::vector<Trade>& trades) {
    for (auto& t : trades) {
        std::cout << absl::StrFormat("  Trade #%u: %s %d @ %.4f (aggressor=%u, passive=%u)\n",
            t.trade_id, side_to_string(t.aggressor_side), t.quantity,
            price_to_double(t.price), t.aggressor_id, t.passive_id);
    }
}

void print_result(const MatchResult& r, const char* label) {
    std::cout << absl::StrFormat("\n--- %s ---\n", label);
    std::cout << absl::StrFormat("Filled: %d, Remaining: %d, Resting: %s, Cancelled: %s\n",
        r.filled_qty, r.remaining_qty,
        r.was_resting ? "yes" : "no",
        r.was_cancelled ? "yes" : "no");
    if (!r.trades.empty()) {
        std::cout << "Trades:\n";
        print_trades(r.trades);
    } else {
        std::cout << "No trades generated.\n";
    }
}

}  // namespace

int main() {
    std::cout << "=== Phase 2: Matching Engine ===\n\n";

    MatchingEngine engine;

    // --- Build a resting book ---
    std::cout << "Building resting order book...\n";

    // Asks (sell side) -- these will be matched by incoming buy orders
    engine.submit_order(make_order(Side::Sell, 150.12, 300));
    engine.submit_order(make_order(Side::Sell, 150.11, 500));
    engine.submit_order(make_order(Side::Sell, 150.11, 200));  // Second order at same price

    // Bids (buy side) -- these will be matched by incoming sell orders
    engine.submit_order(make_order(Side::Buy,  150.09, 400));
    engine.submit_order(make_order(Side::Buy,  150.08, 1000));

    std::cout << absl::StrFormat("Book has %d resting orders.\n", engine.book().total_orders());

    // --- Test 1: LIMIT buy that partially fills ---
    // BUY 600 @ 150.11 should match 500+200=700 available at 150.11, but only needs 600.
    // Expected: fill 500 from first resting order, then 100 from second. Remaining 100 rests.
    auto r1 = engine.submit_order(make_order(Side::Buy, 150.11, 600));
    print_result(r1, "LIMIT BUY 600 @ 150.11 (should partially fill two resting sells)");

    // --- Test 2: LIMIT buy that sweeps multiple levels ---
    // BUY 500 @ 150.12 should match remaining 100 at 150.11, then up to 300 at 150.12.
    auto r2 = engine.submit_order(make_order(Side::Buy, 150.12, 500));
    print_result(r2, "LIMIT BUY 500 @ 150.12 (should sweep across price levels)");

    // --- Test 3: IOC order ---
    // IOC SELL: fill what you can immediately, cancel the rest.
    auto r3 = engine.submit_order(make_order(Side::Sell, 150.08, 600, OrderType::IOC));
    print_result(r3, "IOC SELL 600 @ 150.08 (fill against bids, cancel remainder)");

    // --- Test 4: FOK order that can't fill ---
    // FOK BUY 10000 @ 150.20: not enough quantity in the book.
    auto r4 = engine.submit_order(make_order(Side::Buy, 150.20, 10000, OrderType::FOK));
    print_result(r4, "FOK BUY 10000 @ 150.20 (should be entirely cancelled)");

    // --- Test 5: FOK order that CAN fill ---
    // First replenish the book
    engine.submit_order(make_order(Side::Sell, 150.15, 100));
    auto r5 = engine.submit_order(make_order(Side::Buy, 150.15, 100, OrderType::FOK));
    print_result(r5, "FOK BUY 100 @ 150.15 (should fill completely)");

    std::cout << absl::StrFormat("\nTotal trades generated: %u\n", engine.trade_count() - 1);
    std::cout << absl::StrFormat("Remaining orders in book: %d\n", engine.book().total_orders());

    std::cout << "\n=== Phase 2 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - Price-time priority: best price first, then FIFO at same price\n";
    std::cout << "  - Execution price = resting (maker) price, not incoming (taker) price\n";
    std::cout << "  - LIMIT rests on book, IOC cancels remainder, FOK is all-or-nothing\n";
    std::cout << "  - Every fill generates a Trade event (used by clearing, risk, market data)\n";

    return 0;
}
