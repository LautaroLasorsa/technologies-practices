// =============================================================================
// Phase 1: Limit Order Book
//
// The order book is the central data structure in every exchange and trading
// system. It stores all resting (unmatched) orders, organized by price level
// and arrival time.
//
// You'll implement:
//   1. PriceLevel::add_order()    -- FIFO queue management
//   2. PriceLevel::remove_order() -- Cancel by ID within a level
//   3. PriceLevel::total_quantity()
//   4. OrderBook::add_order()     -- Insert into the correct side
//   5. OrderBook::cancel_order()  -- O(1) lookup + removal
//   6. OrderBook::get_l2_snapshot() -- Market data generation
//
// After implementing, run this file. It tests your order book with a
// realistic sequence of orders and prints the L2 snapshot.
//
// REAL-WORLD CONTEXT:
//   Nasdaq processes ~1 billion order book messages per day.
//   The cancel-to-fill ratio is ~30:1 -- meaning for every trade,
//   roughly 30 orders are placed and then cancelled. This is why
//   O(1) cancel performance (via flat_hash_map) is critical.
// =============================================================================

#include <iostream>

#include "absl/strings/str_format.h"
#include "order_book.h"

// ── PriceLevel implementation ───────────────────────────────────────────────

Order& PriceLevel::add_order(Order order) {
    // TODO(human): Add the order to the back of the FIFO queue.
    //
    // Steps:
    //   1. Set order.remaining_qty = order.quantity (full quantity available)
    //   2. Push the order to the back of the deque
    //   3. Return a reference to the newly added order
    //
    // This is straightforward -- the interesting logic is in OrderBook::add_order().

    // --- Placeholder: compiles but doesn't do the right thing ---
    order.remaining_qty = order.quantity;
    orders.push_back(std::move(order));
    return orders.back();

    // TODO(human): Replace the placeholder above with your implementation.
    // The placeholder happens to be correct for this simple case, but make
    // sure you understand WHY: deque::push_back gives FIFO ordering,
    // and returning back() gives the reference to what we just inserted.
}

bool PriceLevel::remove_order(OrderId order_id) {
    // TODO(human): Find and remove the order with the given ID.
    //
    // Steps:
    //   1. Linear scan through the deque to find the order with matching order_id
    //   2. If found, erase it and return true
    //   3. If not found, return false
    //
    // Performance note: This is O(N) where N = orders at this price level.
    // In production, you'd use an intrusive doubly-linked list for O(1) removal.
    // But N is typically small (10-50), so linear scan is fine for learning.
    //
    // Hint: Use std::find_if + deque::erase, or a simple for loop with index.

    // --- Placeholder: always returns false ---
    (void)order_id;
    return false;

    // TODO(human): Replace with your implementation.
}

qty_t PriceLevel::total_quantity() const {
    // TODO(human): Sum the remaining_qty of all orders at this level.
    //
    // This is used for L2 snapshots (aggregate quantity per price level).
    // In production, you'd cache this value and update it incrementally
    // (+= on add, -= on fill/cancel) to avoid the O(N) scan.
    //
    // Hint: std::accumulate or a simple range-for loop.

    // --- Placeholder: returns 0 ---
    return 0;

    // TODO(human): Replace with your implementation.
}

// ── OrderBook implementation ────────────────────────────────────────────────

void OrderBook::add_order(Order order) {
    // TODO(human): Add a resting order to the correct side of the book.
    //
    // Steps:
    //   1. Choose the correct side map: bids_ for Buy, asks_ for Sell
    //   2. Access (or create) the PriceLevel at order.price using operator[]
    //      - If the level is new, set its price field
    //   3. Call PriceLevel::add_order() to insert the order
    //   4. Record the order's location in order_locations_:
    //      order_locations_[order.order_id] = {order.side, order.price}
    //
    // Why std::map and not unordered_map for price levels?
    //   We need sorted iteration: best bid = highest price (rbegin),
    //   best ask = lowest price (begin). std::map gives us this for free.
    //   The number of active price levels is small (~50-100), so O(log N) is fast.

    // --- Placeholder: does nothing ---
    (void)order;

    // TODO(human): Replace with your implementation.
}

bool OrderBook::cancel_order(OrderId order_id) {
    // TODO(human): Remove an order by ID using the O(1) lookup.
    //
    // Steps:
    //   1. Look up order_id in order_locations_. If not found, return false.
    //   2. From the OrderLocation, get the side and price.
    //   3. Find the PriceLevel in the correct side map (bids_ or asks_).
    //   4. Call PriceLevel::remove_order(order_id).
    //   5. If the PriceLevel is now empty, erase it from the side map.
    //      (Empty levels waste memory and confuse best_bid/best_ask queries.)
    //   6. Erase order_id from order_locations_.
    //   7. Return true.
    //
    // This is the key operation that justifies the flat_hash_map:
    // Without it, cancel would require scanning every price level on the
    // correct side to find the order. With the hash map, it's O(1).

    // --- Placeholder: always returns false ---
    (void)order_id;
    return false;

    // TODO(human): Replace with your implementation.
}

std::optional<L2Level> OrderBook::best_bid() const {
    if (bids_.empty()) return std::nullopt;
    // Best bid = highest price = last element in the map (rbegin)
    auto& [price, level] = *bids_.rbegin();
    return L2Level{price, level.total_quantity(), level.order_count()};
}

std::optional<L2Level> OrderBook::best_ask() const {
    if (asks_.empty()) return std::nullopt;
    // Best ask = lowest price = first element in the map (begin)
    auto& [price, level] = *asks_.begin();
    return L2Level{price, level.total_quantity(), level.order_count()};
}

L2Snapshot OrderBook::get_l2_snapshot(int depth) const {
    // TODO(human): Build an L2 snapshot with the top `depth` levels on each side.
    //
    // Steps:
    //   1. For bids: iterate from rbegin() (highest price first)
    //      - For each level, create an L2Level{price, total_quantity(), order_count()}
    //      - Stop after `depth` levels or when you run out of levels
    //   2. For asks: iterate from begin() (lowest price first)
    //      - Same as above
    //   3. Return L2Snapshot{bids, asks}
    //
    // This is what Nasdaq ITCH's "System Event Message" + "Add Order" sequence
    // effectively communicates. CME MDP 3.0 has a dedicated "Market Data Snapshot"
    // message type that looks exactly like this struct.

    // --- Placeholder: returns empty snapshot ---
    (void)depth;
    return L2Snapshot{};

    // TODO(human): Replace with your implementation.
}

// ── Test harness ────────────────────────────────────────────────────────────

namespace {

void print_l2(const L2Snapshot& snap) {
    std::cout << "\n=== L2 Order Book Snapshot ===\n";
    std::cout << absl::StrFormat("%-6s %12s %10s %8s\n", "Side", "Price", "Qty", "Orders");
    std::cout << std::string(40, '-') << "\n";

    // Print asks in reverse (highest first, so it looks like a trading screen)
    for (int i = static_cast<int>(snap.asks.size()) - 1; i >= 0; --i) {
        auto& lvl = snap.asks[i];
        std::cout << absl::StrFormat("%-6s %12.4f %10d %8d\n",
            "ASK", price_to_double(lvl.price), lvl.total_qty, lvl.order_count);
    }
    std::cout << std::string(40, '=') << "  <-- spread\n";
    for (auto& lvl : snap.bids) {
        std::cout << absl::StrFormat("%-6s %12.4f %10d %8d\n",
            "BID", price_to_double(lvl.price), lvl.total_qty, lvl.order_count);
    }
    std::cout << "\n";
}

uint64_t next_order_id() {
    static uint64_t id = 1;
    return id++;
}

Order make_order(Side side, double price, qty_t qty) {
    Order o;
    o.order_id = next_order_id();
    o.side     = side;
    o.price    = price_from_double(price);
    o.quantity = qty;
    o.type     = OrderType::Limit;
    return o;
}

}  // namespace

int main() {
    std::cout << "=== Phase 1: Limit Order Book ===\n\n";

    OrderBook book;

    // --- Build a realistic order book for "AAPL" ---
    // In reality, AAPL has 100+ price levels on each side.
    // We'll build a small but representative book.

    std::cout << "Adding orders to build the book...\n";

    // Bid side (buyers): prices below the spread
    auto b1 = make_order(Side::Buy,  150.10, 500);
    auto b2 = make_order(Side::Buy,  150.10, 300);  // Same price as b1 (tests FIFO)
    auto b3 = make_order(Side::Buy,  150.09, 1200);
    auto b4 = make_order(Side::Buy,  150.08, 400);
    auto b5 = make_order(Side::Buy,  150.07, 800);
    auto b6 = make_order(Side::Buy,  150.06, 200);

    // Ask side (sellers): prices above the spread
    auto a1 = make_order(Side::Sell, 150.11, 600);
    auto a2 = make_order(Side::Sell, 150.11, 200);  // Same price as a1
    auto a3 = make_order(Side::Sell, 150.12, 900);
    auto a4 = make_order(Side::Sell, 150.13, 1500);
    auto a5 = make_order(Side::Sell, 150.14, 350);

    // Add all orders
    for (auto& o : {b1, b2, b3, b4, b5, b6, a1, a2, a3, a4, a5}) {
        book.add_order(o);
    }

    std::cout << absl::StrFormat("Total orders in book: %d\n", book.total_orders());

    // Print top-of-book
    if (auto bid = book.best_bid(); bid.has_value()) {
        std::cout << absl::StrFormat("Best bid: %.4f x %d (%d orders)\n",
            price_to_double(bid->price), bid->total_qty, bid->order_count);
    } else {
        std::cout << "Best bid: (empty)\n";
    }

    if (auto ask = book.best_ask(); ask.has_value()) {
        std::cout << absl::StrFormat("Best ask: %.4f x %d (%d orders)\n",
            price_to_double(ask->price), ask->total_qty, ask->order_count);
    } else {
        std::cout << "Best ask: (empty)\n";
    }

    // Print full L2 snapshot (top 5 levels)
    auto snap = book.get_l2_snapshot(5);
    print_l2(snap);

    // --- Test cancel ---
    std::cout << "Cancelling order b2 (BUY 300 @ 150.10)...\n";
    bool cancelled = book.cancel_order(b2.order_id);
    std::cout << absl::StrFormat("Cancel result: %s\n", cancelled ? "success" : "failed");
    std::cout << absl::StrFormat("Total orders after cancel: %d\n", book.total_orders());

    // Print updated snapshot
    snap = book.get_l2_snapshot(5);
    print_l2(snap);

    // --- Test cancelling a non-existent order ---
    std::cout << "Cancelling non-existent order 99999...\n";
    cancelled = book.cancel_order(99999);
    std::cout << absl::StrFormat("Cancel result: %s (expected: failed)\n",
        cancelled ? "success" : "failed");

    std::cout << "\n=== Phase 1 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - std::map gives sorted price levels (O(log N) insert, O(1) best)\n";
    std::cout << "  - absl::flat_hash_map gives O(1) cancel lookups\n";
    std::cout << "  - FIFO within each price level enforces time priority\n";
    std::cout << "  - L2 snapshots aggregate orders at each price\n";

    return 0;
}
