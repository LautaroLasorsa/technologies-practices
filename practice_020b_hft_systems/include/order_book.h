#pragma once

// =============================================================================
// order_book.h -- Limit Order Book (LOB)
//
// The order book is THE central data structure in any exchange or trading system.
// It maintains all resting (unmatched) orders organized by price and time.
//
// DATA STRUCTURE DESIGN:
//
//   std::map<price_t, PriceLevel>  -- one per side (bids, asks)
//     |
//     +-- PriceLevel contains a std::deque<Order> (FIFO queue)
//
//   absl::flat_hash_map<OrderId, OrderLocation>  -- O(1) cancel lookup
//
// WHY THIS DESIGN:
//   - std::map gives O(log N) insert/erase and O(1) access to best price
//     (begin() for asks, rbegin() for bids). In practice, N (number of distinct
//     price levels) is small (~20-100 active levels), so log N ~ 5-7.
//
//   - absl::flat_hash_map for cancel lookups: when a cancel arrives, we need
//     to find the order's position instantly. Real exchanges receive millions
//     of cancels/second. O(1) lookup is non-negotiable.
//
//   - std::deque for orders at each level: maintains FIFO (time priority).
//     Real exchanges use intrusive doubly-linked lists for O(1) removal,
//     but deque is simpler and correct for learning.
//
// REAL-WORLD CONTEXT:
//   - Nasdaq processes ~1 billion messages/day through its order book
//   - CME's Globex matching engine handles 100M+ orders/day
//   - The "WK Selph" blog post is the canonical reference for LOB design
//   - Production LOBs use memory pools (from 020a) to avoid allocation
//
// PRICE-TIME PRIORITY (FIFO):
//   Orders are matched in two dimensions:
//   1. PRICE priority: better prices match first (higher bid, lower ask)
//   2. TIME priority: at the same price, earlier orders match first (FIFO)
//   This is used by NYSE, Nasdaq, CME, ICE, and most major exchanges.
//   (Some venues use pro-rata or hybrid priority -- not covered here.)
// =============================================================================

#include <deque>
#include <map>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "types.h"

// ── PriceLevel ──────────────────────────────────────────────────────────────

// All resting orders at a single price point, in FIFO order.
// This is the building block of the order book.
//
// In production:
//   - Use an intrusive linked list (from 020a) for O(1) removal
//   - Pre-allocate from a memory pool to avoid malloc in hot path
//   - Keep aggregate quantity cached (don't recompute on every query)
struct PriceLevel {
    price_t price = 0;
    std::deque<Order> orders;  // FIFO: front() is oldest (highest time priority)

    // Add an order to the back of the queue (lowest time priority at this price).
    // Returns: reference to the added order.
    //
    // TODO(human): Implement this.
    // Hint: push the order to the back of the deque. Update remaining_qty = quantity.
    // This is simple -- the interesting logic is in OrderBook::add_order().
    Order& add_order(Order order);

    // Remove a specific order by ID. Returns true if found and removed.
    //
    // TODO(human): Implement this.
    // Hint: Linear scan through the deque. O(N) where N = orders at this price.
    // In production, an intrusive list makes this O(1). For learning, O(N) is fine
    // because N is typically small (10-50 orders at one price level).
    bool remove_order(OrderId order_id);

    // Total quantity across all orders at this level.
    // Used for L2 snapshots. In production, cache this value and update incrementally.
    //
    // TODO(human): Implement this.
    // Hint: Sum remaining_qty of all orders in the deque.
    qty_t total_quantity() const;

    // Number of orders at this level.
    int order_count() const { return static_cast<int>(orders.size()); }

    // Is this level empty? (all orders cancelled/filled)
    bool empty() const { return orders.empty(); }

    // Peek at the front order (highest time priority).
    Order& front() { return orders.front(); }
    const Order& front() const { return orders.front(); }
};

// ── OrderLocation ───────────────────────────────────────────────────────────

// Tracks where an order lives in the book (for O(1) cancel lookup).
// The flat_hash_map maps OrderId -> OrderLocation.
struct OrderLocation {
    Side    side  = Side::Buy;
    price_t price = 0;
};

// ── OrderBook ───────────────────────────────────────────────────────────────

// The full two-sided order book.
//
// INTERFACE:
//   add_order(order)           -- Insert a new resting order
//   cancel_order(order_id)     -- Remove an order by ID
//   best_bid() / best_ask()    -- Top of book
//   get_l2_snapshot(depth)     -- L2 market data snapshot
//
// INVARIANTS:
//   - Bids are sorted descending (best = highest price = rbegin)
//   - Asks are sorted ascending (best = lowest price = begin)
//   - Every order in the book has an entry in order_locations_
//   - Empty price levels are removed immediately
class OrderBook {
public:
    // ── Core operations ─────────────────────────────────────────────────

    // Add a resting order to the book.
    //
    // TODO(human): Implement this.
    // Hint (step by step):
    //   1. Determine which side map to use (bids_ or asks_) based on order.side
    //   2. If the price level doesn't exist in the map, create it (operator[] auto-creates)
    //   3. Set the PriceLevel's price field if it's new
    //   4. Call PriceLevel::add_order() to insert the order
    //   5. Record the order's location in order_locations_ for O(1) cancel
    //
    // Think about: why do we use std::map and not unordered_map for price levels?
    // Answer: we need sorted iteration (best bid = highest, best ask = lowest).
    void add_order(Order order);

    // Cancel (remove) an order by ID.
    //
    // TODO(human): Implement this.
    // Hint (step by step):
    //   1. Look up the order in order_locations_ (flat_hash_map). If not found, return false.
    //   2. From the location, get the side and price to find the correct PriceLevel.
    //   3. Call PriceLevel::remove_order() to remove it from the FIFO queue.
    //   4. If the PriceLevel is now empty, erase it from the side map (cleanup).
    //   5. Erase the order from order_locations_.
    //   6. Return true.
    //
    // This is why flat_hash_map is critical: cancel is the most frequent operation
    // on an exchange. Nasdaq sees ~30 cancels for every fill.
    bool cancel_order(OrderId order_id);

    // ── Market data queries ─────────────────────────────────────────────

    // Best bid price and quantity (top of book, buy side).
    // Returns nullopt if the bid side is empty.
    std::optional<L2Level> best_bid() const;

    // Best ask price and quantity (top of book, sell side).
    // Returns nullopt if the ask side is empty.
    std::optional<L2Level> best_ask() const;

    // L2 snapshot: top `depth` price levels on each side.
    //
    // TODO(human): Implement this.
    // Hint:
    //   - For bids: iterate from rbegin() (highest price) and take `depth` levels
    //   - For asks: iterate from begin() (lowest price) and take `depth` levels
    //   - For each level: record price, total_quantity(), order_count()
    //   - Return an L2Snapshot with both sides populated
    //
    // This is what market data feeds broadcast to participants.
    // Nasdaq ITCH provides full book depth. CME provides top 10 by default.
    L2Snapshot get_l2_snapshot(int depth) const;

    // ── Accessors (used by matching engine) ─────────────────────────────

    // Direct access to the bid/ask maps (used by MatchingEngine to walk levels).
    // In production, the matching engine and order book are tightly coupled
    // (often the same class). Here we keep them separate for clarity.
    std::map<price_t, PriceLevel>& bids() { return bids_; }
    std::map<price_t, PriceLevel>& asks() { return asks_; }
    const std::map<price_t, PriceLevel>& bids() const { return bids_; }
    const std::map<price_t, PriceLevel>& asks() const { return asks_; }

    // Check if an order exists in the book.
    bool has_order(OrderId id) const { return order_locations_.contains(id); }

    // Total number of orders in the book (both sides).
    size_t total_orders() const { return order_locations_.size(); }

    // Remove the location tracking for an order (called after fill by matching engine).
    void erase_order_location(OrderId id) { order_locations_.erase(id); }

private:
    // Bid side: sorted ascending by price. Best bid = rbegin() (highest price).
    // Using std::map because we need sorted iteration for matching and L2 snapshots.
    std::map<price_t, PriceLevel> bids_;

    // Ask side: sorted ascending by price. Best ask = begin() (lowest price).
    std::map<price_t, PriceLevel> asks_;

    // O(1) order lookup for cancels.
    // Maps OrderId -> {side, price} so we can find the order's PriceLevel instantly.
    // This is the key insight from the WK Selph article:
    //   "Use a hash map from order ID to order location for O(1) cancel."
    absl::flat_hash_map<OrderId, OrderLocation> order_locations_;
};
