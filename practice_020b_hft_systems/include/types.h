#pragma once

// =============================================================================
// types.h -- Core domain types for an HFT trading system
//
// DESIGN DECISIONS (real-world rationale):
//
// 1. PRICE AS INTEGER (fixed-point), NOT floating point.
//    Floating point is BANNED in HFT pricing. IEEE 754 cannot represent
//    0.1 exactly: (0.1 + 0.2 != 0.3). In a matching engine processing
//    millions of orders, rounding errors accumulate and cause mismatches
//    between participants. Every real exchange (Nasdaq, CME, LSE) uses
//    integer-based fixed-point prices.
//
//    Convention: price_t stores price * PRICE_SCALE.
//    If PRICE_SCALE = 10000, then $123.4567 is stored as 1234567.
//    This gives 4 decimal places of precision (sufficient for most markets).
//
// 2. STRONG TYPES via enum class.
//    Prevents accidentally passing a Quantity where a Price is expected.
//    Real systems use even stronger typing (tagged types / phantom types).
//
// 3. COMPACT LAYOUT.
//    Order struct is designed to fit in ~64 bytes (one cache line on x86).
//    HFT systems process millions of orders -- cache efficiency matters.
//    Members are ordered to minimize padding (largest alignment first).
//
// REAL-WORLD PARALLELS:
//    - Nasdaq ITCH protocol uses 4-byte integer prices (price * 10000)
//    - CME MDP 3.0 uses 8-byte mantissa + 1-byte exponent (fixed-point)
//    - FIX protocol transmits prices as strings, parsed to fixed-point internally
// =============================================================================

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"

// ── Fixed-point price constants ─────────────────────────────────────────────

// Price is stored as integer ticks. 1 tick = $0.0001 (0.01 cent).
// This matches Nasdaq ITCH's 4-decimal-place convention.
// Max representable price: int64_t max / 10000 ~ $922 trillion. Sufficient.
constexpr int64_t PRICE_SCALE = 10000;

// Helper: convert a double price to fixed-point (use ONLY for test setup, never in hot path)
constexpr int64_t price_from_double(double p) {
    return static_cast<int64_t>(p * PRICE_SCALE + 0.5); // round to nearest tick
}

// Helper: convert fixed-point back to double (use ONLY for display, never for arithmetic)
constexpr double price_to_double(int64_t p) {
    return static_cast<double>(p) / PRICE_SCALE;
}

// ── Type aliases ────────────────────────────────────────────────────────────

// Strong type aliases. In production HFT, these would be tagged/phantom types
// to prevent mixing OrderId with other uint64_t values. For this practice,
// simple aliases suffice.
using OrderId  = uint64_t;
using price_t  = int64_t;   // Fixed-point: actual_price * PRICE_SCALE
using qty_t    = int64_t;   // Quantity in lots (shares, contracts, etc.)
using TradeId  = uint64_t;
using SeqNum   = uint64_t;  // Market data sequence number

// ── Enums ───────────────────────────────────────────────────────────────────

// Side: Buy (bid) or Sell (ask/offer).
// uint8_t for compact storage -- an Order needs only 1 byte for side.
enum class Side : uint8_t {
    Buy,   // Bid side -- wants to purchase
    Sell   // Ask/Offer side -- wants to sell
};

inline const char* side_to_string(Side s) {
    return s == Side::Buy ? "BUY" : "SELL";
}

inline Side opposite_side(Side s) {
    return s == Side::Buy ? Side::Sell : Side::Buy;
}

// Order type determines matching behavior:
//   LIMIT: Rests on the book if not immediately filled (most common, ~60-70% of orders)
//   IOC:   Immediate-Or-Cancel -- fill what you can, cancel the rest (used by aggressive algos)
//   FOK:   Fill-Or-Kill -- fill the ENTIRE quantity or cancel entirely (used for large blocks)
//
// Real exchanges support many more types (Stop, StopLimit, MarketOnClose, Pegged, Iceberg...)
// but these three cover the core matching engine logic.
enum class OrderType : uint8_t {
    Limit,  // Passive: rests on book if unmatched
    IOC,    // Aggressive: fill immediately or cancel remainder
    FOK     // Aggressive: fill entirely or cancel entirely
};

inline const char* order_type_to_string(OrderType t) {
    switch (t) {
        case OrderType::Limit: return "LIMIT";
        case OrderType::IOC:   return "IOC";
        case OrderType::FOK:   return "FOK";
    }
    return "UNKNOWN";
}

// Order state in the lifecycle (used by OMS in Phase 5).
// Transitions:
//   New -> Acknowledged -> (PartiallyFilled)* -> Filled
//   New -> Acknowledged -> (PartiallyFilled)* -> Cancelled
//   New -> Rejected
enum class OrderState : uint8_t {
    New,              // Created locally, not yet sent
    PendingSend,      // Sent to exchange, awaiting ack
    Acknowledged,     // Exchange confirmed receipt
    PartiallyFilled,  // Some quantity executed
    Filled,           // Fully executed (terminal)
    PendingCancel,    // Cancel request sent
    Cancelled,        // Cancelled (terminal)
    Rejected          // Exchange rejected (terminal)
};

inline const char* order_state_to_string(OrderState s) {
    switch (s) {
        case OrderState::New:             return "NEW";
        case OrderState::PendingSend:     return "PENDING_SEND";
        case OrderState::Acknowledged:    return "ACKED";
        case OrderState::PartiallyFilled: return "PARTIAL_FILL";
        case OrderState::Filled:          return "FILLED";
        case OrderState::PendingCancel:   return "PENDING_CANCEL";
        case OrderState::Cancelled:       return "CANCELLED";
        case OrderState::Rejected:        return "REJECTED";
    }
    return "UNKNOWN";
}

// Signal direction for Phase 4.
enum class Signal : uint8_t {
    Buy,
    Sell,
    Hold
};

inline const char* signal_to_string(Signal s) {
    switch (s) {
        case Signal::Buy:  return "BUY";
        case Signal::Sell: return "SELL";
        case Signal::Hold: return "HOLD";
    }
    return "UNKNOWN";
}

// ── Order struct ────────────────────────────────────────────────────────────

// Represents a single order in the book.
//
// LAYOUT: Members ordered by alignment to minimize padding.
// On a 64-bit system with this layout:
//   order_id (8) + price (8) + quantity (8) + remaining_qty (8) +
//   timestamp_ns (8) + side (1) + type (1) + padding (6) = 48 bytes
//
// Fits comfortably in a single cache line (64 bytes), leaving room
// for intrusive list pointers if we add them later.
struct Order {
    OrderId   order_id      = 0;
    price_t   price         = 0;          // Fixed-point price
    qty_t     quantity      = 0;          // Original quantity
    qty_t     remaining_qty = 0;          // Remaining unfilled quantity
    uint64_t  timestamp_ns  = 0;          // Nanosecond timestamp (for time priority)
    Side      side          = Side::Buy;
    OrderType type          = OrderType::Limit;

    // Convenience: is this order fully filled?
    bool is_filled() const { return remaining_qty <= 0; }

    // Convenience: how much has been filled?
    qty_t filled_qty() const { return quantity - remaining_qty; }

    std::string to_string() const {
        return absl::StrFormat("Order{id=%u %s %s px=%d qty=%d/%d}",
            order_id, side_to_string(side), order_type_to_string(type),
            price, remaining_qty, quantity);
    }
};

// ── Trade struct ────────────────────────────────────────────────────────────

// Generated by the matching engine when two orders cross.
//
// In real exchanges, a trade has two sides:
//   - Aggressor (taker): the incoming order that initiated the match
//   - Passive (maker): the resting order already on the book
//
// Exchanges typically charge different fees:
//   - Maker rebate: -$0.002/share (you get PAID for providing liquidity)
//   - Taker fee:    +$0.003/share (you pay for consuming liquidity)
// This maker-taker model incentivizes resting limit orders (liquidity provision).
struct Trade {
    TradeId  trade_id         = 0;
    OrderId  aggressor_id     = 0;   // Incoming order (taker)
    OrderId  passive_id       = 0;   // Resting order (maker)
    price_t  price            = 0;   // Execution price (always the resting order's price)
    qty_t    quantity          = 0;   // Executed quantity
    Side     aggressor_side   = Side::Buy;
    uint64_t timestamp_ns     = 0;

    std::string to_string() const {
        return absl::StrFormat("Trade{id=%u %s aggressor=%u passive=%u px=%d qty=%d}",
            trade_id, side_to_string(aggressor_side),
            aggressor_id, passive_id, price, quantity);
    }
};

// ── L2 Price Level (snapshot) ───────────────────────────────────────────────

// A single level in an L2 (Level 2) market data snapshot.
// L2 shows aggregate quantity at each price level (not individual orders).
// L3 (Level 3) shows individual orders -- only available on some exchanges.
//
// Example L2 snapshot for AAPL:
//   Bid: $150.10 x 500,  $150.09 x 1200,  $150.08 x 300
//   Ask: $150.11 x 800,  $150.12 x 400,   $150.13 x 2000
struct L2Level {
    price_t price       = 0;
    qty_t   total_qty   = 0;   // Aggregate quantity across all orders at this price
    int     order_count = 0;   // Number of orders at this price
};

// Full L2 snapshot with top-of-book on both sides.
struct L2Snapshot {
    std::vector<L2Level> bids;  // Sorted: best (highest) bid first
    std::vector<L2Level> asks;  // Sorted: best (lowest) ask first
};
