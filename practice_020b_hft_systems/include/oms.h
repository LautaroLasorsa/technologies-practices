#pragma once

// =============================================================================
// oms.h -- Order Management System (OMS)
//
// The OMS is the gateway between the trading strategy and the exchange.
// It manages the entire lifecycle of orders and enforces risk limits.
//
// ORDER LIFECYCLE:
//   Strategy decides to buy 100 shares at $150.10:
//     1. Strategy calls OMS::send_order(BUY, 150.10, 100)
//     2. OMS runs pre-trade risk checks (position limits, notional limits)
//     3. If checks pass, order state = PendingSend, send to exchange
//     4. Exchange acknowledges: state = Acknowledged
//     5. Partial fill of 50 shares: state = PartiallyFilled, position updated
//     6. Fill of remaining 50: state = Filled, position updated
//   OR:
//     4. Strategy decides to cancel: OMS::cancel_order(id)
//     5. Exchange confirms cancel: state = Cancelled
//
// PRE-TRADE RISK CHECKS:
//   These MUST be synchronous (on the critical path before order submission).
//   Why? If you check asynchronously, the order might reach the exchange
//   before the risk check completes. By then it's too late.
//
//   Real risk checks (simplified here):
//   - Max order size: single order can't exceed N shares (prevents fat-finger errors)
//   - Max position: net position can't exceed N shares (limits directional risk)
//   - Max notional: total value at risk can't exceed $N (limits capital exposure)
//   - Rate limiting: max N orders per second (prevents runaway algos)
//
//   Famous disasters from inadequate risk checks:
//   - Knight Capital (2012): Software bug sent millions of erroneous orders,
//     lost $440M in 45 minutes. No kill switch or position limit caught it.
//   - Flash Crash (2010): Automated selling with no risk limits cascaded.
//
// POSITION TRACKING:
//   - Net position: sum of all fills (positive = long, negative = short)
//   - Realized PnL: profit/loss from completed round-trips
//   - Unrealized PnL: mark-to-market value of open position
//   - Total PnL = Realized + Unrealized
//
// REAL-WORLD CONTEXT:
//   Every trading firm has an OMS. Bloomberg EMSX, FlexTrade, and Fidessa
//   are commercial OMS products. HFT firms build custom OMS in C++ for speed.
// =============================================================================

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "types.h"

// ── Risk Limits Configuration ───────────────────────────────────────────────

struct RiskLimits {
    qty_t   max_order_size   = 1000;       // Max shares per order
    qty_t   max_position     = 5000;       // Max absolute net position
    int64_t max_notional     = 10000000;   // Max notional value (price * qty), in fixed-point
    // In real systems: also max orders/second, max loss per day, circuit breakers, etc.
};

// ── Managed Order ───────────────────────────────────────────────────────────

// An order as tracked by the OMS (augments the raw Order with lifecycle state).
struct ManagedOrder {
    Order      order;
    OrderState state        = OrderState::New;
    qty_t      cum_fill_qty = 0;   // Cumulative filled quantity
    price_t    avg_fill_px  = 0;   // Volume-weighted average fill price (fixed-point)
    // Internal tracking for VWAP calculation
    int64_t    fill_notional = 0;  // Sum of (fill_price * fill_qty) for VWAP
};

// ── Position Tracker ────────────────────────────────────────────────────────

// Tracks net position and PnL for a single instrument.
//
// Uses the "FIFO cost basis" method for realized PnL:
//   - When you buy 100 @ $10 and sell 50 @ $12, realized PnL = 50 * ($12 - $10) = $100
//   - When you're flat (position = 0), all PnL is realized
//
// In production, you'd track per-instrument, per-account, and aggregate.
class PositionTracker {
public:
    // Record a fill.
    //
    // TODO(human): Implement this.
    // Hint:
    //   - If the fill is in the same direction as current position (adding):
    //     position_ += qty (or -= for sell), update cost_basis_
    //   - If the fill is in the opposite direction (reducing/flipping):
    //     Calculate realized PnL on the closed portion
    //     realized_pnl_ += closed_qty * (fill_price - avg_cost) for longs
    //     realized_pnl_ += closed_qty * (avg_cost - fill_price) for shorts
    //     Reduce position, and if it flips, start new cost basis
    //
    // This is the trickiest part of the OMS. Get the sign conventions right!
    // Convention: positive position = long, negative = short.
    void on_fill(Side side, price_t price, qty_t quantity);

    // Mark-to-market: calculate unrealized PnL at a given market price.
    //
    // TODO(human): Implement this.
    // Hint:
    //   - If long (position_ > 0): unrealized = position_ * (mark_price - avg_cost_)
    //   - If short (position_ < 0): unrealized = abs(position_) * (avg_cost_ - mark_price)
    //   - If flat: unrealized = 0
    //
    // avg_cost_ = cost_basis_notional_ / abs(position_) (be careful with division by zero!)
    int64_t unrealized_pnl(price_t mark_price) const;

    // Total PnL = realized + unrealized.
    int64_t total_pnl(price_t mark_price) const {
        return realized_pnl_ + unrealized_pnl(mark_price);
    }

    // Accessors
    qty_t   position() const { return position_; }
    int64_t realized_pnl() const { return realized_pnl_; }

    // Average cost of current position (fixed-point price).
    // Returns 0 if flat.
    price_t avg_cost() const {
        if (position_ == 0) return 0;
        auto abs_pos = position_ > 0 ? position_ : -position_;
        return cost_basis_notional_ / abs_pos;
    }

private:
    qty_t   position_            = 0;   // Net position (+ = long, - = short)
    int64_t cost_basis_notional_ = 0;   // Total cost of current position (price * qty sum)
    int64_t realized_pnl_        = 0;   // Cumulative realized PnL
};

// ── Order Manager ───────────────────────────────────────────────────────────

class OrderManager {
public:
    explicit OrderManager(RiskLimits limits = {}) : limits_(limits) {}

    // Submit a new order (with risk checks).
    //
    // TODO(human): Implement this.
    // Hint:
    //   1. Run pre-trade risk checks (call check_risk())
    //   2. If checks fail, return the order with state = Rejected
    //   3. Create a ManagedOrder with state = PendingSend
    //   4. Store it in orders_ map
    //   5. Return the managed order
    //
    // In production, this would also serialize the order and send it over
    // a network connection to the exchange. Here we just track state.
    ManagedOrder send_order(Order order);

    // Called when exchange acknowledges the order.
    //
    // TODO(human): Implement this.
    // Hint: Find the order in orders_, update state to Acknowledged.
    // If not found or not in PendingSend state, ignore (or log warning).
    void on_ack(OrderId order_id);

    // Called when a fill is received from the exchange.
    //
    // TODO(human): Implement this.
    // Hint:
    //   1. Find the order in orders_
    //   2. Update cum_fill_qty and avg_fill_px (VWAP)
    //   3. Update state: PartiallyFilled or Filled (if cum_fill_qty == quantity)
    //   4. Call position_.on_fill() to update position tracking
    void on_fill(OrderId order_id, price_t fill_price, qty_t fill_qty);

    // Called when exchange confirms a cancel.
    //
    // TODO(human): Implement this.
    // Hint: Find order, update state to Cancelled.
    void on_cancel(OrderId order_id);

    // Pre-trade risk check.
    //
    // TODO(human): Implement this.
    // Hint: Check all of:
    //   1. order.quantity <= limits_.max_order_size
    //   2. abs(position_.position() + signed_qty) <= limits_.max_position
    //      where signed_qty = +qty for BUY, -qty for SELL
    //   3. order.price * order.quantity <= limits_.max_notional
    // Return true if ALL checks pass.
    bool check_risk(const Order& order) const;

    // Accessors
    const PositionTracker& position() const { return position_; }
    PositionTracker& position() { return position_; }
    const RiskLimits& limits() const { return limits_; }

    // Look up a managed order by ID.
    const ManagedOrder* get_order(OrderId id) const {
        auto it = orders_.find(id);
        return it != orders_.end() ? &it->second : nullptr;
    }

    size_t active_order_count() const { return orders_.size(); }

private:
    RiskLimits limits_;
    PositionTracker position_;
    absl::flat_hash_map<OrderId, ManagedOrder> orders_;
};
