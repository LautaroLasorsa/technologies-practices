// =============================================================================
// Phase 5: Order Management System (OMS)
//
// The OMS sits between the strategy and the exchange. It manages order
// lifecycle, enforces risk limits, and tracks positions/PnL.
//
// You'll implement:
//   1. PositionTracker::on_fill()      -- Update position and realized PnL
//   2. PositionTracker::unrealized_pnl() -- Mark-to-market
//   3. OrderManager::check_risk()      -- Pre-trade risk checks
//   4. OrderManager::send_order()      -- Submit with risk validation
//   5. OrderManager::on_ack/fill/cancel() -- Lifecycle callbacks
//
// WHY RISK CHECKS MATTER:
//   Knight Capital (2012): A software deployment bug activated old test code
//   that flooded the market with erroneous orders. In 45 minutes, Knight
//   lost $440 MILLION. They had no effective kill switch, no position limit
//   that could stop the bleeding. Knight was bankrupt within days.
//
//   Lesson: Pre-trade risk checks are not optional. They are the last line
//   of defense between a bug and financial ruin.
//
// POSITION TRACKING:
//   Convention: positive = long (you own the asset), negative = short.
//
//   Example round-trip:
//     Buy 100 @ $150.10  -> position = +100, cost_basis = $15,010
//     Sell 100 @ $150.20 -> position = 0, realized_pnl = 100 * ($150.20 - $150.10) = $10
//
//   Example partial close:
//     Buy 100 @ $150.10  -> position = +100
//     Sell 50 @ $150.20  -> position = +50, realized_pnl = 50 * $0.10 = $5
//     Market @ $150.15   -> unrealized_pnl = 50 * ($150.15 - $150.10) = $2.50
//     Total PnL = $5 + $2.50 = $7.50
// =============================================================================

#include <iostream>

#include "absl/strings/str_format.h"
#include "oms.h"

// ── PositionTracker implementation ──────────────────────────────────────────

void PositionTracker::on_fill(Side side, price_t price, qty_t quantity) {
    // TODO(human): Update position and calculate realized PnL.
    //
    // This is the trickiest function in the OMS. Sign conventions matter!
    //
    // Convention: BUY increases position, SELL decreases.
    //   signed_qty = (side == Side::Buy) ? +quantity : -quantity;
    //
    // Case 1: ADDING to position (same direction):
    //   If position_ >= 0 and buying, or position_ <= 0 and selling.
    //   Just update position and cost basis:
    //     cost_basis_notional_ += price * quantity;
    //     position_ += signed_qty;
    //
    // Case 2: REDUCING position (opposite direction):
    //   Close part (or all) of the existing position.
    //   close_qty = min(quantity, abs(position_))
    //   avg_cost = cost_basis_notional_ / abs(position_)
    //
    //   If we were long (position_ > 0) and now selling:
    //     realized_pnl_ += close_qty * (price - avg_cost);
    //   If we were short (position_ < 0) and now buying:
    //     realized_pnl_ += close_qty * (avg_cost - price);
    //
    //   Reduce cost basis proportionally:
    //     cost_basis_notional_ -= avg_cost * close_qty;
    //   Update position:
    //     position_ += signed_qty;
    //
    // Case 3: FLIPPING position (reduce past zero):
    //   If quantity > abs(position_), we close the old position entirely
    //   and open a new one in the opposite direction.
    //   Remaining after close: open_qty = quantity - close_qty
    //   New cost basis: cost_basis_notional_ = price * open_qty
    //
    // Hint: Handle Case 2+3 together. After closing, check if we flipped.

    // --- Placeholder: only updates position, no PnL tracking ---
    qty_t signed_qty = (side == Side::Buy) ? quantity : -quantity;
    position_ += signed_qty;
    cost_basis_notional_ += price * quantity;  // wrong for closing trades, but compiles

    // TODO(human): Replace with correct implementation.
}

int64_t PositionTracker::unrealized_pnl(price_t mark_price) const {
    // TODO(human): Calculate unrealized PnL at the given mark price.
    //
    // Steps:
    //   If position_ == 0: return 0 (flat, no unrealized PnL)
    //   avg_cost = cost_basis_notional_ / abs(position_)
    //   If long (position_ > 0):  return position_ * (mark_price - avg_cost)
    //   If short (position_ < 0): return (-position_) * (avg_cost - mark_price)
    //
    // Be careful: all arithmetic is in fixed-point. The result is also in
    // fixed-point "notional" units. To get dollars, divide by PRICE_SCALE.
    //
    // In production, mark_price would be the current mid-price or last trade price.

    // --- Placeholder: returns 0 ---
    (void)mark_price;
    return 0;

    // TODO(human): Replace with your implementation.
}

// ── OrderManager implementation ─────────────────────────────────────────────

bool OrderManager::check_risk(const Order& order) const {
    // TODO(human): Pre-trade risk checks.
    //
    // Check ALL of the following (return false if ANY fails):
    //
    //   1. Max order size:
    //      order.quantity <= limits_.max_order_size
    //      (Prevents fat-finger errors: accidentally typing 1,000,000 instead of 1,000)
    //
    //   2. Max position:
    //      Calculate what the new position would be after this order fills:
    //      qty_t new_pos = position_.position();
    //      new_pos += (order.side == Side::Buy) ? order.quantity : -order.quantity;
    //      abs(new_pos) <= limits_.max_position
    //      (Prevents accumulating too much directional risk)
    //
    //   3. Max notional:
    //      order.price * order.quantity <= limits_.max_notional
    //      (Prevents overexposure in dollar terms -- important for high-priced stocks)
    //
    // Return true if ALL checks pass.
    //
    // In production, you'd also check:
    //   - Rate limits (max orders per second)
    //   - Daily loss limits (stop trading after losing $X)
    //   - Instrument-specific limits
    //   - Circuit breakers (halt if market moves too fast)

    // --- Placeholder: always returns true (no risk checks) ---
    (void)order;
    return true;

    // TODO(human): Replace with your implementation.
}

ManagedOrder OrderManager::send_order(Order order) {
    // TODO(human): Submit an order with risk validation.
    //
    // Steps:
    //   1. Call check_risk(order)
    //   2. If fails: return ManagedOrder with state = Rejected
    //   3. If passes:
    //      - Create ManagedOrder with state = PendingSend
    //      - Store in orders_[order.order_id]
    //      - Return the managed order
    //
    // In production, step 3 would also serialize the order to a FIX message
    // and send it over TCP to the exchange. The state would transition to
    // PendingSend -> Acknowledged when the exchange sends back an ExecutionReport.

    // --- Placeholder: always accepts ---
    ManagedOrder mo;
    mo.order = order;
    mo.state = OrderState::PendingSend;
    orders_[order.order_id] = mo;
    return mo;

    // TODO(human): Replace with your implementation (add risk check).
}

void OrderManager::on_ack(OrderId order_id) {
    // TODO(human): Handle exchange acknowledgment.
    //
    // Steps:
    //   1. Find order in orders_. If not found, log warning and return.
    //   2. If state is not PendingSend, log warning and return (unexpected ack).
    //   3. Update state to Acknowledged.
    //
    // In production, the ack also carries the exchange's internal order ID,
    // timestamp, and potentially a modified price/quantity.

    // --- Placeholder: does nothing ---
    (void)order_id;

    // TODO(human): Replace with your implementation.
}

void OrderManager::on_fill(OrderId order_id, price_t fill_price, qty_t fill_qty) {
    // TODO(human): Handle a fill from the exchange.
    //
    // Steps:
    //   1. Find order in orders_. If not found, log warning and return.
    //   2. Update cumulative fill quantity:
    //      mo.cum_fill_qty += fill_qty
    //   3. Update VWAP (volume-weighted average price):
    //      mo.fill_notional += fill_price * fill_qty
    //      mo.avg_fill_px = mo.fill_notional / mo.cum_fill_qty
    //   4. Update state:
    //      If cum_fill_qty >= order.quantity: state = Filled (terminal)
    //      Else: state = PartiallyFilled
    //   5. Update position tracker:
    //      position_.on_fill(mo.order.side, fill_price, fill_qty)

    // --- Placeholder: only updates position ---
    auto it = orders_.find(order_id);
    if (it == orders_.end()) return;
    auto& mo = it->second;
    position_.on_fill(mo.order.side, fill_price, fill_qty);

    // TODO(human): Add state management and VWAP tracking.
}

void OrderManager::on_cancel(OrderId order_id) {
    // TODO(human): Handle cancel confirmation from exchange.
    //
    // Steps:
    //   1. Find order in orders_. If not found, return.
    //   2. Update state to Cancelled.

    // --- Placeholder: does nothing ---
    (void)order_id;

    // TODO(human): Replace with your implementation.
}

// ── Test harness ────────────────────────────────────────────────────────────

namespace {

void print_position(const PositionTracker& pos, price_t mark) {
    std::cout << absl::StrFormat("  Position: %d | AvgCost: %.4f | Realized: %.4f | Unrealized: %.4f | Total: %.4f\n",
        pos.position(),
        price_to_double(pos.avg_cost()),
        static_cast<double>(pos.realized_pnl()) / PRICE_SCALE,
        static_cast<double>(pos.unrealized_pnl(mark)) / PRICE_SCALE,
        static_cast<double>(pos.total_pnl(mark)) / PRICE_SCALE);
}

void print_order_state(const OrderManager& oms, OrderId id) {
    auto* mo = oms.get_order(id);
    if (mo) {
        std::cout << absl::StrFormat("  Order %u: state=%s, filled=%d/%d, avg_px=%.4f\n",
            id, order_state_to_string(mo->state),
            mo->cum_fill_qty, mo->order.quantity,
            price_to_double(mo->avg_fill_px));
    }
}

}  // namespace

int main() {
    std::cout << "=== Phase 5: Order Management System ===\n\n";

    RiskLimits limits;
    limits.max_order_size = 1000;
    limits.max_position   = 5000;
    limits.max_notional   = static_cast<int64_t>(200000.0 * PRICE_SCALE);  // $200k

    OrderManager oms(limits);
    price_t mark = price_from_double(150.15);

    // --- Test 1: Normal order lifecycle ---
    std::cout << "--- Test 1: Normal order lifecycle ---\n";
    Order o1;
    o1.order_id = 1;
    o1.side = Side::Buy;
    o1.price = price_from_double(150.10);
    o1.quantity = 200;
    o1.type = OrderType::Limit;

    auto mo1 = oms.send_order(o1);
    std::cout << absl::StrFormat("Sent order %u: state=%s\n", o1.order_id, order_state_to_string(mo1.state));

    oms.on_ack(o1.order_id);
    print_order_state(oms, o1.order_id);

    oms.on_fill(o1.order_id, price_from_double(150.10), 100);
    print_order_state(oms, o1.order_id);
    print_position(oms.position(), mark);

    oms.on_fill(o1.order_id, price_from_double(150.10), 100);
    print_order_state(oms, o1.order_id);
    print_position(oms.position(), mark);

    // --- Test 2: Close position (sell what we bought) ---
    std::cout << "\n--- Test 2: Closing position ---\n";
    Order o2;
    o2.order_id = 2;
    o2.side = Side::Sell;
    o2.price = price_from_double(150.20);
    o2.quantity = 200;
    o2.type = OrderType::Limit;

    oms.send_order(o2);
    oms.on_ack(o2.order_id);
    oms.on_fill(o2.order_id, price_from_double(150.20), 200);
    print_order_state(oms, o2.order_id);
    print_position(oms.position(), mark);

    std::cout << "Expected: position=0, realized_pnl=$20 (200 shares * $0.10 profit)\n";

    // --- Test 3: Risk check -- order too large ---
    std::cout << "\n--- Test 3: Risk rejection -- order too large ---\n";
    Order o3;
    o3.order_id = 3;
    o3.side = Side::Buy;
    o3.price = price_from_double(150.00);
    o3.quantity = 5000;  // Exceeds max_order_size of 1000
    o3.type = OrderType::Limit;

    auto mo3 = oms.send_order(o3);
    std::cout << absl::StrFormat("Order %u state: %s (expected: REJECTED)\n",
        o3.order_id, order_state_to_string(mo3.state));

    // --- Test 4: Risk check -- position limit ---
    std::cout << "\n--- Test 4: Risk rejection -- position limit ---\n";
    // Buy near max position first
    Order o4a;
    o4a.order_id = 4;
    o4a.side = Side::Buy;
    o4a.price = price_from_double(150.00);
    o4a.quantity = 900;
    o4a.type = OrderType::Limit;
    oms.send_order(o4a);
    oms.on_ack(o4a.order_id);
    oms.on_fill(o4a.order_id, price_from_double(150.00), 900);
    print_position(oms.position(), mark);

    // Now try to buy more -- should be near the limit
    Order o4b;
    o4b.order_id = 5;
    o4b.side = Side::Buy;
    o4b.price = price_from_double(150.00);
    o4b.quantity = 800;
    o4b.type = OrderType::Limit;

    // This should check: |current_position + 800| <= 5000
    auto mo4b = oms.send_order(o4b);
    std::cout << absl::StrFormat("Order %u state: %s (position would be %d)\n",
        o4b.order_id, order_state_to_string(mo4b.state),
        oms.position().position() + 800);

    // --- Test 5: Cancel ---
    std::cout << "\n--- Test 5: Cancel an active order ---\n";
    Order o5;
    o5.order_id = 6;
    o5.side = Side::Buy;
    o5.price = price_from_double(149.00);
    o5.quantity = 100;
    o5.type = OrderType::Limit;
    oms.send_order(o5);
    oms.on_ack(o5.order_id);
    print_order_state(oms, o5.order_id);

    oms.on_cancel(o5.order_id);
    print_order_state(oms, o5.order_id);

    std::cout << "\n=== Phase 5 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - Risk checks are synchronous and MANDATORY before every order\n";
    std::cout << "  - Position tracking: signed qty, cost basis, realized/unrealized PnL\n";
    std::cout << "  - Order lifecycle: New -> PendingSend -> Acked -> PartialFill -> Filled\n";
    std::cout << "  - Knight Capital lost $440M in 45 min from inadequate risk controls\n";

    return 0;
}
