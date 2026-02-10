// =============================================================================
// Phase 4: Signal Generation
//
// Signals transform raw order book data into trading decisions.
// This is the simplest form of "alpha generation" -- predicting short-term
// price movement from the current state of the book.
//
// You'll implement:
//   1. calculate_mid_price()  -- Fair value estimate
//   2. calculate_spread()     -- Liquidity measure
//   3. book_imbalance()       -- Order flow predictor
//   4. SignalGenerator::generate() -- Combine into BUY/SELL/HOLD
//
// BOOK IMBALANCE INTUITION:
//   If the bid side has 10,000 shares and the ask side has 2,000 shares,
//   there's strong buying pressure (imbalance = +0.67). The next trade is
//   likely to consume ask liquidity, pushing the ask price up. A market
//   maker seeing this signal might:
//     - Move their ask quote higher (anticipating price increase)
//     - Buy ahead of the anticipated move
//     - Reduce their offer size to avoid being picked off
//
//   Empirically, book imbalance at top 5 levels predicts the direction of
//   the next mid-price move with ~55-60% accuracy (depends on instrument
//   and time horizon). That sounds small, but at HFT frequencies with
//   thousands of trades per day, even 1-2% edge compounds enormously.
//
//   Reference: Cont, Stoikov, Talreja (2010) "A Stochastic Model for
//   Order Book Dynamics" -- foundational paper on LOB dynamics.
// =============================================================================

#include <iostream>

#include "absl/strings/str_format.h"
#include "order_book.h"
#include "signal.h"

// ── Signal functions implementation ─────────────────────────────────────────

std::optional<price_t> calculate_mid_price(const OrderBook& book) {
    // TODO(human): Calculate mid-price = (best_bid + best_ask) / 2.
    //
    // Steps:
    //   1. Get best_bid() and best_ask() from the book
    //   2. If either is nullopt, return nullopt (can't compute mid without both sides)
    //   3. Return (bid.price + ask.price) / 2
    //
    // Note: integer division truncates. (10001 + 10002) / 2 = 10001.
    // For more precision, some systems use: mid = (bid + ask + 1) / 2 (round up),
    // or store as double. For signal generation, truncation is fine.
    //
    // In a locked market (bid == ask), mid = bid = ask.
    // In a crossed market (bid > ask), mid is between them -- this shouldn't
    // happen in a correctly functioning matching engine.

    // --- Placeholder: always returns nullopt ---
    (void)book;
    return std::nullopt;

    // TODO(human): Replace with your implementation.
}

std::optional<price_t> calculate_spread(const OrderBook& book) {
    // TODO(human): Calculate spread = best_ask - best_bid.
    //
    // Steps:
    //   1. Get best_bid() and best_ask()
    //   2. If either is nullopt, return nullopt
    //   3. Return ask.price - bid.price (should be >= 0)
    //
    // Spread interpretation (for a $150 stock with PRICE_SCALE=10000):
    //   1 tick  = $0.0001 = 0.07 bps -- extremely liquid (SPY, QQQ)
    //   10 ticks = $0.001  = 0.7 bps  -- very liquid (large caps)
    //   100 ticks = $0.01  = 6.7 bps  -- liquid (most listed stocks)
    //   1000+ ticks         = wide     -- illiquid (small caps, exotic options)
    //
    // Market makers profit from the spread: buy at bid, sell at ask.
    // Tighter spread = more competitive, less profit per trade.

    // --- Placeholder: always returns nullopt ---
    (void)book;
    return std::nullopt;

    // TODO(human): Replace with your implementation.
}

double book_imbalance(const OrderBook& book, int depth) {
    // TODO(human): Calculate order book imbalance at top `depth` levels.
    //
    // Formula: imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    // Range: [-1.0, +1.0]
    //
    // Steps:
    //   1. Get L2 snapshot with given depth: book.get_l2_snapshot(depth)
    //   2. Sum total bid quantity across all levels
    //   3. Sum total ask quantity across all levels
    //   4. If (bid_qty + ask_qty) == 0, return 0.0 (empty book, no signal)
    //   5. Return (bid_qty - ask_qty) / double(bid_qty + ask_qty)
    //
    // Why use multiple levels, not just top-of-book?
    //   Top-of-book is easily manipulated. A spoofer can place a large fake
    //   order at the best bid to create the illusion of buying pressure, then
    //   cancel it before it trades. Using deeper levels makes the signal more
    //   robust. The SEC fined Navinder Sarao ~$25M for this exact strategy
    //   in S&P 500 E-mini futures (contributing to the 2010 Flash Crash).

    // --- Placeholder: returns 0 ---
    (void)book;
    (void)depth;
    return 0.0;

    // TODO(human): Replace with your implementation.
}

Signal SignalGenerator::generate(const OrderBook& book) {
    // TODO(human): Generate a trading signal from book state.
    //
    // Steps:
    //   1. Calculate spread. If can't compute or spread > config_.max_spread:
    //      return Signal::Hold (don't trade in illiquid conditions)
    //
    //   2. Calculate imbalance at config_.imbalance_depth levels
    //
    //   3. Decision:
    //      if imbalance > +config_.imbalance_threshold: return Signal::Buy
    //      if imbalance < -config_.imbalance_threshold: return Signal::Sell
    //      else: return Signal::Hold
    //
    // In production, you'd combine dozens of signals (imbalance, trade flow,
    // volatility, correlation with related instruments, time-of-day effects)
    // into a single alpha score, often using ML models trained on historical
    // data. This simple threshold-based signal is the starting point.

    // --- Placeholder: always returns Hold ---
    (void)book;
    return Signal::Hold;

    // TODO(human): Replace with your implementation.
}

// ── Test harness ────────────────────────────────────────────────────────────

namespace {

uint64_t next_id() {
    static uint64_t id = 1;
    return id++;
}

void add_order(OrderBook& book, Side side, double price, qty_t qty) {
    Order o;
    o.order_id = next_id();
    o.side     = side;
    o.price    = price_from_double(price);
    o.quantity = qty;
    o.type     = OrderType::Limit;
    book.add_order(o);
}

void print_signals(const OrderBook& book, SignalGenerator& gen) {
    auto mid = calculate_mid_price(book);
    auto spread = calculate_spread(book);
    double imb = book_imbalance(book, 5);
    Signal sig = gen.generate(book);

    std::cout << absl::StrFormat("  Mid: %s  Spread: %s  Imbalance: %+.3f  Signal: %s\n",
        mid.has_value() ? absl::StrFormat("%.4f", price_to_double(*mid)) : "N/A",
        spread.has_value() ? absl::StrFormat("%.4f", price_to_double(*spread)) : "N/A",
        imb,
        signal_to_string(sig));
}

}  // namespace

int main() {
    std::cout << "=== Phase 4: Signal Generation ===\n\n";

    SignalGenerator gen(SignalConfig{
        .imbalance_depth = 5,
        .imbalance_threshold = 0.3,
        .max_spread = price_from_double(0.05)  // Max 5 cents spread
    });

    // --- Scenario 1: Balanced book ---
    std::cout << "--- Scenario 1: Balanced book ---\n";
    OrderBook book1;
    add_order(book1, Side::Buy,  150.10, 500);
    add_order(book1, Side::Buy,  150.09, 500);
    add_order(book1, Side::Sell, 150.11, 500);
    add_order(book1, Side::Sell, 150.12, 500);
    print_signals(book1, gen);

    // --- Scenario 2: Heavy bid side (buying pressure) ---
    std::cout << "\n--- Scenario 2: Heavy bid side (buying pressure) ---\n";
    OrderBook book2;
    add_order(book2, Side::Buy,  150.10, 3000);
    add_order(book2, Side::Buy,  150.09, 2000);
    add_order(book2, Side::Buy,  150.08, 1500);
    add_order(book2, Side::Sell, 150.11, 200);
    add_order(book2, Side::Sell, 150.12, 300);
    print_signals(book2, gen);

    // --- Scenario 3: Heavy ask side (selling pressure) ---
    std::cout << "\n--- Scenario 3: Heavy ask side (selling pressure) ---\n";
    OrderBook book3;
    add_order(book3, Side::Buy,  150.10, 100);
    add_order(book3, Side::Buy,  150.09, 200);
    add_order(book3, Side::Sell, 150.11, 3000);
    add_order(book3, Side::Sell, 150.12, 2000);
    add_order(book3, Side::Sell, 150.13, 1500);
    print_signals(book3, gen);

    // --- Scenario 4: Wide spread (illiquid) ---
    std::cout << "\n--- Scenario 4: Wide spread (illiquid, should be HOLD) ---\n";
    OrderBook book4;
    add_order(book4, Side::Buy,  149.50, 1000);  // Bid far from ask
    add_order(book4, Side::Sell, 150.50, 1000);  // 1 dollar spread
    print_signals(book4, gen);

    // --- Scenario 5: One-sided book ---
    std::cout << "\n--- Scenario 5: Only bids (no asks) ---\n";
    OrderBook book5;
    add_order(book5, Side::Buy, 150.10, 1000);
    print_signals(book5, gen);

    // --- Scenario 6: Dynamic -- watch imbalance shift ---
    std::cout << "\n--- Scenario 6: Dynamic -- adding orders and watching signal change ---\n";
    OrderBook book6;
    add_order(book6, Side::Buy,  100.00, 500);
    add_order(book6, Side::Sell, 100.01, 500);
    std::cout << "Initial (balanced):\n";
    print_signals(book6, gen);

    // Large buy order arrives -- imbalance shifts
    add_order(book6, Side::Buy, 100.00, 3000);
    std::cout << "After large buy order:\n";
    print_signals(book6, gen);

    // Large sell order arrives -- imbalance shifts back
    add_order(book6, Side::Sell, 100.01, 5000);
    std::cout << "After large sell order:\n";
    print_signals(book6, gen);

    std::cout << "\n=== Phase 4 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - Mid-price = (best_bid + best_ask) / 2 is the simplest fair value\n";
    std::cout << "  - Book imbalance predicts short-term price direction (~55-60% accuracy)\n";
    std::cout << "  - Don't trade when spread is too wide (illiquid conditions)\n";
    std::cout << "  - Real strategies combine many signals, not just imbalance\n";

    return 0;
}
