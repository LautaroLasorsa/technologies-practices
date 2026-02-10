#pragma once

// =============================================================================
// signal.h -- Signal Generation from Order Book Data
//
// Signal generation transforms raw market data into trading decisions.
// This is the "alpha" layer -- the part that tries to predict short-term
// price movements and decide whether to buy, sell, or hold.
//
// SIGNALS IMPLEMENTED HERE:
//
// 1. MID-PRICE:
//    mid = (best_bid + best_ask) / 2
//    The "fair value" of the instrument. Changes in mid-price are the
//    most basic measure of price movement.
//
// 2. SPREAD:
//    spread = best_ask - best_bid
//    Measures liquidity. Tight spread = liquid market. Wide spread = illiquid.
//    In HFT, you typically only trade when spread <= threshold.
//
// 3. BOOK IMBALANCE:
//    imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)
//    Range: [-1.0, +1.0]
//    +1.0 = all liquidity on bid side (strong buying pressure)
//    -1.0 = all liquidity on ask side (strong selling pressure)
//
//    WHY IT'S PREDICTIVE:
//    If there's much more resting quantity on the bid side than the ask side,
//    the next trade is more likely to be a buy (consuming ask liquidity),
//    which pushes the price up. This is the simplest form of order flow
//    prediction and is used (in much more sophisticated forms) by every
//    market-making firm.
//
//    Reference: "Queue-Reactive Models" by Cont, Stoikov, Talreja (2010)
//    quantifies the predictive power of book imbalance.
//
// 4. SIMPLE SIGNAL:
//    Combines imbalance and spread into a BUY/SELL/HOLD decision.
//    This is a toy signal -- real strategies use hundreds of features,
//    ML models, and much more sophisticated logic. But the structure
//    (features -> signal -> order decision) is the same.
//
// REAL-WORLD CONTEXT:
//   - Citadel Securities, Jump Trading, Two Sigma all start from book imbalance
//   - Virtu Financial's 10-K filing mentions "market microstructure signals"
//   - Renaissance Technologies (Medallion Fund) pioneered statistical microstructure signals
// =============================================================================

#include "order_book.h"
#include "types.h"

// ── Signal calculation functions ────────────────────────────────────────────

// Calculate the mid-price: (best_bid + best_ask) / 2
//
// TODO(human): Implement this.
// Hint:
//   - Get best_bid() and best_ask() from the order book
//   - If either side is empty, return std::nullopt (can't compute mid without both sides)
//   - Return (bid.price + ask.price) / 2 as a price_t
//   - Note: integer division truncates. For a more precise mid, you could
//     return a double, but in HFT we keep everything in fixed-point.
//     (bid + ask) / 2 is fine for signal generation.
std::optional<price_t> calculate_mid_price(const OrderBook& book);

// Calculate the bid-ask spread: best_ask - best_bid
//
// TODO(human): Implement this.
// Hint:
//   - Same nullopt handling as mid-price
//   - spread = ask.price - bid.price (always >= 0 in a valid book)
//   - A "locked" market (spread = 0) or "crossed" market (spread < 0) indicates
//     an error or a momentary state that should be matched immediately
std::optional<price_t> calculate_spread(const OrderBook& book);

// Calculate book imbalance at the top `depth` levels.
//
// TODO(human): Implement this.
// Hint:
//   - Get L2 snapshot with given depth
//   - Sum total bid quantity and total ask quantity across all levels
//   - imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty) as a double
//   - If both sides are empty (bid_qty + ask_qty == 0), return 0.0
//   - Range: [-1.0, +1.0]
//
// Think about: why use multiple levels instead of just top-of-book?
// Answer: top-of-book can be easily manipulated (spoofing). Deeper levels
// provide a more robust signal. The SEC fined Navinder Sarao $25M+ for
// spoofing S&P 500 futures by placing large fake orders at deeper levels.
double book_imbalance(const OrderBook& book, int depth);

// ── SignalGenerator ─────────────────────────────────────────────────────────

// Configuration for the signal generator.
struct SignalConfig {
    int     imbalance_depth     = 5;       // Number of levels for imbalance calc
    double  imbalance_threshold = 0.3;     // |imbalance| > this triggers a signal
    price_t max_spread          = 50;      // Don't trade if spread > this (in ticks)
    // In real systems, these would be calibrated per-instrument and adjusted
    // dynamically based on volatility, time of day, etc.
};

// Generates trading signals from order book state.
//
// This is a stateless signal generator -- it looks at the current book state
// and produces a signal. Real generators maintain state (EMA of imbalance,
// recent trade flow, volatility estimates, etc.)
class SignalGenerator {
public:
    explicit SignalGenerator(SignalConfig config = {}) : config_(config) {}

    // Generate a signal from the current order book state.
    //
    // TODO(human): Implement this.
    // Hint:
    //   1. Calculate spread. If spread > max_spread or can't compute -> HOLD
    //   2. Calculate imbalance at configured depth
    //   3. If imbalance > +threshold  -> BUY  (bid pressure, price likely to rise)
    //      If imbalance < -threshold  -> SELL (ask pressure, price likely to fall)
    //      Otherwise                  -> HOLD (no clear signal)
    //
    // In production, you'd also check:
    //   - Volatility regime (don't trade in extreme volatility)
    //   - Time of day (avoid open/close auctions)
    //   - Inventory risk (don't accumulate too much position)
    //   - Recent fill rate (are your orders actually executing?)
    Signal generate(const OrderBook& book);

    const SignalConfig& config() const { return config_; }

private:
    SignalConfig config_;
};
