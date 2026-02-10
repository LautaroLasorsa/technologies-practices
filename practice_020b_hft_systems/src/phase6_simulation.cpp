// =============================================================================
// Phase 6: End-to-End Trading Simulation
//
// Wire everything together into a complete (simplified) trading loop:
//
//   Market Data Generator
//         |
//         v
//   Feed Handler  -->  Order Book
//                        |
//                        v
//                  Signal Generator
//                        |
//                        v
//                  Order Manager  -->  Matching Engine
//                        |
//                        v
//                  Position Tracker / PnL
//
// This is a simulation loop that:
//   1. Generates synthetic market data (random walk with mean reversion)
//   2. Feeds it through the FeedHandler to update the local OrderBook
//   3. Runs SignalGenerator to produce BUY/SELL/HOLD signals
//   4. Uses the OMS to submit orders when signals fire
//   5. Matches orders in the MatchingEngine
//   6. Tracks position and PnL throughout
//
// At the end, prints summary statistics:
//   - Number of ticks processed
//   - Number of trades executed
//   - Final PnL (realized + unrealized)
//   - Signal distribution (how many BUY/SELL/HOLD)
//   - Average latency per tick (simulated)
//
// IN PRODUCTION:
//   - Market data comes from a real exchange feed (Nasdaq ITCH, CME MDP3)
//   - The matching engine is ON the exchange, not in your code
//   - Your code: FeedHandler -> Book -> Signal -> OMS -> send to exchange
//   - Latency budget: ~1-10 microseconds from data arrival to order sent
//   - This simulation compresses what happens across multiple machines
//     into a single process for learning purposes
// =============================================================================

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include "absl/strings/str_format.h"
#include "feed_handler.h"
#include "matching_engine.h"
#include "oms.h"
#include "signal.h"

// ── Synthetic Market Data Generator ─────────────────────────────────────────

// Generates a stream of market data messages simulating a liquid stock.
//
// Model: Geometric Brownian Motion with mean reversion (Ornstein-Uhlenbeck).
//   - Base price: $150
//   - Volatility: ~0.5% per tick (scaled)
//   - Mean reversion: price pulled back toward $150 (prevents drift to infinity)
//   - Random orders arrive at random prices near the mid
//
// This is a simplified version of the models used to backtest trading strategies.
// Real backtesting uses historical tick data (e.g., from Lobster Data, TickData).
class MarketDataGenerator {
public:
    explicit MarketDataGenerator(double base_price = 150.0, uint64_t seed = 42)
        : base_price_(base_price)
        , current_price_(base_price)
        , rng_(seed)
        , price_dist_(0.0, 0.02)          // ~2 cents std dev per tick
        , qty_dist_(50, 500)               // Random quantity 50-500
        , side_dist_(0.0, 1.0)
        , spread_dist_(0.01, 0.05)         // Random spread 1-5 cents
    {}

    // Generate the next batch of market data messages for one "tick".
    //
    // TODO(human): Implement this.
    // Hint:
    //   1. Update current_price_ with mean-reverting random walk:
    //      double noise = price_dist_(rng_);
    //      double reversion = mean_reversion_strength_ * (base_price_ - current_price_);
    //      current_price_ += noise + reversion;
    //
    //   2. Generate a spread:
    //      double half_spread = spread_dist_(rng_) / 2.0;
    //      price_t bid_price = price_from_double(current_price_ - half_spread);
    //      price_t ask_price = price_from_double(current_price_ + half_spread);
    //
    //   3. Create messages:
    //      - Cancel some old orders (simulate churn -- HFT firms replace orders constantly)
    //      - Add new bid and ask orders at/near the current price
    //      - Occasionally generate a trade message
    //
    //   4. Assign sequential sequence numbers
    //
    //   5. Return the vector of messages
    //
    // For simplicity, generate 2-4 messages per tick:
    //   - 1 cancel (of oldest outstanding order, if any)
    //   - 1 new bid order
    //   - 1 new ask order
    //   - Occasionally 1 trade
    std::vector<MarketDataMessage> generate_tick();

    price_t current_mid_price() const { return price_from_double(current_price_); }
    SeqNum current_seq() const { return seq_num_; }

private:
    double base_price_;
    double current_price_;
    double mean_reversion_strength_ = 0.01;

    std::mt19937_64 rng_;
    std::normal_distribution<double> price_dist_;
    std::uniform_int_distribution<qty_t> qty_dist_;
    std::uniform_real_distribution<double> side_dist_;
    std::uniform_real_distribution<double> spread_dist_;

    SeqNum  seq_num_ = 0;
    OrderId next_order_id_ = 1000;
    TradeId next_trade_id_ = 1;

    // Track outstanding orders for cancellation
    std::vector<OrderId> outstanding_order_ids_;
};

std::vector<MarketDataMessage> MarketDataGenerator::generate_tick() {
    // TODO(human): Generate synthetic market data for one tick.
    //
    // Suggested implementation:
    //
    //   std::vector<MarketDataMessage> msgs;
    //
    //   // 1. Price evolution (mean-reverting random walk)
    //   double noise = price_dist_(rng_);
    //   double reversion = mean_reversion_strength_ * (base_price_ - current_price_);
    //   current_price_ += noise + reversion;
    //
    //   // 2. Generate spread
    //   double half_spread = spread_dist_(rng_) / 2.0;
    //   price_t bid_px = price_from_double(current_price_ - half_spread);
    //   price_t ask_px = price_from_double(current_price_ + half_spread);
    //
    //   // 3. Cancel an old order (if any)
    //   if (!outstanding_order_ids_.empty()) {
    //       OrderId cancel_id = outstanding_order_ids_.front();
    //       outstanding_order_ids_.erase(outstanding_order_ids_.begin());
    //       msgs.push_back(MsgCancelOrder{++seq_num_, cancel_id});
    //   }
    //
    //   // 4. Add new bid order
    //   OrderId bid_id = next_order_id_++;
    //   qty_t bid_qty = qty_dist_(rng_);
    //   msgs.push_back(MsgAddOrder{++seq_num_, bid_id, Side::Buy, bid_px, bid_qty});
    //   outstanding_order_ids_.push_back(bid_id);
    //
    //   // 5. Add new ask order
    //   OrderId ask_id = next_order_id_++;
    //   qty_t ask_qty = qty_dist_(rng_);
    //   msgs.push_back(MsgAddOrder{++seq_num_, ask_id, Side::Sell, ask_px, ask_qty});
    //   outstanding_order_ids_.push_back(ask_id);
    //
    //   return msgs;

    // --- Placeholder: generates two simple orders per tick ---
    std::vector<MarketDataMessage> msgs;

    double noise = price_dist_(rng_);
    double reversion = mean_reversion_strength_ * (base_price_ - current_price_);
    current_price_ += noise + reversion;

    price_t bid_px = price_from_double(current_price_ - 0.01);
    price_t ask_px = price_from_double(current_price_ + 0.01);

    OrderId bid_id = next_order_id_++;
    OrderId ask_id = next_order_id_++;

    msgs.push_back(MsgAddOrder{++seq_num_, bid_id, Side::Buy, bid_px, 100});
    msgs.push_back(MsgAddOrder{++seq_num_, ask_id, Side::Sell, ask_px, 100});

    outstanding_order_ids_.push_back(bid_id);
    outstanding_order_ids_.push_back(ask_id);

    return msgs;

    // TODO(human): Replace with the fuller implementation above.
}

// ── Simulation Runner ───────────────────────────────────────────────────────

// Simulation statistics.
struct SimStats {
    int    total_ticks      = 0;
    int    total_trades     = 0;
    int    signals_buy      = 0;
    int    signals_sell     = 0;
    int    signals_hold     = 0;
    int    orders_sent      = 0;
    int    orders_rejected  = 0;
    int    orders_filled    = 0;
    double total_latency_us = 0.0;
};

// Run the full simulation.
//
// TODO(human): Implement the main simulation loop.
// Hint:
//   1. Create all components: generator, feed_handler, signal_gen, oms, engine
//   2. For each tick (e.g., 1000 ticks):
//      a. Start timer (std::chrono::high_resolution_clock)
//      b. Generate market data: generator.generate_tick()
//      c. Feed messages to feed_handler: for (auto& msg : msgs) fh.on_message(msg);
//      d. Get signal: signal_gen.generate(fh.book())
//      e. If signal is BUY/SELL:
//         - Create an order (LIMIT or IOC) at the current mid price
//         - Submit through OMS: oms.send_order(order)
//         - If accepted, match in engine: engine.submit_order(order)
//         - Process fills: for each trade, call oms.on_fill(...)
//      f. Stop timer, accumulate latency
//      g. Update stats
//   3. Print summary
//
// Key decision: in real trading, the matching engine is ON THE EXCHANGE.
// Your code sends the order and waits for fills. Here, for simplicity,
// we run the matching engine locally. The loop structure is the same.
void run_simulation(int num_ticks) {
    std::cout << absl::StrFormat("Running simulation: %d ticks\n\n", num_ticks);

    // --- Component initialization ---
    MarketDataGenerator generator(150.0, 42);

    FeedHandler feed_handler;

    SignalConfig sig_config;
    sig_config.imbalance_depth = 3;
    sig_config.imbalance_threshold = 0.25;
    sig_config.max_spread = price_from_double(0.10);
    SignalGenerator signal_gen(sig_config);

    RiskLimits risk_limits;
    risk_limits.max_order_size = 500;
    risk_limits.max_position = 2000;
    risk_limits.max_notional = static_cast<int64_t>(100000.0 * PRICE_SCALE);
    OrderManager oms(risk_limits);

    MatchingEngine engine;

    SimStats stats;
    uint64_t next_order_id = 1;

    // --- Simulation loop ---
    // TODO(human): Implement the main loop.
    //
    // The structure should be:
    // for (int tick = 0; tick < num_ticks; ++tick) {
    //     auto t_start = std::chrono::high_resolution_clock::now();
    //
    //     // 1. Generate and process market data
    //     auto msgs = generator.generate_tick();
    //     for (auto& msg : msgs) {
    //         feed_handler.on_message(msg);
    //     }
    //
    //     // 2. Generate signal
    //     Signal sig = signal_gen.generate(feed_handler.book());
    //     // Update signal counts...
    //
    //     // 3. Act on signal
    //     if (sig != Signal::Hold) {
    //         Order order;
    //         order.order_id = next_order_id++;
    //         order.side = (sig == Signal::Buy) ? Side::Buy : Side::Sell;
    //         // Use mid price or aggressive price
    //         auto mid = calculate_mid_price(feed_handler.book());
    //         if (mid.has_value()) {
    //             order.price = *mid;
    //             order.quantity = 100;  // Fixed size for simplicity
    //             order.type = OrderType::IOC;  // IOC: don't rest, fill or cancel
    //
    //             auto mo = oms.send_order(order);
    //             if (mo.state != OrderState::Rejected) {
    //                 stats.orders_sent++;
    //                 // Submit to our local matching engine
    //                 auto result = engine.submit_order(order);
    //                 for (auto& trade : result.trades) {
    //                     oms.on_fill(order.order_id, trade.price, trade.quantity);
    //                     stats.total_trades++;
    //                     stats.orders_filled++;
    //                 }
    //             } else {
    //                 stats.orders_rejected++;
    //             }
    //         }
    //     }
    //
    //     // 4. Measure latency
    //     auto t_end = std::chrono::high_resolution_clock::now();
    //     double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
    //     stats.total_latency_us += us;
    //     stats.total_ticks++;
    //
    //     // 5. Periodic progress report (every 100 ticks)
    //     if ((tick + 1) % 100 == 0) {
    //         auto mark = generator.current_mid_price();
    //         std::cout << absl::StrFormat("Tick %4d | Mid: %.4f | Pos: %d | PnL: %.2f\n",
    //             tick + 1, price_to_double(mark), oms.position().position(),
    //             static_cast<double>(oms.position().total_pnl(mark)) / PRICE_SCALE);
    //     }
    // }

    // --- Placeholder loop: processes data but doesn't trade ---
    for (int tick = 0; tick < num_ticks; ++tick) {
        auto t_start = std::chrono::high_resolution_clock::now();

        auto msgs = generator.generate_tick();
        for (auto& msg : msgs) {
            feed_handler.on_message(msg);
        }

        Signal sig = signal_gen.generate(feed_handler.book());
        if (sig == Signal::Buy)  stats.signals_buy++;
        if (sig == Signal::Sell) stats.signals_sell++;
        if (sig == Signal::Hold) stats.signals_hold++;

        // TODO(human): Add order submission and matching here (see hint above)

        auto t_end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t_end - t_start).count();
        stats.total_latency_us += us;
        stats.total_ticks++;
    }

    // --- Print summary ---
    price_t final_mark = generator.current_mid_price();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "=== SIMULATION SUMMARY ===\n";
    std::cout << std::string(60, '=') << "\n\n";

    std::cout << absl::StrFormat("Ticks processed:    %d\n", stats.total_ticks);
    std::cout << absl::StrFormat("Orders sent:        %d\n", stats.orders_sent);
    std::cout << absl::StrFormat("Orders rejected:    %d\n", stats.orders_rejected);
    std::cout << absl::StrFormat("Orders filled:      %d\n", stats.orders_filled);
    std::cout << absl::StrFormat("Total trades:       %d\n", stats.total_trades);
    std::cout << "\n";

    std::cout << absl::StrFormat("Signals: BUY=%d, SELL=%d, HOLD=%d\n",
        stats.signals_buy, stats.signals_sell, stats.signals_hold);
    std::cout << "\n";

    std::cout << absl::StrFormat("Final mid price:    %.4f\n", price_to_double(final_mark));
    std::cout << absl::StrFormat("Final position:     %d\n", oms.position().position());
    std::cout << absl::StrFormat("Realized PnL:       $%.2f\n",
        static_cast<double>(oms.position().realized_pnl()) / PRICE_SCALE);
    std::cout << absl::StrFormat("Unrealized PnL:     $%.2f\n",
        static_cast<double>(oms.position().unrealized_pnl(final_mark)) / PRICE_SCALE);
    std::cout << absl::StrFormat("Total PnL:          $%.2f\n",
        static_cast<double>(oms.position().total_pnl(final_mark)) / PRICE_SCALE);
    std::cout << "\n";

    double avg_latency = stats.total_ticks > 0 ? stats.total_latency_us / stats.total_ticks : 0.0;
    std::cout << absl::StrFormat("Avg latency/tick:   %.2f us\n", avg_latency);
    std::cout << absl::StrFormat("Total time:         %.2f ms\n", stats.total_latency_us / 1000.0);
    std::cout << "\n";

    std::cout << "Book state at end:\n";
    std::cout << absl::StrFormat("  Resting orders: %d\n", feed_handler.book().total_orders());
    std::cout << absl::StrFormat("  Feed gaps:      %d\n", feed_handler.gap_count());
}

int main() {
    std::cout << "=== Phase 6: End-to-End Trading Simulation ===\n\n";

    // Run simulation with 500 ticks (each tick = one market data update cycle).
    // In real HFT, you'd process millions of ticks per day.
    run_simulation(500);

    std::cout << "\n=== Phase 6 Complete ===\n";
    std::cout << "Key takeaways:\n";
    std::cout << "  - The full trading loop: Data -> Book -> Signal -> Risk -> Order -> Fill -> PnL\n";
    std::cout << "  - Each component is independently testable and replaceable\n";
    std::cout << "  - In production: feed handler + signal + OMS run locally;\n";
    std::cout << "    matching engine runs on the exchange\n";
    std::cout << "  - Latency per tick should be single-digit microseconds in production\n";
    std::cout << "  - This simulation compresses a multi-machine architecture into one process\n";

    return 0;
}
