/// QuantLib European Option Pricing with Black-Scholes
///
/// This program prices a European call option using the analytic
/// Black-Scholes formula and computes the Greeks (delta, gamma,
/// theta, vega, rho).
///
/// QuantLib architecture (from bottom to top):
///   Quote           -> observable market data (spot price, vol, rates)
///   TermStructure   -> curves built from quotes (yield curve, vol surface)
///   Process         -> stochastic process combining the curves (e.g., GBM)
///   Engine          -> pricing algorithm that uses the process
///   Instrument      -> the derivative being priced; delegates to the engine
///
/// Reference: https://www.implementingquantlib.com/2023/11/black-scholes.html

#include <ql/qldefines.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/exercise.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/settings.hpp>

#include <iostream>
#include <iomanip>

using namespace QuantLib;

// =============================================================================
// Market data setup (fully implemented -- boilerplate)
// =============================================================================

/// Create market data for the Black-Scholes model.
/// Returns handles to: spot quote, risk-free curve, volatility surface.
struct MarketData {
    ext::shared_ptr<SimpleQuote> spot_quote;
    Handle<YieldTermStructure> risk_free_curve;
    Handle<BlackVolTermStructure> vol_surface;
};

MarketData setup_market_data(
    const Date& eval_date,
    double spot_price,
    double risk_free_rate,
    double volatility)
{
    MarketData data;

    // SimpleQuote is an observable: when its value changes, instruments
    // that depend on it are automatically notified to recalculate.
    data.spot_quote = ext::make_shared<SimpleQuote>(spot_price);

    // FlatForward: a flat yield curve at a constant rate.
    // Actual360 is a common day count convention in money markets.
    auto flat_rate = ext::make_shared<FlatForward>(
        eval_date, risk_free_rate, Actual360()
    );
    data.risk_free_curve = Handle<YieldTermStructure>(flat_rate);

    // BlackConstantVol: a flat volatility surface (constant across
    // strikes and maturities). TARGET() is the Trans-European
    // Automated Real-time Gross settlement Express Transfer calendar.
    auto const_vol = ext::make_shared<BlackConstantVol>(
        eval_date, TARGET(), volatility, Actual360()
    );
    data.vol_surface = Handle<BlackVolTermStructure>(const_vol);

    return data;
}

// =============================================================================
// Option pricing -- the core exercise
// =============================================================================

void price_european_option()
{
    std::cout << "=== European Call Option Pricing (Black-Scholes) ===\n\n";

    // --- Configuration ---
    Date eval_date(8, February, 2026);
    Settings::instance().evaluationDate() = eval_date;

    double spot_price    = 100.0;   // Current stock price
    double strike_price  = 105.0;   // Option strike price
    double risk_free_rate = 0.05;   // 5% annual risk-free rate
    double volatility    = 0.20;    // 20% annual volatility
    Date maturity_date(8, August, 2026);  // 6-month option

    std::cout << "Market data:\n"
              << "  Spot:       " << spot_price << "\n"
              << "  Strike:     " << strike_price << "\n"
              << "  Risk-free:  " << risk_free_rate * 100 << "%\n"
              << "  Volatility: " << volatility * 100 << "%\n"
              << "  Maturity:   " << maturity_date << "\n\n";

    // --- Market data setup (boilerplate) ---
    auto market = setup_market_data(eval_date, spot_price, risk_free_rate, volatility);

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches QuantLib's process abstraction and the separation
    // between market data (quotes, curves) and pricing logic (engines). The
    // BlackScholesProcess encapsulates geometric Brownian motion (GBM), the
    // foundation of the Black-Scholes model. Understanding this separation is
    // critical for swapping models (e.g., from Black-Scholes to Heston) without
    // changing instrument definitions.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Create the Black-Scholes process.
    //
    // A BlackScholesProcess models geometric Brownian motion (GBM):
    //   dS/S = r*dt + sigma*dW
    //
    // Constructor:
    //   ext::make_shared<BlackScholesProcess>(
    //       Handle<Quote>(market.spot_quote),  // spot price (observable)
    //       market.risk_free_curve,             // risk-free yield curve
    //       market.vol_surface                  // volatility surface
    //   );
    //
    // Store it in: auto bs_process = ...
    //
    // Reference: https://www.quantlib.org/reference/class_quant_lib_1_1_black_scholes_process.html

    // --- Your code here (create bs_process) ---

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches the VanillaOption structure: separating payoff
    // (what you receive at expiry) from exercise style (when you can exercise).
    // This design enables code reuse: the same pricing engines work for calls
    // and puts, American and European styles, just with different payoff and
    // exercise objects. This is a practical application of the Strategy pattern.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Create the European option instrument.
    //
    // Two components:
    //   1. Payoff: PlainVanillaPayoff(Option::Call, strike_price)
    //      - This defines max(S-K, 0) at expiry
    //   2. Exercise: EuropeanExercise(maturity_date)
    //      - Can only be exercised at maturity
    //
    // Combine them into a VanillaOption:
    //   VanillaOption option(
    //       ext::make_shared<PlainVanillaPayoff>(Option::Call, strike_price),
    //       ext::make_shared<EuropeanExercise>(maturity_date)
    //   );
    //
    // Reference: https://www.quantlib.org/reference/class_quant_lib_1_1_vanilla_option.html

    // --- Your code here (create option) ---

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches the instrument-engine pattern: instruments delegate
    // pricing to engines. AnalyticEuropeanEngine uses closed-form Black-Scholes,
    // which is exact and fast. For American options, you'd swap to a numerical
    // engine (binomial tree, finite differences). This flexibility is why QuantLib
    // separates "what" (instrument) from "how" (engine).
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Set the pricing engine and compute the price.
    //
    // The AnalyticEuropeanEngine uses the closed-form Black-Scholes formula.
    // Attach it to the option:
    //   option.setPricingEngine(
    //       ext::make_shared<AnalyticEuropeanEngine>(bs_process)
    //   );
    //
    // Then compute:
    //   double npv = option.NPV();      // Net Present Value (option price)
    //
    // Print the result:
    //   std::cout << "Option price (NPV): " << npv << "\n";

    // --- Your code here (set engine, compute NPV) ---

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches Greeks computation and interpretation. Greeks are
    // the foundation of risk management: delta for hedging spot exposure, gamma
    // for hedging delta risk, vega for volatility risk. Production trading systems
    // aggregate Greeks across portfolios to compute net exposure and hedge ratios.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Compute and print the Greeks.
    //
    // After setting the engine, VanillaOption exposes Greek accessors:
    //   option.delta()  -- dV/dS: price sensitivity to spot
    //   option.gamma()  -- d2V/dS2: delta sensitivity to spot
    //   option.theta()  -- dV/dt: price sensitivity to time (per day)
    //   option.vega()   -- dV/dsigma: price sensitivity to volatility
    //   option.rho()    -- dV/dr: price sensitivity to interest rate
    //
    // Print them in a formatted table:
    //   std::cout << std::fixed << std::setprecision(6);
    //   std::cout << "  Delta: " << option.delta() << "\n";
    //   std::cout << "  Gamma: " << option.gamma() << "\n";
    //   std::cout << "  Theta: " << option.theta() << "\n";
    //   std::cout << "  Vega:  " << option.vega()  << "\n";
    //   std::cout << "  Rho:   " << option.rho()   << "\n";
    //
    // Interpretation:
    //   - Delta ~0.5 means option price moves ~$0.50 per $1 move in spot
    //   - Positive gamma means delta increases as spot rises (convexity)
    //   - Negative theta means time decay erodes option value daily
    //   - Positive vega means higher vol increases option value

    // --- Your code here (print Greeks) ---

    std::cout << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    try {
        price_european_option();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
