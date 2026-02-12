/// QuantLib Fixed-Rate Bond Pricing
///
/// This program prices a fixed-rate bond using a flat yield curve
/// and computes clean price, dirty price, yield, and duration.
///
/// Key concepts:
///   - Clean price:  bond price WITHOUT accrued interest (what you see quoted)
///   - Dirty price:  bond price WITH accrued interest (what you actually pay)
///   - Yield:        the discount rate that makes PV(cash flows) = dirty price
///   - Duration:     sensitivity of price to yield changes (in years)
///
/// Reference: https://github.com/lballabio/QuantLib/blob/master/Examples/Bonds/Bonds.cpp

#include <ql/qldefines.hpp>
#include <ql/instruments/bonds/fixedratebond.hpp>
#include <ql/pricingengines/bond/discountingbondengine.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/calendars/unitedstates.hpp>
#include <ql/time/daycounters/actualactual.hpp>
#include <ql/time/daycounters/thirty360.hpp>
#include <ql/time/schedule.hpp>
#include <ql/settings.hpp>

#include <iostream>
#include <iomanip>
#include <vector>

using namespace QuantLib;

// =============================================================================
// Bond parameters (fully implemented -- boilerplate)
// =============================================================================

struct BondParams {
    Date issue_date;
    Date maturity_date;
    double face_value;       // Par value (e.g., 1000)
    double coupon_rate;      // Annual coupon rate (e.g., 0.05 for 5%)
    int settlement_days;     // Days between trade and settlement
    Frequency coupon_freq;   // How often coupons are paid (Semiannual, Annual)
};

BondParams default_bond_params()
{
    return BondParams{
        .issue_date     = Date(15, January, 2024),
        .maturity_date  = Date(15, January, 2029),
        .face_value     = 1000.0,
        .coupon_rate    = 0.05,    // 5% annual coupon
        .settlement_days = 2,
        .coupon_freq    = Semiannual,
    };
}

// =============================================================================
// Bond pricing -- the core exercise
// =============================================================================

void price_fixed_rate_bond()
{
    std::cout << "=== Fixed-Rate Bond Pricing ===\n\n";

    // --- Evaluation date ---
    Date eval_date(8, February, 2026);
    Settings::instance().evaluationDate() = eval_date;

    auto params = default_bond_params();

    std::cout << "Bond parameters:\n"
              << "  Issue date:       " << params.issue_date << "\n"
              << "  Maturity date:    " << params.maturity_date << "\n"
              << "  Face value:       " << params.face_value << "\n"
              << "  Coupon rate:      " << params.coupon_rate * 100 << "%\n"
              << "  Coupon frequency: Semiannual\n"
              << "  Settlement days:  " << params.settlement_days << "\n\n";

    // --- Calendar and day count conventions (boilerplate) ---
    Calendar calendar = UnitedStates(UnitedStates::GovernmentBond);
    DayCounter day_counter = ActualActual(ActualActual::Bond);

    // --- Coupon schedule (boilerplate) ---
    // Schedule generates the coupon payment dates from issue to maturity.
    Schedule schedule(
        params.issue_date,
        params.maturity_date,
        Period(params.coupon_freq),
        calendar,
        Unadjusted,           // Business day convention for generation
        Unadjusted,           // Business day convention for termination date
        DateGeneration::Backward,  // Generate dates backward from maturity
        false                 // End of month convention
    );

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches yield curve construction, the foundation of all
    // fixed-income pricing. A flat curve is the simplest term structure; production
    // systems use bootstrapped curves (from deposits, futures, swaps) with
    // interpolation (linear, cubic spline, Nelson-Siegel). Understanding that
    // yield curves are live objects (observable via QuantLib's Observer pattern)
    // is key for real-time pricing systems.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Build the flat yield curve.
    //
    // A FlatForward creates a constant-rate discount curve. This is the
    // simplest term structure -- all maturities are discounted at the same rate.
    //
    // Create a yield term structure handle:
    //   double market_yield = 0.045;  // 4.5% market yield (different from coupon!)
    //
    //   auto flat_curve = ext::make_shared<FlatForward>(
    //       eval_date,
    //       market_yield,
    //       day_counter    // ActualActual for government bonds
    //   );
    //
    //   Handle<YieldTermStructure> yield_curve(flat_curve);
    //
    // Think about it: if market_yield < coupon_rate, the bond trades at a
    // premium (above par). If market_yield > coupon_rate, it trades at a
    // discount (below par). Why?
    //
    // Reference: https://www.quantlib.org/reference/class_quant_lib_1_1_flat_forward.html

    // --- Your code here (create yield_curve) ---

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches bond construction: schedules (coupon dates), day
    // count conventions (ActualActual, 30/360), and settlement conventions (T+2).
    // These details matter in production: a wrong day count convention can cause
    // pricing discrepancies of several basis points, leading to P&L breaks.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Create the FixedRateBond instrument.
    //
    // Constructor:
    //   FixedRateBond(
    //       settlement_days,           // int: days between trade and settlement
    //       face_value,                // double: par/face value
    //       schedule,                  // Schedule: coupon dates
    //       std::vector<Rate>{coupon_rate},  // coupon rates (one per period)
    //       day_counter                // DayCounter: for accrual calculation
    //   );
    //
    // Example:
    //   FixedRateBond bond(
    //       params.settlement_days,
    //       params.face_value,
    //       schedule,
    //       std::vector<Rate>{params.coupon_rate},
    //       day_counter
    //   );
    //
    // Reference: https://www.quantlib.org/reference/class_quant_lib_1_1_fixed_rate_bond.html

    // --- Your code here (create bond) ---

    // ── Exercise Context ──────────────────────────────────────────────────
    // This exercise teaches bond analytics: clean vs dirty price (market quoting
    // convention vs actual payment), yield to maturity (inverse calculation from
    // price to yield), and accrued interest (prorated coupon since last payment).
    // Understanding these conventions is essential for fixed-income trading systems:
    // counterparties trade clean prices but settle dirty prices.
    // ──────────────────────────────────────────────────────────────────────

    // TODO(human): Set the pricing engine and compute bond analytics.
    //
    // The DiscountingBondEngine discounts all future cash flows (coupons +
    // principal) using the yield curve to compute the present value.
    //
    // Set the engine:
    //   bond.setPricingEngine(
    //       ext::make_shared<DiscountingBondEngine>(yield_curve)
    //   );
    //
    // Compute and print:
    //   double clean = bond.cleanPrice();   // Price without accrued interest
    //   double dirty = bond.dirtyPrice();   // Price with accrued interest
    //   double accrued = bond.accruedAmount();  // Accrued interest
    //
    //   // Yield from clean price (inverse calculation):
    //   double ytm = bond.yield(
    //       clean,
    //       day_counter,
    //       Compounded,      // Compounding convention
    //       params.coupon_freq  // Compounding frequency
    //   );
    //
    // Print results:
    //   std::cout << std::fixed << std::setprecision(4);
    //   std::cout << "Results:\n";
    //   std::cout << "  Clean price:      " << clean << "\n";
    //   std::cout << "  Dirty price:      " << dirty << "\n";
    //   std::cout << "  Accrued interest: " << accrued << "\n";
    //   std::cout << "  Yield to maturity:" << ytm * 100 << "%\n";
    //
    // Verify: the computed yield should match market_yield (4.5%)
    // since we used a flat curve at that rate.
    //
    // Reference: https://cppforquants.com/a-bonds-pricer-implementation-in-c-with-quantlib/

    // --- Your code here (set engine, compute analytics) ---

    std::cout << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main()
{
    try {
        price_fixed_rate_bond();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
