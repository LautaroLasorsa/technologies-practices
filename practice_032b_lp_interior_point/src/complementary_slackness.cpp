// ============================================================================
// Practice 032b — Phase 2: Complementary Slackness & Shadow Prices
// ============================================================================
// Given known optimal primal and dual solutions for a production planning LP,
// verify complementary slackness conditions and interpret dual variables as
// shadow prices (the marginal value of each resource).
//
// Production planning example:
//   A factory makes 2 products (chairs, tables) using 3 resources
//   (labor hours, wood units, machine hours).
//
//   Primal (maximize profit, converted to min for our convention):
//     minimize  -5*chairs - 8*tables
//     subject to:
//       2*chairs + 3*tables <= 120  (labor)     →  -2c -3t >= -120
//       4*chairs + 2*tables <= 100  (wood)      →  -4c -2t >= -100
//       1*chairs + 2*tables <=  80  (machine)   →  -1c -2t >= -80
//       chairs, tables >= 0
//
//   Optimal primal:  chairs = 10, tables = 33.333...
//   Optimal value:   -(5*10 + 8*33.333) = -316.667
//   Optimal dual:    y_labor = 2.333, y_wood = 0, y_machine = 0.667  (maybe not exact, but illustrative)
//   (NOTE: the user will verify these numerically.)
// ============================================================================

#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cmath>
#include <stdexcept>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// ─── Problem data ────────────────────────────────────────────────────────────

struct ProductionLP {
    MatrixXd A;
    VectorXd b;
    VectorXd c;
    std::vector<std::string> product_names;
    std::vector<std::string> resource_names;

    // Known optimal solutions (from solving the LP)
    VectorXd x_star;  // optimal primal: production quantities
    VectorXd y_star;  // optimal dual: shadow prices per resource
};

ProductionLP make_production_problem() {
    ProductionLP lp;

    // Constraint matrix (in >= form: -A_orig x >= -b_orig)
    lp.A.resize(3, 2);
    lp.A << -2, -3,   // labor:   2*chairs + 3*tables <= 120
            -4, -2,   // wood:    4*chairs + 2*tables <= 100
            -1, -2;   // machine: 1*chairs + 2*tables <= 80

    lp.b.resize(3);
    lp.b << -120, -100, -80;

    lp.c.resize(2);
    lp.c << -5, -8;   // minimize -5c - 8t  (= maximize 5c + 8t)

    lp.product_names = {"chairs", "tables"};
    lp.resource_names = {"labor_hours", "wood_units", "machine_hours"};

    // Known optimal primal solution:
    //   From the constraints:
    //     -2c - 3t >= -120  ↔  2c + 3t <= 120
    //     -4c - 2t >= -100  ↔  4c + 2t <= 100
    //     -1c - 2t >= -80   ↔  c + 2t  <= 80
    //
    //   Optimal: c = 6, t = 36 (labor and machine binding, wood slack)
    //     labor:   2*6 + 3*36 = 12 + 108 = 120  ✓ (binding)
    //     wood:    4*6 + 2*36 = 24 + 72  = 96   (slack = 4)
    //     machine: 1*6 + 2*36 = 6 + 72   = 78   ... let me recalculate
    //
    //   Actually, solving properly:
    //     labor binding:   2c + 3t = 120
    //     machine binding: c + 2t  = 80  →  c = 80 - 2t
    //     Substitute: 2(80-2t) + 3t = 120 → 160 - 4t + 3t = 120 → t = 40
    //     Then c = 80 - 80 = 0
    //     Check wood: 4*0 + 2*40 = 80 <= 100 ✓ (slack = 20)
    //     Obj: 5*0 + 8*40 = 320
    //
    //   Or labor + wood binding:
    //     2c + 3t = 120
    //     4c + 2t = 100  →  2c + t = 50  →  t = 50 - 2c
    //     2c + 3(50 - 2c) = 120 → 2c + 150 - 6c = 120 → -4c = -30 → c = 7.5
    //     t = 50 - 15 = 35
    //     Check machine: 7.5 + 70 = 77.5 <= 80 ✓ (slack = 2.5)
    //     Obj: 5*7.5 + 8*35 = 37.5 + 280 = 317.5
    //
    //   Compare: labor+machine → obj=320, labor+wood → obj=317.5
    //   Need the actual optimum. Let's check all vertices:
    //     (0, 0): obj = 0
    //     (0, 40) [labor+machine]: obj = 320
    //     (25, 0) [labor+wood?]: 4*25 = 100 ✓, 25 <= 80 ✓. obj = 125
    //     (7.5, 35) [labor+wood]: obj = 317.5
    //     (0, 40) [c=0, labor binding]: obj = 320
    //
    //   The optimum is c=0, t=40, obj=320.
    //   (labor: 3*40=120 ✓ binding, wood: 2*40=80 <= 100, slack=20, machine: 2*40=80 ✓ binding)
    lp.x_star.resize(2);
    lp.x_star << 0.0, 40.0;

    // Dual: maximize -120*y1 - 100*y2 - 80*y3
    //   s.t.  -2y1 - 4y2 - 1y3 <= -5    (dual constraint for chairs)
    //         -3y1 - 2y2 - 2y3 <= -8    (dual constraint for tables)
    //         y >= 0
    //
    // Equivalently (multiply by -1): maximize -120y1 -100y2 -80y3
    //   s.t. 2y1 + 4y2 + y3 >= 5
    //        3y1 + 2y2 + 2y3 >= 8
    //
    // By complementary slackness:
    //   x_chairs = 0 → dual constraint for chairs can have slack
    //   x_tables = 40 > 0 → dual constraint for tables must be tight: 3y1 + 2y2 + 2y3 = 8
    //   y_wood (y2): wood has slack (96 < 100) → y2 = 0
    //   y_labor (y1): labor is binding → y1 can be > 0
    //   y_machine (y3): machine is binding → y3 can be > 0
    //
    //   From 3y1 + 2*0 + 2y3 = 8 → 3y1 + 2y3 = 8
    //   Dual obj: -120y1 - 80y3 = -320  (must equal primal obj)
    //   120y1 + 80y3 = 320 → 3y1 + 2y3 = 8  (dividing by 40)
    //   This is the same equation! So we have one equation, two unknowns.
    //   Check the chairs dual constraint: 2y1 + y3 >= 5
    //   Since chairs = 0, the constraint can have slack.
    //
    //   We need another condition. Let's parameterize: y3 = (8 - 3y1)/2
    //   For y3 >= 0: y1 <= 8/3
    //   The chairs constraint: 2y1 + (8-3y1)/2 >= 5 → 4y1 + 8 - 3y1 >= 10 → y1 >= 2
    //   So y1 in [2, 8/3]. The dual has multiple optima (degenerate).
    //   Pick y1 = 2, y3 = (8-6)/2 = 1.
    //   Verify: dual obj = 120*2 + 80*1 = 320 ✓
    lp.y_star.resize(3);
    lp.y_star << 2.0, 0.0, 1.0;

    return lp;
}

// ─── Utility: compute slacks ─────────────────────────────────────────────────

/// Compute primal slack: t = Ax - b (for Ax >= b, slack >= 0 if feasible)
VectorXd compute_primal_slack(const MatrixXd& A, const VectorXd& b, const VectorXd& x) {
    return A * x - b;
}

/// Compute dual slack: s = c - A^T y (for A^T y <= c, slack >= 0 if feasible)
VectorXd compute_dual_slack(const MatrixXd& A, const VectorXd& c, const VectorXd& y) {
    return c - A.transpose() * y;
}

// ─── TODO(human) implementations ────────────────────────────────────────────

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Verify complementary slackness conditions                  ║
// ║                                                                         ║
// ║ Complementary slackness (CS) is the bridge between primal and dual      ║
// ║ solutions. At optimality, the following must hold:                       ║
// ║                                                                         ║
// ║   Primal CS:  x_j * s_j = 0  for all j = 1..n                          ║
// ║     where s_j = c_j - (A^T y)_j  is the dual slack (reduced cost)      ║
// ║     Meaning: if variable x_j > 0 (we produce product j), then s_j = 0  ║
// ║     (the product is priced exactly at its marginal cost). If s_j > 0    ║
// ║     (the product would "lose money"), then x_j must be 0.               ║
// ║                                                                         ║
// ║   Dual CS:    y_i * t_i = 0  for all i = 1..m                          ║
// ║     where t_i = (Ax)_i - b_i  is the primal slack                      ║
// ║     Meaning: if constraint i has slack (t_i > 0, resource not fully     ║
// ║     used), then y_i = 0 (that resource's shadow price is zero — an     ║
// ║     extra unit has no value). If y_i > 0 (the resource IS valuable),    ║
// ║     then t_i = 0 (the resource must be fully consumed).                 ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   x — optimal primal solution (n-vector)                                ║
// ║   dual_slack — s = c - A^T y (n-vector, slack in dual constraints)      ║
// ║   y — optimal dual solution (m-vector)                                  ║
// ║   primal_slack — t = Ax - b (m-vector, slack in primal constraints)     ║
// ║   tol — numerical tolerance (use ~1e-6)                                 ║
// ║                                                                         ║
// ║ Steps:                                                                  ║
// ║   1. For each j: check |x(j) * dual_slack(j)| < tol. Print result.     ║
// ║   2. For each i: check |y(i) * primal_slack(i)| < tol. Print result.   ║
// ║   3. Return true only if ALL conditions hold.                           ║
// ║                                                                         ║
// ║ Print a clear line for each condition, e.g.:                            ║
// ║   "  Primal CS[j]:  x_j=40.0 * s_j=0.0 = 0.0  ✓"                     ║
// ║   "  Dual CS[i]:    y_i=2.0  * t_i=0.0 = 0.0  ✓"                     ║
// ║                                                                         ║
// ║ Hint: use std::abs() for the check. Remember that floating-point        ║
// ║ arithmetic may give very small but non-zero products.                   ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
bool check_complementary_slackness(
    const VectorXd& x,
    const VectorXd& dual_slack,
    const VectorXd& y,
    const VectorXd& primal_slack,
    double tol = 1e-6)
{
    (void)x;
    (void)dual_slack;
    (void)y;
    (void)primal_slack;
    (void)tol;
    throw std::runtime_error("TODO(human): implement check_complementary_slackness");
}

// ╔═══════════════════════════════════════════════════════════════════════════╗
// ║ TODO(human): Interpret shadow prices (dual variables)                   ║
// ║                                                                         ║
// ║ Each dual variable y_i is the shadow price of resource i:               ║
// ║   y_i = dZ*/db_i                                                        ║
// ║ meaning "if we had one more unit of resource i, the optimal objective   ║
// ║ would improve by y_i."                                                  ║
// ║                                                                         ║
// ║ For our production problem (maximizing profit):                         ║
// ║   - A shadow price of y=2 for labor means: one extra labor hour would   ║
// ║     let us earn $2 more profit.                                         ║
// ║   - A shadow price of y=0 for wood means: we already have more wood     ║
// ║     than we need; extra wood is worthless.                              ║
// ║                                                                         ║
// ║ Parameters:                                                             ║
// ║   y — optimal dual solution (m-vector)                                  ║
// ║   resource_names — names of each resource                               ║
// ║   primal_slack — slack in each constraint (to identify binding ones)    ║
// ║   is_maximization — if true, interpret in profit terms (our LP is       ║
// ║     actually a maximization disguised as min by negating c)             ║
// ║                                                                         ║
// ║ For each resource i, print:                                             ║
// ║   1. The resource name and its shadow price y_i                         ║
// ║   2. Whether the constraint is binding (slack ~ 0) or has slack         ║
// ║   3. Economic interpretation:                                           ║
// ║      - If y_i > 0: "Scarce resource. Each additional unit improves      ║
// ║        profit by $y_i. Worth acquiring if cost < $y_i per unit."        ║
// ║      - If y_i ≈ 0: "Abundant resource (has slack). Additional units     ║
// ║        have no value at current production levels."                      ║
// ║                                                                         ║
// ║ Hint: since our LP has c negated (min -profit = max profit), the        ║
// ║ shadow prices y_i correspond to the PROFIT increase per unit of         ║
// ║ resource (not cost decrease). When is_maximization is true, multiply    ║
// ║ interpretation values by 1 (shadow prices are already positive for      ║
// ║ binding constraints of a resource that limits profit).                   ║
// ╚═══════════════════════════════════════════════════════════════════════════╝
void interpret_shadow_prices(
    const VectorXd& y,
    const std::vector<std::string>& resource_names,
    const VectorXd& primal_slack,
    bool is_maximization = true)
{
    (void)y;
    (void)resource_names;
    (void)primal_slack;
    (void)is_maximization;
    throw std::runtime_error("TODO(human): implement interpret_shadow_prices");
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "================================================================\n";
    std::cout << "  Phase 2: Complementary Slackness & Shadow Prices\n";
    std::cout << "================================================================\n\n";

    auto prob = make_production_problem();

    // ── Display problem ──────────────────────────────────────────────────
    std::cout << "Production Planning Problem:\n";
    std::cout << "  Products: ";
    for (size_t i = 0; i < prob.product_names.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << prob.product_names[i];
    }
    std::cout << "\n  Resources: ";
    for (size_t i = 0; i < prob.resource_names.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << prob.resource_names[i];
    }
    std::cout << "\n\n";

    // ── Display constraints in original (<=) form ────────────────────────
    std::cout << "Constraints (original form):\n";
    for (int i = 0; i < static_cast<int>(prob.resource_names.size()); ++i) {
        std::cout << "  " << prob.resource_names[i] << ": ";
        for (int j = 0; j < static_cast<int>(prob.product_names.size()); ++j) {
            if (j > 0) std::cout << " + ";
            std::cout << -prob.A(i, j) << "*" << prob.product_names[j];
        }
        std::cout << " <= " << -prob.b(i) << "\n";
    }
    std::cout << "\n";

    // ── Optimal solutions ────────────────────────────────────────────────
    std::cout << "Optimal primal solution (x*):\n";
    for (int j = 0; j < static_cast<int>(prob.product_names.size()); ++j) {
        std::cout << "  " << prob.product_names[j] << " = " << prob.x_star(j) << "\n";
    }
    double primal_obj = prob.c.dot(prob.x_star);
    std::cout << "  Objective (min form): " << primal_obj << "\n";
    std::cout << "  Profit (max form):    " << -primal_obj << "\n\n";

    std::cout << "Optimal dual solution (y*):\n";
    for (int i = 0; i < static_cast<int>(prob.resource_names.size()); ++i) {
        std::cout << "  " << prob.resource_names[i] << " shadow price = " << prob.y_star(i) << "\n";
    }
    double dual_obj = prob.b.dot(prob.y_star);
    std::cout << "  Dual objective: " << dual_obj << "\n";
    std::cout << "  (Should equal primal obj = " << primal_obj << " by strong duality)\n\n";

    // ── Compute slacks ───────────────────────────────────────────────────
    VectorXd primal_slack = compute_primal_slack(prob.A, prob.b, prob.x_star);
    VectorXd dual_slack = compute_dual_slack(prob.A, prob.c, prob.y_star);

    std::cout << "Primal slacks (t = Ax - b, should be >= 0):\n";
    for (int i = 0; i < static_cast<int>(prob.resource_names.size()); ++i) {
        std::cout << "  " << prob.resource_names[i] << ": t = " << primal_slack(i);
        if (std::abs(primal_slack(i)) < 1e-6)
            std::cout << "  (BINDING)";
        else
            std::cout << "  (slack)";
        std::cout << "\n";
    }
    std::cout << "\n";

    std::cout << "Dual slacks (s = c - A^T y, should be >= 0):\n";
    for (int j = 0; j < static_cast<int>(prob.product_names.size()); ++j) {
        std::cout << "  " << prob.product_names[j] << ": s = " << dual_slack(j);
        if (std::abs(dual_slack(j)) < 1e-6)
            std::cout << "  (BINDING)";
        else
            std::cout << "  (slack)";
        std::cout << "\n";
    }
    std::cout << "\n";

    // ── Complementary Slackness ──────────────────────────────────────────
    std::cout << "--- Complementary Slackness Check ---\n\n";
    bool cs_holds = check_complementary_slackness(
        prob.x_star, dual_slack, prob.y_star, primal_slack);

    std::cout << "\nComplementary slackness "
              << (cs_holds ? "HOLDS" : "VIOLATED") << "\n\n";

    // ── Shadow Price Interpretation ──────────────────────────────────────
    std::cout << "--- Shadow Price Interpretation ---\n\n";
    interpret_shadow_prices(prob.y_star, prob.resource_names, primal_slack, true);

    std::cout << "\n================================================================\n";
    std::cout << "  Phase 2 complete.\n";
    std::cout << "================================================================\n";

    return 0;
}
