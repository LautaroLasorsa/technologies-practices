# Practice 046: Robust Optimization

## Technologies

- **CVXPY** — Domain-specific language for convex optimization embedded in Python. Formulates robust counterparts as conic programs (LP, SOCP, SDP) with automatic DCP verification.
- **NumPy** — Numerical arrays and linear algebra for data generation and Monte Carlo evaluation.
- **SciPy** — Matrix square root (`sqrtm`), statistical distributions for uncertainty set calibration.
- **matplotlib** — Visualization of efficient frontiers, price of robustness curves, and out-of-sample performance.
- **Python 3.12+** — Runtime with `uv` for dependency management.

## Stack

- Python 3.12+
- CVXPY >= 1.4 (modeling layer + CLARABEL/SCS bundled)
- NumPy >= 1.26, SciPy >= 1.12
- matplotlib >= 3.8
- uv (package manager)

## Theoretical Context

### What Robust Optimization Is and the Problem It Solves

**Robust Optimization (RO)** is a methodology for optimization under uncertainty where the uncertain parameters are modeled as belonging to a deterministic **uncertainty set** rather than following a known probability distribution. The decision maker seeks a solution that is feasible and optimal for the **worst-case realization** of the uncertain parameters within that set.

The central problem RO addresses: real-world optimization data is never exact. Constraint coefficients, objective coefficients, and right-hand sides are estimated from data, forecasts, or expert judgment. A solution that is optimal for nominal data may become infeasible or severely suboptimal when the true data deviates even slightly. RO provides **guaranteed feasibility** for all realizations within the uncertainty set, at the cost of a controlled reduction in objective value (the "price of robustness").

The general robust optimization problem takes the form:

```
minimize    f(x)
subject to  g(x, u) <= 0   for ALL u in U
```

where `U` is the uncertainty set. The challenge is that "for all u in U" is an infinite set of constraints. The key insight of modern RO (Ben-Tal & Nemirovski, 1998) is that for specific uncertainty set geometries and problem classes, this semi-infinite program can be reformulated as a **tractable finite-dimensional convex program** — the **robust counterpart**.

### Uncertainty Sets: The Core Design Choice

The geometry of the uncertainty set determines both the conservatism and the computational complexity of the robust counterpart. Three fundamental uncertainty set types are used in practice:

**Box (Interval) Uncertainty — L-infinity ball:**
```
U_box = { u : |u_j| <= 1 for all j }
```
Each uncertain parameter varies independently within an interval. The robust counterpart of an LP under box uncertainty remains an LP. This is the simplest and most conservative geometry: it assumes all parameters can deviate simultaneously at their worst-case values. For a constraint `a^T x <= b` with `a_j in [a_nom_j - delta_j, a_nom_j + delta_j]`, the worst case is `(a_nom + delta * sign(x))^T x <= b`, which for non-negative `x` simplifies to `(a_nom + delta)^T x <= b`.

**Ellipsoidal Uncertainty — L2-norm ball:**
```
U_ell = { u : ||u||_2 <= rho }
```
The perturbation vector has bounded Euclidean norm. This is less conservative than box because it constrains the total "energy" of the perturbation — not all parameters can be at their worst simultaneously. The robust counterpart of an LP under ellipsoidal uncertainty becomes a **Second-Order Cone Program (SOCP)**, because the worst-case perturbation introduces a norm term via the Cauchy-Schwarz inequality. Specifically, the robust version of `a^T x <= b` with `a = a_nom + D u`, `||u|| <= rho` becomes:
```
a_nom^T x + rho * ||D x||_2 <= b
```
This is the approach introduced by [Ben-Tal and Nemirovski (1998)](https://www2.isye.gatech.edu/~nemirovs/stablpn.pdf) in their foundational work on tractable robust counterparts.

**Polyhedral (Budget) Uncertainty — Bertsimas-Sim:**
```
U_budget = { u : |u_j| <= 1, sum_j |u_j| <= Gamma }
```
Introduced by [Bertsimas and Sim (2004)](https://www.robustopt.com/references/Price%20of%20Robustness.pdf) in "The Price of Robustness," this set adds a **budget constraint** limiting how many parameters can deviate simultaneously. The parameter `Gamma` (0 to n) directly controls the tradeoff: `Gamma = 0` recovers the nominal problem; `Gamma = n` recovers full box uncertainty. The crucial property: the robust counterpart of an LP under budget uncertainty **remains an LP** (with additional dual variables), preserving computational tractability. Bertsimas and Sim also proved probabilistic guarantees: if each uncertain coefficient is independent and symmetric around its nominal value, the probability of constraint violation is bounded by `exp(-Gamma^2 / 2n)`, providing a principled way to choose `Gamma` based on a desired confidence level.

### The Robust Counterpart Methodology

The robust counterpart approach works by converting the semi-infinite constraint into a finite one through duality:

1. **Start** with a nominal constraint that must hold under uncertainty: `g(x, u) <= 0 for all u in U`
2. **Identify** the worst-case inner problem: `max_{u in U} g(x, u)`
3. **Apply duality** (or Cauchy-Schwarz for ellipsoidal sets) to the inner maximization
4. **Obtain** a finite set of constraints equivalent to the semi-infinite one

The problem class of the robust counterpart depends on the uncertainty set geometry:

| Nominal Problem | Uncertainty Set | Robust Counterpart |
|----------------|----------------|-------------------|
| LP | Box (interval) | LP |
| LP | Ellipsoidal (L2 ball) | SOCP |
| LP | Polyhedral (budget) | LP (with extra variables) |
| QP | Box | QP |
| QP | Ellipsoidal | SOCP |
| SOCP | Ellipsoidal | SDP |

### Price of Robustness

The **price of robustness** quantifies the degradation in objective value caused by protecting against uncertainty:

```
Price = (Nominal_Obj - Robust_Obj) / |Nominal_Obj| * 100%
```

For a maximization problem, the robust solution has a lower objective (you sacrifice profit for protection). For minimization, the robust solution has a higher objective (you pay more for guaranteed feasibility). The price of robustness is monotonically increasing in the size of the uncertainty set: larger sets require more conservative solutions but provide stronger guarantees.

Bertsimas and Sim showed that the budget parameter `Gamma` provides fine-grained control over this tradeoff. In practice, even moderate values of `Gamma` (much smaller than `n`) provide high probabilistic protection, so the price of robustness can be kept small while achieving strong guarantees.

### Distributionally Robust Optimization (DRO)

**DRO** bridges the gap between stochastic programming (assumes a fully known probability distribution) and classical robust optimization (assumes no distributional information, only a support set). DRO works with an **ambiguity set** of probability distributions — a family of distributions "close" to the empirical distribution — and optimizes against the **worst-case distribution** in this family:

```
minimize_x  sup_{P in A}  E_P[h(x, xi)]
```

where `A` is the ambiguity set and `h(x, xi)` is the cost function under uncertainty `xi`.

Two main types of ambiguity sets:

**Moment-based ambiguity sets** constrain the mean and covariance of the distribution:
```
A_moment = { P : E[xi] in M,  E[xi xi^T] <= S }
```
[Delage and Ye (2010)](https://pubsonline.informs.org/doi/10.1287/opre.1090.0741) showed that for a wide range of cost functions, the min-max problem over moment-based ambiguity sets can be reformulated as a tractable semidefinite program (SDP). The classical [Scarf (1958)](https://www.researchgate.net/publication/311153932_A_Min-Max_Solution_of_an_Inventory_Problem) newsvendor result is a special case: given only the mean and variance of demand, the worst-case expected cost is achieved by a two-point distribution, and the optimal order quantity has a closed-form expression (Scarf's bound).

**Wasserstein ambiguity sets** constrain the Wasserstein distance from the empirical distribution:
```
A_wass = { P : W_p(P, P_hat_N) <= epsilon }
```
[Mohajerin Esfahani and Kuhn (2018)](https://link.springer.com/article/10.1007/s10107-017-1172-1) proved that for piecewise-linear convex loss functions, the worst-case expectation over a type-1 Wasserstein ball has a tractable finite convex reformulation. The key advantage: Wasserstein balls contain both discrete and continuous distributions, and the radius `epsilon` has a natural statistical interpretation (it controls the confidence level for out-of-sample performance). As `epsilon -> 0`, Wasserstein DRO recovers SAA; as `epsilon -> infinity`, it converges to worst-case robust optimization.

For a loss function that is L-Lipschitz in the uncertain parameter, the Wasserstein DRO adds a regularization term `L * epsilon` to the SAA objective, providing a clean interpretation as **distributional regularization**.

### Adjustable Robust Optimization (ARO)

In **static** robust optimization, all decisions are made before uncertainty is revealed. In **adjustable** (adaptive) robust optimization, some decisions are **wait-and-see**: they can be made after observing (part of) the uncertainty. This models two-stage problems where:
- **Stage 1** (here-and-now): Decide `x` before uncertainty is revealed
- **Stage 2** (wait-and-see): After observing uncertain parameter `u`, decide `y(u)`

The general ARO problem is:
```
minimize_x  c^T x + max_{u in U} min_{y(u)}  q^T y(u)
subject to  Ax + By(u) >= d(u)   for all u in U
            x >= 0, y(u) >= 0    for all u in U
```

The fully adjustable problem (where `y(u)` can be any function of `u`) is computationally intractable (NP-hard even for LPs). [Ben-Tal, Goryashko, Guslitzer, and Nemirovski (2004)](https://www2.isye.gatech.edu/~nemirovs/MP_Elana_2004.pdf) proposed the **Affinely Adjustable Robust Counterpart (AARC)**: restrict the adjustable variables to be **affine functions** of the uncertain data:

```
y(u) = Y0 + Y1 * u
```

where `Y0` is a constant vector and `Y1` is a sensitivity matrix. Under this restriction, the AARC of a fixed-recourse LP with a polyhedral uncertainty set is a tractable LP or SOCP. The affine decision rule always gives a better (or equal) worst-case objective than the static solution, because static is a special case (`Y1 = 0`).

The quality of the affine approximation depends on the problem structure. For many practical problems (inventory, production planning, network design), affine rules capture most of the benefit of full adaptivity. Extensions to piecewise affine and nonlinear decision rules can further close the gap but at higher computational cost.

### Comparison with Stochastic Programming

| Aspect | Stochastic Programming | Robust Optimization | DRO |
|--------|----------------------|--------------------|----|
| **Uncertainty model** | Known probability distribution | Deterministic uncertainty set | Ambiguity set of distributions |
| **Objective** | Expected cost | Worst-case cost | Worst-case expected cost |
| **Key assumption** | Distribution is correct | Uncertainty set is correct | Ambiguity set is correct |
| **Conservatism** | Low (trusts distribution) | High (worst-case) | Medium (between the two) |
| **Data requirement** | Full distributional knowledge | Only support/bounds | Partial distributional info |
| **Computational** | Scenario-dependent | Set geometry-dependent | Ambiguity set-dependent |
| **Out-of-sample** | Can fail if distribution is wrong | Guaranteed within set | Guaranteed within ambiguity set |

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Uncertainty set** | Deterministic set `U` of possible values for uncertain parameters; the decision must be feasible for all `u in U` |
| **Robust counterpart** | Tractable finite-dimensional reformulation of the semi-infinite robust problem |
| **Price of robustness** | Relative loss in objective value due to protection against uncertainty |
| **Box uncertainty** | Each parameter varies independently in an interval; worst case: all deviate simultaneously |
| **Ellipsoidal uncertainty** | Perturbation vector has bounded L2 norm; less conservative than box; leads to SOCP |
| **Budget uncertainty** | Bertsimas-Sim: L-infinity + L1 budget constraint; controls how many parameters deviate; stays LP |
| **Gamma (budget)** | Parameter in `[0, n]` controlling tradeoff between nominal (`Gamma=0`) and full box (`Gamma=n`) |
| **DRO** | Distributionally Robust Optimization — worst-case over a family of probability distributions |
| **Ambiguity set** | Family of distributions considered plausible (moment-based, Wasserstein, etc.) |
| **Wasserstein distance** | Metric on probability distributions measuring the "cost of transporting" one distribution to another |
| **Moment-based DRO** | Ambiguity set defined by constraints on mean and covariance |
| **Scarf's bound** | Worst-case `E[max(x-d, 0)]` over all distributions with given mean and variance (1958) |
| **Adjustable robust** | Wait-and-see decisions can depend on realized uncertainty |
| **Affine decision rule** | Restriction `y(u) = Y0 + Y1 u` that makes ARO tractable |
| **Static robust** | All decisions made before uncertainty is revealed (special case: `Y1 = 0`) |
| **SAA** | Sample Average Approximation — optimize over empirical distribution directly |

## Description

Formulate and solve robust optimization problems using CVXPY. Progress from uncertainty sets and robust LP counterparts (Phase 1) through robust portfolio optimization (Phase 2) to distributionally robust optimization (Phase 3) and adjustable robust optimization with affine decision rules (Phase 4). Each phase demonstrates how different uncertainty models lead to different problem classes (LP, SOCP, SDP) and different tradeoffs between conservatism and optimality.

### What you'll build

1. **Robust LP with uncertainty sets** — Box, ellipsoidal, and budget uncertainty for a production planning LP. Quantify the price of robustness and validate with Monte Carlo feasibility checks.
2. **Robust portfolio optimization** — Box and ellipsoidal uncertainty on expected returns. Compare nominal Markowitz, robust portfolios, efficient frontiers, and out-of-sample performance.
3. **Distributionally robust optimization** — Moment-based and Wasserstein DRO for the newsvendor problem. Compare SAA, classical robust, and DRO methods on out-of-sample cost.
4. **Adjustable robust optimization** — Static vs affine decision rules for two-stage production planning. Demonstrate how adaptivity reduces worst-case cost.

## Instructions

### Phase 1: Uncertainty Sets & Robust LP (~30 min)

**File:** `src/phase1_robust_lp.py`

This phase teaches the foundational robust optimization workflow: starting from a nominal LP, you derive and implement the robust counterpart under three different uncertainty set geometries. The key learning is how uncertainty set shape determines both the conservatism and the computational class of the robust counterpart.

**What you implement:**

- **`solve_robust_box(data)`** — Robust counterpart under box (interval) uncertainty. The worst case for non-negative variables is simply replacing `A_nom` with `A_nom + delta`. The robust counterpart stays LP. This teaches the simplest robust counterpart derivation: the adversary maximizes `a^T x` over independent intervals.

- **`solve_robust_ellipsoidal(data, rho)`** — Robust counterpart under ellipsoidal uncertainty. The worst case uses Cauchy-Schwarz: `max_{||u|| <= rho} u^T (D x) = rho * ||D x||_2`. The robust constraint becomes `a_nom^T x + rho * ||D x||_2 <= b`, an SOCP constraint. This teaches the Ben-Tal & Nemirovski insight that ellipsoidal uncertainty lifts LP to SOCP.

- **`solve_robust_budget(data, Gamma)`** — Robust counterpart under Bertsimas-Sim budget uncertainty. The inner maximization is dualized using LP duality, introducing auxiliary variables `z` (dual of budget constraint) and `q` (dual of box constraints). The robust counterpart remains LP with extra variables. This teaches how budget uncertainty interpolates between nominal and box via `Gamma`.

**Why it matters:** These three uncertainty sets are the building blocks of all robust optimization. Understanding how the set geometry maps to problem class (LP -> LP, LP -> SOCP) is essential for formulating tractable robust models. The Monte Carlo feasibility check validates that robust solutions are indeed protected against perturbations.

### Phase 2: Robust Portfolio Optimization (~30 min)

**File:** `src/phase2_robust_portfolio.py`

This phase applies robust optimization to a finance problem where the uncertain parameter is the expected return vector `mu`. You build on Phase 1's uncertainty set concepts but now in a quadratic programming context (Markowitz portfolio optimization).

**What you implement:**

- **`solve_robust_box_portfolio(mu_hat, Sigma, delta, target_return)`** — Box uncertainty on expected returns. The worst-case return for a long-only portfolio is `(mu_hat - delta)^T w`. The robust formulation is a QP (same class as nominal Markowitz, just with a penalized return estimate). This teaches how robustness penalizes assets with high estimation uncertainty.

- **`solve_robust_ellipsoidal_portfolio(mu_hat, Sigma, S, kappa, target_return)`** — Ellipsoidal uncertainty on expected returns, accounting for correlation in estimation errors. The worst-case return is `mu_hat^T w - kappa * ||S^{1/2} w||_2` (via Cauchy-Schwarz). The robust return constraint is an SOCP constraint. This teaches the ellipsoidal model's advantage: it captures correlated estimation errors, making it less conservative than box when correlations are present.

**Why it matters:** Portfolio optimization is the canonical application of robust optimization in finance. The out-of-sample simulation demonstrates a key empirical finding: robust portfolios often have *better* out-of-sample Sharpe ratios than nominal portfolios because they avoid overfitting to noisy return estimates.

### Phase 3: Distributionally Robust Optimization (~30 min)

**File:** `src/phase3_dro.py`

This phase introduces DRO via the newsvendor problem — a single-product inventory problem where you must order before demand is known. DRO provides a middle ground between SAA (trusts the empirical distribution exactly) and classical robust (only uses support information).

**What you implement:**

- **`solve_newsvendor_moment_dro(data, gamma1, gamma2)`** — Moment-based DRO using Scarf's bound. Given bounds on the mean and second moment, the worst-case expected overage cost is computed via the classical Chebyshev-type inequality. Uses CVXPY with SOCP constraints (Scarf's bound involves a norm). This teaches how partial distributional information (moments only) still provides protection against distribution misspecification.

- **`solve_newsvendor_wasserstein_dro(data, epsilon)`** — Wasserstein DRO. For the newsvendor with a Lipschitz cost function, the worst-case expected cost over a Wasserstein ball simplifies to `SAA_cost + L * epsilon` where `L` is the Lipschitz constant. The epsilon parameter acts as a distributional regularizer. This teaches the elegant connection between Wasserstein DRO and regularization.

**Why it matters:** DRO is the state-of-the-art approach for data-driven optimization under uncertainty. Moment-based DRO requires only summary statistics; Wasserstein DRO works directly with empirical samples and provides finite-sample performance guarantees. The newsvendor comparison shows that DRO achieves better out-of-sample cost than both SAA and classical robust.

### Phase 4: Adjustable Robust Optimization (~30 min)

**File:** `src/phase4_adjustable_robust.py`

This phase introduces two-stage optimization under uncertainty, where some decisions can adapt to the realized uncertainty. The static robust approach (all decisions before uncertainty) is compared with affine decision rules (production adapts linearly to observed demand).

**What you implement:**

- **`solve_static_robust(data)`** — Static robust formulation where both raw material purchase `x` and production `y` are fixed before demand is revealed. Uses LP duality to reformulate the worst-case penalty over Bertsimas-Sim uncertainty. This teaches the limitation of static robust: the inability to adapt production to demand leads to excessive conservatism.

- **`solve_affine_adjustable(data)`** — Affine decision rule `y(d) = Y0 + Y1 @ d` where production adapts linearly to observed demand. Robustified against all vertices of the uncertainty set. This teaches the AARC methodology: restricting adjustable variables to affine functions makes the problem tractable while capturing most of the benefit of full adaptivity.

**Why it matters:** Many real-world problems are multi-stage: some decisions must be made now, others can wait. Adjustable robust optimization is the standard framework for these problems. The comparison between static and affine solutions demonstrates the **value of adaptivity** — the percentage improvement in worst-case cost from allowing decisions to respond to observed data.

## Motivation

Robust optimization is essential for any optimization practitioner working with real-world data:

- **Ubiquitous uncertainty**: Constraint coefficients, demand forecasts, return estimates, and processing times are never exact. Nominal solutions can be infeasible in practice.
- **Industry adoption**: Robust optimization is standard in supply chain planning (Amazon, Procter & Gamble), portfolio management (BlackRock, Goldman Sachs), network design, and energy systems.
- **Complements stochastic programming**: Where stochastic programming requires a full distributional model, RO works with minimal uncertainty information (bounds, moments, or Wasserstein balls).
- **Computational tractability**: Robust counterparts of LPs under budget uncertainty are LPs; under ellipsoidal uncertainty are SOCPs. No scenario trees, no sampling — deterministic reformulations solved by standard solvers.
- **DRO bridges the gap**: Distributionally robust optimization combines the data-driven nature of stochastic programming with the worst-case guarantees of robust optimization, and is the state-of-the-art in data-driven prescriptive analytics.

After implementing convex optimization algorithms from scratch (practices 033a/033b) and mastering CVXPY's modeling language (practice 041a), this practice shows how to handle the reality that optimization data is uncertain. The techniques here — uncertainty sets, robust counterparts, DRO, affine decision rules — are the tools that move optimization from textbook to production.

## Commands

All commands are run from the `practice_046_robust_optimization/` folder root.

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install dependencies (CVXPY, NumPy, SciPy, matplotlib) into the virtual environment |

### Run

| Command | Description |
|---------|-------------|
| `uv run python src/phase1_robust_lp.py` | Phase 1: Robust LP with box, ellipsoidal, and budget uncertainty sets |
| `uv run python src/phase2_robust_portfolio.py` | Phase 2: Robust portfolio optimization with box and ellipsoidal uncertainty on returns |
| `uv run python src/phase3_dro.py` | Phase 3: Distributionally robust newsvendor — moment-based and Wasserstein DRO |
| `uv run python src/phase4_adjustable_robust.py` | Phase 4: Adjustable robust optimization — static vs affine decision rules |

## References

### Foundational Papers

- [Ben-Tal, A. & Nemirovski, A. (1998). "Robust Convex Optimization."](https://www2.isye.gatech.edu/~nemirovs/stablpn.pdf) — Foundational paper proving tractability of robust counterparts under ellipsoidal uncertainty.
- [Ben-Tal, A. & Nemirovski, A. (2002). "Robust Optimization — Methodology and Applications."](https://link.springer.com/article/10.1007/s101070100286) — Survey of the robust counterpart methodology and applications.
- [Bertsimas, D. & Sim, M. (2004). "The Price of Robustness."](https://www.robustopt.com/references/Price%20of%20Robustness.pdf) — Budget uncertainty sets with probabilistic guarantees and controlled price of robustness.
- [Ben-Tal, A., Goryashko, A., Guslitzer, E., & Nemirovski, A. (2004). "Adjustable Robust Solutions of Uncertain Linear Programs."](https://www2.isye.gatech.edu/~nemirovs/MP_Elana_2004.pdf) — Affinely Adjustable Robust Counterpart (AARC) for multi-stage problems.
- [Scarf, H. (1958). "A Min-Max Solution of an Inventory Problem."](https://www.researchgate.net/publication/311153932_A_Min-Max_Solution_of_an_Inventory_Problem) — Foundational work on distribution-free newsvendor with moment information.

### Distributionally Robust Optimization

- [Delage, E. & Ye, Y. (2010). "Distributionally Robust Optimization Under Moment Uncertainty."](https://pubsonline.informs.org/doi/10.1287/opre.1090.0741) — Moment-based ambiguity sets with SDP reformulation.
- [Mohajerin Esfahani, P. & Kuhn, D. (2018). "Data-Driven Distributionally Robust Optimization Using the Wasserstein Metric."](https://link.springer.com/article/10.1007/s10107-017-1172-1) — Wasserstein ambiguity sets with tractable finite convex reformulations and finite-sample guarantees.

### Textbooks and Surveys

- [Ben-Tal, A., El Ghaoui, L., & Nemirovski, A. (2009). *Robust Optimization*. Princeton University Press.](https://press.princeton.edu/books/hardcover/9780691143682/robust-optimization) — Definitive textbook on the robust counterpart methodology.
- [Bertsimas, D., Brown, D., & Caramanis, C. (2011). "Theory and Applications of Robust Optimization."](https://arxiv.org/pdf/1010.5445) — Comprehensive survey in SIAM Review.
- [Gorissen, B., Yanikoglu, I., & den Hertog, D. (2015). "A Practical Guide to Robust Optimization."](https://arxiv.org/pdf/1501.02634) — Practical guide covering modeling techniques and common pitfalls.

### CVXPY

- [CVXPY Documentation](https://www.cvxpy.org/) — Official documentation, DCP rules, and atom reference.
- [Boyd, S. & Vandenberghe, L. *Convex Optimization*.](https://web.stanford.edu/~boyd/cvxbook/) — Textbook behind CVXPY's design; Chapter 6 covers robust optimization.

## State

`not-started`
