"""Phase 3: Constraints & Mixed Variables.

Defines constrained multi-objective optimization problems using pymoo.
Implements the welded beam design problem with 4 inequality constraints,
and demonstrates constraint satisfaction analysis on the Pareto front.

pymoo API patterns used:
  - ElementwiseProblem with n_ieq_constr for inequality constraints
  - out["G"] = [...] for constraint values (negative = satisfied)
  - Constraint handling: feasible >> infeasible, lower CV preferred
"""

import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize


# ============================================================================
# TODO(human): Welded beam design problem
# ============================================================================

class WeldedBeamProblem(ElementwiseProblem):
    """Bi-objective welded beam design problem.

    A beam is welded to a rigid member and must support a vertical load P.
    The designer must choose 4 dimensions to minimize cost and deflection
    while satisfying stress, buckling, and geometric constraints.

    Decision variables:
      x[0] = h  — weld thickness         (0.125 to 5.0)
      x[1] = l  — weld length            (0.1 to 10.0)
      x[2] = t  — beam height            (0.1 to 10.0)
      x[3] = b  — beam width             (0.125 to 5.0)

    Objectives (minimize both):
      f_1 = cost    = 1.10471 * h^2 * l + 0.04811 * t * b * (14.0 + l)
      f_2 = deflection = 4 * P * L^3 / (E * t^3 * b)

    Constraints (g_i <= 0):
      g_1: shear stress τ(x) <= τ_max
      g_2: bending stress σ(x) <= σ_max
      g_3: h <= b  (weld thickness <= beam width)
      g_4: buckling load P_c(x) >= P  (must not buckle)

    Constants:
      P = 6000 lb, L = 14 in, E = 30e6 psi, G = 12e6 psi
      τ_max = 13600 psi, σ_max = 30000 psi

    This is a well-known engineering benchmark from Deb (1991).
    """

    # Problem constants
    P = 6000.0       # Applied load (lb)
    L = 14.0         # Beam length (in)
    E = 30e6         # Young's modulus (psi)
    G_SHEAR = 12e6   # Shear modulus (psi)
    TAU_MAX = 13600.0   # Max shear stress (psi)
    SIGMA_MAX = 30000.0  # Max bending stress (psi)

    def __init__(self):
        # TODO(human): Initialize the constrained problem
        #
        # Call super().__init__() with:
        #   n_var=4         — four design variables (h, l, t, b)
        #   n_obj=2         — two objectives (cost, deflection)
        #   n_ieq_constr=4  — four inequality constraints
        #   xl=np.array([0.125, 0.1, 0.1, 0.125])  — lower bounds
        #   xu=np.array([5.0, 10.0, 10.0, 5.0])     — upper bounds
        #
        # n_ieq_constr tells pymoo how many constraint values to expect
        # in out["G"]. pymoo's constraint handling rule:
        #   - out["G"][i] <= 0 means constraint i is SATISFIED
        #   - out["G"][i] > 0 means constraint i is VIOLATED
        #
        # The constraint violation (CV) of a solution is:
        #   CV = Σ max(0, G[i])
        #
        # During selection, pymoo enforces:
        #   1. Feasible solutions ALWAYS dominate infeasible ones
        #   2. Among infeasible: prefer lower CV
        #   3. Among feasible: use normal Pareto dominance
        raise NotImplementedError("TODO(human): call super().__init__ with n_var, n_obj, n_ieq_constr, xl, xu")

    def _evaluate(self, x, out, *args, **kwargs):
        # TODO(human): Compute objectives and constraints
        #
        # Extract variables:
        #   h, l, t, b = x[0], x[1], x[2], x[3]
        #
        # --- Objectives ---
        # f1 = cost = 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
        #   Material cost: weld volume (h^2 * l) + beam volume (t * b * (14+l))
        #
        # f2 = deflection = 4.0 * P * L**3 / (E * t**3 * b)
        #   Tip deflection from Euler-Bernoulli beam theory.
        #   Larger t and b reduce deflection (stiffer beam).
        #
        # --- Stress calculations ---
        # The weld experiences combined direct shear and torsional shear:
        #
        # tau_prime = P / (sqrt(2) * h * l)
        #   Direct shear stress in the weld.
        #
        # M = P * (L + l/2)
        #   Bending moment at the weld.
        #
        # R = sqrt(l**2/4 + ((h + t)/2)**2)
        #   Distance from weld centroid to critical point.
        #
        # J = 2 * (sqrt(2) * h * l * (l**2/12 + ((h + t)/2)**2))
        #   Polar moment of inertia of the weld group.
        #
        # tau_double_prime = M * R / J
        #   Torsional shear stress.
        #
        # tau = sqrt(tau_prime**2 + 2*tau_prime*tau_double_prime*(l/(2*R)) + tau_double_prime**2)
        #   Combined shear stress (vector sum at the critical point).
        #
        # sigma = 6 * P * L / (b * t**2)
        #   Bending stress in the beam.
        #
        # P_c = (4.013 * E * sqrt(t**2 * b**6 / 36)) / L**2 * (1 - t/(2*L) * sqrt(E/(4*G_SHEAR)))
        #   Critical buckling load (Euler formula with correction).
        #
        # --- Constraints (all in form g_i <= 0) ---
        # g1 = tau - TAU_MAX           (shear stress must not exceed max)
        # g2 = sigma - SIGMA_MAX       (bending stress must not exceed max)
        # g3 = h - b                   (weld thickness must not exceed beam width)
        # g4 = P - P_c                 (applied load must not exceed buckling load)
        #
        # Store: out["F"] = [f1, f2], out["G"] = [g1, g2, g3, g4]
        #
        # If g_i <= 0, constraint is satisfied. If g_i > 0, it's violated.
        # For example, g1 = tau - TAU_MAX: if tau <= TAU_MAX then g1 <= 0 (satisfied).
        raise NotImplementedError("TODO(human): compute objectives f1, f2 and constraints g1-g4")


# ============================================================================
# TODO(human): Run constrained optimization
# ============================================================================

def run_constrained_optimization(
    pop_size: int = 200,
    n_gen: int = 300,
    seed: int = 42,
):
    """Run NSGA-II on the constrained welded beam problem.

    Uses a larger population and more generations than unconstrained problems
    because constraint satisfaction reduces the effective feasible region.

    Args:
        pop_size: Population size (larger helps find feasible solutions).
        n_gen: Number of generations.
        seed: Random seed.

    Returns:
        result: pymoo Result object.
    """
    # TODO(human): Create the welded beam problem and run NSGA-II
    #
    # Step 1: Instantiate the problem
    #   problem = WeldedBeamProblem()
    #
    # Step 2: Configure NSGA-II with a larger population
    #   algorithm = NSGA2(pop_size=pop_size)
    #
    #   Why pop_size=200 instead of 100?
    #   Constrained problems have a smaller feasible region. Many randomly
    #   initialized solutions will be infeasible. A larger population
    #   increases the chance of finding feasible solutions early, which
    #   then guide the search toward the constrained Pareto front.
    #
    # Step 3: Run with more generations
    #   result = minimize(problem, algorithm, ("n_gen", n_gen),
    #                     seed=seed, verbose=False)
    #
    #   More generations are needed because:
    #   1. The algorithm must first find feasible solutions (exploration).
    #   2. Then optimize within the feasible region (exploitation).
    #   3. The feasible PF may be a small subset of the unconstrained PF.
    #
    # Return result.
    raise NotImplementedError("TODO(human): create WeldedBeamProblem, run NSGA2, return result")


# ============================================================================
# Analysis helpers
# ============================================================================

def analyze_constraint_satisfaction(result) -> None:
    """Analyze how well the Pareto-optimal solutions satisfy constraints."""
    F = result.F
    G = result.G  # Constraint values for each solution

    n_solutions = F.shape[0]
    n_constraints = G.shape[1] if G is not None and len(G.shape) > 1 else 0

    print(f"\n{'=' * 60}")
    print("  Constraint Satisfaction Analysis")
    print(f"{'=' * 60}")
    print(f"  Pareto-optimal solutions: {n_solutions}")

    if G is None or n_constraints == 0:
        print("  No constraint data available.")
        return

    constraint_names = [
        "g1: shear stress (tau <= tau_max)",
        "g2: bending stress (sigma <= sigma_max)",
        "g3: geometry (h <= b)",
        "g4: buckling (P <= P_c)",
    ]

    for i in range(n_constraints):
        g_values = G[:, i]
        n_satisfied = np.sum(g_values <= 0)
        name = constraint_names[i] if i < len(constraint_names) else f"g{i+1}"
        max_violation = max(0, g_values.max())
        max_margin = abs(min(0, g_values.min()))

        print(f"\n  {name}")
        print(f"    Satisfied: {n_satisfied}/{n_solutions} "
              f"({100 * n_satisfied / n_solutions:.1f}%)")
        print(f"    Range: [{g_values.min():.2f}, {g_values.max():.2f}]")
        if max_violation > 0:
            print(f"    Max violation: {max_violation:.2f}")
        print(f"    Max margin (slack): {max_margin:.2f}")

    # Overall feasibility
    feasible_mask = np.all(G <= 0, axis=1)
    n_feasible = feasible_mask.sum()
    print(f"\n  Fully feasible solutions: {n_feasible}/{n_solutions} "
          f"({100 * n_feasible / n_solutions:.1f}%)")

    # Total constraint violation
    cv = np.sum(np.maximum(0, G), axis=1)
    print(f"  Constraint violation: mean={cv.mean():.4f}, max={cv.max():.4f}")


def plot_constrained_pareto(result) -> None:
    """Plot the constrained Pareto front colored by constraint satisfaction."""
    F = result.F
    G = result.G

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Pareto front
    ax = axes[0]
    ax.scatter(F[:, 0], F[:, 1], c="C0", s=30, alpha=0.7, edgecolors="none")
    ax.set_xlabel("f₁ (cost)", fontsize=12)
    ax.set_ylabel("f₂ (deflection)", fontsize=12)
    ax.set_title("Welded Beam — Constrained Pareto Front", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Right: constraint violation
    ax = axes[1]
    if G is not None:
        cv = np.sum(np.maximum(0, G), axis=1)
        scatter = ax.scatter(F[:, 0], F[:, 1], c=cv, cmap="RdYlGn_r",
                             s=30, alpha=0.7, edgecolors="none")
        plt.colorbar(scatter, ax=ax, label="Constraint Violation")
    else:
        ax.scatter(F[:, 0], F[:, 1], c="C0", s=30, alpha=0.7)
    ax.set_xlabel("f₁ (cost)", fontsize=12)
    ax.set_ylabel("f₂ (deflection)", fontsize=12)
    ax.set_title("Colored by Constraint Violation", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_design_variables(result) -> None:
    """Plot parallel coordinates of design variables for PF solutions."""
    X = result.X
    var_names = ["h (weld thickness)", "l (weld length)",
                 "t (beam height)", "b (beam width)"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize each variable to [0, 1] for parallel coordinates
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    X_norm = (X - X_min) / X_range

    # Color by first objective (cost)
    F = result.F
    colors = plt.cm.viridis((F[:, 0] - F[:, 0].min()) / (F[:, 0].max() - F[:, 0].min()))

    for i in range(X_norm.shape[0]):
        ax.plot(range(4), X_norm[i], c=colors[i], alpha=0.3, linewidth=0.8)

    ax.set_xticks(range(4))
    ax.set_xticklabels(var_names, fontsize=10)
    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title("Design Variables — Parallel Coordinates (color = cost)", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Phase 3: Constraints & Mixed Variables")
    print("=" * 60)

    # --- Constrained welded beam ---
    print("\n--- Welded Beam Design Problem ---")
    print("  4 variables, 2 objectives, 4 inequality constraints")
    result = run_constrained_optimization(pop_size=200, n_gen=300)

    # Summary
    F = result.F
    X = result.X
    print(f"\n  Solutions found:   {F.shape[0]}")
    print(f"  Cost range:        [{F[:, 0].min():.2f}, {F[:, 0].max():.2f}]")
    print(f"  Deflection range:  [{F[:, 1].min():.6f}, {F[:, 1].max():.6f}]")
    print(f"  Execution time:    {result.exec_time:.2f}s")

    # Design variable ranges
    var_names = ["h", "l", "t", "b"]
    print("\n  Design variable ranges on the Pareto front:")
    for i, name in enumerate(var_names):
        print(f"    {name}: [{X[:, i].min():.4f}, {X[:, i].max():.4f}]")

    # Constraint analysis
    analyze_constraint_satisfaction(result)

    # Plots
    plot_constrained_pareto(result)
    plot_design_variables(result)

    # Show a few representative solutions
    print(f"\n{'=' * 60}")
    print("  Sample Pareto-optimal Designs")
    print(f"{'=' * 60}")
    indices = [0, len(F) // 4, len(F) // 2, 3 * len(F) // 4, len(F) - 1]
    # Sort by cost first
    sort_idx = np.argsort(F[:, 0])
    for rank, idx in enumerate(indices):
        si = sort_idx[idx]
        print(f"\n  Design {rank + 1}:")
        print(f"    h={X[si, 0]:.4f}, l={X[si, 1]:.4f}, "
              f"t={X[si, 2]:.4f}, b={X[si, 3]:.4f}")
        print(f"    Cost={F[si, 0]:.2f}, Deflection={F[si, 1]:.6f}")

    print("\n[Phase 3 complete]")


if __name__ == "__main__":
    main()
