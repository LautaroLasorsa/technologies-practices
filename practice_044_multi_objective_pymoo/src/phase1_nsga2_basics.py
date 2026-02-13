"""Phase 1: pymoo Basics — Problem Definition & NSGA-II.

Defines custom multi-objective problems by subclassing ElementwiseProblem,
runs NSGA-II on custom and benchmark (ZDT1, ZDT3) problems, and compares
obtained Pareto fronts against known analytical solutions.

pymoo API patterns used:
  - ElementwiseProblem: subclass, define _evaluate(self, x, out)
  - NSGA2(pop_size=...): configure the algorithm
  - minimize(problem, algorithm, ("n_gen", N)): run optimization
  - result.F: objective values of Pareto-optimal solutions
  - result.X: decision variables of Pareto-optimal solutions
  - get_problem("zdt1"): load benchmark problems
  - problem.pareto_front(): get analytical Pareto front
"""

import numpy as np
import matplotlib.pyplot as plt

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.problems import get_problem


# ============================================================================
# TODO(human): Custom bi-objective problem
# ============================================================================

class CustomBiObjectiveProblem(ElementwiseProblem):
    """A custom bi-objective test problem with 2 decision variables.

    Problem (Schaffer-like):
      minimize  f_1(x) = x_1^2 + (x_2 - 1)^2
      minimize  f_2(x) = (x_1 - 1)^2 + x_2^2

    Decision variables: x_1, x_2 in [-2, 2]

    The Pareto front lies on the line segment from (0, 1) to (1, 0) in
    decision space, producing a convex front in objective space.
    """

    def __init__(self):
        # TODO(human): Initialize the ElementwiseProblem base class
        #
        # Call super().__init__() with the following keyword arguments:
        #   n_var=2       — number of decision variables (x_1 and x_2)
        #   n_obj=2       — number of objectives (f_1 and f_2)
        #   xl=np.array([-2.0, -2.0])  — lower bounds for each variable
        #   xu=np.array([2.0, 2.0])    — upper bounds for each variable
        #
        # These four parameters fully define the problem's structure.
        # pymoo uses n_var to know how many elements x has in _evaluate,
        # n_obj to know how many values out["F"] should contain,
        # and xl/xu to initialize the population within bounds and to
        # enforce bound constraints during evolution.
        #
        # Why ElementwiseProblem instead of Problem?
        #   ElementwiseProblem evaluates ONE solution at a time — x is a
        #   1D array of shape (n_var,). Problem evaluates the ENTIRE
        #   population at once — x is 2D of shape (pop_size, n_var).
        #   ElementwiseProblem is simpler to implement; pymoo handles
        #   the vectorization automatically via a loop internally.
        raise NotImplementedError("TODO(human): call super().__init__ with n_var, n_obj, xl, xu")

    def _evaluate(self, x, out, *args, **kwargs):
        # TODO(human): Compute both objectives and store in out["F"]
        #
        # Arguments:
        #   x: 1D numpy array of shape (n_var,) = (2,) — one candidate solution
        #   out: dictionary to write results into
        #
        # Compute:
        #   f1 = x[0]**2 + (x[1] - 1)**2       (minimize: pushes x toward (0, 1))
        #   f2 = (x[0] - 1)**2 + x[1]**2        (minimize: pushes x toward (1, 0))
        #
        # Store as: out["F"] = [f1, f2]
        #
        # The two objectives conflict because f1 wants x_1=0, x_2=1 while
        # f2 wants x_1=1, x_2=0. The set of Pareto-optimal solutions is
        # the segment connecting these two ideal points. Any solution on
        # this segment cannot improve one objective without worsening the other.
        #
        # Note: out["F"] can be a list or numpy array. pymoo converts it
        # internally. For constraints, you'd also set out["G"] here.
        raise NotImplementedError("TODO(human): compute f1, f2 and set out['F'] = [f1, f2]")


# ============================================================================
# TODO(human): Run NSGA-II on the custom problem
# ============================================================================

def run_nsga2_on_custom(pop_size: int = 100, n_gen: int = 100, seed: int = 42):
    """Run NSGA-II on the custom bi-objective problem.

    Args:
        pop_size: Population size for NSGA-II.
        n_gen: Number of generations to evolve.
        seed: Random seed for reproducibility.

    Returns:
        result: pymoo Result object with .F (objectives) and .X (variables).
    """
    # TODO(human): Create problem, configure NSGA-II, and run minimize
    #
    # Step 1: Instantiate the problem
    #   problem = CustomBiObjectiveProblem()
    #
    # Step 2: Configure NSGA-II
    #   algorithm = NSGA2(pop_size=pop_size)
    #
    #   NSGA2 parameters you can tune (defaults are usually good):
    #     pop_size: population size. Larger = better coverage but slower.
    #               Rule of thumb: 100-200 for 2 objectives.
    #     The algorithm uses SBX crossover (eta=15) and polynomial
    #     mutation (eta=20) by default — standard settings from the
    #     NSGA-II paper. pymoo auto-configures these based on n_var.
    #
    # Step 3: Run the optimization
    #   result = minimize(
    #       problem,
    #       algorithm,
    #       ("n_gen", n_gen),   # termination criterion: number of generations
    #       seed=seed,          # reproducibility
    #       verbose=False,      # suppress per-generation output
    #   )
    #
    # The minimize() function returns a Result object:
    #   result.F — shape (n_solutions, n_obj) — objective values of Pareto front
    #   result.X — shape (n_solutions, n_var) — decision variables of PF solutions
    #   result.exec_time — wall-clock time in seconds
    #
    # Return the result object.
    raise NotImplementedError("TODO(human): create problem, NSGA2, call minimize, return result")


# ============================================================================
# TODO(human): Run NSGA-II on benchmark ZDT1
# ============================================================================

def run_nsga2_on_zdt1(pop_size: int = 100, n_gen: int = 200, seed: int = 42):
    """Run NSGA-II on the ZDT1 benchmark problem.

    ZDT1 has 30 decision variables, 2 objectives, and a convex Pareto front
    defined by f_2 = 1 - sqrt(f_1) for f_1 in [0, 1].

    Args:
        pop_size: Population size.
        n_gen: Number of generations.
        seed: Random seed.

    Returns:
        result: pymoo Result object.
        pf_true: Analytical Pareto front from the problem definition.
    """
    # TODO(human): Load ZDT1, run NSGA-II, and return result + true PF
    #
    # Step 1: Load the benchmark problem
    #   problem = get_problem("zdt1")
    #
    #   get_problem() loads pre-defined test problems. ZDT1 is a standard
    #   bi-objective benchmark with n_var=30 variables in [0,1].
    #   The difficulty: only x_1 determines f_1, but ALL 30 variables
    #   affect f_2 through a helper function g(x). The algorithm must
    #   simultaneously push x_2...x_30 toward 0 (for g→1) and find
    #   the Pareto-optimal trade-off between f_1 and f_2.
    #
    # Step 2: Get the analytical Pareto front
    #   pf_true = problem.pareto_front()
    #
    #   Returns an (n_points, 2) array of the true PF. For ZDT1:
    #   f_2 = 1 - sqrt(f_1), sampled at n_points=100 by default.
    #   This is used for visual comparison and metrics in Phase 4.
    #
    # Step 3: Run NSGA-II (same pattern as above)
    #   algorithm = NSGA2(pop_size=pop_size)
    #   result = minimize(problem, algorithm, ("n_gen", n_gen),
    #                     seed=seed, verbose=False)
    #
    # Return (result, pf_true) as a tuple.
    raise NotImplementedError("TODO(human): load zdt1, get pareto_front, run NSGA2, return both")


# ============================================================================
# Visualization helpers
# ============================================================================

def plot_pareto_front_2d(
    F: np.ndarray,
    title: str,
    pf_true: np.ndarray | None = None,
    xlabel: str = "f₁",
    ylabel: str = "f₂",
) -> None:
    """Plot a 2D Pareto front with optional analytical comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if pf_true is not None:
        ax.plot(
            pf_true[:, 0], pf_true[:, 1],
            "k-", linewidth=2, alpha=0.6, label="True Pareto Front",
        )

    ax.scatter(
        F[:, 0], F[:, 1],
        c="C0", s=30, alpha=0.7, edgecolors="none", label="NSGA-II Solutions",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_space(X: np.ndarray, title: str) -> None:
    """Plot decision variables of the Pareto front (first 2 dimensions)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X[:, 0], X[:, 1], c="C1", s=30, alpha=0.7, edgecolors="none")
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_result_summary(result, name: str) -> None:
    """Print summary statistics of a pymoo Result object."""
    F = result.F
    X = result.X
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Solutions found:   {F.shape[0]}")
    print(f"  Objectives:        {F.shape[1]}")
    print(f"  Decision vars:     {X.shape[1]}")
    print(f"  f₁ range:          [{F[:, 0].min():.4f}, {F[:, 0].max():.4f}]")
    print(f"  f₂ range:          [{F[:, 1].min():.4f}, {F[:, 1].max():.4f}]")
    print(f"  Execution time:    {result.exec_time:.2f}s")


# ============================================================================
# ZDT3 demonstration (provided — disconnected Pareto front)
# ============================================================================

def run_nsga2_on_zdt3(pop_size: int = 100, n_gen: int = 200, seed: int = 42):
    """Run NSGA-II on ZDT3 — disconnected Pareto front.

    ZDT3 has a Pareto front consisting of 5 disconnected convex segments.
    This tests whether NSGA-II can distribute solutions across all segments.
    """
    problem = get_problem("zdt3")
    pf_true = problem.pareto_front()
    algorithm = NSGA2(pop_size=pop_size)
    result = minimize(problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=False)
    return result, pf_true


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 60)
    print("Phase 1: pymoo Basics — Problem Definition & NSGA-II")
    print("=" * 60)

    # --- Custom problem ---
    print("\n--- Custom Bi-Objective Problem ---")
    res_custom = run_nsga2_on_custom(pop_size=100, n_gen=100)
    print_result_summary(res_custom, "Custom Problem (NSGA-II)")
    plot_pareto_front_2d(
        res_custom.F,
        "Custom Problem — NSGA-II Pareto Front",
    )
    plot_decision_space(
        res_custom.X,
        "Custom Problem — Decision Space",
    )

    # --- ZDT1 benchmark ---
    print("\n--- ZDT1 Benchmark (Convex Front) ---")
    res_zdt1, pf_zdt1 = run_nsga2_on_zdt1(pop_size=100, n_gen=200)
    print_result_summary(res_zdt1, "ZDT1 (NSGA-II)")
    plot_pareto_front_2d(
        res_zdt1.F,
        "ZDT1 — NSGA-II vs Analytical Pareto Front",
        pf_true=pf_zdt1,
    )

    # --- ZDT3 benchmark (disconnected) ---
    print("\n--- ZDT3 Benchmark (Disconnected Front) ---")
    res_zdt3, pf_zdt3 = run_nsga2_on_zdt3(pop_size=100, n_gen=200)
    print_result_summary(res_zdt3, "ZDT3 (NSGA-II)")
    plot_pareto_front_2d(
        res_zdt3.F,
        "ZDT3 — NSGA-II vs Analytical Pareto Front (Disconnected)",
        pf_true=pf_zdt3,
    )

    print("\n[Phase 1 complete]")


if __name__ == "__main__":
    main()
