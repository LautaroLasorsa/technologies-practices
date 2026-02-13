"""Phase 1: N-Queens with AllDifferent — CP-SAT.

Place N queens on an N×N board so no two attack each other.
Demonstrates: CpModel workflow, AllDifferent global constraint, solution enumeration.
"""

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_board(queens: list[int]) -> None:
    """Print an N-Queens solution as an ASCII board.

    Args:
        queens: List where queens[i] = column of queen in row i.
    """
    n = len(queens)
    for row in range(n):
        line = ""
        for col in range(n):
            line += " Q" if queens[row] == col else " ."
        print(line)
    print()


def print_summary(n: int, solutions: list[list[int]], show_max: int = 4) -> None:
    """Print a summary of N-Queens solutions.

    Args:
        n: Board size.
        solutions: List of solutions (each a list of column indices).
        show_max: Maximum number of solutions to display in full.
    """
    print(f"=== {n}-Queens: {len(solutions)} solution(s) found ===\n")
    for i, sol in enumerate(solutions[:show_max]):
        print(f"Solution {i + 1}:")
        print_board(sol)
    if len(solutions) > show_max:
        print(f"... and {len(solutions) - show_max} more solutions.\n")


# ---------------------------------------------------------------------------
# Solution collector callback
# ---------------------------------------------------------------------------

class SolutionCollector(cp_model.CpSolverSolutionCallback):
    """Callback that collects all solutions found by the solver.

    Usage:
        collector = SolutionCollector(queens)
        solver.parameters.enumerate_all_solutions = True
        solver.solve(model, collector)
        print(collector.solutions)  # list of lists of column indices
    """

    def __init__(self, variables: list[cp_model.IntVar]) -> None:
        super().__init__()
        self._variables = variables
        self.solutions: list[list[int]] = []

    def on_solution_callback(self) -> None:
        self.solutions.append([self.value(v) for v in self._variables])


# ---------------------------------------------------------------------------
# TODO(human): Implement the solver
# ---------------------------------------------------------------------------

def solve_n_queens(n: int, find_all: bool = False) -> list[list[int]]:
    """Solve the N-Queens problem using CP-SAT.

    Args:
        n: Board size (number of queens).
        find_all: If True, enumerate all solutions. If False, return first found.

    Returns:
        List of solutions. Each solution is a list of length n where
        solution[i] = column of queen in row i.
    """
    # TODO(human): N-Queens with CP-SAT
    #
    # Place n queens on an n×n board so no two attack each other.
    #
    # Variables: queens[i] = column of queen in row i (IntVar, domain [0, n-1])
    #   This representation automatically ensures one queen per row.
    #
    # Constraints:
    #   1. AllDifferent(queens) — no two queens in same column
    #   2. AllDifferent(queens[i] + i for i in range(n)) — no two on same diagonal
    #   3. AllDifferent(queens[i] - i for i in range(n)) — no two on same anti-diagonal
    #
    # In CP-SAT:
    #   model = cp_model.CpModel()
    #   queens = [model.new_int_var(0, n-1, f'q{i}') for i in range(n)]
    #   model.add_all_different(queens)
    #   model.add_all_different(queens[i] + i for i in range(n))
    #   model.add_all_different(queens[i] - i for i in range(n))
    #
    # To find ALL solutions: use CpSolverSolutionCallback
    #   collector = SolutionCollector(queens)
    #   solver.parameters.enumerate_all_solutions = True
    #   solver.solve(model, collector)
    #   return collector.solutions
    #
    # For single solution: solver.solve(model), then read solver.value(q) for each q.
    #
    # Expected counts: n=4→2, n=8→92, n=12→14200.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Single solution for n=8
    print("--- Single solution for 8-Queens ---")
    solutions = solve_n_queens(8, find_all=False)
    if solutions:
        print_board(solutions[0])

    # All solutions for small boards
    for n in [4, 8]:
        solutions = solve_n_queens(n, find_all=True)
        print_summary(n, solutions)

    # Count for n=12 (should be 14200)
    solutions = solve_n_queens(12, find_all=True)
    print(f"12-Queens: {len(solutions)} solutions found.")


if __name__ == "__main__":
    main()
