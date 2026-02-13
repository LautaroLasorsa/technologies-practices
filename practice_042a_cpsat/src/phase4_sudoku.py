"""Phase 4: Sudoku Solver with CP-SAT.

Model Sudoku as a CSP with AllDifferent constraints on rows, columns, and 3x3 boxes.
Demonstrates: pure constraint satisfaction (no objective), AllDifferent propagation power.
"""

import time

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Puzzles (0 = empty cell)
# ---------------------------------------------------------------------------

# Easy — many givens, solvable with basic propagation
EASY_PUZZLE = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# Medium — fewer givens, requires some search
MEDIUM_PUZZLE = [
    [0, 0, 0, 6, 0, 0, 4, 0, 0],
    [7, 0, 0, 0, 0, 3, 6, 0, 0],
    [0, 0, 0, 0, 9, 1, 0, 8, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 5, 0, 1, 8, 0, 0, 0, 3],
    [0, 0, 0, 3, 0, 6, 0, 4, 5],
    [0, 4, 0, 2, 0, 0, 0, 6, 0],
    [9, 0, 3, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 1, 0, 0],
]

# Hard — minimal givens (17), requires significant search
HARD_PUZZLE = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 8, 5],
    [0, 0, 1, 0, 2, 0, 0, 0, 0],
    [0, 0, 0, 5, 0, 7, 0, 0, 0],
    [0, 0, 4, 0, 0, 0, 1, 0, 0],
    [0, 9, 0, 0, 0, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0, 0, 7, 3],
    [0, 0, 2, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 4, 0, 0, 0, 9],
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_grid(grid: list[list[int]], title: str = "") -> None:
    """Print a 9x9 Sudoku grid with box separators.

    Args:
        grid: 9x9 list of integers (0 = empty).
        title: Optional title to print above the grid.
    """
    if title:
        print(f"\n{title}")
    print("+-------+-------+-------+")
    for i in range(9):
        if i > 0 and i % 3 == 0:
            print("+-------+-------+-------+")
        row_str = "| "
        for j in range(9):
            val = grid[i][j]
            row_str += str(val) if val != 0 else "."
            row_str += " | " if (j + 1) % 3 == 0 else " "
        print(row_str)
    print("+-------+-------+-------+")


def count_givens(grid: list[list[int]]) -> int:
    """Count the number of non-zero cells in the grid."""
    return sum(1 for row in grid for val in row if val != 0)


# ---------------------------------------------------------------------------
# TODO(human): Implement the solver
# ---------------------------------------------------------------------------

def solve_sudoku(grid: list[list[int]]) -> list[list[int]] | None:
    """Solve a 9x9 Sudoku puzzle using CP-SAT.

    Args:
        grid: 9x9 list of integers. 0 means empty cell, 1-9 means given clue.

    Returns:
        Solved 9x9 grid, or None if no solution exists.
    """
    # TODO(human): Sudoku Solver with CP-SAT
    #
    # Variables: cells[i][j] = IntVar(1, 9) for each cell
    #
    # Constraints:
    #   1. Given clues: model.add(cells[i][j] == grid[i][j]) for filled cells
    #   2. Row uniqueness: model.add_all_different(cells[i][j] for j in range(9))
    #      for each row i
    #   3. Column uniqueness: model.add_all_different(cells[i][j] for i in range(9))
    #      for each column j
    #   4. Box uniqueness: for each 3x3 box (br, bc) where br,bc in {0,1,2}:
    #      model.add_all_different(
    #          cells[3*br + r][3*bc + c] for r in range(3) for c in range(3)
    #      )
    #
    # No objective — this is a pure satisfaction problem.
    # Solve and extract: solver.value(cells[i][j]) for each cell.
    #
    # CP-SAT with AllDifferent + propagation solves even hard Sudoku instantly.
    # The AllDifferent constraint is a GLOBAL constraint — it propagates much
    # more effectively than decomposing into pairwise != constraints.
    # Under the hood, AllDifferent uses matching algorithms from graph theory
    # (Hall's theorem) to detect infeasibility early.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Sudoku Solver ===")

    puzzles = [
        ("Easy", EASY_PUZZLE),
        ("Medium", MEDIUM_PUZZLE),
        ("Hard", HARD_PUZZLE),
    ]

    for name, puzzle in puzzles:
        givens = count_givens(puzzle)
        print(f"\n--- {name} ({givens} givens) ---")
        print_grid(puzzle, title="Input:")

        t0 = time.perf_counter()
        solution = solve_sudoku(puzzle)
        elapsed = time.perf_counter() - t0

        if solution:
            print_grid(solution, title="Solution:")
            print(f"  Solved in {elapsed * 1000:.2f} ms")
        else:
            print("  No solution found.")
        print()


if __name__ == "__main__":
    main()
