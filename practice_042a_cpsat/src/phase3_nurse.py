"""Phase 3: Nurse Scheduling with CP-SAT.

Assign nurses to shifts over multiple days with coverage, fairness, and temporal constraints.
Demonstrates: BoolVar, AddExactlyOne, AddAtMostOne, AddImplication, fairness objectives.
"""

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Scheduling data
# ---------------------------------------------------------------------------

SHIFT_NAMES = ["Morning", "Afternoon", "Night"]
MORNING, AFTERNOON, NIGHT = 0, 1, 2

# Default problem parameters
DEFAULT_N_NURSES = 4
DEFAULT_N_DAYS = 7
DEFAULT_SHIFTS_PER_DAY = 3  # Morning, Afternoon, Night


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_schedule(
    n_nurses: int,
    n_days: int,
    shifts_per_day: int,
    assignments: dict[tuple[int, int, int], bool],
) -> None:
    """Print the nurse schedule as a formatted table.

    Args:
        n_nurses: Number of nurses.
        n_days: Number of days.
        shifts_per_day: Number of shifts per day.
        assignments: Dictionary mapping (nurse, day, shift) -> True/False.
    """
    shift_chars = ["M", "A", "N"]

    # Header
    print(f"\n{'Nurse':>8}", end="")
    for d in range(n_days):
        print(f"  Day{d}", end="")
    print("  Total")
    print("-" * (8 + n_days * 6 + 7))

    # Each nurse row
    for n in range(n_nurses):
        print(f"{'N' + str(n):>8}", end="")
        total = 0
        for d in range(n_days):
            assigned = ""
            for s in range(shifts_per_day):
                if assignments.get((n, d, s), False):
                    assigned = shift_chars[s]
                    total += 1
            print(f"{'  ' + assigned if assigned else '   -':>6}", end="")
        print(f"  {total:>4}")
    print()

    # Coverage check
    print("Coverage per shift:")
    for d in range(n_days):
        for s in range(shifts_per_day):
            count = sum(
                1 for n in range(n_nurses) if assignments.get((n, d, s), False)
            )
            status = "OK" if count == 1 else f"ISSUE({count})"
            print(f"  Day{d} {SHIFT_NAMES[s]}: {status}")
    print()


def print_solver_stats(solver: cp_model.CpSolver) -> None:
    """Print solver statistics."""
    print(f"  Status       : {solver.status_name()}")
    print(f"  Wall time    : {solver.wall_time:.4f}s")
    print(f"  Branches     : {solver.num_branches}")
    print(f"  Conflicts    : {solver.num_conflicts}")
    print()


# ---------------------------------------------------------------------------
# TODO(human): Implement the solver
# ---------------------------------------------------------------------------

def solve_nurse_scheduling(
    n_nurses: int = DEFAULT_N_NURSES,
    n_days: int = DEFAULT_N_DAYS,
    shifts_per_day: int = DEFAULT_SHIFTS_PER_DAY,
) -> dict[tuple[int, int, int], bool] | None:
    """Solve a nurse scheduling problem using CP-SAT.

    Args:
        n_nurses: Number of nurses.
        n_days: Number of days to schedule.
        shifts_per_day: Number of shifts per day (3: morning, afternoon, night).

    Returns:
        Dictionary mapping (nurse, day, shift) -> True/False for each
        assignment. Returns None if infeasible.
    """
    # TODO(human): Nurse Scheduling with CP-SAT
    #
    # Assign nurses to shifts (morning/afternoon/night) over multiple days.
    #
    # Variables: shifts[n][d][s] = BoolVar — nurse n works shift s on day d
    #   Create as: model.new_bool_var(f'shift_n{n}_d{d}_s{s}')
    #
    # Constraints:
    #   1. Each shift each day has exactly one nurse:
    #      model.add_exactly_one(shifts[n][d][s] for n in range(n_nurses))
    #      for each (d, s) pair
    #   2. Each nurse works at most one shift per day:
    #      model.add_at_most_one(shifts[n][d][s] for s in range(shifts_per_day))
    #      for each (n, d) pair
    #   3. No night shift followed by morning shift next day:
    #      model.add_implication(shifts[n][d][NIGHT], shifts[n][d+1][MORNING].Not())
    #      for each nurse n and day d where d+1 < n_days
    #   4. Even distribution: compute total shifts per nurse, then constrain
    #      min_shifts <= total[n] <= max_shifts for each nurse
    #      where min_shifts = (n_days * shifts_per_day) // n_nurses
    #      and max_shifts = min_shifts + 1
    #
    # No explicit objective needed (satisfaction problem) — but you can
    # optionally minimize the difference between max and min workload.
    #
    # After solving, extract: solver.value(shifts[n][d][s]) for all (n, d, s).
    # Return as dict {(n, d, s): bool}.
    #
    # AddImplication is the key CP constraint for temporal/logical rules.
    # In MIP this requires big-M constraints, which are weaker.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Nurse Scheduling ===\n")

    # Default: 4 nurses, 7 days, 3 shifts
    print(f"--- {DEFAULT_N_NURSES} nurses, {DEFAULT_N_DAYS} days, {DEFAULT_SHIFTS_PER_DAY} shifts/day ---")
    assignments = solve_nurse_scheduling()
    if assignments:
        print_schedule(DEFAULT_N_NURSES, DEFAULT_N_DAYS, DEFAULT_SHIFTS_PER_DAY, assignments)

    # Tighter: 3 nurses, 5 days
    print("--- 3 nurses, 5 days, 3 shifts/day ---")
    assignments = solve_nurse_scheduling(n_nurses=3, n_days=5)
    if assignments:
        print_schedule(3, 5, DEFAULT_SHIFTS_PER_DAY, assignments)


if __name__ == "__main__":
    main()
