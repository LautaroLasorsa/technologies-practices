"""Phase 2: Job-Shop Scheduling with CP-SAT.

Schedule jobs (each a sequence of tasks) on machines, minimizing makespan.
Demonstrates: IntervalVar, NoOverlap, precedence constraints, optimization.
"""

from collections import namedtuple

from ortools.sat.python import cp_model


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Task = namedtuple("Task", ["machine", "duration"])
# A job is a list of Tasks that must be executed in order (precedence).

# Classic 3-job, 3-machine instance (Fisher & Thompson ft03).
# Optimal makespan = 11.
JOBS_3x3: list[list[Task]] = [
    [Task(0, 3), Task(1, 2), Task(2, 2)],  # Job 0: machine 0 for 3, then 1 for 2, then 2 for 2
    [Task(0, 2), Task(2, 1), Task(1, 4)],  # Job 1: machine 0 for 2, then 2 for 1, then 1 for 4
    [Task(1, 4), Task(2, 3)],              # Job 2: machine 1 for 4, then 2 for 3
]

# Larger instance: 4 jobs, 3 machines.
JOBS_4x3: list[list[Task]] = [
    [Task(0, 3), Task(1, 2), Task(2, 2)],
    [Task(0, 2), Task(2, 1), Task(1, 4)],
    [Task(1, 4), Task(2, 3), Task(0, 1)],
    [Task(2, 1), Task(0, 3), Task(1, 2)],
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_gantt_text(
    jobs: list[list[Task]],
    n_machines: int,
    starts: dict[tuple[int, int], int],
    makespan: int,
) -> None:
    """Print a text-based Gantt chart of the schedule.

    Args:
        jobs: Job definitions (list of list of Task).
        n_machines: Number of machines.
        starts: Dictionary mapping (job_id, task_id) -> start time.
        makespan: Optimal makespan value.
    """
    print(f"\nMakespan: {makespan}")
    print(f"{'Time':>6}", end="")
    for t in range(makespan):
        print(f"{t:>3}", end="")
    print()

    for m in range(n_machines):
        row = ["."] * makespan
        for j, job in enumerate(jobs):
            for t_idx, task in enumerate(job):
                if task.machine == m:
                    s = starts[(j, t_idx)]
                    for dt in range(task.duration):
                        if s + dt < makespan:
                            row[s + dt] = str(j)
        print(f"M{m:>4} |", end="")
        for ch in row:
            print(f"  {ch}", end="")
        print()
    print()


def print_solver_stats(solver: cp_model.CpSolver) -> None:
    """Print solver statistics."""
    print(f"  Wall time    : {solver.wall_time:.4f}s")
    print(f"  Branches     : {solver.num_branches}")
    print(f"  Conflicts    : {solver.num_conflicts}")
    print()


# ---------------------------------------------------------------------------
# TODO(human): Implement the solver
# ---------------------------------------------------------------------------

def solve_job_shop(
    jobs: list[list[Task]],
    n_machines: int,
) -> tuple[int, dict[tuple[int, int], int]] | None:
    """Solve a job-shop scheduling problem using CP-SAT.

    Args:
        jobs: List of jobs. Each job is a list of Task(machine, duration)
              that must be executed in order.
        n_machines: Number of machines.

    Returns:
        Tuple of (makespan, starts) where starts maps (job_id, task_id)
        to the start time of that task. Returns None if infeasible.
    """
    # TODO(human): Job-Shop Scheduling with CP-SAT
    #
    # Each job consists of tasks that must be done in order (precedence).
    # Each task runs on a specific machine for a given duration.
    # No two tasks can run on the same machine at the same time.
    # Minimize makespan (completion time of all jobs).
    #
    # Variables: for each task (job j, task t):
    #   start[j][t] = IntVar(0, horizon)
    #   end[j][t] = IntVar(0, horizon)
    #   interval[j][t] = model.new_interval_var(start, duration, end, name)
    #   where horizon = sum of all task durations (safe upper bound)
    #
    # Constraints:
    #   1. Precedence: end[j][t] <= start[j][t+1] for consecutive tasks in same job
    #   2. No overlap: model.add_no_overlap(intervals on same machine)
    #      Collect all interval vars assigned to each machine, then add one
    #      NoOverlap constraint per machine.
    #   3. Makespan: create makespan = IntVar(0, horizon), then
    #      model.add(makespan >= end[j][last_task]) for each job j
    #
    # Objective: model.minimize(makespan)
    #
    # After solving, extract start times: solver.value(start[j][t])
    # Return (makespan_value, {(j, t): start_time, ...})
    #
    # The NoOverlap constraint is what makes this a CP problem — it is a
    # global constraint that reasons about time intervals on a single machine.
    # CP-SAT is state-of-the-art for job-shop: solves benchmark instances
    # with 100+ jobs efficiently.
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=== Job-Shop Scheduling ===\n")

    # 3x3 instance (optimal = 11)
    print("--- 3 jobs, 3 machines (optimal makespan = 11) ---")
    result = solve_job_shop(JOBS_3x3, n_machines=3)
    if result:
        makespan, starts = result
        print_gantt_text(JOBS_3x3, 3, starts, makespan)

    # 4x3 instance
    print("--- 4 jobs, 3 machines ---")
    result = solve_job_shop(JOBS_4x3, n_machines=3)
    if result:
        makespan, starts = result
        print_gantt_text(JOBS_4x3, 3, starts, makespan)


if __name__ == "__main__":
    main()
