"""Phase 2 — CP-SAT Solver: ScheduleRequest → ScheduleSolution / InfeasibleResult

This is the **symbolic core** of the hybrid agent.  No LLM is involved
here, by design.  CP-SAT (Google OR-Tools) is a state-of-the-art
constraint programming solver that combines:

  - **Constraint propagation** — each new variable assignment prunes the
    domains of related variables, narrowing the search.
  - **Conflict-driven backtracking (CDCL)** — when the search hits a
    dead end, it learns a no-good clause so it never revisits the same
    failure.
  - **Search heuristics** — variable/value ordering chosen automatically
    for combinatorial problems.

Why CP-SAT vs. an LLM "solver":
  - LLMs cannot reliably enumerate exponential-size assignment spaces.
  - CP-SAT is *complete* — if it returns INFEASIBLE the problem really
    has no solution.
  - It returns in milliseconds for problems with ~100 variables.

Why CP-SAT vs. MILP:
  - CP-SAT excels at logical/Boolean constraints (this is a Boolean
    matrix problem); MILP shines for linear-numeric problems.

We use **assumption literals** for every hard constraint.  When the
problem is infeasible, the solver's ``SufficientAssumptionsForInfeasibility``
returns a subset of those literals — that's what Phase 3 turns into a
human-readable conflict.

Run on its own to sanity-check the solver:
    uv run python -m src._02_cpsat_solver
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model

from .models import (
    Assignment,
    HardConstraint,
    InfeasibleResult,
    ScheduleRequest,
    ScheduleSolution,
)

# Status names for human-readable output
_STATUS_NAME = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
}


@dataclass
class _BuiltModel:
    """Container for the model and its decision variables.

    ``x[(emp_id, shift_id)] == 1`` iff that employee is assigned to that
    shift.  ``assumption_lits[i]`` is the assumption literal attached to
    the i-th hard constraint of the request — used by Phase 3 to
    reconstruct which constraints participate in the conflict.
    """
    model: cp_model.CpModel
    x: dict[tuple[str, str], cp_model.IntVar]
    assumption_lits: list[cp_model.IntVar]
    constraints_in_order: list[HardConstraint]


def _build_decision_vars(
    request: ScheduleRequest, model: cp_model.CpModel
) -> dict[tuple[str, str], cp_model.IntVar]:
    """One BoolVar per (employee, shift) pair."""
    x: dict[tuple[str, str], cp_model.IntVar] = {}
    for e in request.employees:
        for s in request.shifts:
            x[(e.id, s.id)] = model.NewBoolVar(f"x_{e.id}_{s.id}")
    return x


# ---------------------------------------------------------------------------
# TODO(human) — Add hard constraints with assumption literals
# ---------------------------------------------------------------------------
# Goal: encode each item in `request.hard_constraints` as one or more
# CP-SAT constraints, gated by an assumption literal.  Returning the
# list of assumption literals (in order) is what makes infeasibility
# explainable later.
#
# Pattern for each constraint:
#   1. lit = model.NewBoolVar(f"assume_{i}_{constraint.kind}")
#   2. Encode the constraint as one or more `model.Add(...)` clauses,
#      each with `.OnlyEnforceIf(lit)`.  This means the constraint is
#      active iff `lit == True`.
#   3. Append `lit` to `assumption_lits` in the same order as the
#      `request.hard_constraints` list.
#
# Encoding cookbook for the constraint variants in models.py:
#
#   CoverageConstraint:
#     For each shift s:  sum(x[e.id, s.id] for e in employees) == s.demand
#
#   MaxShiftsPerEmployee:
#     For each employee e:  sum(x[e.id, s.id] for s in shifts) <= e.max_shifts
#
#   ForbiddenAssignment(employee_id, shift_id):
#     x[employee_id, shift_id] == 0
#
#   RequiredAssignment(employee_id, shift_id):
#     x[employee_id, shift_id] == 1
#
#   QualificationConstraint:
#     For each (employee, shift) pair where the shift has a
#     required_qualification that the employee LACKS:
#         x[employee.id, shift.id] == 0
#
# Use `.OnlyEnforceIf(lit)` on EVERY `model.Add`.  That's how the solver
# gets a chance to relax a single constraint to prove infeasibility.
#
# Reference: https://developers.google.com/optimization/cp/cp_solver
# (look for "Solution found by assumption" / SufficientAssumptionsForInfeasibility).
# ---------------------------------------------------------------------------
def _add_hard_constraints(
    request: ScheduleRequest,
    model: cp_model.CpModel,
    x: dict[tuple[str, str], cp_model.IntVar],
) -> list[cp_model.IntVar]:
    """Add hard constraints to the model and return their assumption literals."""
    raise NotImplementedError(
        "TODO(human): for each hard constraint, create an assumption literal, "
        "encode the constraint with .OnlyEnforceIf(lit), and append lit to the result."
    )


def _add_soft_objective(
    request: ScheduleRequest,
    model: cp_model.CpModel,
    x: dict[tuple[str, str], cp_model.IntVar],
) -> None:
    """Maximise the weighted sum of satisfied soft preferences."""
    if not request.soft_preferences:
        return
    terms = []
    for p in request.soft_preferences:
        var = x.get((p.employee_id, p.shift_id))
        if var is None:
            # The preference references an unknown employee/shift; skip.
            continue
        terms.append(p.weight * var)
    if terms:
        model.Maximize(sum(terms))


def _build_model(request: ScheduleRequest) -> _BuiltModel:
    model = cp_model.CpModel()
    x = _build_decision_vars(request, model)
    assumption_lits = _add_hard_constraints(request, model, x)
    model.AddAssumptions(assumption_lits)
    _add_soft_objective(request, model, x)
    return _BuiltModel(model, x, assumption_lits, list(request.hard_constraints))


def _extract_assignments(
    built: _BuiltModel, solver: cp_model.CpSolver
) -> list[Assignment]:
    out: list[Assignment] = []
    for (emp_id, shift_id), var in built.x.items():
        if solver.Value(var) == 1:
            out.append(Assignment(employee_id=emp_id, shift_id=shift_id))
    return out


def solve(request: ScheduleRequest) -> ScheduleSolution | InfeasibleResult:
    """Solve a ScheduleRequest with CP-SAT.

    Returns a ``ScheduleSolution`` if feasible, else an
    ``InfeasibleResult`` carrying the assumption indices needed by the
    infeasibility analyser.
    """
    built = _build_model(request)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(built.model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignments = _extract_assignments(built, solver)
        objective = int(solver.ObjectiveValue()) if request.soft_preferences else 0
        return ScheduleSolution(
            assignments=assignments,
            objective_value=objective,
            solver_status=_STATUS_NAME.get(status, str(status)),
        )

    # Infeasible — record indices of all assumption literals so Phase 3
    # can ask the solver which subset is sufficient for infeasibility.
    return InfeasibleResult(
        assumption_indices=list(range(len(built.assumption_lits))),
        solver_status=_STATUS_NAME.get(status, str(status)),
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


def _demo_request() -> ScheduleRequest:
    from .golden_cases import simple_feasible
    return simple_feasible()


def main() -> None:
    print("Sanity-checking the CP-SAT solver...\n")
    request = _demo_request()
    result = solve(request)
    if isinstance(result, ScheduleSolution):
        print(f"Status: {result.solver_status}")
        print(f"Objective: {result.objective_value}")
        for a in result.assignments:
            print(f"  {a.employee_id}  ->  {a.shift_id}")
    else:
        print(f"Status: {result.solver_status}")
        print(f"Assumption indices in conflict candidate set: {result.assumption_indices}")


if __name__ == "__main__":
    main()
