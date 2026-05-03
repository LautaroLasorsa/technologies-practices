"""Phase 3 — Infeasibility Analyser: InfeasibleResult → Conflict

When the CP-SAT solver returns INFEASIBLE, it has proved no schedule
exists that satisfies all hard constraints.  But "INFEASIBLE" alone is
useless to the user — they need to know *which* constraints conflict.

This is the killer feature of hybrid systems and the reason we used
**assumption literals** in Phase 2.  CP-SAT exposes a method that
returns a *sufficient* subset of assumptions whose presence already
makes the model infeasible:

    solver.SufficientAssumptionsForInfeasibility() -> list[int]

That's a Minimum (or near-minimum) Unsatisfiable Subset (MUS), and it's
exactly what the LLM explainer in Phase 4 needs: a small, focused list
of conflicting constraints, not the entire request.

References:
- OR-Tools CP-SAT docs:
  https://developers.google.com/optimization/cp/cp_solver
- Background on MUS / IIS:
  https://en.wikipedia.org/wiki/Unsatisfiable_core
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from .models import (
    Conflict,
    ForbiddenAssignment,
    HardConstraint,
    InfeasibleResult,
    QualificationConstraint,
    RequiredAssignment,
    ScheduleRequest,
)


def _short_label(constraints: list[HardConstraint]) -> str:
    """Human-friendly one-line summary of the conflict shape."""
    kinds = sorted({c.kind for c in constraints})
    if "coverage" in kinds and "max_shifts_per_employee" in kinds:
        return "demand exceeds employee capacity"
    if "coverage" in kinds and "qualification" in kinds:
        return "not enough qualified employees for required shifts"
    if "required_assignment" in kinds and "forbidden_assignment" in kinds:
        return "required and forbidden assignments contradict"
    return ", ".join(kinds)


# ---------------------------------------------------------------------------
# TODO(human) — Reproduce the model and ask CP-SAT for a sufficient subset
# ---------------------------------------------------------------------------
# `solve()` already returned an InfeasibleResult, but the CpSolver object
# is gone — we need to rebuild the model to query it.  This is a
# deliberate design choice: the solver state is not portable across
# function calls, so we reconstruct.
#
# What to do:
#   1. Import `solve`'s helper from Phase 2 to rebuild:
#          from ._02_cpsat_solver import _build_model
#      and call `built = _build_model(request)`.
#   2. Create a fresh `cp_model.CpSolver()`, set
#          solver.parameters.max_time_in_seconds = 5.0
#      and call `solver.Solve(built.model)`.
#   3. Call `indices = solver.SufficientAssumptionsForInfeasibility()`.
#      This returns the *indices* (into the model's assumption list,
#      i.e. `built.assumption_lits`) of a subset of assumptions
#      sufficient to make the model infeasible.
#   4. Map those indices back to the original HardConstraint objects via
#      `built.constraints_in_order` and collect them into a list.
#   5. Return `Conflict(conflicting_constraints=..., summary=_short_label(...))`.
#
# Edge case: if `indices` is empty (solver couldn't extract one — e.g.
# MODEL_INVALID), fall back to returning the full
# `request.hard_constraints` list as the conflict; the explainer can
# still produce a coarse message.
#
# Note: we accept the ``InfeasibleResult`` parameter to keep the API
# symmetric with the rest of the pipeline, but the core work is on
# ``request`` because that's what we rebuild from.
# ---------------------------------------------------------------------------
def analyse(request: ScheduleRequest, infeasible: InfeasibleResult) -> Conflict:
    """Extract the minimum unsatisfiable subset of hard constraints."""

    from ._02_cpsat_solver import _build_model

    built = _build_model(request)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.Solve(built.model)

    lit_to_idx = {lit.Index():i for i,lit in enumerate(built.assumption_lits)}
    indices = list[int](solver.SufficientAssumptionsForInfeasibility())

    conflicts = [built.constraints_in_order[lit_to_idx[i]] for i in indices] if indices else request.hard_constraints
    return Conflict(conflicting_constraints=conflicts, summary = _short_label(conflicts))
# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    """Run the analyser on a known-infeasible golden case."""
    from ._02_cpsat_solver import solve
    from .golden_cases import infeasible_overdemand

    request = infeasible_overdemand()
    result = solve(request)
    if not isinstance(result, InfeasibleResult):
        print("Expected an InfeasibleResult but got a feasible solution.")
        print("Pick a different golden case.")
        return

    print(f"Solver status: {result.solver_status}")
    print(f"All hard constraints: {len(request.hard_constraints)}")
    conflict = analyse(request, result)
    print(f"\nConflict ({conflict.summary}):")
    for c in conflict.conflicting_constraints:
        print(f"  - {c.kind}: {c.model_dump(exclude={'kind'})}")


if __name__ == "__main__":
    main()
