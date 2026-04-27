"""Hand-built ScheduleRequest fixtures used to test the symbolic core.

These bypass the LLM extraction stage entirely so you can exercise the
CP-SAT solver, the infeasibility analyser, and the explainer without
depending on a flaky 7B-parameter local model getting the JSON right.

Each function returns a ``ScheduleRequest`` and is annotated with the
expected solver outcome (feasible / infeasible).  ``demo.py`` and the
unit-style scripts in ``_02_*`` and ``_03_*`` use these.

Don't add a TODO here — these are reference fixtures, not exercises.
"""

from __future__ import annotations

from .models import (
    CoverageConstraint,
    Employee,
    ForbiddenAssignment,
    MaxShiftsPerEmployee,
    QualificationConstraint,
    RequiredAssignment,
    ScheduleRequest,
    Shift,
    SoftPreference,
)


def simple_feasible() -> ScheduleRequest:
    """A small problem with an obvious schedule.

    3 employees, 3 shifts, every shift demand=1.  Bob is the only
    manager and the manager-required shift is Tue morning, so Bob has
    to take it.  Alice can't work mornings, so Carol takes Mon morning,
    and Alice takes Mon evening.

    Expected: FEASIBLE.
    """
    return ScheduleRequest(
        employees=[
            Employee(id="alice", name="Alice", max_shifts=3, qualifications=["barista"]),
            Employee(id="bob", name="Bob", max_shifts=3, qualifications=["barista", "manager"]),
            Employee(id="carol", name="Carol", max_shifts=2, qualifications=["barista"]),
        ],
        shifts=[
            Shift(id="mon_morning", day="mon", label="morning", required_qualification="barista"),
            Shift(id="mon_evening", day="mon", label="evening", required_qualification="barista"),
            Shift(id="tue_morning", day="tue", label="morning", required_qualification="manager"),
        ],
        hard_constraints=[
            CoverageConstraint(),
            MaxShiftsPerEmployee(),
            QualificationConstraint(),
            ForbiddenAssignment(employee_id="alice", shift_id="mon_morning"),
            ForbiddenAssignment(employee_id="alice", shift_id="tue_morning"),
        ],
        soft_preferences=[
            SoftPreference(employee_id="bob", shift_id="tue_morning", weight=3),
        ],
    )


def feasible_with_required() -> ScheduleRequest:
    """Same as simple_feasible but with a pre-assigned shift.

    Carol is required to be on mon_morning. The solver must respect it.

    Expected: FEASIBLE.
    """
    base = simple_feasible()
    base.hard_constraints.append(
        RequiredAssignment(employee_id="carol", shift_id="mon_morning")
    )
    return base


def infeasible_overdemand() -> ScheduleRequest:
    """Demand exceeds total available capacity.

    4 shifts of demand=1 each, but the only two employees have
    max_shifts=1, so total capacity is 2 < 4.

    Expected: INFEASIBLE.  The conflict should mention coverage
    against max_shifts_per_employee.
    """
    return ScheduleRequest(
        employees=[
            Employee(id="alice", name="Alice", max_shifts=1, qualifications=["barista"]),
            Employee(id="bob", name="Bob", max_shifts=1, qualifications=["barista"]),
        ],
        shifts=[
            Shift(id=f"slot_{i}", day="mon", label=f"slot_{i}", required_qualification="barista")
            for i in range(4)
        ],
        hard_constraints=[
            CoverageConstraint(),
            MaxShiftsPerEmployee(),
            QualificationConstraint(),
        ],
    )


def infeasible_qualification() -> ScheduleRequest:
    """No-one is qualified for the manager shift.

    Two baristas, no manager, but a manager-required shift.

    Expected: INFEASIBLE.  The conflict should mention coverage and
    qualification.
    """
    return ScheduleRequest(
        employees=[
            Employee(id="alice", name="Alice", max_shifts=3, qualifications=["barista"]),
            Employee(id="bob", name="Bob", max_shifts=3, qualifications=["barista"]),
        ],
        shifts=[
            Shift(id="tue_morning", day="tue", label="morning", required_qualification="manager"),
        ],
        hard_constraints=[
            CoverageConstraint(),
            MaxShiftsPerEmployee(),
            QualificationConstraint(),
        ],
    )


def infeasible_required_vs_forbidden() -> ScheduleRequest:
    """Direct contradiction: Alice is both required and forbidden on mon_morning.

    Expected: INFEASIBLE.  The conflict should be {required_assignment,
    forbidden_assignment}.
    """
    return ScheduleRequest(
        employees=[
            Employee(id="alice", name="Alice", max_shifts=3, qualifications=["barista"]),
        ],
        shifts=[
            Shift(id="mon_morning", day="mon", label="morning", required_qualification="barista"),
        ],
        hard_constraints=[
            CoverageConstraint(),
            MaxShiftsPerEmployee(),
            QualificationConstraint(),
            RequiredAssignment(employee_id="alice", shift_id="mon_morning"),
            ForbiddenAssignment(employee_id="alice", shift_id="mon_morning"),
        ],
    )


ALL_CASES = [
    ("simple_feasible", simple_feasible, "feasible"),
    ("feasible_with_required", feasible_with_required, "feasible"),
    ("infeasible_overdemand", infeasible_overdemand, "infeasible"),
    ("infeasible_qualification", infeasible_qualification, "infeasible"),
    ("infeasible_required_vs_forbidden", infeasible_required_vs_forbidden, "infeasible"),
]
