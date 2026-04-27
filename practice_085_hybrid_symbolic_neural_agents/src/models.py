"""Shared Pydantic data models for the hybrid symbolic-neural pipeline.

These types are the *interface* between the LLM stages and the CP-SAT
stage.  Keeping them small, typed, and JSON-serialisable is what lets
each stage be tested in isolation:

    natural language ─► ScheduleRequest ─► (CP-SAT) ─► ScheduleSolution
                                                  └─► InfeasibleResult ─► Conflict
                                                                        ─► explanation

Conventions:
- Employees are identified by short string IDs ("alice", "bob", ...).
- Shifts are identified by ``ShiftId`` strings ("mon_morning", ...).
- Hard constraints MUST hold; soft constraints carry a positive weight
  and contribute to the objective when violated.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmployeeId = str
ShiftId = str


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


class Employee(BaseModel):
    """An employee available for the schedule."""
    id: EmployeeId
    name: str
    max_shifts: int = Field(ge=0, description="Max shifts this employee can be assigned across the planning horizon.")
    qualifications: list[str] = Field(default_factory=list, description="Tags like 'barista', 'manager'.")


class Shift(BaseModel):
    """A shift slot to be filled."""
    id: ShiftId
    day: str = Field(description="e.g. 'mon', 'tue', ... or an ISO date.")
    label: str = Field(description="Human-readable label, e.g. 'morning', 'evening'.")
    required_qualification: str | None = None
    demand: int = Field(default=1, ge=1, description="Number of employees required for this shift.")


# Hard constraints supported by the CP-SAT model. Each is a discriminated
# union variant so the LLM has a closed taxonomy to extract into.

class CoverageConstraint(BaseModel):
    """Each shift must have exactly ``shift.demand`` employees assigned."""
    kind: Literal["coverage"] = "coverage"


class MaxShiftsPerEmployee(BaseModel):
    """No employee exceeds their personal ``max_shifts`` cap."""
    kind: Literal["max_shifts_per_employee"] = "max_shifts_per_employee"


class ForbiddenAssignment(BaseModel):
    """Employee ``employee_id`` cannot work shift ``shift_id``."""
    kind: Literal["forbidden_assignment"] = "forbidden_assignment"
    employee_id: EmployeeId
    shift_id: ShiftId


class RequiredAssignment(BaseModel):
    """Employee ``employee_id`` MUST work shift ``shift_id``."""
    kind: Literal["required_assignment"] = "required_assignment"
    employee_id: EmployeeId
    shift_id: ShiftId


class QualificationConstraint(BaseModel):
    """An employee can only work a shift if they have its required qualification."""
    kind: Literal["qualification"] = "qualification"


HardConstraint = (
    CoverageConstraint
    | MaxShiftsPerEmployee
    | ForbiddenAssignment
    | RequiredAssignment
    | QualificationConstraint
)


class SoftPreference(BaseModel):
    """A weighted preference used in the objective.

    ``weight`` > 0 rewards the assignment, ``weight`` < 0 penalises it.
    Soft preferences never make a model infeasible.
    """
    employee_id: EmployeeId
    shift_id: ShiftId
    weight: int = Field(description="Positive = preferred, negative = avoid.")


class ScheduleRequest(BaseModel):
    """Structured form of the user's natural-language scheduling problem.

    This is what the LLM extraction stage emits and what the CP-SAT
    solver consumes.  Both stages are validated by Pydantic, so a
    malformed extraction is caught before it reaches the solver.
    """
    employees: list[Employee]
    shifts: list[Shift]
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
    soft_preferences: list[SoftPreference] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Solver outputs
# ---------------------------------------------------------------------------


class Assignment(BaseModel):
    employee_id: EmployeeId
    shift_id: ShiftId


class ScheduleSolution(BaseModel):
    """A feasible assignment found by CP-SAT."""
    assignments: list[Assignment]
    objective_value: int = Field(default=0, description="Sum of soft-preference weights satisfied.")
    solver_status: str = Field(description="'OPTIMAL' | 'FEASIBLE'")


class InfeasibleResult(BaseModel):
    """The solver could not find any feasible assignment.

    ``assumption_indices`` are the indices into the *original* request's
    ``hard_constraints`` list that were attached as CP-SAT assumption
    literals.  They are what the infeasibility analyser inspects to
    extract a minimum unsatisfiable subset.
    """
    assumption_indices: list[int]
    solver_status: str = Field(description="'INFEASIBLE' | 'UNKNOWN' | 'MODEL_INVALID'")


class Conflict(BaseModel):
    """A minimal subset of constraints that cannot be jointly satisfied.

    Produced by the infeasibility analyser from an ``InfeasibleResult``.
    The ``conflicting_constraints`` are a subset of the original request's
    ``hard_constraints`` — small enough that the LLM explainer can turn
    them into a useful natural-language message.
    """
    conflicting_constraints: list[HardConstraint]
    summary: str = Field(default="", description="Optional short label, e.g. 'understaffed coverage'.")


# ---------------------------------------------------------------------------
# Orchestrator output
# ---------------------------------------------------------------------------


class AgentResult(BaseModel):
    """Top-level result returned by the orchestrator."""
    solved: bool
    solution: ScheduleSolution | None = None
    conflict: Conflict | None = None
    explanation: str = ""
