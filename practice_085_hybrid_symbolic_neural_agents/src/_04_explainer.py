"""Phase 4 — Explainer: ScheduleSolution / Conflict → natural language

The last LLM stage in the pipeline.  Its job is to turn a structured
output from the symbolic core into a message a human will actually
read.  Crucially, the LLM is NOT making any decisions here — every
fact in the explanation is already in the structured input.

Two modes:
- **Solution explainer** — given a ``ScheduleSolution``, summarise who
  works what, and call out which preferences were honoured.
- **Conflict explainer** — given a ``Conflict``, tell the user *why*
  the schedule is impossible and suggest which constraint to relax.

We use a low temperature (0.0) and a fixed system prompt so the same
input gives the same output (mostly).  The system prompt is kept stable
so the LLM provider's prompt cache can hit on repeated calls — see
https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching for
why a stable prefix matters.

Run on its own to see both modes:
    uv run python -m src._04_explainer
"""

from __future__ import annotations

import textwrap

from .llm_config import LMConfig, chat, get_lm
from .models import Conflict, ScheduleSolution

EXPLAINER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a scheduling assistant.  You will be given either a
    feasible schedule (list of assignments + objective value) or a
    conflict (a small set of constraints that cannot be jointly
    satisfied).  Produce a short, friendly natural-language message
    (3–6 sentences):

    - For a feasible schedule: list the assignments grouped by shift,
      then mention which preferences were satisfied (objective_value > 0).
    - For a conflict: explain in plain English which constraints clash
      and recommend ONE concrete relaxation (e.g. "raise Alice's
      max_shifts to 4", "drop the manager qualification on Tue
      morning").  Do NOT invent constraints that aren't in the input.

    Never produce JSON or code blocks. Only prose.
    """
)


def _format_solution(solution: ScheduleSolution) -> str:
    lines = [f"Status: {solution.solver_status}", f"Objective: {solution.objective_value}"]
    for a in solution.assignments:
        lines.append(f"  {a.employee_id} -> {a.shift_id}")
    return "\n".join(lines)


def _format_conflict(conflict: Conflict) -> str:
    lines = [f"Summary: {conflict.summary}", "Conflicting constraints:"]
    for c in conflict.conflicting_constraints:
        payload = c.model_dump(exclude={"kind"})
        lines.append(f"  - {c.kind}: {payload}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# TODO(human) — Single explain() entry point
# ---------------------------------------------------------------------------
# Both branches end up calling the same `chat(...)` with the same
# system prompt — only the user message body differs.  That's
# intentional: a stable system prompt is what lets prompt caching hit
# across calls.
#
# What to do:
#   1. Build the user message body:
#        - if `payload` is a ScheduleSolution: body = _format_solution(payload)
#        - if `payload` is a Conflict:         body = _format_conflict(payload)
#        - else: raise TypeError with the offending type name.
#   2. Build messages:
#        [{"role": "system", "content": EXPLAINER_SYSTEM_PROMPT},
#         {"role": "user",   "content": body}]
#   3. Call `chat(cfg, messages, temperature=0.0, max_tokens=400)` and
#      return the stripped string.
#
# Why a single function for both branches: the orchestrator
# (`_05_orchestrator.py`) doesn't know in advance which it'll get from
# the solver, so a uniform `explain(...)` keeps its control flow simple.
# ---------------------------------------------------------------------------
def explain(payload: ScheduleSolution | Conflict, cfg: LMConfig | None = None) -> str:
    """Produce a natural-language explanation of a solver output."""
    cfg = cfg or get_lm()
    raise NotImplementedError(
        "TODO(human): build messages from EXPLAINER_SYSTEM_PROMPT + a formatted "
        "view of `payload`, call chat(), and return the stripped string."
    )


# -- Sanity demo (scaffolded) -----------------------------------------------


def main() -> None:
    from ._02_cpsat_solver import solve
    from ._03_infeasibility_analyzer import analyse
    from .golden_cases import infeasible_overdemand, simple_feasible
    from .models import InfeasibleResult

    print("=== Solution explanation ===\n")
    feasible = solve(simple_feasible())
    if isinstance(feasible, ScheduleSolution):
        print(explain(feasible))
    else:
        print("Feasible golden case unexpectedly returned an InfeasibleResult.")

    print("\n=== Conflict explanation ===\n")
    request = infeasible_overdemand()
    result = solve(request)
    if isinstance(result, InfeasibleResult):
        conflict = analyse(request, result)
        print(explain(conflict))
    else:
        print("Infeasible golden case unexpectedly returned a feasible solution.")


if __name__ == "__main__":
    main()
