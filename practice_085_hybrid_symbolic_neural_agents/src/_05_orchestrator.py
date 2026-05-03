"""Phase 5 — Orchestrator: tying the pipeline together

The orchestrator is the *agent* of the practice — the layer that
decides which tool to call when.  Its control flow is deliberately
simple, because the smarts live in the individual stages:

    user request
        │
        ▼  _01_constraint_extraction.extract_schedule_request
    ScheduleRequest
        │
        ▼  _02_cpsat_solver.solve
    ScheduleSolution ─────────────►  _04_explainer.explain
        │ or
        ▼ InfeasibleResult
    _03_infeasibility_analyzer.analyse
        │
        ▼ Conflict
    _04_explainer.explain

DESIGN DISCUSSION (read this before implementing):

You have two reasonable choices for the orchestration layer:

(A) **Hand-rolled state machine** — a plain Python function that calls
    each stage in sequence and branches on the solver result.  Pros:
    zero dependencies, trivial to test, every transition is visible.
    Cons: doesn't compose if you later want to add memory, retries, or
    a human-in-the-loop "relax which constraint?" step.

(B) **LangGraph** (or similar agent framework) — model the pipeline as
    a graph of nodes with an explicit state object.  Pros: gives you
    streaming, checkpointing, conditional edges, and a nice mental
    model when the pipeline grows.  Cons: extra dependency for a
    pipeline that is fundamentally linear at this stage.

For this practice the pipeline is linear with exactly one branch
(feasible vs infeasible), so (A) is the right call.  Implement
`run_agent()` as a hand-rolled state machine.  When you do practice
029b / 030c later you'll see what the LangGraph version looks like.

(If you disagree, this is a great place to argue back in the docstring
when you implement.)

Run on its own to drive the whole pipeline end-to-end:
    uv run python -m src._05_orchestrator
"""

from __future__ import annotations

from .llm_config import LMConfig, get_lm
from .models import (
    AgentResult,
    InfeasibleResult,
    ScheduleSolution,
)


# ---------------------------------------------------------------------------
# TODO(human) — The agent loop
# ---------------------------------------------------------------------------
# Wire the four stages into a single function.  No new logic — just
# orchestration.
#
# What to do:
#   1. Resolve the LM config (`cfg = cfg or get_lm()`) so all LLM stages
#      share the same provider for this run.
#   2. Stage 1 — extraction:
#        from ._01_constraint_extraction import extract_schedule_request
#        request = extract_schedule_request(user_request, cfg)
#   3. Stage 2 — solve:
#        from ._02_cpsat_solver import solve
#        result = solve(request)
#   4. Branch on the result:
#        - ScheduleSolution:
#            from ._04_explainer import explain
#            explanation = explain(result, cfg)
#            return AgentResult(solved=True, solution=result, explanation=explanation)
#        - InfeasibleResult:
#            from ._03_infeasibility_analyzer import analyse
#            from ._04_explainer import explain
#            conflict = analyse(request, result)
#            explanation = explain(conflict, cfg)
#            return AgentResult(solved=False, conflict=conflict, explanation=explanation)
#
# Notice: this function is the *only* place that knows about the whole
# pipeline.  Each stage is independently testable, and swapping one out
# (e.g. a different solver) only touches one node.  That's the
# Single-Responsibility / Indirection lesson of the practice.
# ---------------------------------------------------------------------------
def run_agent(user_request: str, cfg: LMConfig | None = None) -> AgentResult:
    """Run the full hybrid symbolic-neural pipeline on a user request."""
    cfg = cfg or get_lm()

    from ._01_constraint_extraction import extract_schedule_request
    request = extract_schedule_request(user_request, cfg)

    from ._02_cpsat_solver import solve
    result = solve(request)


    from ._04_explainer import explain
    match result:
        case ScheduleSolution():
            explanation = explain(result, cfg)
            return AgentResult(solved=True, solution=result, explanation = explanation)

        case InfeasibleResult():
            from ._03_infeasibility_analyzer import analyse
            conflict = analyse(request, result)
            explanation = explain(conflict, cfg)
            return AgentResult(solved=False, conflict=conflict, explanation=explanation)

# -- Sanity demo (scaffolded) -----------------------------------------------


_EXAMPLE = (
    "Schedule the coffee shop for Monday and Tuesday. "
    "Monday morning needs 1 barista, Monday evening needs 1 barista, "
    "Tuesday morning needs 1 manager. "
    "Alice is a barista (max 3 shifts) and cannot work mornings. "
    "Bob is a barista and a manager (max 3 shifts) and prefers Tuesday morning. "
    "Carol is a barista (max 2 shifts)."
)


def main() -> None:
    print("Running the hybrid symbolic-neural agent end-to-end...\n")
    print("USER REQUEST:")
    print(f"  {_EXAMPLE}\n")
    result = run_agent(_EXAMPLE)
    print(f"solved = {result.solved}")
    if result.solution is not None:
        print(f"  {len(result.solution.assignments)} assignments, "
              f"objective = {result.solution.objective_value}")
    if result.conflict is not None:
        print(f"  conflict: {result.conflict.summary}")
    print("\nEXPLANATION:")
    print(result.explanation)


if __name__ == "__main__":
    main()
