"""End-to-end demo of the hybrid symbolic-neural agent.

Drives the whole pipeline (extract → solve → analyse → explain) on one
example natural-language scheduling request.  This script is intentionally
small — the work happens inside the ``src/_NN_*.py`` modules.

Until you implement the TODO(human) functions in those modules, this
script will raise ``NotImplementedError`` from the first stage that
isn't done yet.  That's expected — the order of failures matches the
order in which you should attack the exercises.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from src._05_orchestrator import run_agent

USER_REQUEST = """
Schedule the coffee shop for Monday and Tuesday.

Shifts:
  - Monday morning (1 barista)
  - Monday evening (1 barista)
  - Tuesday morning (1 manager)

Employees:
  - Alice, barista, max 3 shifts. Cannot work mornings.
  - Bob, barista AND manager, max 3 shifts. Prefers Tuesday morning.
  - Carol, barista, max 2 shifts.
""".strip()


def main() -> None:
    print("=" * 60)
    print("Practice 085 — Hybrid Symbolic-Neural Agent")
    print("=" * 60)
    print("\nUSER REQUEST")
    print("-" * 60)
    print(USER_REQUEST)

    print("\nRUNNING PIPELINE")
    print("-" * 60)
    result = run_agent(USER_REQUEST)

    print("\nRESULT")
    print("-" * 60)
    print(f"solved        = {result.solved}")
    if result.solution is not None:
        print(f"objective     = {result.solution.objective_value}")
        print(f"assignments   = {len(result.solution.assignments)}")
        for a in result.solution.assignments:
            print(f"  {a.employee_id:>8} -> {a.shift_id}")
    if result.conflict is not None:
        print(f"conflict      = {result.conflict.summary}")
        for c in result.conflict.conflicting_constraints:
            print(f"  - {c.kind}: {c.model_dump(exclude={'kind'})}")

    print("\nEXPLANATION")
    print("-" * 60)
    print(result.explanation)


if __name__ == "__main__":
    main()
