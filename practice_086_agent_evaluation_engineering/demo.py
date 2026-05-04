"""End-to-end demo of the agent-evaluation harness.

Runs the full eval (system A vs system B) on the first 5 golden tasks
with k=3 rollouts each, then prints a comparison table.

Until you implement the TODO(human) functions in
``src/_01_*.py`` … ``src/_05_*.py``, this script will raise
``NotImplementedError`` from the first stage that isn't done yet.
That's expected — the order of failures matches the order in which
you should attack the exercises.

Run:
    uv run python demo.py
"""

from __future__ import annotations

from src._06_run_eval import print_report, run_eval
from src.golden_cases import GOLDEN_TASKS


def main() -> None:
    print("=" * 60)
    print("Practice 086 — Agent Evaluation Engineering")
    print("=" * 60)
    tasks = GOLDEN_TASKS[:5]
    print(f"\nEvaluating {len(tasks)} tasks, k=3 rollouts per task, 2 systems (A, B)\n")
    report = run_eval(tasks, k=3, seed=0)
    print_report(report)


if __name__ == "__main__":
    main()
