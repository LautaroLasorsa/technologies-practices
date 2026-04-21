"""Exercise 03: Compare GraphRAG query modes.

Implement the single core operation:
  - `run_question_across_modes`: for one question, run it through each
    mode via `run_graphrag_query()` and return a `{mode: response}` dict.

The CLI wrapper, per-question printout, and tabulated summary table are
all scaffolded.  The point is to *see* how local, global, and basic
search answer differently on entity-specific vs thematic questions.

Run (after `graphrag index --root .`):
    uv run python -m src._03_query_comparison
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local + basic are fast enough; global re-runs the LLM over every community
# report, so on a 7B model it is the slowest mode.  drift is skipped by
# default (most expensive).
MODES = ("local", "global", "basic")

# The first two questions reward local search (entity-centric); the last
# two reward global search (corpus-wide thematic).
QUESTIONS = [
    "What breakthrough did QuantumCore Technologies achieve and who led the research?",
    "What is the relationship between NeuralPath Systems and Cascade Robotics?",
    "What are the major themes and trends in Meridian City's tech ecosystem?",
    "How do the different organizations in Meridian City collaborate with each other?",
]


# -- CLI wrapper (scaffolded) ---------------------------------------------


def run_graphrag_query(question: str, method: str) -> str:
    """Run a graphrag CLI query and return the response text (or an error)."""
    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(PROJECT_ROOT),
        "--method", method,
        "--query", question,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return f"[ERROR] {result.stderr[:200]}"
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Query timed out after 120s"
    except Exception as e:
        return f"[ERROR] {e}"


# -- TODO -----------------------------------------------------------------


def run_question_across_modes(question: str, modes: tuple[str, ...] = MODES) -> dict[str, str]:
    """Run *question* through every mode in *modes* and collect the answers.

    For each mode, call `run_graphrag_query(question, mode)` and store the
    response under that mode's name.  Print a progress line before each
    call since queries on local models take 30-60 s each.
    """
    # TODO(human): iterate modes, call run_graphrag_query, return {mode: response}.
    raise NotImplementedError("Implement run_question_across_modes()")


# -- Presentation (scaffolded) --------------------------------------------


def _print_per_question(question: str, responses: dict[str, str]) -> None:
    print("\n" + "=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    for mode, text in responses.items():
        print(f"\n--- {mode} ---")
        truncated = text if len(text) <= 500 else text[:500] + "..."
        print(truncated)


def _print_summary_table(all_results: list[tuple[str, dict[str, str]]]) -> None:
    headers = ["question"] + list(MODES)
    rows = []
    for question, responses in all_results:
        row = [question[:50]]
        for mode in MODES:
            text = responses.get(mode, "")
            row.append(text[:100] if text else "")
        rows.append(row)
    print("\n\nSummary table (first 100 chars per mode):")
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# -- Main orchestrator ----------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Exercise 3: Query Mode Comparison")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Questions: {len(QUESTIONS)}  modes: {MODES}")
    print("Expect several minutes with a local 7B model.")
    print("-" * 60)

    all_results: list[tuple[str, dict[str, str]]] = []
    for i, question in enumerate(QUESTIONS, start=1):
        print(f"\n[{i}/{len(QUESTIONS)}] running modes for: {question[:60]}...")
        responses = run_question_across_modes(question)
        _print_per_question(question, responses)
        all_results.append((question, responses))

    _print_summary_table(all_results)


if __name__ == "__main__":
    main()
