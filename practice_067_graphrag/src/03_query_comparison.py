"""Exercise 3: Compare GraphRAG Query Modes.

GraphRAG provides four query modes, each suited to different question types:

- **Local Search**: Embeds the query, finds nearest entities, traverses the
  graph to build rich context (entities + relationships + text units +
  community reports), then generates an answer. Best for entity-specific
  questions like "What did Dr. James Chen discover?"

- **Global Search**: Map-reduce over pre-computed community summaries.
  Best for corpus-wide thematic questions like "What are the main themes
  in this dataset?" Expensive (processes all community reports).

- **DRIFT Search**: Hybrid of local + global. Starts entity-centric but
  generates follow-up sub-questions to broaden scope. Best quality but
  highest cost.

- **Basic Search**: Standard vector similarity over TextUnits — naive RAG
  baseline for comparison.

This exercise runs the same questions through multiple modes and compares
the results to build intuition about when each mode excels.
"""

import subprocess
import sys
from pathlib import Path

from tabulate import tabulate


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Questions designed to test different query mode strengths:
# - Entity-specific questions (local search should excel)
# - Corpus-wide thematic questions (global search should excel)
QUESTIONS = [
    # Entity-specific (local search advantage)
    "What breakthrough did QuantumCore Technologies achieve and who led the research?",
    "What is the relationship between NeuralPath Systems and Cascade Robotics?",
    # Corpus-wide thematic (global search advantage)
    "What are the major themes and trends in Meridian City's tech ecosystem?",
    "How do the different organizations in Meridian City collaborate with each other?",
]


def run_graphrag_query(question: str, method: str) -> str:
    """Run a graphrag query via CLI and return the response text.

    Args:
        question: The question to ask.
        method: One of "local", "global", "drift", "basic".

    Returns:
        The query response as a string.
    """
    cmd = [
        sys.executable, "-m", "graphrag", "query",
        "--root", str(PROJECT_ROOT),
        "--method", method,
        "--query", question,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            return f"[ERROR] {result.stderr[:200]}"
        # graphrag query outputs the response to stdout
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        return "[ERROR] Query timed out after 120s"
    except Exception as e:
        return f"[ERROR] {e}"


# TODO(human): Implement compare_query_modes
#
# This exercise builds intuition about when each GraphRAG query mode excels.
# You'll see that local search gives detailed, entity-focused answers while
# global search provides high-level thematic summaries — and basic search
# (naive RAG) often misses connections that the graph captures.
#
# 1. For each question in the QUESTIONS list, run it through three modes:
#    "local", "global", and "basic". Use the run_graphrag_query() helper
#    function provided above.
#
#    Skip "drift" for now — it's slow and expensive. If you have time later,
#    try it on one question to see how it generates follow-up sub-questions.
#
# 2. For each question, print a formatted comparison showing:
#    - The question
#    - The response from each mode (truncated to first 500 characters
#      if longer, to keep output readable)
#    - A brief label indicating which mode should theoretically excel
#      for this type of question
#
# 3. After all questions are processed, print a summary table using the
#    tabulate library showing:
#    | Question (first 50 chars) | Local (first 100 chars) | Global (first 100 chars) | Basic (first 100 chars) |
#
# Function signature:
#   def compare_query_modes() -> None
#
# Hints:
#   - The first 2 questions are entity-specific (local should excel)
#   - The last 2 questions are thematic (global should excel)
#   - run_graphrag_query() returns the full response text
#   - Use tabulate(rows, headers=headers, tablefmt="grid") for nice formatting
#   - Queries are slow with local models — expect 30-60s per query
#   - Consider printing a progress indicator: "Running question 1/4 with local..."


def main() -> None:
    print("=" * 60)
    print("Exercise 3: Query Mode Comparison")
    print("=" * 60)
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Questions to test: {len(QUESTIONS)}")
    print("Modes: local, global, basic")
    print("\nThis will take several minutes with local models...")
    print("-" * 60)

    compare_query_modes()


if __name__ == "__main__":
    main()
