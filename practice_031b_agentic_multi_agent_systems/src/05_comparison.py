"""
Phase 5: Architecture Comparison Benchmark

Runs the same queries through all three architectures (supervisor, swarm, agent-as-tool)
and measures: wall-clock time, output length, and subjective quality.

CrewAI uses a different task structure (topic-based vs query-based), so it's benchmarked
separately with comparable topics.

This is the capstone exercise: after implementing all patterns, you evaluate them
side-by-side to build intuition for when to use which architecture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage

# ---------------------------------------------------------------------------
# Benchmark data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Result from running a single query through one architecture."""

    architecture: str
    query: str
    response: str
    wall_clock_seconds: float
    response_length: int = 0
    quality_rating: int = 0  # 1-5, filled in by user

    def __post_init__(self) -> None:
        self.response_length = len(self.response)


@dataclass
class BenchmarkSuite:
    """Collects results across architectures for comparison."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    def print_comparison_table(self) -> None:
        """Print a formatted comparison table of all results."""
        print("\n" + "=" * 90)
        print("COMPARISON TABLE")
        print("=" * 90)
        print(f"{'Architecture':<20} {'Query':<35} {'Time (s)':<10} {'Length':<10} {'Rating':<8}")
        print("-" * 90)
        for r in self.results:
            query_short = r.query[:32] + "..." if len(r.query) > 35 else r.query
            print(
                f"{r.architecture:<20} {query_short:<35} "
                f"{r.wall_clock_seconds:<10.2f} {r.response_length:<10} {r.quality_rating:<8}"
            )

    def print_summary(self) -> None:
        """Print aggregated metrics per architecture."""
        print("\n" + "=" * 70)
        print("AGGREGATE SUMMARY")
        print("=" * 70)

        architectures = sorted(set(r.architecture for r in self.results))
        print(f"{'Architecture':<20} {'Avg Time (s)':<15} {'Avg Length':<15} {'Queries':<10}")
        print("-" * 70)
        for arch in architectures:
            arch_results = [r for r in self.results if r.architecture == arch]
            avg_time = sum(r.wall_clock_seconds for r in arch_results) / len(arch_results)
            avg_len = sum(r.response_length for r in arch_results) / len(arch_results)
            print(f"{arch:<20} {avg_time:<15.2f} {avg_len:<15.0f} {len(arch_results):<10}")


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

BENCHMARK_QUERIES = [
    "What is the capital of France and what is its population?",
    "Calculate the integral of x^2 from 0 to 5.",
    "Write a haiku about distributed systems.",
    "What is quantum computing? Then write a limerick about it.",
    "How many prime numbers are there between 1 and 50?",
]


# ---------------------------------------------------------------------------
# Architecture runners
# ---------------------------------------------------------------------------


def run_all_benchmarks() -> BenchmarkSuite:
    """Run all benchmark queries through all architectures.

    Returns a BenchmarkSuite with all results.
    """
    suite = BenchmarkSuite()

    # TODO(human) #1: Import and invoke all three architectures.
    #
    # This is where you bring together everything from Phases 1, 2, and 4.
    # For each architecture, run all 5 BENCHMARK_QUERIES and collect results.
    #
    # Pattern for each architecture:
    #   1. Import the runner function from the corresponding module:
    #      - from src.01_supervisor import run_supervisor
    #      - from src.02_swarm import run_swarm
    #      - from src.04_agent_as_tool import run_agent_as_tool
    #      (Adjust imports based on your project structure — you may need relative
    #      imports or to add the src directory to sys.path)
    #
    #   2. For each query in BENCHMARK_QUERIES:
    #      start = time.time()
    #      response = run_supervisor(query)  # or run_swarm, run_agent_as_tool
    #      elapsed = time.time() - start
    #
    #      suite.add(BenchmarkResult(
    #          architecture="supervisor",  # or "swarm", "agent-as-tool"
    #          query=query,
    #          response=response,
    #          wall_clock_seconds=elapsed,
    #      ))
    #
    #   3. Repeat for all three architectures.
    #
    # Handle errors gracefully: wrap each invocation in try/except, and if an
    # architecture fails on a query, add a result with response="ERROR: {e}"
    # and wall_clock_seconds=0. This prevents one failure from blocking the
    # entire benchmark.
    #
    # Note: CrewAI (Phase 3) uses topic-based input rather than direct queries,
    # so it's not directly comparable here. You could adapt it by using the
    # queries as topics, but the comparison won't be apples-to-apples.
    raise NotImplementedError("TODO(human) #1: Run benchmarks across all architectures")

    return suite


def rate_results(suite: BenchmarkSuite) -> None:
    """Interactively rate each result's quality (1-5).

    This is optional — skip if you prefer to just compare times and lengths.
    """
    # TODO(human) #2: Implement quality rating collection.
    #
    # After running all benchmarks, review each response and assign a quality
    # rating from 1 (poor) to 5 (excellent). This adds the subjective dimension
    # that automated metrics can't capture.
    #
    # For each result in suite.results:
    #   1. Print the architecture name, query, and response (truncated to ~500 chars)
    #   2. Ask user: "Rate quality (1-5): "
    #   3. Read input, validate it's 1-5, assign to result.quality_rating
    #   4. If input is empty or invalid, default to 0 (unrated)
    #
    # This teaches an important evaluation principle: in multi-agent systems,
    # speed and token count are easy to measure, but output quality requires
    # human judgment. Production systems often combine automated metrics (latency,
    # cost) with periodic human evaluation (accuracy, helpfulness, coherence).
    #
    # Skip implementation if you prefer just automated metrics — the comparison
    # table and summary will still show time and length.
    print("\n[Skipping quality ratings — implement TODO(human) #2 to enable]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Phase 5: Architecture Comparison Benchmark")
    print("=" * 70)
    print(f"Running {len(BENCHMARK_QUERIES)} queries x 3 architectures...")
    print("This may take several minutes with a local LLM.\n")

    suite = run_all_benchmarks()

    rate_results(suite)

    suite.print_comparison_table()
    suite.print_summary()

    print("\n" + "=" * 70)
    print("Discussion Questions:")
    print("  1. Which architecture was fastest? Why?")
    print("  2. Which produced the longest responses? Is longer better?")
    print("  3. For which query types did each architecture excel?")
    print("  4. What would change at scale (100+ agents, long conversations)?")
    print("=" * 70)


if __name__ == "__main__":
    main()
