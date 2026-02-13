"""Phase 6 — Error Recovery & Evaluation: Resilience Patterns + LLM-as-Judge.

Demonstrates: retry with exponential backoff (tenacity), fallback chains,
and agent evaluation using an LLM as a judge to score responses 1-5.

Run:
    uv run python src/06_recovery_eval.py
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from langchain_ollama import ChatOllama
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ── Simulated flaky service ──────────────────────────────────────────

class ServiceUnavailableError(Exception):
    """Simulates a transient external service failure."""
    pass


def flaky_external_tool(query: str) -> str:
    """Simulates an external tool that fails ~50% of the time.

    This represents real-world conditions: API rate limits, network timeouts,
    upstream service degradation. The tool works fine when it works, but
    intermittently raises ServiceUnavailableError.
    """
    if random.random() < 0.5:
        raise ServiceUnavailableError(f"Service temporarily unavailable for query: {query[:30]}")
    return f"Tool result for '{query}': The answer is 42."


# ── TODO(human) #1: Retry with Exponential Backoff ────────────────────

# Wrap the flaky_external_tool with tenacity's @retry decorator.
#
# tenacity is the standard Python library for retry logic. The @retry decorator
# wraps a function so that if it raises a specific exception, tenacity
# automatically retries the call with configurable delays. Exponential backoff
# (1s, 2s, 4s, ...) prevents overwhelming a recovering service — this is the
# industry standard pattern used by AWS SDKs, Google Cloud, and gRPC.
#
# Create a function `resilient_tool(query: str) -> str` decorated with @retry:
#   - retry=retry_if_exception_type(ServiceUnavailableError)
#     Only retry on transient errors, not on all exceptions (e.g., ValueError
#     from bad input should fail immediately — retrying won't fix it)
#   - wait=wait_exponential(multiplier=1, min=1, max=10)
#     Wait 1s, 2s, 4s, 8s, 10s between retries (capped at 10s)
#   - stop=stop_after_attempt(5)
#     Give up after 5 attempts (don't retry forever — that's a resource leak)
#   - before_sleep=before_sleep_log(logger, logging.WARNING)
#     Log each retry attempt (essential for debugging in production)
#
# Inside the function body, just call flaky_external_tool(query) and return
# the result. tenacity handles the retry logic transparently.
#
# Docs: https://tenacity.readthedocs.io/en/latest/#waiting-before-retrying

def resilient_tool(query: str) -> str:
    """Call flaky_external_tool with automatic retry on transient failures.

    TODO(human): Implement this function with the @retry decorator.
    """
    raise NotImplementedError


# ── TODO(human) #2: Fallback Chain ────────────────────────────────────

async def fallback_query(query: str) -> tuple[str, str]:
    """Try primary agent, fall back to simpler agent, then to static error.

    TODO(human): Implement this function.

    Fallback chains ensure the user always gets a response, even during
    partial outages. The strategy is layered: try the best option first,
    then progressively simpler alternatives. This is how production systems
    maintain availability — Netflix's Zuul gateway, AWS's Route 53 health
    checks, and Kubernetes' readiness probes all use variants of this pattern.

    Steps:
      1. Try the primary agent:
         - Create a ChatOllama with model=MODEL_NAME and temperature=0.7
         - Invoke with a detailed prompt: f"You are an expert assistant. {query}"
         - If successful, return (response.content, "primary")
      2. If the primary agent raises any exception, catch it and try the fallback:
         - Create a ChatOllama with model=MODEL_NAME and temperature=0.0
         - Invoke with a simpler prompt: f"Answer briefly: {query}"
         - If successful, return (response.content, "fallback")
      3. If the fallback also fails, return a static error message:
         - return ("I'm sorry, I'm unable to process your request at this time. "
                   "Please try again later.", "error")

    The tuple (response, source) lets callers know which tier answered —
    useful for logging, monitoring, and setting user expectations (e.g.,
    "This response may be less detailed due to high load").

    Returns:
        Tuple of (response_text, source) where source is "primary", "fallback", or "error".
    """
    raise NotImplementedError


# ── TODO(human) #3: Evaluation with LLM-as-Judge ─────────────────────

# Evaluation dataset: pairs of (question, expected_answer_keywords)
EVAL_DATASET: list[dict[str, str]] = [
    {"question": "What is Python?", "expected": "high-level programming language, interpreted, Guido van Rossum"},
    {"question": "What is Docker?", "expected": "containerization platform, containers, images, deployment"},
    {"question": "What is REST?", "expected": "architectural style, HTTP, stateless, resources, API"},
    {"question": "What is SQL?", "expected": "query language, databases, relational, tables"},
    {"question": "What is Git?", "expected": "version control, distributed, commits, branches"},
    {"question": "What is Kubernetes?", "expected": "container orchestration, pods, services, scaling"},
    {"question": "What is TCP?", "expected": "transport protocol, reliable, connection-oriented, packets"},
    {"question": "What is JSON?", "expected": "data format, key-value, lightweight, JavaScript"},
    {"question": "What is OAuth?", "expected": "authorization framework, tokens, access, delegated"},
    {"question": "What is WebSocket?", "expected": "full-duplex communication, persistent connection, real-time"},
]


async def evaluate_agent() -> list[dict[str, Any]]:
    """Run evaluation dataset through the agent and score with LLM-as-judge.

    TODO(human): Implement this function.

    Evaluation is how you know if your agent is good (or getting worse after
    changes). The LLM-as-judge pattern uses one LLM to evaluate another's output.
    The judge sees: the question, the expected answer keywords, and the actual
    response, then scores 1-5 on relevance and accuracy. This is the same approach
    used in academic LLM benchmarks (MT-Bench, Arena) and production eval pipelines.

    Steps:
      1. Create a ChatOllama for the "student" (the agent being evaluated):
         student_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)
      2. Create a ChatOllama for the "judge" (evaluator):
         judge_llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)
         In production you'd use a stronger model as judge, but for this exercise
         the same model is fine.
      3. For each item in EVAL_DATASET:
         a. Get the student's response: await student_llm.ainvoke(item["question"])
         b. Build a judge prompt:
            "You are an evaluation judge. Score the following response 1-5.
             Question: {item['question']}
             Expected keywords: {item['expected']}
             Actual response: {student_response}
             Score 1=completely wrong, 3=partially correct, 5=excellent.
             Reply with ONLY a single digit (1-5) followed by a brief reason."
         c. Get the judge's verdict: await judge_llm.ainvoke(judge_prompt)
         d. Parse the score: extract the first digit from the judge's response
         e. Append to results: {"question": ..., "response": ..., "score": ..., "judge_reason": ...}
      4. Print a summary: average score, min, max, and per-question breakdown
      5. Return the results list

    Returns:
        List of evaluation result dicts.
    """
    raise NotImplementedError


# ── Orchestration ────────────────────────────────────────────────────

async def test_retry() -> None:
    """Test the retry mechanism with the flaky tool."""
    print("\n--- Testing Retry with Exponential Backoff ---")
    for i in range(5):
        try:
            result = resilient_tool(f"test query {i}")
            print(f"  Query {i}: SUCCESS — {result}")
        except ServiceUnavailableError as e:
            print(f"  Query {i}: FAILED after all retries — {e}")


async def test_fallback() -> None:
    """Test the fallback chain."""
    print("\n--- Testing Fallback Chain ---")
    queries = [
        "Explain the CAP theorem",
        "What is a B-tree?",
        "How does TLS handshake work?",
    ]
    for query in queries:
        response, source = await fallback_query(query)
        print(f"  [{source:8s}] {query}: {response[:100]}...")


async def test_evaluation() -> None:
    """Run the full evaluation suite."""
    print("\n--- Running Evaluation Suite ---")
    results = await evaluate_agent()

    scores = [r["score"] for r in results if isinstance(r["score"], int)]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\n  Average score: {avg:.1f}/5")
        print(f"  Min: {min(scores)}, Max: {max(scores)}")
        print(f"  Evaluated: {len(scores)}/{len(EVAL_DATASET)} questions")


async def main() -> None:
    print("=" * 60)
    print("Phase 6 — Error Recovery & Evaluation")
    print("=" * 60)

    await test_retry()
    await test_fallback()
    await test_evaluation()

    print(f"\n{'=' * 60}")
    print("Phase 6 complete.")
    print("Review: retry success rate, fallback tier distribution, eval scores.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
