"""Exercise 3 — Golden Dataset & Evaluation.

Agents are non-deterministic: the same prompt may produce different tool-call
sequences or wordings across runs.  Traditional ``assert output == expected``
tests don't work.  Instead, we evaluate agents with *golden datasets* —
collections of verified-correct interactions — and **flexible metrics** that
measure "close enough" rather than exact match.

This file teaches you to:
1. Define a golden dataset (list of test-case dicts).
2. Run each case through the agent and capture tool calls + final response.
3. Score results with two metrics:
   - **Tool accuracy**: did the agent call the right tool(s) with the right args?
   - **Response quality**: does the final text contain expected keywords?
4. Produce a summary report.

Key API points:
- ``runner.run_async()`` returns an async event stream.
- Events with ``event.actions.tool_calls`` (or equivalent) indicate tool usage.
  In practice, look for events whose ``content.parts`` contain a
  ``function_call`` (the LLM requesting a tool) or ``function_response``
  (the tool result).  The simplest approach: collect *all* events and
  inspect ``part.function_call`` on each part.
- ``event.is_final_response()`` identifies the last text response.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from task_agent.agent import add_task, complete_task, delete_task, list_tasks

MODEL = "ollama_chat/qwen2.5:7b"


# ============================================================================
# Golden Dataset
# ============================================================================
#
# A golden dataset is a list of dicts, each describing one test scenario.
#
# Structure per case:
#   {
#       "name":            str   — human-readable test name
#       "prompt":          str   — what the user says to the agent
#       "expected_tools":  list  — tool names the agent *should* call, in order
#       "expected_args":   list  — (optional) dicts of key args for each tool
#       "expected_keywords": list — substrings the final response must contain
#   }
#
# ``expected_tools`` checks *what* the agent does (action-level correctness).
# ``expected_keywords`` checks *what* it says (response-level correctness).
# Together they cover the two most common evaluation dimensions.
# ============================================================================


# TODO(human): Define the golden dataset.
#
# Create a list called ``GOLDEN_DATASET`` containing 5-8 test-case dicts.
#
# Guidelines for good test cases:
#   - Cover each tool at least once (add_task, list_tasks, complete_task,
#     delete_task).
#   - Include at least one case where the agent should call multiple tools
#     (e.g., "Add a task and then show me all tasks").
#   - Include at least one negative case — a prompt that should NOT trigger
#     any tool (e.g., "What's the weather?" — the agent has no weather tool
#     and should just respond with text).
#   - Keep prompts natural but unambiguous enough for a 7B model.
#
# Example entry (replace / extend with your own):
#
#   {
#       "name": "add_single_task",
#       "prompt": "Add a task called 'Write tests' with high priority",
#       "expected_tools": ["add_task"],
#       "expected_args": [{"task_name": "Write tests", "priority": "high"}],
#       "expected_keywords": ["Write tests", "high"],
#   }
#
# The dataset is a plain Python list — no special ADK types needed.
# Think of it as the "ground truth" that your evaluation functions compare
# the agent's actual behavior against.

GOLDEN_DATASET: list[dict[str, Any]] = []  # TODO(human): populate this list


# ============================================================================
# Evaluation helpers
# ============================================================================


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""

    name: str
    prompt: str
    tool_accuracy: float = 0.0  # 0.0 – 1.0
    response_quality: float = 0.0  # 0.0 – 1.0
    actual_tools: list[str] = field(default_factory=list)
    actual_response: str = ""
    details: str = ""


def evaluate_tool_accuracy(
    expected_tools: list[str],
    actual_tools: list[str],
    expected_args: list[dict[str, Any]] | None = None,
    actual_args: list[dict[str, Any]] | None = None,
) -> tuple[float, str]:
    """Compare expected vs actual tool calls and return (score, explanation).

    Scoring rubric:
    - If expected_tools is empty and actual_tools is empty → 1.0 (correct no-op)
    - If expected_tools is empty but agent called tools → 0.0 (false positive)
    - Otherwise: score = |intersection| / |union|  (Jaccard similarity on names)
    - Bonus: if ``expected_args`` is provided and matches, add detail note.

    The Jaccard similarity (|A & B| / |A | B|) is a standard set-overlap
    metric.  It's 1.0 when the two sets are identical and 0.0 when they share
    nothing.  It penalizes both missing tools (false negatives) and extra
    tools (false positives).
    """

    # TODO(human): Implement tool accuracy evaluation.
    #
    # Steps:
    #   1. Handle the edge case: if both lists are empty, return (1.0, "No tools expected or called").
    #   2. Handle false positive: if expected is empty but actual is not,
    #      return (0.0, f"Expected no tools but got: {actual_tools}").
    #   3. Compute Jaccard similarity:
    #      - ``expected_set = set(expected_tools)``
    #      - ``actual_set = set(actual_tools)``
    #      - ``intersection = expected_set & actual_set``
    #      - ``union = expected_set | actual_set``
    #      - ``score = len(intersection) / len(union)``
    #   4. Build an explanation string:
    #      - Which tools matched (intersection)
    #      - Which were missing from actual (expected_set - actual_set)
    #      - Which were unexpected (actual_set - expected_set)
    #   5. (Optional) If expected_args is provided, compare each arg dict
    #      against actual_args for matching tool names and note discrepancies.
    #   6. Return ``(score, explanation)``.
    raise NotImplementedError


def evaluate_response_quality(
    expected_keywords: list[str],
    actual_response: str,
) -> tuple[float, str]:
    """Check if the agent's final response contains expected keywords.

    Scoring: fraction of expected keywords found (case-insensitive).

    This is a simple but effective metric for agent responses.  More
    sophisticated approaches use LLM-as-judge (ask another LLM to score the
    response) or ROUGE/BERTScore for semantic similarity.  For this practice,
    keyword matching is sufficient and deterministic.
    """

    # TODO(human): Implement response quality evaluation.
    #
    # Steps:
    #   1. If ``expected_keywords`` is empty, return (1.0, "No keywords to check").
    #   2. Normalize: ``response_lower = actual_response.lower()``
    #   3. For each keyword in ``expected_keywords``, check if
    #      ``keyword.lower() in response_lower``.
    #   4. Count hits and misses.
    #   5. ``score = hits / len(expected_keywords)``
    #   6. Build explanation: list which keywords were found and which were missing.
    #   7. Return ``(score, explanation)``.
    raise NotImplementedError


# ============================================================================
# Evaluation runner
# ============================================================================


async def run_single_case(
    runner: Runner, user_id: str, case: dict[str, Any]
) -> EvalResult:
    """Run one test case through the agent and evaluate it.

    For each case:
    1. Create a fresh session (so test cases don't interfere).
    2. Send the prompt via ``runner.run_async()``.
    3. Collect tool calls from the event stream.
    4. Collect the final text response.
    5. Score with ``evaluate_tool_accuracy`` and ``evaluate_response_quality``.
    """

    # TODO(human): Implement the single-case evaluation runner.
    #
    # Steps:
    #   1. Create a fresh session:
    #      ``session = await runner.session_service.create_session(
    #            app_name="eval", user_id=user_id)``
    #
    #   2. Build the user message:
    #      ``content = types.Content(
    #            role="user", parts=[types.Part(text=case["prompt"])])``
    #
    #   3. Initialize collectors:
    #      ``actual_tools = []``
    #      ``actual_args = []``
    #      ``final_response = ""``
    #
    #   4. Iterate over events:
    #      ``async for event in runner.run_async(
    #            user_id=user_id, session_id=session.id, new_message=content):``
    #
    #      For each event, inspect ``event.content.parts`` (if present):
    #        - If a part has ``part.function_call``:
    #            * Append ``part.function_call.name`` to ``actual_tools``
    #            * Append ``dict(part.function_call.args)`` to ``actual_args``
    #        - If the event is the final response (``event.is_final_response()``):
    #            * Concatenate all text parts into ``final_response``
    #
    #   5. Score:
    #      ``tool_score, tool_detail = evaluate_tool_accuracy(
    #            case.get("expected_tools", []), actual_tools,
    #            case.get("expected_args"), actual_args)``
    #      ``resp_score, resp_detail = evaluate_response_quality(
    #            case.get("expected_keywords", []), final_response)``
    #
    #   6. Return an ``EvalResult`` with all fields populated.
    raise NotImplementedError


async def run_evaluation() -> list[EvalResult]:
    """Run all golden dataset cases and print a summary report.

    This is the top-level evaluation function.  It:
    1. Creates a fresh agent + runner for isolation.
    2. Iterates over ``GOLDEN_DATASET``.
    3. Calls ``run_single_case`` for each entry.
    4. Prints a formatted report and returns the results.
    """

    # TODO(human): Implement the evaluation loop.
    #
    # Steps:
    #   1. Create the agent (same as task_agent but standalone):
    #      ``agent = Agent(
    #            name="eval_agent", model=MODEL,
    #            instruction="You are a task-management assistant...",
    #            tools=[add_task, list_tasks, complete_task, delete_task])``
    #
    #   2. Create session service and runner:
    #      ``session_service = InMemorySessionService()``
    #      ``runner = Runner(agent=agent, app_name="eval",
    #                        session_service=session_service)``
    #
    #   3. Iterate over ``GOLDEN_DATASET``:
    #      ``results = []``
    #      ``for i, case in enumerate(GOLDEN_DATASET):``
    #          - Print progress: ``Running case {i+1}/{len(GOLDEN_DATASET)}: {case['name']}``
    #          - ``result = await run_single_case(runner, "eval_user", case)``
    #          - Append to results.
    #
    #   4. Print a summary table:
    #      For each result, print:
    #        ``{name:<30} tool_acc={tool_accuracy:.2f}  resp_qual={response_quality:.2f}``
    #      Also print overall averages.
    #
    #   5. Return the list of EvalResult objects.
    raise NotImplementedError


if __name__ == "__main__":
    if not GOLDEN_DATASET:
        print("ERROR: GOLDEN_DATASET is empty. Define test cases first!")
        print("See the TODO(human) block above for instructions.")
    else:
        asyncio.run(run_evaluation())
