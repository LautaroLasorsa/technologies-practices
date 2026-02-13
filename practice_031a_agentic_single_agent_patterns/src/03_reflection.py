"""Reflection Agent — Self-critique and iterative refinement.

Pattern: Generate → Evaluate → Refine → Iterate until quality threshold met

The agent produces an initial draft, then a separate evaluation step critiques
it and assigns a score. If the score is below a threshold, the agent loops back
to refine the draft using the critique as guidance. This pattern significantly
improves output quality for tasks like code generation, writing, and reasoning.

Run:
    uv run python src/03_reflection.py
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
QUALITY_THRESHOLD = 7
MAX_ITERATIONS = 3

GENERATOR_PROMPT = """You are an expert Python developer. Your task is to write
high-quality Python code based on the given requirements.

If you receive critique/feedback, carefully address EVERY point in your revision.
Show the complete improved code, not just the changes."""

EVALUATOR_PROMPT = """You are a senior code reviewer. Evaluate the following Python code
on these criteria:
1. Correctness: Does it solve the problem correctly?
2. Readability: Is it clear and well-structured?
3. Edge cases: Does it handle edge cases?
4. Style: Does it follow Python best practices (type hints, docstrings)?

Output your evaluation in this EXACT format:
SCORE: <number from 1-10>
FEEDBACK: <specific, actionable feedback>

Be strict. Only score 7+ if the code is genuinely good.
A score of 10 means production-ready code with no issues."""


# ── State ────────────────────────────────────────────────────────────


class ReflectionState(TypedDict):
    """State for the Reflection agent.

    task: The code generation task description (e.g., "Write a function
          that reverses a linked list").
    draft: The current code draft. Updated each time the generator refines.
    critique: The evaluator's textual feedback. Passed to the generator
              on the next refinement iteration.
    score: Numeric quality score (1-10) from the evaluator. The conditional
           edge uses this to decide whether to loop or finish.
    iteration: Current iteration number. Bounded by MAX_ITERATIONS.
    """

    task: str
    draft: str
    critique: str
    score: int
    iteration: int


# ── LLM ──────────────────────────────────────────────────────────────

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement these two functions + conditional edge ────


def generator_node(state: ReflectionState) -> dict:
    """Generate or refine a code draft based on the task and any critique.

    TODO(human): Implement this function.

    On the FIRST call (no critique yet), the generator produces an initial
    draft from the task description alone. On SUBSEQUENT calls, it receives
    the previous critique and must address each point.

    Steps:
      1. Build the user message. If state["critique"] is empty/falsy, use:
         "Write the following:\n{state['task']}"
         If critique exists, use:
         "Revise the following code based on the feedback.\n\n"
         "Current code:\n{state['draft']}\n\n"
         "Feedback:\n{state['critique']}\n\n"
         "Provide the complete improved code."
      2. Build messages: SystemMessage with GENERATOR_PROMPT + HumanMessage.
      3. Invoke the LLM.
      4. Increment iteration by 1.
      5. Return {"draft": response.content, "iteration": new_iteration}.

    Why self-refinement works: LLMs are better at fixing code given specific
    feedback than generating perfect code on the first try. The evaluator
    identifies gaps the generator missed, creating a feedback loop analogous
    to code review in software teams. Research shows 2-3 iterations often
    doubles the quality of LLM-generated code.
    """
    raise NotImplementedError("TODO(human): implement generator_node")


def evaluator_node(state: ReflectionState) -> dict:
    """Evaluate the current draft and produce a score + critique.

    TODO(human): Implement this function.

    The evaluator acts as an independent reviewer. It reads the draft WITHOUT
    knowing the previous critique (to avoid anchoring bias) and produces
    fresh feedback. The score determines whether the loop continues.

    Steps:
      1. Build the user message:
         "Task: {state['task']}\n\nCode to review:\n{state['draft']}"
      2. Build messages: SystemMessage with EVALUATOR_PROMPT + HumanMessage.
      3. Invoke the LLM.
      4. Parse the response to extract score and feedback:
         - Look for "SCORE:" line and extract the number
         - Everything after "FEEDBACK:" is the critique text
         - If parsing fails, default to score=5 and use the full response
           as critique (graceful degradation).
      5. Return {"score": parsed_score, "critique": parsed_feedback}.

    Parsing hint:
      for line in response.content.split("\\n"):
          if line.strip().upper().startswith("SCORE:"):
              score = int("".join(c for c in line if c.isdigit()) or "5")
          if line.strip().upper().startswith("FEEDBACK:"):
              feedback_start = response.content.index(line)
              critique = response.content[feedback_start + len(line):]

    Key design choice: Using a structured output format (SCORE: / FEEDBACK:)
    rather than free-form text makes parsing reliable. In production, you'd
    use Pydantic structured output for guaranteed schema compliance.
    """
    raise NotImplementedError("TODO(human): implement evaluator_node")


def should_continue(state: ReflectionState) -> Literal["generator", "__end__"]:
    """Decide whether to refine again or accept the current draft.

    TODO(human): Implement this function.

    This conditional edge implements the quality gate. It checks two
    conditions to decide whether to loop back for another refinement
    pass or accept the draft as final output.

    Steps:
      1. If state["score"] >= QUALITY_THRESHOLD → return "__end__"
         (quality is sufficient, accept the draft).
      2. If state["iteration"] >= MAX_ITERATIONS → return "__end__"
         (we've refined enough times, accept whatever we have).
      3. Otherwise → return "generator" (loop back for another refinement).

    Why both conditions: The score threshold ensures we stop when quality
    is good enough. The iteration limit ensures we stop even if the model
    is a harsh self-critic that never scores above threshold. Without
    the iteration limit, a strict evaluator could create an infinite loop.
    """
    raise NotImplementedError("TODO(human): implement should_continue")


# ── Graph wiring (provided) ─────────────────────────────────────────


def build_reflection_graph():
    """Wire the Reflection graph: generator → evaluator → conditional → loop or END."""
    graph = StateGraph(ReflectionState)

    graph.add_node("generator", generator_node)
    graph.add_node("evaluator", evaluator_node)

    graph.add_edge(START, "generator")
    graph.add_edge("generator", "evaluator")
    graph.add_conditional_edges(
        "evaluator",
        should_continue,
        ["generator", END],
    )

    return graph.compile()


# ── Orchestration ────────────────────────────────────────────────────


def run_reflection(task: str) -> None:
    """Run the Reflection agent and print each iteration."""
    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print("=" * 60)

    agent = build_reflection_graph()
    initial_state: ReflectionState = {
        "task": task,
        "draft": "",
        "critique": "",
        "score": 0,
        "iteration": 0,
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} (iteration {node_output.get('iteration', '?')}) ---")
            if node_name == "generator" and "draft" in node_output:
                draft = node_output["draft"]
                # Show first 400 chars of draft
                print(f"  Draft ({len(draft)} chars):")
                for line in draft[:400].split("\n"):
                    print(f"    {line}")
                if len(draft) > 400:
                    print("    ...")
            elif node_name == "evaluator":
                print(f"  Score: {node_output.get('score', '?')}/10")
                critique = node_output.get("critique", "")
                print(f"  Critique: {critique[:300]}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    print("Practice 031a — Phase 3: Reflection / Self-Critique Agent")
    print("Generate → Evaluate → Refine loop for quality improvement\n")

    # Test 1: Function that the model might get wrong on first try
    run_reflection(
        "Write a Python function `merge_sorted_lists(list1: list[int], list2: list[int]) -> list[int]` "
        "that merges two sorted lists into one sorted list. Do NOT use the built-in sorted() function. "
        "Include type hints, a docstring, and handle edge cases (empty lists, lists of different lengths)."
    )

    # Test 2: More complex task
    run_reflection(
        "Write a Python class `LRUCache` that implements a Least Recently Used cache with "
        "get(key) and put(key, value) methods, both in O(1) time. Use type hints and include "
        "a docstring explaining the data structure choice."
    )


if __name__ == "__main__":
    main()
