"""
Practice 029b — Phase 4: Parallel Execution with Send API
Build a map-reduce workflow using LangGraph's Send primitive.

The Send API enables dynamic fan-out: a conditional edge function returns
a list of Send(node_name, custom_state) objects. Each Send dispatches an
independent invocation of the target node with its own input state. All
dispatched invocations run in the same superstep (in parallel).

Combined with a reducer (e.g., Annotated[list, operator.add]), the results
from all parallel workers are automatically collected into the shared state.
This is LangGraph's native map-reduce primitive.

Pattern:
    split --> [Send("worker", task1), Send("worker", task2), Send("worker", task3)]
                  |                       |                       |
                  v                       v                       v
               worker(task1)          worker(task2)          worker(task3)
                  |                       |                       |
                  +--------> aggregate <--+----------- <----------+
                            (reducer collects all worker outputs)
"""

import operator
from typing import Annotated

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import TypedDict

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ---------------------------------------------------------------------------
# State schemas (provided — study the reducer on 'results')
# ---------------------------------------------------------------------------


class OverallState(TypedDict):
    """Main graph state for the map-reduce workflow.

    - question: the original research question
    - subtasks: list of subtask descriptions (produced by split node)
    - results: list of worker outputs — uses operator.add REDUCER so that
      each worker's return {"results": ["my result"]} is APPENDED to the
      list rather than replacing it. This is how parallel results are
      automatically collected.
    - summary: final aggregated answer
    """

    question: str
    subtasks: list[str]
    results: Annotated[list[str], operator.add]
    summary: str


class WorkerState(TypedDict):
    """State for individual worker nodes (via Send).

    This is SEPARATE from OverallState — each Send("worker", {"task": ...})
    creates a worker with this state. The worker's return value is merged
    back into OverallState via the reducer.

    The Send API allows worker state to differ from the main graph state.
    Workers only see what they need (their specific task), not the entire
    graph state. This is good design — each worker is isolated.
    """

    task: str


# ---------------------------------------------------------------------------
# TODO(human): Build the map-reduce graph with Send API
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   Three functions and the graph wiring.
#
# Node 1 — split_into_subtasks(state: OverallState) -> dict:
#   - Read state["question"]
#   - Ask the LLM to break it into 3 research subtasks:
#     Prompt: "Break this research question into exactly 3 independent
#              subtasks that can be researched separately. Return ONLY
#              the 3 subtasks, one per line, numbered 1-3.
#
#              Question: {question}"
#   - Parse the response to extract a list of subtask strings
#     HINT: Split by newlines, strip whitespace, filter empty lines.
#           Remove numbering prefixes like "1. " or "1) " if present.
#   - Return {"subtasks": parsed_subtasks}
#
# Routing function — route_to_workers(state: OverallState) -> list[Send]:
#   - Read state["subtasks"]
#   - Return a list of Send objects, one per subtask:
#       [Send("worker", {"task": subtask}) for subtask in state["subtasks"]]
#   - Each Send dispatches an independent worker with its own WorkerState
#   - All workers run in parallel (same superstep)
#
#   IMPORTANT: This function is used as a conditional edge. Instead of
#   returning a node name (string), it returns a list of Send objects.
#   LangGraph detects this and dispatches all sends in parallel.
#
# Node 2 — worker(state: WorkerState) -> dict:
#   - Read state["task"]
#   - Research the subtask using the LLM:
#     Prompt: "Research this topic and provide a concise but informative
#              summary (3-5 sentences):
#
#              Topic: {task}"
#   - Return {"results": [response.content]}
#     NOTE: Wrap in a list! The operator.add reducer on OverallState["results"]
#     will append this single-element list to the collected results.
#     If you return {"results": response.content} (a string, not a list),
#     the reducer will try to concatenate characters instead of appending.
#
# Node 3 — aggregate(state: OverallState) -> dict:
#   - Read state["results"] — this now contains ALL worker outputs
#     (collected by the reducer from parallel workers)
#   - Read state["question"] for context
#   - Ask the LLM to synthesize a final answer:
#     Prompt: "Based on the following research findings, provide a
#              comprehensive answer to the original question.
#
#              Original question: {question}
#
#              Research findings:
#              {numbered_findings}
#
#              Synthesize these into a coherent, comprehensive answer."
#   - Return {"summary": response.content}
#
# Graph wiring:
#   builder = StateGraph(OverallState)
#   Add nodes: "split", "worker", "aggregate"
#   Edges:
#     START -> "split"
#     "split" -> conditional: route_to_workers  (fan-out via Send)
#     "worker" -> "aggregate"  (normal edge — all workers flow to aggregate)
#     "aggregate" -> END
#
#   For the conditional edge from "split":
#     builder.add_conditional_edges("split", route_to_workers, ["worker"])
#   The third argument ["worker"] tells LangGraph which nodes the Send
#   objects will target — this is needed for graph validation.
#
#   Compile normally (no checkpointer needed for this exercise).
#
# WHY THIS MATTERS:
#   Map-reduce is the fundamental pattern for parallelism in LangGraph.
#   Production use cases:
#   - Researching multiple sources simultaneously
#   - Generating content in parallel (multiple sections of a report)
#   - Evaluating an answer from multiple perspectives (multi-judge scoring)
#   - Processing batch items (each item gets its own worker)
#
#   The Send API decouples the NUMBER of workers from the graph structure.
#   The graph has one "worker" node definition, but at runtime it's
#   invoked N times in parallel — one per Send. The reducer automatically
#   collects all results. This is declarative parallelism.
#
# EXPECTED BEHAVIOR:
#   Question: "What are the key factors in building reliable distributed systems?"
#   -> split produces 3 subtasks
#   -> 3 workers run in parallel, each researching one subtask
#   -> aggregate combines all 3 research results into one comprehensive answer


def split_into_subtasks(state: OverallState) -> dict:
    response = llm.invoke(
        [
            "Break this research question into exactly 3 independent"
            "subtasks that can be researched separately. Return ONLY 3 lines,"
            "one with each of the subtasks. Each subtask must be a single line"
            f"\n\n Question:{state['question']}"
        ]
    )

    return {
        "subtasks": list(filter(lambda s: len(s) > 0, response.content.split("\n")))
    }


def route_to_workers(state: OverallState) -> list[Send]:
    return [Send("worker", {"task": subtask}) for subtask in state["subtasks"]]


def worker(state: WorkerState) -> dict:
    return {
        "results": [
            llm.invoke(
                [
                    "Research this topic and provide a concise but informative summary (3-5 sentences):\n"
                    f"Topic: {state['task']}"
                ]
            ).content
        ]
    }


def aggregate(state: OverallState) -> dict:
    numbered_findings = "\n".join(
        [f"{i + 1}: {finding}\n" for (i, finding) in enumerate(state["results"])]
    )

    return {
        "summary": llm.invoke(
            "Based on the following research findings, provide a"
            "comprehensive answer to the original question.\n"
            f"Original question: {state['question']}\n"
            "Research findings:\n"
            f"{numbered_findings}\n"
            "Synthesize these into a coherent, comprehensive answer."
        ).content
    }


def build_map_reduce_graph():
    builder = StateGraph(OverallState)
    builder.add_node("split", split_into_subtasks)
    builder.add_node("worker", worker)
    builder.add_node("aggregate", aggregate)

    builder.add_edge(START, "split")
    builder.add_conditional_edges("split", route_to_workers, ["worker"])
    builder.add_edge("worker", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("EXERCISE: Map-Reduce with Send API")
    print("=" * 60)

    graph = build_map_reduce_graph()

    question = "What are the key factors in building reliable distributed systems?"
    print(f"\nResearch question: {question}\n")

    # Stream to see each step as it happens
    print("--- Streaming execution ---\n")
    final_state = None
    for step in graph.stream(
        {"question": question, "subtasks": [], "results": [], "summary": ""},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            if node_name == "split":
                print(f"[split] Generated {len(update.get('subtasks', []))} subtasks:")
                for i, task in enumerate(update.get("subtasks", []), 1):
                    print(f"  {i}. {task}")
                print()
            elif node_name == "worker":
                result = update.get("results", [""])[0]
                print(f"[worker] Completed: {result[:100]}...")
                print()
            elif node_name == "aggregate":
                print(f"[aggregate] Final summary:")
                print(update.get("summary", ""))
                print()

    # Also run with invoke to get the complete final state
    print("\n--- Final State (via invoke) ---\n")
    result = graph.invoke(
        {"question": question, "subtasks": [], "results": [], "summary": ""}
    )
    print(f"Question: {result['question']}")
    print(f"Subtasks: {len(result['subtasks'])}")
    print(f"Worker results: {len(result['results'])}")
    print(f"\nSummary:\n{result['summary']}")

    print("\nPhase 4 complete.")
