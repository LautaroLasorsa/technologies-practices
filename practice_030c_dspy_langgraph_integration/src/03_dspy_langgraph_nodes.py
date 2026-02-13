"""
Practice 030c — Phase 3: DSPy Modules as LangGraph Nodes

This is the core integration pattern: LangGraph orchestrates the workflow
(conditional routing, state management), while DSPy handles each LLM call
with optimized prompts.

Pattern: LangGraph node function wraps a DSPy module call.
  - Node receives graph state (TypedDict)
  - Extracts inputs from state
  - Calls a DSPy module
  - Returns state updates (dict)

Run: uv run python src/03_dspy_langgraph_nodes.py
"""

from typing import Literal, TypedDict

import dspy
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"


def configure_dspy() -> None:
    """Configure DSPy to use the local Ollama model."""
    lm = dspy.LM(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        api_base=f"{OLLAMA_BASE_URL}/v1",
        api_key="",
    )
    dspy.configure(lm=lm)


# ---------------------------------------------------------------------------
# Graph state definition
# ---------------------------------------------------------------------------

class QueryState(TypedDict):
    """State that flows through the LangGraph.

    Fields:
        question: The user's original question.
        category: The classified category (math, factual, or creative).
        answer: The final answer produced by the specialist node.
    """
    question: str
    category: str
    answer: str


# ---------------------------------------------------------------------------
# TODO(human) #4: Build the DSPy classifier module
# ---------------------------------------------------------------------------
# A DSPy ChainOfThought module that classifies a question into one of three
# categories: "math", "factual", or "creative". This classification will
# drive LangGraph's conditional routing — the graph will send the question
# to a different specialist node based on the category.
#
# Steps:
#   1. Create a DSPy signature for classification. The signature should be:
#      "question -> category"
#      where category is one of: math, factual, creative.
#
#   2. Instantiate a ChainOfThought module with this signature:
#      classifier = dspy.ChainOfThought("question -> category")
#
#   3. Create a LangGraph node function `classify_node(state: QueryState) -> dict`
#      that:
#      - Calls the classifier with state["question"]
#      - Extracts the category from the result
#      - Normalizes it to one of "math", "factual", "creative" (lowercase, strip)
#      - Returns {"category": normalized_category}
#
#   4. Create the routing function `route_by_category(state: QueryState) -> str`
#      that returns the node name to route to based on state["category"].
#      Map: "math" -> "math_node", "factual" -> "fact_node", "creative" -> "creative_node"
#
# Why this matters: This is the pattern for DSPy-driven conditional routing.
# The LLM (via DSPy) makes a structured decision, and LangGraph routes based
# on that decision. The DSPy module can later be optimized with few-shot
# examples to improve classification accuracy.

def classify_node(state: QueryState) -> dict:
    raise NotImplementedError("TODO(human): implement DSPy classifier node")


def route_by_category(state: QueryState) -> str:
    raise NotImplementedError("TODO(human): implement routing function")


# ---------------------------------------------------------------------------
# TODO(human) #5: Build specialized DSPy nodes
# ---------------------------------------------------------------------------
# Each specialist node wraps a DSPy ChainOfThought module tailored for its
# category. The node reads the question from graph state, calls its DSPy
# module, and writes the answer back to state.
#
# Create three node functions:
#
# 1. math_node(state: QueryState) -> dict
#    - Uses a ChainOfThought with a math-focused signature, e.g.:
#      "question -> answer"  (you can add a prefix hint in the signature like
#      "question: str, context: str -> answer: str" and pass context="Solve step by step")
#    - Returns {"answer": result.answer}
#
# 2. fact_node(state: QueryState) -> dict
#    - Uses a ChainOfThought for factual Q&A
#    - Returns {"answer": result.answer}
#
# 3. creative_node(state: QueryState) -> dict
#    - Uses a ChainOfThought for creative/open-ended responses
#    - Returns {"answer": result.answer}
#
# Each module can use the same base signature "question -> answer" but you
# can differentiate them by adding context or using different DSPy module
# types. The key learning here is the PATTERN: node function wraps DSPy call.
#
# Tip: instantiate the DSPy modules outside the node functions (at module
# level or in a setup function) so they aren't recreated on every call.

def math_node(state: QueryState) -> dict:
    raise NotImplementedError("TODO(human): implement math specialist node")


def fact_node(state: QueryState) -> dict:
    raise NotImplementedError("TODO(human): implement factual specialist node")


def creative_node(state: QueryState) -> dict:
    raise NotImplementedError("TODO(human): implement creative specialist node")


# ---------------------------------------------------------------------------
# TODO(human) #6: Wire the LangGraph
# ---------------------------------------------------------------------------
# Now connect everything into a LangGraph that routes questions through the
# classifier to the appropriate specialist.
#
# Graph structure:
#   START -> "classify" -> conditional_edge -> "math_node" / "fact_node" / "creative_node" -> END
#
# Steps:
#   1. Create a StateGraph(QueryState)
#   2. Add nodes: "classify", "math_node", "fact_node", "creative_node"
#   3. Add edge: START -> "classify"
#   4. Add conditional edges from "classify" using route_by_category:
#      graph.add_conditional_edges("classify", route_by_category)
#   5. Add edges from each specialist to END:
#      graph.add_edge("math_node", END)
#      graph.add_edge("fact_node", END)
#      graph.add_edge("creative_node", END)
#   6. Compile: app = graph.compile()
#   7. Test with different questions:
#      - "What is 15 * 23 + 7?"              -> should route to math_node
#      - "What is the capital of France?"     -> should route to fact_node
#      - "Write a haiku about programming"    -> should route to creative_node
#   8. Print the category and answer for each test.

def build_and_run_graph() -> None:
    raise NotImplementedError("TODO(human): build LangGraph with DSPy nodes")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 3: DSPy Modules as LangGraph Nodes")
    print("=" * 60)

    build_and_run_graph()


if __name__ == "__main__":
    main()
