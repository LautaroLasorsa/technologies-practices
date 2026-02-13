"""
Practice 029b — Phase 2: Conditional Routing & Cycles
Build graphs with conditional edges that branch and loop.

Conditional edges are the core mechanism that makes LangGraph a GRAPH
instead of a linear chain. A routing function examines the current state
and returns the name of the next node. When a conditional edge points
backward (to an earlier node), it creates a cycle — enabling iterative
refinement until a quality threshold is met.
"""

from typing import Literal

from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ===================================================================
# EXERCISE 1: Quality-Check Loop
# ===================================================================

class QualityState(TypedDict):
    """State for the quality-check loop.

    - topic: what to write about
    - content: the current draft (updated each iteration)
    - quality_score: numeric score from the evaluator (1-10)
    - iteration: current loop count (for the max-iterations guard)
    """
    topic: str
    content: str
    quality_score: int
    iteration: int


MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 7


# ---------------------------------------------------------------------------
# TODO(human) #1: Build the quality-check loop graph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   Three node functions and a conditional edge that creates a cycle.
#
# Node 1 — generate(state: QualityState) -> dict:
#   - If state["iteration"] == 0: generate fresh content about state["topic"]
#     Prompt: "Write a short paragraph (3-4 sentences) about: {topic}"
#   - If state["iteration"] > 0: improve the existing content based on its score
#     Prompt: "This text scored {quality_score}/10. Rewrite it to be better.
#              Focus on clarity and depth:\n\n{content}"
#   - Increment the iteration counter
#   - Return {"content": response.content, "iteration": state["iteration"] + 1}
#
# Node 2 — evaluate(state: QualityState) -> dict:
#   - Ask the LLM to rate the content on a scale of 1-10
#     Prompt: "Rate this text from 1 to 10 for quality (clarity, depth,
#              engagement). Reply with ONLY a single number:\n\n{content}"
#   - Parse the response to extract the integer score
#     HINT: Use a try/except; if parsing fails, default to 5
#   - Return {"quality_score": parsed_score}
#
# Routing function — should_continue(state: QualityState) -> str:
#   - If state["quality_score"] >= QUALITY_THRESHOLD: return END
#   - If state["iteration"] >= MAX_ITERATIONS: return END (guard against infinite loops)
#   - Otherwise: return "generate" (loop back for another attempt)
#   - Type hint the return as Literal["generate", "__end__"]
#     (LangGraph uses "__end__" as the string constant for END)
#
# Graph wiring:
#   builder = StateGraph(QualityState)
#   Add nodes: "generate", "evaluate"
#   Edges:
#     START -> "generate"
#     "generate" -> "evaluate"  (normal edge — always evaluate after generating)
#     "evaluate" -> conditional: should_continue  (loop or end)
#   Compile.
#
# WHY THIS MATTERS:
#   The generate-evaluate-loop pattern is the foundation of self-improving
#   agents. In production, this same structure is used for:
#   - Code generation with test validation (generate code, run tests, retry)
#   - Research with fact-checking (draft answer, verify facts, refine)
#   - Content creation with quality gates (write, review, polish)
#   The MAX_ITERATIONS guard is critical — without it, a cycle can loop
#   forever if the LLM never produces satisfactory output.
#
# EXPECTED BEHAVIOR:
#   The graph should iterate 1-3 times. Each iteration prints the
#   current content and score. It stops when score >= 7 or after 3 tries.

def build_quality_loop() -> StateGraph:
    raise NotImplementedError("TODO(human): Implement the quality-check loop graph")


# ===================================================================
# EXERCISE 2: Query Router
# ===================================================================

class RouterState(TypedDict):
    """State for the query router.

    - query: the user's input question
    - category: detected category (math, creative, general)
    - response: the specialist's answer
    """
    query: str
    category: str
    response: str


# ---------------------------------------------------------------------------
# Specialist nodes (provided — these handle each category)
# ---------------------------------------------------------------------------

def math_node(state: RouterState) -> dict:
    """Handle math/logic queries with a focused system prompt."""
    response = llm.invoke(
        f"You are a precise math tutor. Solve step by step.\n\nQuestion: {state['query']}"
    )
    return {"response": response.content}


def creative_node(state: RouterState) -> dict:
    """Handle creative writing queries."""
    response = llm.invoke(
        f"You are a creative writer. Be imaginative and vivid.\n\nPrompt: {state['query']}"
    )
    return {"response": response.content}


def general_node(state: RouterState) -> dict:
    """Handle general knowledge queries."""
    response = llm.invoke(
        f"You are a helpful assistant. Be clear and informative.\n\nQuestion: {state['query']}"
    )
    return {"response": response.content}


# ---------------------------------------------------------------------------
# TODO(human) #2: Build the query router graph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   A classify node, a routing function, and the wired graph.
#
# Node — classify(state: RouterState) -> dict:
#   - Ask the LLM to classify the query into one of three categories
#     Prompt: "Classify this query into exactly one category.
#              Reply with ONLY one word: math, creative, or general.
#
#              Query: {query}"
#   - Parse the response: extract the category string, lowercase it
#     If parsing fails or category is unrecognized, default to "general"
#   - Return {"category": parsed_category}
#
# Routing function — route_by_category(state: RouterState) -> str:
#   - Read state["category"]
#   - Return the corresponding node name: "math_node", "creative_node",
#     or "general_node"
#   - Type hint return as Literal["math_node", "creative_node", "general_node"]
#
# Graph wiring:
#   builder = StateGraph(RouterState)
#   Add nodes: "classify", "math_node", "creative_node", "general_node"
#   Edges:
#     START -> "classify"
#     "classify" -> conditional: route_by_category
#     "math_node" -> END
#     "creative_node" -> END
#     "general_node" -> END
#   Compile.
#
# WHY THIS MATTERS:
#   The router pattern is the building block for multi-agent systems.
#   Instead of a single monolithic LLM call, you decompose the problem:
#   1. A lightweight classifier determines intent
#   2. Specialized handlers process each intent
#   This improves quality (focused prompts) and lets you use different
#   models for different tasks (small model for routing, large for generation).
#
#   The add_conditional_edges method accepts:
#     - source node name
#     - routing function
#   The routing function's return value is used as the target node name.
#
# EXPECTED BEHAVIOR:
#   "What is 15 * 23?" -> routes to math_node
#   "Write a haiku about rain" -> routes to creative_node
#   "What is the capital of France?" -> routes to general_node

def build_query_router() -> StateGraph:
    raise NotImplementedError("TODO(human): Implement the query router graph")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Exercise 1: Quality Loop ---
    print("=" * 60)
    print("EXERCISE 1: Quality-Check Loop")
    print("=" * 60)

    quality_graph = build_quality_loop()
    result = quality_graph.invoke({
        "topic": "why iterative refinement improves AI output quality",
        "content": "",
        "quality_score": 0,
        "iteration": 0,
    })

    print(f"\nFinal content (after {result['iteration']} iterations):")
    print(f"Score: {result['quality_score']}/10")
    print(f"Content:\n{result['content']}\n")

    # --- Exercise 2: Query Router ---
    print("=" * 60)
    print("EXERCISE 2: Query Router")
    print("=" * 60)

    router_graph = build_query_router()

    test_queries = [
        "What is 15 * 23 + 7?",
        "Write a haiku about programming",
        "Explain how TCP/IP works",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router_graph.invoke({
            "query": query,
            "category": "",
            "response": "",
        })
        print(f"Category: {result['category']}")
        print(f"Response: {result['response'][:200]}...")

    print("\nPhase 2 complete.")
