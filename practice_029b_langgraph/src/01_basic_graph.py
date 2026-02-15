"""
Practice 029b — Phase 1: Basic 2-Node Graph
Build the simplest possible LangGraph workflow: generate content, then refine it.

This exercise teaches the fundamental LangGraph cycle:
    1. Define a state schema (TypedDict)
    2. Write node functions that receive state and return updates
    3. Wire nodes with edges (START -> A -> B -> END)
    4. Compile and invoke
"""

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# LLM setup (provided — not part of the exercise)
# ---------------------------------------------------------------------------

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ---------------------------------------------------------------------------
# State schema (provided — study how it defines the shared data)
# ---------------------------------------------------------------------------


class ContentState(TypedDict):
    """Shared state for the generate-refine pipeline.

    - topic: the subject the user wants content about
    - draft: the initial generated text (written by 'generate' node)
    - refined: the improved text (written by 'refine' node)
    """

    topic: str
    draft: str
    refined: str


# ---------------------------------------------------------------------------
# TODO(human): Build the 2-node generate-refine graph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   Two node functions and a compiled graph.
#
# Node 1 — generate(state: ContentState) -> dict:
#   - Read state["topic"]
#   - Use `llm.invoke()` with a prompt like:
#       "Write a short paragraph (3-4 sentences) about: {topic}"
#   - Return {"draft": response.content}
#   - The LLM returns an AIMessage; access the text via .content
#
# Node 2 — refine(state: ContentState) -> dict:
#   - Read state["draft"]
#   - Use `llm.invoke()` with a prompt like:
#       "Improve the following text. Make it more engaging and concise:\n\n{draft}"
#   - Return {"refined": response.content}
#
# Graph wiring:
#   - Create a StateGraph(ContentState) builder
#   - Add both nodes with builder.add_node("generate", generate)
#   - Wire: START -> "generate" -> "refine" -> END
#   - Compile with builder.compile()
#
# WHY THIS MATTERS:
#   This is the atomic unit of LangGraph — every complex agent is built
#   by composing this exact pattern: state + nodes + edges + compile.
#   The state schema is the contract between nodes. Each node only reads
#   what it needs and writes what it produces. The graph handles the flow.
#
# EXPECTED BEHAVIOR:
#   invoke({"topic": "...", "draft": "", "refined": ""}) should return
#   a state dict with all three fields populated. The "refined" text
#   should be a polished version of "draft".
#
# HINT:
#   llm.invoke("your prompt") returns an AIMessage.
#   Access the text with response.content (it's a string).


def build_graph() -> StateGraph:
    def generate(state: ContentState) -> dict:
        return {
            "draft": llm.invoke(
                [
                    (
                        "human",
                        f"Write a short paragraph (3-4 sentences) about:{state['topic']}",
                    )
                ]
            ).content
        }

    def refine(state: ContentState) -> dict:
        return {
            "refined": llm.invoke(
                [
                    (
                        "human",
                        f"Improve the following text. Make it more engaging and concise:\n\n{state['draft']}",
                    )
                ]
            ).content
        }

    builder = StateGraph(ContentState)
    builder.add_node("generate", generate)
    builder.add_node("refine", refine)
    builder.add_edge(START, "generate")
    builder.add_edge("generate", "refine")
    builder.add_edge("refine", END)
    return builder.compile()


# ---------------------------------------------------------------------------
# Main — run the graph
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = build_graph()

    print("=== Phase 1: Basic 2-Node Graph ===\n")

    topic = "the benefits of graph-based AI agent architectures"
    result = graph.invoke({"topic": topic, "draft": "", "refined": ""})

    print(f"Topic: {topic}\n")
    print(f"--- Draft ---\n{result['draft']}\n")
    print(f"--- Refined ---\n{result['refined']}\n")
    print("Phase 1 complete.")
