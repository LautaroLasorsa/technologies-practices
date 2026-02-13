"""
Practice 029b — Verify Setup
Confirms that Ollama is reachable and LangGraph imports work correctly.
Builds and runs a trivial 1-node graph as a smoke test.
"""

from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


# ---------------------------------------------------------------------------
# 1. Test Ollama connectivity
# ---------------------------------------------------------------------------

def test_ollama_connection() -> None:
    """Send a single prompt to ChatOllama and print the response."""
    print("=== Testing Ollama connectivity ===")
    llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)
    response = llm.invoke("Say 'LangGraph is ready' and nothing else.")
    print(f"Model response: {response.content}")
    print("Ollama connection OK.\n")


# ---------------------------------------------------------------------------
# 2. Trivial 1-node graph smoke test
# ---------------------------------------------------------------------------

class TrivialState(TypedDict):
    message: str


def echo_node(state: TrivialState) -> dict:
    """Trivial node that uppercases the message."""
    return {"message": state["message"].upper()}


def test_trivial_graph() -> None:
    """Build a 1-node graph: START -> echo -> END."""
    print("=== Testing trivial LangGraph execution ===")

    builder = StateGraph(TrivialState)
    builder.add_node("echo", echo_node)
    builder.add_edge(START, "echo")
    builder.add_edge("echo", END)

    graph = builder.compile()
    result = graph.invoke({"message": "hello langgraph"})

    print(f"Input:  'hello langgraph'")
    print(f"Output: '{result['message']}'")
    assert result["message"] == "HELLO LANGGRAPH", "Graph output mismatch!"
    print("Trivial graph OK.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_ollama_connection()
    test_trivial_graph()
    print("All checks passed. Setup is ready.")
