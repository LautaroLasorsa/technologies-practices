"""
Practice 029b — Verify Setup
Confirms that Ollama is reachable and LangGraph imports work correctly.
Builds and runs a trivial 1-node graph as a smoke test.
"""

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from llm_config import LLM_MODEL, LLM_PROVIDER, get_chat_model


# ---------------------------------------------------------------------------
# 1. Test LLM connectivity
# ---------------------------------------------------------------------------


def test_ollama_connection() -> None:
    """Send a single prompt to the configured chat model and print the response."""
    print(f"=== Testing LLM connectivity (provider={LLM_PROVIDER}, model={LLM_MODEL}) ===")
    llm = get_chat_model(temperature=0)
    response = llm.invoke("Say 'LangGraph is ready' and nothing else.")
    print(f"Model response: {response.content}")
    print("LLM connection OK.\n")


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
