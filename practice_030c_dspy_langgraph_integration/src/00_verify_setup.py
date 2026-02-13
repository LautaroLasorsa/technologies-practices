"""
Practice 030c — Verify Setup
Checks that DSPy, LangGraph, and Ollama are all installed and reachable.
"""

import sys

OLLAMA_MODEL = "qwen2.5:7b"
OLLAMA_BASE_URL = "http://localhost:11434"


def verify_dspy() -> None:
    """Verify DSPy is importable and can configure an Ollama LM."""
    import dspy  # noqa: F811

    lm = dspy.LM(
        model=f"ollama_chat/{OLLAMA_MODEL}",
        api_base=f"{OLLAMA_BASE_URL}/v1",
        api_key="",
    )
    dspy.configure(lm=lm)

    # Quick generation test
    result = dspy.ChainOfThought("question -> answer")(
        question="What is 2 + 2?"
    )
    print(f"  [DSPy]      ChainOfThought answer: {result.answer}")


def verify_langgraph() -> None:
    """Verify LangGraph is importable and can build a trivial graph."""
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class TestState(TypedDict):
        value: str

    def node_a(state: TestState) -> dict:
        return {"value": state["value"] + " -> node_a"}

    graph = StateGraph(TestState)
    graph.add_node("a", node_a)
    graph.add_edge(START, "a")
    graph.add_edge("a", END)
    app = graph.compile()

    result = app.invoke({"value": "start"})
    print(f"  [LangGraph] Graph result: {result['value']}")


def verify_langchain_ollama() -> None:
    """Verify LangChain can reach the Ollama server."""
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    response = llm.invoke("Say 'hello' and nothing else.")
    print(f"  [ChatOllama] Response: {response.content[:80]}")


def verify_dspy_langchain_extra() -> None:
    """Verify the dspy[langchain] extra is installed (Tool.from_langchain)."""
    from dspy import Tool  # noqa: F401

    # Tool.from_langchain is available if dspy[langchain] extra is installed.
    assert hasattr(Tool, "from_langchain"), (
        "Tool.from_langchain not found — reinstall with: uv add 'dspy-ai[langchain]'"
    )
    print("  [DSPy extra] Tool.from_langchain is available")


def main() -> None:
    checks = [
        ("DSPy + Ollama", verify_dspy),
        ("LangGraph", verify_langgraph),
        ("LangChain + Ollama", verify_langchain_ollama),
        ("DSPy LangChain extra", verify_dspy_langchain_extra),
    ]

    print("=" * 60)
    print("Practice 030c — Setup Verification")
    print("=" * 60)

    failed = []
    for name, fn in checks:
        try:
            print(f"\nChecking {name}...")
            fn()
            print(f"  OK")
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED checks: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All checks passed. Ready to start practicing!")


if __name__ == "__main__":
    main()
