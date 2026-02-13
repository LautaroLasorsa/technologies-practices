"""
Phase 0: Verify that all dependencies and infrastructure are working.

Checks:
1. LangGraph imports and version
2. LangGraph Supervisor library
3. LangGraph Swarm library
4. CrewAI imports and version
5. Ollama connectivity and model availability
6. Quick LLM inference test via ChatOllama
"""

import sys


def check_langgraph() -> bool:
    """Verify LangGraph core imports."""
    try:
        import langgraph  # noqa: F401
        from langgraph.graph import StateGraph, MessagesState  # noqa: F401
        from langgraph.prebuilt import create_react_agent  # noqa: F401

        print(f"  LangGraph version: {langgraph.__version__}")
        print("  StateGraph, MessagesState, create_react_agent: OK")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def check_langgraph_supervisor() -> bool:
    """Verify LangGraph Supervisor library."""
    try:
        from langgraph_supervisor import create_supervisor  # noqa: F401

        print("  create_supervisor: OK")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def check_langgraph_swarm() -> bool:
    """Verify LangGraph Swarm library."""
    try:
        from langgraph_swarm import create_swarm, create_handoff_tool  # noqa: F401

        print("  create_swarm, create_handoff_tool: OK")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def check_crewai() -> bool:
    """Verify CrewAI imports."""
    try:
        import crewai  # noqa: F401
        from crewai import Agent, Task, Crew, Process, LLM  # noqa: F401

        print(f"  CrewAI version: {crewai.__version__}")
        print("  Agent, Task, Crew, Process, LLM: OK")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def check_langchain_ollama() -> bool:
    """Verify LangChain-Ollama integration."""
    try:
        from langchain_ollama import ChatOllama  # noqa: F401

        print("  ChatOllama: OK")
        return True
    except ImportError as e:
        print(f"  FAILED: {e}")
        return False


def check_ollama_connectivity() -> bool:
    """Verify Ollama server is reachable and model is available."""
    try:
        import urllib.request
        import json

        url = "http://localhost:11434/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())

        models = [m["name"] for m in data.get("models", [])]
        print(f"  Ollama reachable at localhost:11434")
        print(f"  Available models: {models}")

        has_qwen = any("qwen2.5" in m for m in models)
        if has_qwen:
            print("  qwen2.5 model: FOUND")
        else:
            print("  WARNING: qwen2.5 model not found. Run:")
            print("    docker exec ollama ollama pull qwen2.5:7b")
        return has_qwen
    except Exception as e:
        print(f"  FAILED to connect to Ollama: {e}")
        print("  Make sure Docker is running: docker compose up -d")
        return False


def check_llm_inference() -> bool:
    """Quick inference test — send a trivial prompt to ChatOllama."""
    try:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model="qwen2.5:7b", base_url="http://localhost:11434")
        response = llm.invoke("Say 'hello' and nothing else.")
        content = response.content.strip()
        print(f"  LLM response: {content[:80]}")
        print("  Inference: OK")
        return True
    except Exception as e:
        print(f"  Inference FAILED: {e}")
        return False


def main() -> None:
    checks = [
        ("LangGraph Core", check_langgraph),
        ("LangGraph Supervisor", check_langgraph_supervisor),
        ("LangGraph Swarm", check_langgraph_swarm),
        ("CrewAI", check_crewai),
        ("LangChain-Ollama", check_langchain_ollama),
        ("Ollama Connectivity", check_ollama_connectivity),
        ("LLM Inference", check_llm_inference),
    ]

    print("=" * 60)
    print("Practice 031b — Multi-Agent Systems: Setup Verification")
    print("=" * 60)

    results: list[tuple[str, bool]] = []
    for name, check_fn in checks:
        print(f"\n[{name}]")
        passed = check_fn()
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("Summary:")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll checks passed! Ready to start the practice.")
    else:
        print("\nSome checks failed. Fix the issues above before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
