"""
Practice 029a — Verify Setup
Connects to Ollama, runs a simple prompt, and prints the result.
This file is fully implemented — no TODO(human) blocks.
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"


def verify_ollama_connection() -> None:
    """Test that the Ollama server is reachable and the model is loaded."""
    print(f"Connecting to Ollama at {OLLAMA_BASE_URL}...")
    print(f"Using model: {MODEL_NAME}")
    print("-" * 50)

    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    response = llm.invoke([HumanMessage(content="Say 'Hello, LangChain!' and nothing else.")])

    print(f"Response type: {type(response).__name__}")
    print(f"Response content: {response.content}")
    print("-" * 50)

    if response.content:
        print("Setup verified successfully! Ollama is running and the model responds.")
    else:
        print("WARNING: Got empty response. Check that the model is pulled correctly.")


def verify_model_metadata() -> None:
    """Print model metadata to confirm configuration."""
    llm = ChatOllama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
    )

    print(f"\nModel identifier: {llm.model}")
    print(f"Base URL: {llm.base_url}")
    print(f"Temperature: {llm.temperature}")
    print(f"Type: {type(llm).__name__}")


if __name__ == "__main__":
    try:
        verify_ollama_connection()
        verify_model_metadata()
    except Exception as e:
        print(f"\nERROR: Could not connect to Ollama: {e}")
        print("\nTroubleshooting:")
        print("  1. Is the Ollama container running?  docker compose up -d")
        print(f"  2. Is the model pulled?  docker exec ollama ollama pull {MODEL_NAME}")
        print(f"  3. Is port 11434 accessible?  curl {OLLAMA_BASE_URL}/api/tags")
        raise SystemExit(1)
