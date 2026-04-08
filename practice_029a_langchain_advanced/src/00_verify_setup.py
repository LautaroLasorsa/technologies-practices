"""
Practice 029a — Verify Setup
Connects to the configured LLM provider, runs a simple prompt, and prints the result.
This file is fully implemented — no TODO(human) blocks.
"""

from langchain_core.messages import HumanMessage

from llm_config import LLM_MODEL, LLM_PROVIDER, get_chat_model


def verify_llm_connection() -> None:
    """Test that the LLM provider is reachable and the model is loaded."""
    print(f"Provider: {LLM_PROVIDER}")
    print(f"Using model: {LLM_MODEL}")
    print("-" * 50)

    llm = get_chat_model(temperature=0)

    response = llm.invoke(
        [HumanMessage(content="Say 'Hello, LangChain!' and nothing else.")]
    )

    print(f"Response type: {type(response).__name__}")
    print(f"Response content: {response.content}")
    print("-" * 50)

    if response.content:
        print("Setup verified successfully! The model responds.")
    else:
        print("WARNING: Got empty response. Check that the model is pulled correctly.")


def verify_model_metadata() -> None:
    """Print model metadata to confirm configuration."""
    llm = get_chat_model()

    print(f"\nProvider: {LLM_PROVIDER}")
    print(f"Model: {LLM_MODEL}")
    print(f"Type: {type(llm).__name__}")


if __name__ == "__main__":
    try:
        verify_llm_connection()
        verify_model_metadata()
    except Exception as e:
        print(f"\nERROR: Could not connect to LLM provider '{LLM_PROVIDER}': {e}")
        print("\nTroubleshooting:")
        if LLM_PROVIDER == "ollama":
            print("  1. Is the Ollama container running?  docker compose up -d")
            print(f"  2. Is the model pulled?  docker exec ollama ollama pull {LLM_MODEL}")
            print("  3. Is port 11434 accessible?  curl http://localhost:11434/api/tags")
        elif LLM_PROVIDER == "lmstudio":
            print("  1. Is LM Studio running with the local server enabled?")
            print(f"  2. Is model '{LLM_MODEL}' loaded in LM Studio?")
        else:
            print(f"  1. Check LLM_API_KEY is set for provider '{LLM_PROVIDER}'")
            print(f"  2. Check LLM_MODEL='{LLM_MODEL}' is a valid model name")
        raise SystemExit(1)
