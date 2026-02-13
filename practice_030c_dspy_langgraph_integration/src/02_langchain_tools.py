"""
Practice 030c — Phase 2: LangChain Tool Integration

DSPy can consume LangChain tools via Tool.from_langchain(). This bridges
the LangChain ecosystem (hundreds of pre-built tools) into DSPy's optimized
agent framework.

Run: uv run python src/02_langchain_tools.py
"""

import dspy
from dspy import Tool

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
# Native DSPy tool (for mixing with LangChain tools)
# ---------------------------------------------------------------------------

def calculator(expression: str) -> float:
    """Evaluate a mathematical expression and return the numeric result."""
    return float(eval(expression))


# ---------------------------------------------------------------------------
# TODO(human) #3: Create a LangChain tool, convert it, and use in ReAct
# ---------------------------------------------------------------------------
# The bridge between LangChain and DSPy is Tool.from_langchain(). It takes a
# LangChain tool object and wraps it so DSPy's ReAct agent can call it just
# like a native DSPy tool. This is important because LangChain has a massive
# ecosystem of pre-built tools (web search, databases, APIs, etc.) that you
# can reuse without reimplementing them.
#
# Steps:
#   1. Import the @tool decorator from langchain_core.tools
#
#   2. Define a LangChain tool using the @tool decorator. For example, a
#      dictionary lookup tool:
#        @tool
#        def dictionary_lookup(word: str) -> str:
#            """Look up the definition of an English word."""
#            definitions = {
#                "python": "A large constricting snake or a programming language",
#                "graph": "A diagram showing relationships between quantities",
#                "agent": "A person or program that acts on behalf of another",
#            }
#            return definitions.get(word.lower(), f"No definition found for '{word}'")
#
#   3. Convert it to a DSPy tool:
#        dspy_dict_tool = Tool.from_langchain(dictionary_lookup)
#
#   4. Build a ReAct agent that uses BOTH the native calculator tool AND the
#      converted LangChain tool:
#        agent = dspy.ReAct("question -> answer", tools=[calculator, dspy_dict_tool], max_iters=5)
#
#   5. Test with questions that require each tool:
#      - "What is 42 * 17?"                    -> should use calculator
#      - "What does the word 'graph' mean?"     -> should use dictionary_lookup
#      - "What is 100 / 4, and define 'agent'?" -> may use both tools
#
#   6. Print the results for each test question.

def test_langchain_tool_integration() -> None:
    raise NotImplementedError("TODO(human): create LangChain tool, convert, and test in ReAct")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 2: LangChain Tool Integration")
    print("=" * 60)

    test_langchain_tool_integration()


if __name__ == "__main__":
    main()
