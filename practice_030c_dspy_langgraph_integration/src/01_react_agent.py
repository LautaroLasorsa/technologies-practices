"""
Practice 030c — Phase 1: ReAct Agent with Custom Tools

DSPy's ReAct module implements the Reason-Act-Observe loop automatically.
You define tools as plain Python functions with type hints and docstrings,
and DSPy extracts the tool schemas for the agent to use.

Run: uv run python src/01_react_agent.py
"""

import dspy

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
# TODO(human) #1: Define Python tools for the ReAct agent
# ---------------------------------------------------------------------------
# DSPy's tool system works by introspecting Python functions: it reads the
# function name, docstring, and type hints to build a tool schema that the
# ReAct agent uses to decide which tool to call and with what arguments.
#
# Define three tool functions:
#
# 1. calculator(expression: str) -> float
#    - Evaluates a mathematical expression string and returns the result.
#    - The docstring should clearly state what the tool does (e.g.,
#      "Evaluate a mathematical expression and return the numeric result.").
#    - Hint: you can use Python's eval() for simplicity in this practice.
#
# 2. word_count(text: str) -> int
#    - Counts the number of words in the given text.
#    - Split on whitespace and return the count.
#
# 3. text_reverse(text: str) -> str
#    - Reverses the input text string.
#    - Returns the reversed string.
#
# Why docstrings matter: DSPy reads the docstring to tell the LLM what each
# tool does. A vague or missing docstring means the agent won't know when to
# use the tool. Type hints tell DSPy the expected input/output types.

def calculator(expression: str) -> float:
    raise NotImplementedError("TODO(human): implement calculator tool")


def word_count(text: str) -> int:
    raise NotImplementedError("TODO(human): implement word_count tool")


def text_reverse(text: str) -> str:
    raise NotImplementedError("TODO(human): implement text_reverse tool")


# ---------------------------------------------------------------------------
# TODO(human) #2: Build and test the ReAct agent
# ---------------------------------------------------------------------------
# dspy.ReAct is a built-in agent module that implements the Reason-Act-Observe
# loop. You provide a signature (input -> output fields) and a list of tools.
# The agent will:
#   1. Reason about the question
#   2. Select a tool and arguments
#   3. Execute the tool and observe the result
#   4. Repeat until it has enough info, then produce the final answer
#
# Steps:
#   1. Create the agent:
#      agent = dspy.ReAct("question -> answer", tools=[calculator, word_count, text_reverse], max_iters=5)
#
#   2. Test with questions that exercise each tool:
#      - "What is (15 * 7) + 23?"           -> should use calculator
#      - "How many words are in 'the quick brown fox jumps'?" -> should use word_count
#      - "Reverse the text 'hello world'"    -> should use text_reverse
#
#   3. Print the agent's answer for each question.
#
# Note: max_iters=5 limits the reasoning loops to prevent infinite cycles.
# The agent may take 1-3 iterations depending on the complexity of the question.

def build_and_test_agent() -> None:
    raise NotImplementedError("TODO(human): build ReAct agent and test it")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 1: ReAct Agent with Custom Tools")
    print("=" * 60)

    build_and_test_agent()


if __name__ == "__main__":
    main()
