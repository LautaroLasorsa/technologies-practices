"""Exercise 1: LLM-Based Delegation — Specialist Router.

This exercise demonstrates the most natural ADK multi-agent pattern:
a parent LLM agent with several `sub_agents`, where the parent's LLM
reasoning decides which specialist to delegate to based on the user's query.

Architecture:
    RouterAgent (parent — LLM decides routing)
    ├── MathAgent      (tools: calculate, solve_equation)
    ├── TriviaAgent    (tools: lookup_fact, quiz)
    └── TranslatorAgent(tools: translate)

How LLM delegation works:
- The router's instruction tells it "you have these specialist agents"
- The router's LLM sees the user query and decides which child to invoke
- ADK automatically handles the transfer: the child agent takes over,
  processes the request with its own tools, and returns the result
- The parent sees the child's response and can either return it or
  delegate again

Key concept — `sub_agents`:
    When you pass `sub_agents=[agent1, agent2, ...]` to a parent Agent,
    ADK registers each child so the parent LLM can reference them by name.
    The LLM generates a "transfer" action to route to the chosen child.

Key concept — `description`:
    Each child agent's `description` is critical — it's what the parent LLM
    reads to decide which child to invoke. Think of it as the agent's
    "advertisement" to the router. A vague description leads to misrouting.
"""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

MODEL = LiteLlm(model="ollama_chat/qwen2.5:7b", api_base="http://localhost:11434")


# ---------------------------------------------------------------------------
# Tools for each specialist
# ---------------------------------------------------------------------------
# ADK auto-wraps plain Python functions as FunctionTool when you pass them
# in the `tools` list. The function docstring becomes the tool description
# the LLM sees, and type-annotated parameters become the tool's schema.
# ---------------------------------------------------------------------------


def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result.

    Args:
        expression: A Python-syntax math expression (e.g. "2**10 + 3*7").

    Returns:
        The result as a string, or an error message if evaluation fails.
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"Result: {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


def solve_equation(equation: str) -> str:
    """Solve a simple linear equation for x and return the solution.

    Only handles equations of the form 'ax + b = c' or 'ax = c'.

    Args:
        equation: A simple equation string, e.g. "2x + 3 = 7".

    Returns:
        The solution as a string, e.g. "x = 2.0", or an error message.
    """
    try:
        left, right = equation.replace(" ", "").split("=")
        right_val = float(right)
        if "x" in left:
            parts = left.replace("-", "+-").split("+")
            coeff = 0.0
            const = 0.0
            for part in parts:
                if not part:
                    continue
                if "x" in part:
                    c = part.replace("x", "")
                    coeff += float(c) if c and c != "+" and c != "-" else (1.0 if c != "-" else -1.0)
                else:
                    const += float(part)
            x = (right_val - const) / coeff
            return f"x = {x}"
        return f"No variable 'x' found in: {equation}"
    except Exception as e:
        return f"Error solving '{equation}': {e}"


def lookup_fact(topic: str) -> str:
    """Look up a fun fact about a given topic.

    Args:
        topic: The subject to find a fact about (e.g. "octopus", "Mars").

    Returns:
        A trivia fact string.
    """
    facts: dict[str, str] = {
        "octopus": "An octopus has three hearts and blue blood.",
        "mars": "A day on Mars (a sol) is 24 hours and 37 minutes.",
        "honey": "Honey never spoils. Archaeologists found 3000-year-old edible honey.",
        "bananas": "Bananas are berries, but strawberries are not.",
        "python": "Python is named after Monty Python, not the snake.",
    }
    key = topic.lower().strip()
    for k, v in facts.items():
        if k in key:
            return v
    return f"I don't have a stored fact about '{topic}', but it's surely fascinating!"


def quiz(topic: str) -> str:
    """Generate a simple trivia question about a given topic.

    Args:
        topic: The topic for the trivia question.

    Returns:
        A trivia question string.
    """
    return f"Trivia question: What is an interesting fact about {topic}? (Think about it!)"


def translate(text: str, target_language: str) -> str:
    """Translate text to a target language (simulated).

    In a real system this would call a translation API. Here it returns
    a placeholder showing that the tool was invoked correctly.

    Args:
        text: The text to translate.
        target_language: The language to translate into (e.g. "Spanish").

    Returns:
        A string indicating the translation request was processed.
    """
    return f"[Translation to {target_language}]: '{text}' — (simulated translation result)"


# ---------------------------------------------------------------------------
# TODO(human): Create the specialist agents and the router agent.
# ---------------------------------------------------------------------------
#
# Your task is to create 3 specialist LlmAgent instances and 1 router
# LlmAgent that delegates to them via `sub_agents`.
#
# Step 1 — Create the specialist agents
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create three LlmAgent instances: math_agent, trivia_agent, translator_agent.
#
# Each specialist needs:
#   - name: a unique string identifier (e.g., "math_agent")
#   - model: use the MODEL constant defined above
#   - instruction: a clear system prompt telling the agent its role and
#     how to use its tools. Be specific — the agent only sees its own
#     tools, so the instruction should guide it to use them correctly.
#     Example for math: "You are a math specialist. Use the `calculate`
#     tool for arithmetic and `solve_equation` for algebra."
#   - description: a SHORT string that the router LLM reads to decide
#     whether to delegate here. This is crucial for correct routing.
#     Example: "Handles math calculations and equation solving."
#   - tools: list of tool functions this agent can use
#
# The tools for each specialist:
#   - math_agent:       [calculate, solve_equation]
#   - trivia_agent:     [lookup_fact, quiz]
#   - translator_agent: [translate]
#
# Step 2 — Create the router agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create a LlmAgent named "specialist_router" that acts as the parent.
#
# The router needs:
#   - name: "specialist_router"
#   - model: MODEL
#   - instruction: Tell it that it's a routing agent that analyzes user
#     requests and delegates to the appropriate specialist. It should
#     NOT try to answer directly — it should always transfer to a child.
#     Mention each child by name so the LLM knows what's available.
#   - description: "Routes user queries to the appropriate specialist agent."
#   - sub_agents: [math_agent, trivia_agent, translator_agent]
#   - tools: [] (the router doesn't need its own tools — it delegates)
#
# Step 3 — Export as root_agent
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Assign your router agent to a variable called `root_agent`.
# This is what ADK's `adk web` command looks for when loading the module.
#
# Example structure (fill in the details):
#
#   math_agent = LlmAgent(
#       name="math_agent",
#       model=MODEL,
#       instruction="...",
#       description="...",
#       tools=[calculate, solve_equation],
#   )
#
#   trivia_agent = LlmAgent(...)
#   translator_agent = LlmAgent(...)
#
#   root_agent = LlmAgent(
#       name="specialist_router",
#       model=MODEL,
#       instruction="...",
#       description="...",
#       sub_agents=[math_agent, trivia_agent, translator_agent],
#   )
#
# Test prompts to try after implementation:
#   - "What is 2^10 + 3*7?"          → should route to math_agent
#   - "Tell me a fun fact about Mars" → should route to trivia_agent
#   - "Translate 'hello' to Spanish"  → should route to translator_agent
#   - "Solve 3x + 5 = 20"            → should route to math_agent
# ---------------------------------------------------------------------------

raise NotImplementedError(
    "TODO(human): Create math_agent, trivia_agent, translator_agent, "
    "and the root_agent (specialist_router) with sub_agents=[...]"
)
