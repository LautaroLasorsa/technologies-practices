"""
Phase 4: Agent-as-Tool — Hierarchical Agent Composition

Architecture:
    User Query -> [Supervisor] -> calls research_tool() -> [Researcher Subgraph] -> result -> [Supervisor] -> ...

The "agent-as-tool" pattern wraps an entire agent (or sub-graph) as a callable tool
that another agent can invoke. This enables hierarchical composition: a supervisor
doesn't need to know the internal structure of its specialists — it just calls them
like any other tool function.

This is powerful for reuse: the researcher subgraph from Phase 1 becomes a black-box
tool that any future agent can use.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.1)

# ---------------------------------------------------------------------------
# Specialist system prompts (reused from Phase 1)
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = (
    "You are a research specialist. Your job is to provide accurate, factual "
    "information about any topic. Cite sources when possible. Be thorough but "
    "concise. Stick to facts — do not speculate or create fictional content."
)

MATHEMATICIAN_PROMPT = (
    "You are a mathematics specialist. Your job is to solve math problems step "
    "by step. Show your work clearly. Handle arithmetic, algebra, calculus, "
    "probability, and logic problems. Always verify your answer."
)

WRITER_PROMPT = (
    "You are a creative writing specialist. Your job is to produce engaging, "
    "well-structured text: stories, poems, summaries, emails, or any creative "
    "content. Focus on clarity, style, and tone appropriate to the request."
)


# ---------------------------------------------------------------------------
# Agent-as-Tool: Wrap specialists as callable tools
# ---------------------------------------------------------------------------


def create_agent_tools() -> list:
    """Create tool functions that wrap specialist agents.

    Each tool internally invokes a specialist (via direct LLM call or a compiled
    subgraph) and returns the result as a string.

    Returns a list of tool-decorated functions.
    """
    # TODO(human): Wrap specialist agents as callable tools.
    #
    # The agent-as-tool pattern is one of the most powerful composition techniques
    # in multi-agent systems. Instead of the supervisor routing via graph edges
    # (Phase 1) or agents handing off control (Phase 2), here the supervisor
    # simply CALLS another agent like a function and gets back a result.
    #
    # Implement three tool functions using the @tool decorator:
    #
    #   @tool
    #   def research_tool(query: str) -> str:
    #       """Research factual information about any topic. Use for knowledge questions."""
    #       messages = [
    #           SystemMessage(content=RESEARCHER_PROMPT),
    #           HumanMessage(content=query),
    #       ]
    #       response = llm.invoke(messages)
    #       return response.content
    #
    #   @tool
    #   def math_tool(query: str) -> str:
    #       """Solve math problems step by step. Use for calculations and math questions."""
    #       # Same pattern with MATHEMATICIAN_PROMPT
    #
    #   @tool
    #   def writing_tool(query: str) -> str:
    #       """Write creative content. Use for stories, poems, summaries, emails."""
    #       # Same pattern with WRITER_PROMPT
    #
    # Then return: [research_tool, math_tool, writing_tool]
    #
    # Why this matters: The @tool decorator makes these functions visible to the
    # LLM's tool-calling interface. The supervisor agent sees three tools with
    # descriptions and can call them as needed. The critical difference from
    # Phase 1's supervisor:
    #   - Phase 1: Supervisor returns a routing string, graph edge transfers control
    #   - Phase 4: Supervisor calls a tool function, gets result IN THE SAME TURN
    #
    # This means the supervisor can call multiple tools in sequence within one
    # "turn" (if the LLM supports multi-tool-call), and each tool invocation is
    # isolated — the specialist doesn't see the full conversation history, only
    # the specific query passed as an argument. This is the "tool-based"
    # communication strategy from the Theoretical Context.
    #
    # In a production system, the tool functions would compile and invoke a full
    # LangGraph subgraph instead of a single LLM call. For example:
    #   researcher_graph = build_researcher_subgraph()
    #   compiled = researcher_graph.compile()
    #   result = compiled.invoke({"messages": [HumanMessage(content=query)]})
    # This lets you nest arbitrarily complex agent graphs as tools.
    raise NotImplementedError("TODO(human): Implement agent-as-tool functions")


# ---------------------------------------------------------------------------
# Supervisor with tools (uses create_react_agent for simplicity)
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = """\
You are a supervisor coordinating a team of specialists via tools.

You have access to three specialist tools:
- research_tool: for factual questions and knowledge queries
- math_tool: for math problems and calculations
- writing_tool: for creative writing tasks

For each user request:
1. Determine which specialist(s) are needed
2. Call the appropriate tool(s) with specific sub-queries
3. Synthesize the results into a coherent final answer

You can call multiple tools for complex requests that span multiple domains."""


def build_supervisor_with_tools():
    """Build a supervisor agent that uses specialist agents as tools.

    Returns a compiled LangGraph agent.
    """
    agent_tools = create_agent_tools()

    # create_react_agent is a prebuilt LangGraph pattern that creates an agent
    # capable of reasoning and calling tools in a loop (ReAct pattern).
    # Unlike the manual StateGraph from Phase 1, this handles the
    # tool-calling loop automatically.
    supervisor = create_react_agent(
        model=llm,
        tools=agent_tools,
        prompt=SUPERVISOR_PROMPT,
        name="supervisor",
    )
    return supervisor


# ---------------------------------------------------------------------------
# Test queries
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What is the capital of France and what is its population?",
    "Calculate the integral of x^2 from 0 to 5.",
    "Write a haiku about distributed systems.",
    "What is quantum computing? Then write a limerick about it.",
    "How many prime numbers are there between 1 and 50?",
]


def run_agent_as_tool(query: str) -> str:
    """Run a query through the supervisor-with-tools and return the response."""
    supervisor = build_supervisor_with_tools()
    result = supervisor.invoke({"messages": [HumanMessage(content=query)]})

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    return "(no response)"


def main() -> None:
    print("=" * 70)
    print("Phase 4: Agent-as-Tool — Hierarchical Agent Composition")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print("─" * 70)

        response = run_agent_as_tool(query)
        print(f"\nFinal response:\n{response}")


if __name__ == "__main__":
    main()
