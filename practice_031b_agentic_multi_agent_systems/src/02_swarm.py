"""
Phase 2: Swarm Pattern — Decentralized Agent Handoffs

Architecture:
    User Query -> [Triage Agent] -> handoff -> [Specialist A] -> handoff -> [Specialist B] -> ... -> final answer

Unlike the supervisor pattern, there is NO central coordinator. Each agent decides
autonomously when to hand off control to a peer using handoff tools. The agents form
a peer-to-peer network.

This file uses langgraph-swarm's create_handoff_tool and create_swarm utilities.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph_swarm import create_handoff_tool, create_swarm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.1)

# ---------------------------------------------------------------------------
# Agent system prompts
# ---------------------------------------------------------------------------

TRIAGE_PROMPT = """\
You are a triage agent. Your job is to understand the user's request and hand it off
to the right specialist:
- For factual/knowledge questions: hand off to "researcher"
- For math/calculation problems: hand off to "mathematician"
- For creative writing requests: hand off to "writer"

If the request needs multiple specialists, start with the most relevant one.
After receiving a response, you can hand off to another specialist if needed,
or provide the final answer directly.

IMPORTANT: Always use a handoff tool to transfer to a specialist. Do not try to
answer specialized questions yourself."""

RESEARCHER_PROMPT = """\
You are a research specialist in a swarm of agents. Answer factual questions accurately
and concisely. Cite sources when possible.

After answering, if the user's original request also needs math or creative writing,
hand off to the appropriate specialist. If your answer is complete and no other
specialist is needed, provide your final response without handing off."""

MATHEMATICIAN_PROMPT = """\
You are a mathematics specialist in a swarm of agents. Solve math problems step by step.
Show your work clearly and verify your answer.

After solving, if the user's original request also needs research or creative writing,
hand off to the appropriate specialist. If your answer is complete, provide your
final response without handing off."""

WRITER_PROMPT = """\
You are a creative writing specialist in a swarm of agents. Produce engaging, well-structured
text: stories, poems, summaries, emails, or any creative content.

After writing, if the user's original request also needs research or math,
hand off to the appropriate specialist. If your answer is complete, provide your
final response without handing off."""


# ---------------------------------------------------------------------------
# Agent and swarm construction
# ---------------------------------------------------------------------------


def build_swarm_agents() -> list:
    """Create the three specialist agents with handoff tools.

    Returns a list of agents ready to be passed to create_swarm().
    """
    # TODO(human) #1: Create three peer agents with handoff tools.
    #
    # In the swarm pattern, each agent has handoff tools that let it transfer
    # control to any other agent it might need. This is fundamentally different
    # from the supervisor pattern: there is NO central decision-maker.
    #
    # For each agent you need to:
    #   1. Create handoff tools using create_handoff_tool(agent_name="<name>").
    #      - The triage agent needs handoff tools to: "researcher", "mathematician", "writer"
    #      - The researcher needs handoff tools to: "triage", "mathematician", "writer"
    #      - The mathematician needs handoff tools to: "triage", "researcher", "writer"
    #      - The writer needs handoff tools to: "triage", "researcher", "mathematician"
    #
    #   2. Create each agent using create_react_agent():
    #      create_react_agent(
    #          model=llm,
    #          tools=[...handoff_tools...],
    #          prompt=<system_prompt>,
    #          name="<agent_name>",  # MUST match the name used in create_handoff_tool
    #      )
    #
    #   3. Return a list of all agents: [triage_agent, researcher_agent, mathematician_agent, writer_agent]
    #
    # Key design decision: The triage agent acts as the entry point (default_active_agent
    # in create_swarm). It doesn't answer questions itself — it only routes via handoffs.
    # This is a common swarm pattern: one "router" agent + N specialist agents.
    #
    # Compare with supervisor: In the supervisor pattern, the supervisor explicitly
    # chooses the next agent via conditional edges. Here, each agent autonomously
    # decides whether to hand off and to whom — the LLM itself makes the routing
    # decision as part of its tool-calling behavior.
    raise NotImplementedError("TODO(human) #1: Create swarm agents with handoff tools")


def build_swarm_graph():
    """Build the swarm graph from agents.

    Returns the compiled swarm graph ready for .invoke().
    """
    # TODO(human) #2: Build and compile the swarm.
    #
    # langgraph-swarm provides create_swarm() which wires all agents into a single
    # LangGraph StateGraph where any agent can hand off to any other.
    #
    # Steps:
    #   1. Call build_swarm_agents() to get the list of agents.
    #   2. Create a MemorySaver checkpointer — the swarm library requires this
    #      for tracking which agent is currently active across turns.
    #   3. Call create_swarm(agents, default_active_agent="triage") to create the
    #      swarm workflow. The default_active_agent is the starting agent.
    #   4. Compile the workflow: workflow.compile(checkpointer=checkpointer)
    #   5. Return the compiled graph.
    #
    # Under the hood, create_swarm creates a StateGraph where:
    #   - Each agent is a node
    #   - Handoff tools create dynamic edges between agents
    #   - The active agent tracks which node is currently executing
    #   - When an agent calls a handoff tool, control transfers to the target agent
    #
    # This is much less code than the manual supervisor graph from Phase 1, but
    # you have less control over the routing logic — it's entirely up to the LLM's
    # tool-calling behavior.
    raise NotImplementedError("TODO(human) #2: Build and compile the swarm graph")


# ---------------------------------------------------------------------------
# Test queries (same as Phase 1 for comparison)
# ---------------------------------------------------------------------------

TEST_QUERIES = [
    "What is the capital of France and what is its population?",
    "Calculate the integral of x^2 from 0 to 5.",
    "Write a haiku about distributed systems.",
    "What is quantum computing? Then write a limerick about it.",
    "How many prime numbers are there between 1 and 50?",
]


def run_swarm(query: str) -> str:
    """Run a single query through the swarm and return the final response."""
    graph = build_swarm_graph()

    # Swarm requires a thread_id for the checkpointer
    config = {"configurable": {"thread_id": "swarm-test"}}
    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )

    # Extract the last AI message
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content and not hasattr(msg, "tool_calls"):
            return msg.content
    return "(no response)"


def main() -> None:
    print("=" * 70)
    print("Phase 2: Swarm Pattern — Decentralized Agent Handoffs")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print("─" * 70)

        response = run_swarm(query)
        print(f"\nFinal response:\n{response}")


if __name__ == "__main__":
    main()
