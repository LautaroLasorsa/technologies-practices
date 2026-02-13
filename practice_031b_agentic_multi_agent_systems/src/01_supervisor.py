"""
Phase 1: Supervisor Pattern — Centralized Multi-Agent Routing

Architecture:
    User Query -> [Supervisor] -> routes to -> [Specialist] -> result -> [Supervisor] -> ... -> END

The supervisor is the decision-maker: it receives the conversation so far, decides which
specialist should handle the next step, and eventually synthesizes a final answer.

This file implements the supervisor pattern MANUALLY with StateGraph (not the prebuilt
create_supervisor library) so you understand the graph mechanics from the ground up.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.1)

# ---------------------------------------------------------------------------
# Specialist definitions — system prompts that focus each agent
# ---------------------------------------------------------------------------

SPECIALIST_NAMES = ["researcher", "mathematician", "writer"]

SPECIALIST_PROMPTS: dict[str, str] = {
    "researcher": (
        "You are a research specialist. Your job is to provide accurate, factual "
        "information about any topic. Cite sources when possible. Be thorough but "
        "concise. Stick to facts — do not speculate or create fictional content."
    ),
    "mathematician": (
        "You are a mathematics specialist. Your job is to solve math problems step "
        "by step. Show your work clearly. Handle arithmetic, algebra, calculus, "
        "probability, and logic problems. Always verify your answer."
    ),
    "writer": (
        "You are a creative writing specialist. Your job is to produce engaging, "
        "well-structured text: stories, poems, summaries, emails, or any creative "
        "content. Focus on clarity, style, and tone appropriate to the request."
    ),
}

SUPERVISOR_SYSTEM_PROMPT = f"""\
You are a supervisor coordinating a team of specialists: {', '.join(SPECIALIST_NAMES)}.

Your job:
1. Read the user's query and the conversation history.
2. Decide which specialist should handle the next step.
3. If all needed work is done, respond with "FINISH" to end the conversation.

Rules:
- Route factual/knowledge questions to "researcher".
- Route math/calculation questions to "mathematician".
- Route creative writing requests to "writer".
- If the query needs multiple specialists, route to one at a time.
- After a specialist responds, decide if another specialist is needed or if you should FINISH.
- When finishing, provide a brief synthesis of all specialist responses.

Respond with ONLY the specialist name ({', '.join(SPECIALIST_NAMES)}) or FINISH. Nothing else."""


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class SupervisorState(BaseModel):
    """State for the supervisor graph.

    - messages: full conversation history (appended, never replaced)
    - next_agent: routing decision from the supervisor (specialist name or FINISH)
    """

    messages: Annotated[list[BaseMessage], operator.add] = Field(default_factory=list)
    next_agent: str = ""


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def supervisor_node(state: SupervisorState) -> dict:
    """Supervisor node: decides which specialist handles the next step.

    Reads the conversation history, calls the LLM with the supervisor system prompt,
    and returns a routing decision in `next_agent`.
    """
    # TODO(human) #1: Implement the supervisor's routing logic.
    #
    # The supervisor is the "brain" of the centralized pattern. It must:
    #   1. Build a message list starting with SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
    #      followed by ALL messages currently in state.messages.
    #   2. Call `llm.invoke(messages)` to get the LLM's routing decision.
    #   3. Parse the response content — it should be one of the SPECIALIST_NAMES or "FINISH".
    #      Strip whitespace and convert to lowercase for robustness.
    #   4. If the parsed response isn't a valid specialist name or "FINISH", default to "FINISH"
    #      (defensive programming — LLMs sometimes produce unexpected output).
    #   5. Return a dict with:
    #      - "next_agent": the parsed routing decision (str)
    #      - "messages": a list containing ONE AIMessage with content like
    #        "[Supervisor] Routing to: {decision}" so the conversation log shows the routing.
    #
    # Why this matters: The supervisor pattern's power (and weakness) is that ONE agent
    # sees the full context and makes ALL routing decisions. This is easy to debug and log,
    # but becomes a token-cost bottleneck as conversations grow — the supervisor re-reads
    # the entire history on every turn.
    raise NotImplementedError("TODO(human) #1: Implement supervisor_node")


def researcher_node(state: SupervisorState) -> dict:
    """Researcher specialist: answers factual questions."""
    # TODO(human) #2: Implement the three specialist nodes.
    #
    # Each specialist node follows the same pattern (you'll implement all three):
    #   1. Build a message list: SystemMessage with the specialist's prompt from
    #      SPECIALIST_PROMPTS[name], followed by state.messages.
    #   2. Call `llm.invoke(messages)` to get the specialist's response.
    #   3. Return a dict with "messages": a list containing ONE AIMessage whose content
    #      is prefixed with the specialist's name, e.g., "[Researcher] {response.content}".
    #      This prefix helps distinguish which agent produced which message when reading logs.
    #
    # Why three separate functions instead of one generic one? In a real system you'd
    # likely parameterize this (DRY principle). But for learning, writing each one
    # explicitly makes the graph structure visible — each node is a distinct function
    # that LangGraph can route to independently.
    #
    # Key insight: Each specialist only sees the FULL conversation history (including
    # the supervisor's routing messages and other specialists' responses). This is
    # "message passing" communication — rich context but high token cost. In Phase 5
    # you'll explore alternatives.
    raise NotImplementedError("TODO(human) #2: Implement researcher_node")


def mathematician_node(state: SupervisorState) -> dict:
    """Mathematician specialist: solves math problems step by step."""
    # Same pattern as researcher_node — see TODO(human) #2 comments above.
    # Use SPECIALIST_PROMPTS["mathematician"] and prefix with "[Mathematician]".
    raise NotImplementedError("TODO(human) #2: Implement mathematician_node")


def writer_node(state: SupervisorState) -> dict:
    """Writer specialist: produces creative text."""
    # Same pattern as researcher_node — see TODO(human) #2 comments above.
    # Use SPECIALIST_PROMPTS["writer"] and prefix with "[Writer]".
    raise NotImplementedError("TODO(human) #2: Implement writer_node")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def route_from_supervisor(state: SupervisorState) -> str:
    """Conditional edge function: returns the next node name based on supervisor's decision."""
    if state.next_agent == "FINISH":
        return END
    return state.next_agent


def build_supervisor_graph() -> StateGraph:
    """Build and return the compiled supervisor graph.

    Returns the compiled graph ready for .invoke().
    """
    # TODO(human) #3: Wire the supervisor graph with conditional routing.
    #
    # LangGraph's StateGraph is a directed graph where nodes are functions and edges
    # define the flow. You need to:
    #
    #   1. Create a StateGraph with SupervisorState as the state schema:
    #      `graph = StateGraph(SupervisorState)`
    #
    #   2. Add all four nodes:
    #      graph.add_node("supervisor", supervisor_node)
    #      graph.add_node("researcher", researcher_node)
    #      graph.add_node("mathematician", mathematician_node)
    #      graph.add_node("writer", writer_node)
    #
    #   3. Set the entry point: graph.set_entry_point("supervisor")
    #      Every query starts at the supervisor, who decides the first routing.
    #
    #   4. Add a CONDITIONAL edge from "supervisor" using route_from_supervisor:
    #      graph.add_conditional_edges("supervisor", route_from_supervisor)
    #      This tells LangGraph: "after the supervisor runs, call route_from_supervisor
    #      to determine which node to go to next." The function returns a node name
    #      (or END to terminate).
    #
    #   5. Add normal edges from each specialist BACK to the supervisor:
    #      graph.add_edge("researcher", "supervisor")
    #      graph.add_edge("mathematician", "supervisor")
    #      graph.add_edge("writer", "supervisor")
    #      After a specialist responds, control returns to the supervisor, who
    #      decides whether to call another specialist or FINISH.
    #
    #   6. Compile and return: return graph.compile()
    #
    # The resulting graph topology:
    #   START -> supervisor -> (conditional) -> researcher/mathematician/writer -> supervisor -> ... -> END
    #
    # This is the fundamental LangGraph pattern. Every multi-agent system you build
    # (even with prebuilt libraries) uses this StateGraph + conditional edges mechanism
    # under the hood.
    raise NotImplementedError("TODO(human) #3: Implement build_supervisor_graph")


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


def run_supervisor(query: str) -> str:
    """Run a single query through the supervisor graph and return the final response."""
    graph = build_supervisor_graph()
    initial_state = {"messages": [HumanMessage(content=query)]}
    result = graph.invoke(initial_state)

    # Extract the last AI message as the final response
    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else "(no response)"


def main() -> None:
    print("=" * 70)
    print("Phase 1: Supervisor Pattern — Centralized Multi-Agent Routing")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 70}")
        print(f"Query {i}: {query}")
        print("─" * 70)

        response = run_supervisor(query)
        print(f"\nFinal response:\n{response}")


if __name__ == "__main__":
    main()
