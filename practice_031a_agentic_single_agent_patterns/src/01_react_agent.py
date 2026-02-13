"""ReAct Agent — Reasoning + Acting in an iterative loop.

Pattern: Thought → Action → Observation → Thought → ...

The agent decides which tool to call (or whether to respond directly) based on
the conversation so far. This is the most common single-agent pattern and the
foundation for understanding all other patterns in this practice.

We build the ReAct loop FROM SCRATCH using LangGraph primitives (StateGraph,
conditional edges, cycles) — no prebuilt `create_react_agent`.

Run:
    uv run python src/01_react_agent.py
"""

import operator
from typing import Annotated, Literal

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
MAX_ITERATIONS = 5

SYSTEM_PROMPT = """You are a helpful assistant with access to tools.

When you need to look something up, use the web_search tool.
When you need to compute something, use the calculator tool.

Always reason step by step before deciding which tool to use.
When you have enough information to answer, respond directly without calling tools."""


# ── Tools ────────────────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for information. Returns a text snippet."""
    # Mock search results — in production this would call a real search API
    mock_results = {
        "python": "Python is a high-level programming language created by Guido van Rossum in 1991. It emphasizes readability and simplicity.",
        "langgraph": "LangGraph is a framework by LangChain for building stateful, multi-actor applications with LLMs, using graph-based orchestration.",
        "react pattern": "ReAct (Reasoning + Acting) is an agent pattern where the LLM alternates between reasoning about the current state and taking actions via tools.",
        "population france": "France has a population of approximately 68 million people as of 2024.",
        "capital japan": "Tokyo is the capital city of Japan, with a metropolitan population of about 14 million.",
    }
    query_lower = query.lower()
    for key, result in mock_results.items():
        if key in query_lower:
            return result
    return f"No results found for: {query}. Try a different search query."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Input should be a valid Python math expression."""
    try:
        # Only allow safe math operations
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


TOOLS = [web_search, calculator]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ── State ────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """State for the ReAct agent.

    messages: The conversation history. Uses operator.add so each node's
              return value is APPENDED to the existing list (not replaced).
    iteration_count: Tracks how many agent→tool cycles have occurred.
                     Used by the conditional edge to enforce MAX_ITERATIONS.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    iteration_count: int


# ── LLM ──────────────────────────────────────────────────────────────

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement these three functions ─────────────────────


def agent_node(state: AgentState) -> dict:
    """Call the LLM with the current messages and tools.

    TODO(human): Implement this function.

    This is the "brain" of the ReAct loop. The agent receives the full
    conversation history (including previous tool results) and decides
    what to do next: call a tool, or produce a final answer.

    Steps:
      1. Bind the tools to the LLM using llm.bind_tools(TOOLS). This tells
         the model which tools are available and their schemas. The model
         can then produce structured tool_calls in its response.
      2. Build the input messages: start with SystemMessage(content=SYSTEM_PROMPT),
         then append all messages from state["messages"].
      3. Invoke the model with these messages: model_with_tools.invoke(messages).
         The response is an AIMessage that either contains tool_calls (the model
         wants to use a tool) or just content (the model is done).
      4. Increment iteration_count by 1.
      5. Return {"messages": [response], "iteration_count": new_count}.

    Key concept: The LLM doesn't execute tools — it only DECIDES which tool
    to call and with what arguments. The actual execution happens in tool_node.
    This separation is what makes the ReAct pattern auditable and controllable.

    Hint:
      model_with_tools = llm.bind_tools(TOOLS)
      response = model_with_tools.invoke([SystemMessage(...)] + state["messages"])
    """
    raise NotImplementedError("TODO(human): implement agent_node")


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Decide whether to continue the ReAct loop or stop.

    TODO(human): Implement this function.

    This is the conditional edge that controls the agent's loop. After each
    agent_node call, LangGraph calls this function to decide the next step.

    Steps:
      1. Get the last message from state["messages"] — this is the AIMessage
         that agent_node just produced.
      2. Check if the message has tool_calls (hasattr(last_message, "tool_calls")
         and last_message.tool_calls is truthy).
      3. Also check if iteration_count < MAX_ITERATIONS to prevent infinite loops.
      4. If BOTH conditions are true: return "tools" (continue to tool_node).
         If EITHER is false: return "__end__" (stop the agent).

    Why MAX_ITERATIONS matters: Without it, a confused model could loop forever
    calling tools without making progress. In production, this is a critical
    safety mechanism. The cost grows linearly with each LLM call, so unbounded
    loops can also be expensive.

    Hint:
      last_message = state["messages"][-1]
      has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
      Return "tools" or END (which equals "__end__")
    """
    raise NotImplementedError("TODO(human): implement should_continue")


def build_react_graph() -> StateGraph:
    """Wire the ReAct graph: START → agent → conditional → tools → agent (cycle) or END.

    TODO(human): Implement this function.

    This function constructs the LangGraph state machine that implements the
    ReAct loop. The graph has two nodes (agent, tools) and edges that create
    a cycle: agent calls LLM, conditional edge checks if we should continue,
    if yes → tools node executes the tool and loops back to agent.

    Steps:
      1. Create a StateGraph(AgentState).
      2. Add two nodes:
         - "agent": the agent_node function
         - "tools": the tool_node function (already implemented below)
      3. Add edge from START → "agent" (the agent always runs first).
      4. Add conditional edges from "agent" using should_continue:
         - If should_continue returns "tools" → go to "tools" node
         - If should_continue returns "__end__" → go to END
         Use: graph.add_conditional_edges("agent", should_continue, ["tools", END])
      5. Add edge from "tools" → "agent" (after tool execution, always go back to agent).
      6. Return graph.compile().

    The resulting graph looks like:
        START → agent ──(has tool calls)──→ tools ──→ agent (loop)
                  │
                  └──(no tool calls / max iterations)──→ END

    Hint:
      graph = StateGraph(AgentState)
      graph.add_node("agent", agent_node)
      graph.add_node("tools", tool_node)
      graph.add_edge(START, "agent")
      graph.add_conditional_edges("agent", should_continue, ["tools", END])
      graph.add_edge("tools", "agent")
      return graph.compile()
    """
    raise NotImplementedError("TODO(human): implement build_react_graph")


# ── Tool execution node (provided) ──────────────────────────────────


def tool_node(state: AgentState) -> dict:
    """Execute all tool calls from the last AIMessage.

    This node is GIVEN to you — it handles the mechanics of running tools
    and producing ToolMessage results. Study how it works:

    1. Gets the last message (an AIMessage with tool_calls)
    2. For each tool_call, looks up the tool by name and invokes it
    3. Returns ToolMessages that the agent will see in its next iteration
    """
    last_message = state["messages"][-1]
    results = []
    for tool_call in last_message.tool_calls:
        tool_fn = TOOLS_BY_NAME[tool_call["name"]]
        observation = tool_fn.invoke(tool_call["args"])
        results.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
            )
        )
    return {"messages": results}


# ── Orchestration ────────────────────────────────────────────────────


def run_agent(question: str) -> None:
    """Run the ReAct agent on a single question and print the trace."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    agent = build_react_graph()
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "iteration_count": 0,
    }

    # Stream events to see the agent's reasoning step by step
    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} ---")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  Tool call: {tc['name']}({tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        content = msg.content
                        prefix = msg.__class__.__name__
                        print(f"  [{prefix}] {content[:200]}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    print("Practice 031a — Phase 1: ReAct Agent")
    print("Building a ReAct agent from scratch with LangGraph\n")

    # Test 1: Simple tool usage (web search)
    run_agent("What is the population of France?")

    # Test 2: Multi-step reasoning (search + calculate)
    run_agent("What is the population of France divided by 4?")

    # Test 3: Direct answer (no tools needed)
    run_agent("What is 2 + 2?")


if __name__ == "__main__":
    main()
