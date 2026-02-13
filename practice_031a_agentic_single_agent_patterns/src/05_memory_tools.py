"""Memory & Dynamic Tools — Working memory (scratchpad) and runtime tool creation.

Demonstrates two advanced agent capabilities:
1. Scratchpad memory: Agent writes intermediate notes to a state field,
   references them in later reasoning steps (working memory).
2. Dynamic tool creation: Agent writes a Python function as a string,
   exec() it into a callable, and uses it as a new tool in subsequent iterations.

Run:
    uv run python src/05_memory_tools.py
"""

import operator
from typing import Annotated, Any, Callable, Literal

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
MAX_ITERATIONS = 6

SCRATCHPAD_SYSTEM_PROMPT = """You are a research assistant with a scratchpad for notes.

You have access to these tools:
- web_search(query): Search for information
- note_to_scratchpad(note): Save a note to your scratchpad for later reference
- read_scratchpad(): Read all notes from your scratchpad

IMPORTANT workflow:
1. Search for information using web_search
2. Save key findings to your scratchpad using note_to_scratchpad
3. Read your scratchpad to review all findings before answering
4. When you have enough info, provide your final answer WITHOUT calling tools

Use the scratchpad to track intermediate findings across multiple searches.
This helps you synthesize information from multiple sources."""

DYNAMIC_TOOL_SYSTEM_PROMPT = """You are a problem-solving assistant that can create new tools at runtime.

You have access to these tools:
- create_tool(name, code): Create a new Python function that becomes a tool
  - name: the function name (e.g., "fibonacci")
  - code: the Python function code as a string (e.g., "def fibonacci(n):\\n    if n <= 1: return n\\n    return fibonacci(n-1) + fibonacci(n-2)")
- run_tool(name, args): Run a previously created tool with given arguments
  - name: the tool name you created
  - args: the argument to pass (as string, will be eval'd)

Workflow:
1. Analyze the problem
2. Create a tool (Python function) that solves it
3. Run the tool with the specific input
4. Report the result"""


# ── Mock search tool ─────────────────────────────────────────────────


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    mock_db = {
        "python typing": "Python's typing module (PEP 484) provides type hints. Key types: Optional, Union, List, Dict, TypedDict, Protocol. TypeVar for generics.",
        "python dataclass": "dataclasses (PEP 557) auto-generate __init__, __repr__, __eq__. Use @dataclass decorator. Support default values, field(), frozen=True.",
        "python protocol": "Protocols (PEP 544) enable structural subtyping. Classes match a Protocol if they have matching methods, no inheritance needed.",
        "pydantic": "Pydantic v2 uses Rust-based validation. BaseModel for data classes, Field for constraints, model_validator for cross-field validation.",
        "langgraph state": "LangGraph state uses TypedDict with Annotated fields. operator.add for list append, custom reducers for complex merging.",
    }
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key in query_lower:
            return value
    return f"No specific results for '{query}'. Try narrowing your search."


# ══════════════════════════════════════════════════════════════════════
# PART 1: Scratchpad Memory
# ══════════════════════════════════════════════════════════════════════


class ScratchpadState(TypedDict):
    """State for the scratchpad agent.

    messages: Conversation history (appended via operator.add).
    scratchpad: List of notes the agent has saved. Persists across iterations,
                letting the agent build up knowledge incrementally.
    iteration_count: Loop counter for safety.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    scratchpad: list[str]
    iteration_count: int


# Scratchpad tool functions (these modify state via the agent node)
# We define them as tools so the LLM can call them, but actual state
# mutation happens in the agent node.


@tool
def note_to_scratchpad(note: str) -> str:
    """Save a note to the scratchpad for later reference."""
    return f"Note saved: {note}"


@tool
def read_scratchpad() -> str:
    """Read all notes from the scratchpad."""
    return "Reading scratchpad..."  # Actual content injected by tool_node


SCRATCHPAD_TOOLS = [web_search, note_to_scratchpad, read_scratchpad]
SCRATCHPAD_TOOLS_BY_NAME = {t.name: t for t in SCRATCHPAD_TOOLS}

scratchpad_llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement the scratchpad agent ─────────────────────


def scratchpad_agent_node(state: ScratchpadState) -> dict:
    """Agent node that uses the scratchpad for working memory.

    TODO(human): Implement this function.

    This agent node is similar to the ReAct agent_node from Phase 1, but
    with an important addition: it injects the current scratchpad contents
    into the system prompt so the LLM can reference its previous notes.

    Steps:
      1. Build the system message. Start with SCRATCHPAD_SYSTEM_PROMPT.
         If state["scratchpad"] is non-empty, append:
         "\\n\\nYour current scratchpad notes:\\n"
         followed by each note formatted as "- <note>".
         This makes the scratchpad visible to the LLM in every iteration.
      2. Bind tools to the LLM: scratchpad_llm.bind_tools(SCRATCHPAD_TOOLS).
      3. Build full messages: [SystemMessage(system_content)] + state["messages"].
      4. Invoke the model.
      5. Return {"messages": [response], "iteration_count": state["iteration_count"] + 1}.

    Why inject scratchpad into system prompt: The scratchpad is a state field,
    not a message. The LLM only sees messages. By injecting scratchpad notes
    into the system prompt, we make working memory visible to the LLM without
    cluttering the conversation history. This is a common pattern for giving
    agents access to structured state beyond the message list.
    """
    raise NotImplementedError("TODO(human): implement scratchpad_agent_node")


def scratchpad_tool_node(state: ScratchpadState) -> dict:
    """Execute tool calls, with special handling for scratchpad tools.

    This node is PROVIDED — study how it handles scratchpad state mutations.
    """
    last_message = state["messages"][-1]
    results = []
    new_scratchpad = list(state["scratchpad"])

    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]

        if name == "note_to_scratchpad":
            note = args.get("note", str(args))
            new_scratchpad.append(note)
            content = f"Note saved to scratchpad: {note}"
        elif name == "read_scratchpad":
            if new_scratchpad:
                content = "Scratchpad contents:\n" + "\n".join(
                    f"  {i + 1}. {note}" for i, note in enumerate(new_scratchpad)
                )
            else:
                content = "Scratchpad is empty."
        else:
            # Regular tool (e.g., web_search)
            tool_fn = SCRATCHPAD_TOOLS_BY_NAME[name]
            content = str(tool_fn.invoke(args))

        results.append(
            ToolMessage(content=content, tool_call_id=tool_call["id"], name=name)
        )

    return {"messages": results, "scratchpad": new_scratchpad}


def scratchpad_should_continue(state: ScratchpadState) -> Literal["tools", "__end__"]:
    """Conditional edge for scratchpad agent."""
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    within_limit = state["iteration_count"] < MAX_ITERATIONS

    if has_tool_calls and within_limit:
        return "tools"
    return "__end__"


def build_scratchpad_graph():
    """Wire the scratchpad agent graph."""
    graph = StateGraph(ScratchpadState)

    graph.add_node("agent", scratchpad_agent_node)
    graph.add_node("tools", scratchpad_tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", scratchpad_should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


# ══════════════════════════════════════════════════════════════════════
# PART 2: Dynamic Tool Creation
# ══════════════════════════════════════════════════════════════════════


class DynamicToolState(TypedDict):
    """State for the dynamic tool agent.

    messages: Conversation history.
    dynamic_tools: Dict mapping tool names to callables. Grows as the agent
                   creates new tools at runtime via exec().
    iteration_count: Loop counter.
    """

    messages: Annotated[list[AnyMessage], operator.add]
    dynamic_tools: dict[str, Callable]
    iteration_count: int


@tool
def create_tool(name: str, code: str) -> str:
    """Create a new Python function tool. Provide the function name and complete code."""
    return f"Tool '{name}' creation requested with code: {code}"


@tool
def run_tool(name: str, args: str) -> str:
    """Run a previously created dynamic tool with the given argument."""
    return f"Running tool '{name}' with args: {args}"


DYNAMIC_TOOLS = [create_tool, run_tool]
DYNAMIC_TOOLS_BY_NAME = {t.name: t for t in DYNAMIC_TOOLS}

dynamic_llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement the dynamic tool agent ───────────────────


def dynamic_tool_node(state: DynamicToolState) -> dict:
    """Execute tool calls, with special handling for create_tool and run_tool.

    TODO(human): Implement this function.

    This node handles two meta-tools: create_tool (defines a new Python
    function at runtime) and run_tool (executes a previously created function).
    This demonstrates how agents can extend their own capabilities dynamically.

    Steps:
      1. Get the last message's tool_calls.
      2. Copy state["dynamic_tools"] into a local dict (don't mutate state directly).
      3. For each tool_call:
         a. If name == "create_tool":
            - Extract the "name" and "code" from tool_call["args"].
            - Use exec() to execute the code string in a local namespace:
              namespace = {}
              exec(code, namespace)
            - The function will be in namespace[name].
            - Store it: dynamic_tools[name] = namespace[name]
            - Create a ToolMessage confirming creation.
         b. If name == "run_tool":
            - Extract "name" and "args" from tool_call["args"].
            - Look up the function in dynamic_tools[name].
            - Evaluate the args string: arg_value = eval(args)
            - Call the function: result = dynamic_tools[name](arg_value)
            - Create a ToolMessage with the result.
         c. Handle errors gracefully — wrap in try/except, return error
            as ToolMessage content rather than crashing.
      4. Return {"messages": results, "dynamic_tools": dynamic_tools}.

    Security note: exec() and eval() are DANGEROUS in production — they can
    execute arbitrary code. This is a learning exercise only. In production,
    use sandboxed execution (Docker, E2B, modal) or a restricted code
    interpreter.

    Why dynamic tools matter: In complex agent systems, the required tools
    aren't always known at build time. An agent that can create its own tools
    adapts to novel problems. This pattern appears in advanced systems like
    Voyager (Minecraft agent) and LATM (Language Agent Tree Machine).
    """
    raise NotImplementedError("TODO(human): implement dynamic_tool_node")


def dynamic_agent_node(state: DynamicToolState) -> dict:
    """Agent node for dynamic tool creation."""
    model_with_tools = dynamic_llm.bind_tools(DYNAMIC_TOOLS)

    # Include list of available dynamic tools in system prompt
    dynamic_names = list(state["dynamic_tools"].keys())
    extra = ""
    if dynamic_names:
        extra = f"\n\nPreviously created tools available via run_tool: {dynamic_names}"

    messages = [
        SystemMessage(content=DYNAMIC_TOOL_SYSTEM_PROMPT + extra),
        *state["messages"],
    ]
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }


def dynamic_should_continue(state: DynamicToolState) -> Literal["tools", "__end__"]:
    """Conditional edge for dynamic tool agent."""
    last_message = state["messages"][-1]
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls
    within_limit = state["iteration_count"] < MAX_ITERATIONS

    if has_tool_calls and within_limit:
        return "tools"
    return "__end__"


def build_dynamic_tool_graph():
    """Wire the dynamic tool agent graph."""
    graph = StateGraph(DynamicToolState)

    graph.add_node("agent", dynamic_agent_node)
    graph.add_node("tools", dynamic_tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", dynamic_should_continue, ["tools", END])
    graph.add_edge("tools", "agent")

    return graph.compile()


# ── Orchestration ────────────────────────────────────────────────────


def run_scratchpad_agent(question: str) -> None:
    """Run the scratchpad agent and print the trace."""
    print(f"\n{'=' * 60}")
    print(f"[Scratchpad Agent] Question: {question}")
    print("=" * 60)

    agent = build_scratchpad_graph()
    initial_state: ScratchpadState = {
        "messages": [HumanMessage(content=question)],
        "scratchpad": [],
        "iteration_count": 0,
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} ---")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  Tool call: {tc['name']}({tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        prefix = msg.__class__.__name__
                        print(f"  [{prefix}] {msg.content[:200]}")
            if "scratchpad" in node_output and node_output["scratchpad"]:
                print(f"  Scratchpad ({len(node_output['scratchpad'])} notes):")
                for note in node_output["scratchpad"][-2:]:
                    print(f"    - {note[:100]}")

    print(f"\n{'=' * 60}\n")


def run_dynamic_tool_agent(question: str) -> None:
    """Run the dynamic tool agent and print the trace."""
    print(f"\n{'=' * 60}")
    print(f"[Dynamic Tool Agent] Question: {question}")
    print("=" * 60)

    agent = build_dynamic_tool_graph()
    initial_state: DynamicToolState = {
        "messages": [HumanMessage(content=question)],
        "dynamic_tools": {},
        "iteration_count": 0,
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} ---")
            if "messages" in node_output:
                for msg in node_output["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            print(f"  Tool call: {tc['name']}({tc['args']})")
                    elif hasattr(msg, "content") and msg.content:
                        prefix = msg.__class__.__name__
                        print(f"  [{prefix}] {msg.content[:200]}")
            if "dynamic_tools" in node_output:
                tools = node_output["dynamic_tools"]
                if tools:
                    print(f"  Dynamic tools available: {list(tools.keys())}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    print("Practice 031a — Phase 5: Memory & Dynamic Tools")
    print("Working memory (scratchpad) and runtime tool creation\n")

    # Part 1: Scratchpad memory
    run_scratchpad_agent(
        "Research Python type hints: what are Protocols, how do they differ from ABCs, "
        "and how do dataclasses work with type hints? Save each finding as a note, "
        "then synthesize a summary."
    )

    # Part 2: Dynamic tool creation
    run_dynamic_tool_agent(
        "I need to compute the 10th Fibonacci number. Create a Python function "
        "that calculates Fibonacci numbers, then use it to compute fib(10)."
    )


if __name__ == "__main__":
    main()
