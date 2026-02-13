"""ReWOO Agent — Reason Without Observation.

Pattern: Worker plans with placeholders → Executor runs all tools → Solver integrates

Unlike ReAct (which interleaves LLM calls with tool execution), ReWOO separates
planning from execution entirely. The Worker creates a plan referencing future
tool results as #E1, #E2, etc. The Executor runs ALL tools in batch. The Solver
combines actual results into a final answer. This uses only 2 LLM calls total
(Worker + Solver) regardless of how many tools are needed.

Paper: "ReWOO: Decoupling Reasoning from Observations for Efficient Augmented
Language Models" (Xu et al., 2023) — https://arxiv.org/abs/2305.18323

Run:
    uv run python src/04_rewoo.py
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

WORKER_PROMPT = """You are a planning assistant. Given a question, create a plan that uses
tools to gather information. Reference tool results using placeholders #E1, #E2, etc.

Available tools:
- search(query): Search the web for information
- calculator(expression): Evaluate a math expression
- lookup(topic): Look up detailed information about a topic

Output your plan in this EXACT format (one line per step):
Plan: <description of what this step does>
#E1 = search("query here")
Plan: <description>
#E2 = calculator("expression here, can reference #E1")
Plan: <description>
#E3 = lookup("topic, can reference previous results")

Rules:
- Each #E<n> must be on its own line with the tool call
- Each Plan: line describes what the following tool call does
- Later steps can reference earlier results: #E2 = calculator("#E1 + 10")
- Keep plans short (2-5 steps)"""

SOLVER_PROMPT = """You are a solver assistant. You are given:
1. A question
2. A plan with tool calls and their actual results

Synthesize the results into a clear, concise final answer to the question.
Only use information from the provided evidence. Be factual and direct."""


# ── Mock tools ───────────────────────────────────────────────────────


def mock_search(query: str) -> str:
    """Mock web search."""
    mock_db = {
        "python creator": "Python was created by Guido van Rossum and first released in 1991.",
        "population france": "France has approximately 68 million people (2024).",
        "population japan": "Japan has approximately 125 million people (2024).",
        "capital france": "The capital of France is Paris.",
        "langgraph": "LangGraph is a framework by LangChain for building stateful agents using graph-based orchestration.",
        "rewoo": "ReWOO is a technique that decouples reasoning from observations, reducing LLM calls in agent workflows.",
    }
    query_lower = query.lower()
    for key, value in mock_db.items():
        if key in query_lower:
            return value
    return f"Search result for '{query}': No specific information found."


def mock_calculator(expression: str) -> str:
    """Mock calculator — evaluates math expressions."""
    try:
        # Replace placeholder references with their values (handled by executor)
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


def mock_lookup(topic: str) -> str:
    """Mock lookup for detailed information."""
    mock_details = {
        "python": "Python supports multiple paradigms: procedural, OOP, and functional. Key features: dynamic typing, garbage collection, extensive standard library.",
        "langgraph": "LangGraph uses StateGraph with TypedDict for state, supports cycles (loops), conditional edges, and streaming. Built on top of langchain-core.",
        "react": "ReAct (Reasoning + Acting) interleaves LLM reasoning with tool execution in a loop. Each iteration: Thought → Action → Observation.",
    }
    topic_lower = topic.lower()
    for key, value in mock_details.items():
        if key in topic_lower:
            return value
    return f"Detailed info for '{topic}': No specific details available."


TOOL_REGISTRY = {
    "search": mock_search,
    "calculator": mock_calculator,
    "lookup": mock_lookup,
}


# ── State ────────────────────────────────────────────────────────────


class ReWOOState(TypedDict):
    """State for the ReWOO agent.

    question: The original user question.
    plan: Raw plan text from the Worker LLM (includes Plan: lines and #E<n> lines).
    evidence: Dict mapping placeholder names (#E1, #E2...) to actual tool results.
              Filled by the executor after running all tools.
    final_answer: The Solver LLM's synthesized answer.
    llm_call_count: Tracks total LLM invocations. Should be exactly 2 for ReWOO
                    (Worker + Solver) — compare with ReAct which uses N+1 calls.
    """

    question: str
    plan: str
    evidence: dict[str, str]
    final_answer: str
    llm_call_count: int


# ── LLM ──────────────────────────────────────────────────────────────

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0)


# ── TODO(human): Implement these three functions ─────────────────────


def worker_node(state: ReWOOState) -> dict:
    """Create a plan with #E<n> placeholders for tool results.

    TODO(human): Implement this function.

    The Worker is the first LLM call. It analyzes the question and creates
    a structured plan where each tool call is represented as a placeholder.
    The Worker NEVER sees actual tool results — it reasons about what
    information will be needed and plans the tool calls in advance.

    Steps:
      1. Build messages: SystemMessage with WORKER_PROMPT, then
         HumanMessage with state["question"].
      2. Invoke the LLM.
      3. Increment llm_call_count by 1.
      4. Return {"plan": response.content, "llm_call_count": new_count}.

    The plan output will look like:
      Plan: Search for the population of France
      #E1 = search("population of France")
      Plan: Search for the population of Japan
      #E2 = search("population of Japan")
      Plan: Calculate the difference
      #E3 = calculator("#E1_value - #E2_value")

    Key insight: The Worker makes ONE LLM call to plan ALL tool uses. In
    ReAct, the agent would make one LLM call per tool use (N calls for N
    tools). This is the core efficiency gain of ReWOO — the trade-off is
    that the plan can't adapt based on intermediate results.
    """
    raise NotImplementedError("TODO(human): implement worker_node")


def executor_node(state: ReWOOState) -> dict:
    """Parse the plan, execute all tool calls, and store results in evidence.

    TODO(human): Implement this function.

    The Executor is a PURE CODE node — no LLM calls. It parses the plan
    text to extract tool calls, executes them using the TOOL_REGISTRY,
    and stores results in the evidence dict. Before executing each tool,
    it substitutes any referenced placeholders (#E1, #E2) with their
    actual values from previous steps.

    Steps:
      1. Initialize an empty evidence dict: {}.
      2. Use regex to find all tool call lines in state["plan"]:
         Pattern: r"#(E\\d+)\\s*=\\s*(\\w+)\\((.*)\\)"
         This captures: (placeholder_name, tool_name, arguments_string)
         Example match: "#E1 = search(\"population\")" → ("E1", "search", '"population"')
      3. For each match:
         a. Get the placeholder name (e.g., "E1"), tool name, and raw args.
         b. Clean the args: strip quotes, whitespace.
         c. Substitute any placeholder references in the args:
            For each previously-filled evidence key, replace "#E<n>" in the
            args with the actual result. E.g., if evidence["E1"] = "68 million",
            replace "#E1" in args with "68 million".
         d. Look up the tool function in TOOL_REGISTRY[tool_name].
         e. Call the tool with the processed args.
         f. Store: evidence[placeholder_name] = result.
      4. Return {"evidence": evidence}.

    Why no LLM calls: This is the key difference from ReAct. Tool execution
    is mechanical — parse, substitute, call. No reasoning needed. This keeps
    the total LLM call count at exactly 2 (Worker + Solver).

    Regex hint:
      pattern = r"#(E\\d+)\\s*=\\s*(\\w+)\\((.*)\\)"
      matches = re.findall(pattern, state["plan"])
      # matches = [("E1", "search", '"population of France"'), ...]
    """
    raise NotImplementedError("TODO(human): implement executor_node")


def solver_node(state: ReWOOState) -> dict:
    """Integrate evidence into a final answer.

    TODO(human): Implement this function.

    The Solver is the second (and last) LLM call. It receives the original
    question, the plan, and all evidence (actual tool results), then
    synthesizes everything into a coherent final answer.

    Steps:
      1. Build an evidence summary string. For each key in state["evidence"]:
         "#E1 result: <actual value>"
         "#E2 result: <actual value>"
      2. Build the user message:
         "Question: {state['question']}\n\n"
         "Plan:\n{state['plan']}\n\n"
         "Evidence:\n{evidence_summary}\n\n"
         "Provide a clear, concise answer based on the evidence above."
      3. Build messages: SystemMessage with SOLVER_PROMPT + HumanMessage.
      4. Invoke the LLM.
      5. Increment llm_call_count by 1.
      6. Return {"final_answer": response.content, "llm_call_count": new_count}.

    Comparison with ReAct: For a 3-tool question, ReAct uses ~4 LLM calls
    (initial + 3 tool decisions). ReWOO uses exactly 2 (Worker + Solver).
    The Solver sees ALL evidence at once, which often produces better
    synthesis than ReAct's incremental reasoning.
    """
    raise NotImplementedError("TODO(human): implement solver_node")


# ── Graph wiring (provided) ─────────────────────────────────────────


def build_rewoo_graph():
    """Wire the ReWOO graph: Worker → Executor → Solver → END."""
    graph = StateGraph(ReWOOState)

    graph.add_node("worker", worker_node)
    graph.add_node("executor", executor_node)
    graph.add_node("solver", solver_node)

    # ReWOO is a LINEAR pipeline — no cycles, no conditional edges.
    # This is its defining characteristic vs ReAct's loop.
    graph.add_edge(START, "worker")
    graph.add_edge("worker", "executor")
    graph.add_edge("executor", "solver")
    graph.add_edge("solver", END)

    return graph.compile()


# ── Orchestration ────────────────────────────────────────────────────


def run_rewoo(question: str) -> None:
    """Run the ReWOO agent and print the trace."""
    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    agent = build_rewoo_graph()
    initial_state: ReWOOState = {
        "question": question,
        "plan": "",
        "evidence": {},
        "final_answer": "",
        "llm_call_count": 0,
    }

    for event in agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"\n--- {node_name} ---")
            if node_name == "worker" and "plan" in node_output:
                print("  Plan:")
                for line in node_output["plan"].strip().split("\n"):
                    print(f"    {line}")
            elif node_name == "executor" and "evidence" in node_output:
                print("  Evidence collected:")
                for key, value in node_output["evidence"].items():
                    print(f"    #{key}: {value[:100]}")
            elif node_name == "solver":
                print(f"  Final answer: {node_output.get('final_answer', '?')[:300]}")
                print(f"  Total LLM calls: {node_output.get('llm_call_count', '?')}")

    print(f"\n{'=' * 60}\n")


def main() -> None:
    print("Practice 031a — Phase 4: ReWOO Agent")
    print("Reason Without Observation — batch tool execution\n")

    # Test 1: Multi-tool question
    run_rewoo("What is the population of France and Japan? Which country has more people?")

    # Test 2: Research + synthesis
    run_rewoo("What is LangGraph and how does it relate to the ReAct pattern?")


if __name__ == "__main__":
    main()
