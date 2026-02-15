"""
Practice 029b — Phase 5: Multi-Agent Supervisor
Build a supervisor that coordinates specialized agent nodes.

The supervisor pattern is the standard architecture for multi-agent systems:
  1. A central "supervisor" node examines the query
  2. It decides which specialist agent to invoke
  3. A conditional edge routes to the chosen specialist
  4. The specialist processes the query and returns a result
  5. The result flows back to the supervisor for final synthesis

This is the pattern used by LangGraph's own `langgraph-supervisor` package,
and by most production multi-agent deployments (customer support routing,
research assistants, coding agents with specialized tools, etc.).
"""

from typing import Literal

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ---------------------------------------------------------------------------
# Specialist system prompts (provided — these define each agent's personality)
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = (
    "You are a thorough research assistant. When given a question, provide "
    "well-structured, factual information with specific details. Cite concepts "
    "and explain mechanisms. Be comprehensive but concise."
)

CALCULATOR_PROMPT = (
    "You are a precise math and logic assistant. Show your work step by step. "
    "For calculations, write each step clearly. For logic problems, explain "
    "your reasoning. Always verify your answer at the end."
)

WRITER_PROMPT = (
    "You are a creative writer and communications specialist. Produce engaging, "
    "well-crafted text. Use vivid language, clear structure, and appropriate tone. "
    "Adapt your style to the request (formal, casual, persuasive, etc.)."
)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class SupervisorState(TypedDict):
    """State for the supervisor multi-agent graph.

    - query: the user's original question
    - selected_agent: which specialist the supervisor chose
    - agent_output: the specialist's response
    - final_answer: the supervisor's synthesized final answer
    """

    query: str
    selected_agent: str
    agent_output: str
    final_answer: str


# ===================================================================
# EXERCISE 1: Supervisor Graph
# ===================================================================

# ---------------------------------------------------------------------------
# TODO(human) #1: Build the supervisor graph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   A supervisor node, three specialist nodes, a routing function, and a
#   synthesis node. The supervisor decides which specialist to call, the
#   conditional edge routes to it, and a final node synthesizes the answer.
#
# Node 1 — supervisor(state: SupervisorState) -> dict:
#   - Read state["query"]
#   - Ask the LLM to classify which specialist should handle this:
#     Prompt: "You are a supervisor coordinating a team of specialists.
#              Given the user's query, decide which specialist to assign:
#              - researcher: for factual questions, explanations, 'how does X work'
#              - calculator: for math problems, calculations, logic puzzles
#              - writer: for creative writing, drafting emails, storytelling
#
#              Reply with ONLY one word: researcher, calculator, or writer.
#
#              Query: {query}"
#   - Parse the response: lowercase, strip whitespace
#     Default to "researcher" if the response is unrecognized
#   - Return {"selected_agent": parsed_agent}
#
# Node 2 — researcher_agent(state: SupervisorState) -> dict:
#   - Use llm.invoke() with RESEARCHER_PROMPT as system context:
#     Prompt: f"{RESEARCHER_PROMPT}\n\nQuestion: {state['query']}"
#   - Return {"agent_output": response.content}
#
# Node 3 — calculator_agent(state: SupervisorState) -> dict:
#   - Use llm.invoke() with CALCULATOR_PROMPT:
#     Prompt: f"{CALCULATOR_PROMPT}\n\nProblem: {state['query']}"
#   - Return {"agent_output": response.content}
#
# Node 4 — writer_agent(state: SupervisorState) -> dict:
#   - Use llm.invoke() with WRITER_PROMPT:
#     Prompt: f"{WRITER_PROMPT}\n\nRequest: {state['query']}"
#   - Return {"agent_output": response.content}
#
# Node 5 — synthesize(state: SupervisorState) -> dict:
#   - Combine the supervisor's decision with the specialist's output:
#     Prompt: "You are a supervisor reviewing a specialist's work.
#              The user asked: {query}
#              The {selected_agent} specialist responded:
#              {agent_output}
#
#              Provide a polished final answer. If the specialist's response
#              is good, present it cleanly. If it needs improvement, enhance it."
#   - Return {"final_answer": response.content}
#
# Routing function — route_to_specialist(state: SupervisorState) -> str:
#   - Read state["selected_agent"]
#   - Return the corresponding node name:
#     "researcher" -> "researcher_agent"
#     "calculator" -> "calculator_agent"
#     "writer" -> "writer_agent"
#   - Type hint: Literal["researcher_agent", "calculator_agent", "writer_agent"]
#
# Graph wiring:
#   builder = StateGraph(SupervisorState)
#   Add nodes: "supervisor", "researcher_agent", "calculator_agent",
#              "writer_agent", "synthesize"
#   Edges:
#     START -> "supervisor"
#     "supervisor" -> conditional: route_to_specialist
#     "researcher_agent" -> "synthesize"
#     "calculator_agent" -> "synthesize"
#     "writer_agent" -> "synthesize"
#     "synthesize" -> END
#   Compile.
#
# WHY THIS MATTERS:
#   The supervisor pattern is the most common multi-agent architecture
#   in production. It decouples:
#   - ROUTING (which agent handles this?) from
#   - EXECUTION (how does the agent handle it?) from
#   - SYNTHESIS (how do we present the result?)
#
#   Each specialist can be independently improved, tested, or replaced.
#   The supervisor itself can be a simple classifier (as here) or a
#   complex planning agent that breaks tasks into multi-step workflows.
#
#   In production, specialists often have access to different tools
#   (researcher has web search, calculator has code execution, etc.).
#
# EXPECTED BEHAVIOR:
#   "How does photosynthesis work?" -> supervisor selects researcher
#   "What is 347 * 29?" -> supervisor selects calculator
#   "Write a thank-you email" -> supervisor selects writer
#   Each specialist responds, then synthesize polishes the final answer.


def supervisor(state: SupervisorState) -> dict:
    return {
        "selected_agent": llm.invoke(
            "You are a supervisor coordinating a team of specialists.\n"
            "Given the user's query, decide which specialist to assign:\n"
            "- researcher: for factual questions, explanations, 'how does X work\n"
            "- calculator: for math problems, calculations, logic puzzles\n"
            "- writer: for creative writing, drafting emails, storytelling\n\n"
            "Reply with ONLY one word: researcher, calculator, or writer.\n\n"
            f"Query: {state['query']}"
        )
        .content.strip()
        .lower()
    }


def get_worker_by_prompt(prompt: str):
    def worker(state: SupervisorState) -> dict:
        return {
            "agent_output": llm.invoke(prompt + f"Query : {state['query']}").content
        }

    return worker


def synthesize(state: SupervisorState) -> dict:
    return {
        "final_answer": llm.invoke(
            [
                f"""You are a supervisor reviewing a specialist's work.
                      The user asked: {state["query"]}
                      The {state["selected_agent"]} specialist responded:
                      {state["agent_output"]}

                      Provide a polished final answer. If the specialist's response
                      is good, present it cleanly. If it needs improvement, enhance it."""
            ]
        ).content
    }


def router_fn(
    state: SupervisorState,
) -> Literal["researcher_agent", "calculator_agent", "writer_agent"]:
    match state["selected_agent"]:
        case "researcher":
            return "researcher_agent"
        case "calculator":
            return "calculator_agent"
        case "writer":
            return "writer_agent"
        case _:
            return "researcher_agent"


def build_supervisor_graph():
    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher_agent", get_worker_by_prompt(RESEARCHER_PROMPT))
    builder.add_node("calculator_agent", get_worker_by_prompt(CALCULATOR_PROMPT))
    builder.add_node("writer_agent", get_worker_by_prompt(WRITER_PROMPT))
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", router_fn)
    builder.add_edge("researcher_agent", "synthesize")
    builder.add_edge("calculator_agent", "synthesize")
    builder.add_edge("writer_agent", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# ===================================================================
# EXERCISE 2: Subgraph Extraction
# ===================================================================

# ---------------------------------------------------------------------------
# TODO(human) #2: Extract researcher as a subgraph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   1. A self-contained subgraph for the researcher agent
#   2. Integration of that subgraph into the main supervisor graph
#
# STEP 1 — Define the researcher subgraph:
#
#   The researcher subgraph has its OWN state and internal nodes:
#
#   class ResearcherSubState(TypedDict):
#       query: str          # input: the research question
#       plan: str           # intermediate: research plan
#       raw_research: str   # intermediate: detailed research
#       summary: str        # output: concise summary
#
#   Internal nodes:
#     - plan_research(state) -> {"plan": ...}
#       Prompt: "Create a brief research plan (3 bullet points) for: {query}"
#
#     - execute_research(state) -> {"raw_research": ...}
#       Prompt: "Following this research plan:\n{plan}\n\nProvide detailed
#               findings for: {query}"
#
#     - summarize_research(state) -> {"summary": ...}
#       Prompt: "Summarize these research findings in 3-5 concise sentences:
#               \n\n{raw_research}"
#
#   Subgraph wiring:
#     START -> "plan_research" -> "execute_research" -> "summarize_research" -> END
#
#   Compile the subgraph:
#     researcher_subgraph = sub_builder.compile()
#
# STEP 2 — Integrate into the supervisor:
#
#   Since the subgraph state (ResearcherSubState) differs from the
#   supervisor state (SupervisorState), you need a WRAPPER FUNCTION
#   that transforms state in and out:
#
#   def researcher_agent_with_subgraph(state: SupervisorState) -> dict:
#       # Transform supervisor state -> subgraph state
#       sub_result = researcher_subgraph.invoke({
#           "query": state["query"],
#           "plan": "",
#           "raw_research": "",
#           "summary": "",
#       })
#       # Transform subgraph result -> supervisor state update
#       return {"agent_output": sub_result["summary"]}
#
#   In the supervisor graph, replace the "researcher_agent" node:
#     builder.add_node("researcher_agent", researcher_agent_with_subgraph)
#   Everything else stays the same — the supervisor doesn't know or care
#   that the researcher is now a multi-step subgraph internally.
#
# WHY THIS MATTERS:
#   Subgraphs are the composition primitive for complex agent systems.
#   Benefits:
#   - MODULARITY: Each agent's internal logic is self-contained
#   - TESTABILITY: You can test the researcher subgraph in isolation
#   - REUSABILITY: The same subgraph can be used in multiple parent graphs
#   - ENCAPSULATION: Parent graph only sees input/output, not internal steps
#   - INDEPENDENT EVOLUTION: Change the researcher's internal pipeline
#     without touching the supervisor
#
#   In production, each agent is typically its own subgraph with:
#   - Internal planning nodes
#   - Tool-calling loops
#   - Error recovery logic
#   - Its own checkpointing
#
#   The parent graph composes these subgraphs without knowing their internals.
#   This is the same principle as microservices — each agent has its own
#   internal architecture behind a clean interface.
#
# EXPECTED BEHAVIOR:
#   Same external behavior as Exercise 1, but the researcher agent now
#   runs through 3 internal steps (plan -> execute -> summarize) instead
#   of a single LLM call. The output should be more structured and thorough.


class ResearcherSubState(TypedDict):
    query: str
    plan: str
    raw_research: str
    summary: str


def build_researcher_subgraph():
    def plan_research(state: ResearcherSubState) -> dict:
        return {
            "plan": llm.invoke(
                [
                    f"Create a brief research plan (at most 3 bullet points) for : {state['query']}"
                ]
            ).content
        }

    def execute_research(state: ResearcherSubState) -> dict:
        return {
            "raw_research": llm.invoke(
                [
                    f"Following this plan:\n{state['plan']}\n. Provide an answer for {state['query']}"
                ]
            ).content
        }

    def summarize_research(state: ResearcherSubState) -> dict:
        return {
            "summary": llm.invoke(
                [
                    f"Summarize these research finding in 3-5 concise sentences answering : {state['query']}:\n{state['raw_research']}"
                ]
            ).content
        }

    builder = StateGraph(ResearcherSubState)
    builder.add_node("plan", plan_research)
    builder.add_node("execute", execute_research)
    builder.add_node("summarize", summarize_research)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


def build_supervisor_with_subgraph():
    """Build the supervisor graph using the researcher subgraph."""
    researcher_subgraph = build_researcher_subgraph()

    def researcher_wrapper(state: SupervisorState) -> dict:
        return {
            "agent_output": researcher_subgraph.invoke(
                {"query": state["query"], "plan": "", "raw_research": "", "summary": ""}
            )["summary"]
        }

    builder = StateGraph(SupervisorState)

    builder.add_node("supervisor", supervisor)
    builder.add_node("researcher_agent", researcher_wrapper)
    builder.add_node("calculator_agent", get_worker_by_prompt(CALCULATOR_PROMPT))
    builder.add_node("writer_agent", get_worker_by_prompt(WRITER_PROMPT))
    builder.add_node("synthesize", synthesize)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", router_fn)
    builder.add_edge("researcher_agent", "synthesize")
    builder.add_edge("calculator_agent", "synthesize")
    builder.add_edge("writer_agent", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Exercise 1: Supervisor Graph ---
    print("=" * 60)
    print("EXERCISE 1: Multi-Agent Supervisor")
    print("=" * 60)

    supervisor_graph = build_supervisor_graph()

    test_queries = [
        "How does the TCP three-way handshake work?",
        "Calculate 17^3 + 42 * 15 - 289",
        "Write a professional email declining a meeting invitation",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        result = supervisor_graph.invoke(
            {
                "query": query,
                "selected_agent": "",
                "agent_output": "",
                "final_answer": "",
            }
        )
        print(f"  Routed to: {result['selected_agent']}")
        print(f"  Final answer: {result['final_answer'][:300]}...")
        print()

    # --- Exercise 2: Supervisor with Subgraph ---
    print("=" * 60)
    print("EXERCISE 2: Supervisor with Researcher Subgraph")
    print("=" * 60)

    # First test the subgraph in isolation
    print("\n--- Testing researcher subgraph in isolation ---\n")
    researcher_sub = build_researcher_subgraph()
    sub_result = researcher_sub.invoke(
        {
            "query": "How do distributed consensus algorithms work?",
            "plan": "",
            "raw_research": "",
            "summary": "",
        }
    )
    print(f"Plan:\n{sub_result['plan']}\n")
    print(f"Summary:\n{sub_result['summary']}\n")

    # Now test integrated in the supervisor
    print("--- Testing integrated supervisor with subgraph ---\n")
    supervisor_v2 = build_supervisor_with_subgraph()

    result = supervisor_v2.invoke(
        {
            "query": "Explain how garbage collection works in modern programming languages",
            "selected_agent": "",
            "agent_output": "",
            "final_answer": "",
        }
    )
    print(f"Routed to: {result['selected_agent']}")
    print(f"Final answer:\n{result['final_answer']}")

    print("\nPhase 5 complete.")
