"""
Practice 029b — Phase 3: Checkpointing & Human-in-the-Loop
Add persistence to graphs and implement approval gates.

Checkpointers are LangGraph's persistence layer. They save the full graph
state after every superstep, enabling:
  - Multi-turn conversations that remember history across invocations
  - Fault-tolerant execution that can resume after crashes
  - Human-in-the-loop workflows where execution pauses for approval
  - Time-travel debugging (inspect state at any past superstep)

The key abstraction is the "thread": identified by a thread_id in the
config dict, each thread maintains independent state. Same graph, different
threads = isolated conversations.
"""

import operator
from typing import Annotated

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:3b"


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ===================================================================
# EXERCISE 1: Conversational Memory with Checkpointing
# ===================================================================


class ConversationState(TypedDict):
    """State for a multi-turn conversation.

    Uses Annotated[list, operator.add] as a REDUCER: when a node returns
    {"messages": [new_msg]}, the reducer APPENDS to the existing list
    instead of replacing it. This is essential for maintaining conversation
    history — without the reducer, each node's return would overwrite all
    previous messages.
    """

    messages: Annotated[list[BaseMessage], operator.add]


# ---------------------------------------------------------------------------
# TODO(human) #1: Build a conversational graph with checkpointer
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   A single-node conversational graph with InMemorySaver checkpointer.
#
# Node — chatbot(state: ConversationState) -> dict:
#   - Pass state["messages"] directly to llm.invoke()
#     LangChain chat models accept a list of BaseMessage objects.
#     The model sees the full conversation history and responds.
#   - The response is an AIMessage — wrap it in a list for the reducer:
#     return {"messages": [response]}
#   - The operator.add reducer will APPEND this to the existing messages list
#
# Graph wiring:
#   builder = StateGraph(ConversationState)
#   Add node: "chatbot"
#   Edges: START -> "chatbot" -> END
#   Compile WITH checkpointer:
#     memory = InMemorySaver()
#     graph = builder.compile(checkpointer=memory)
#
# WHY THIS MATTERS:
#   Without a checkpointer, each invoke() starts from scratch — the model
#   has no memory of previous interactions. With a checkpointer:
#   - invoke(input, config={"configurable": {"thread_id": "user1"}})
#     saves state after execution
#   - A second invoke with the SAME thread_id restores the saved state,
#     so the model sees the full conversation history
#   - A different thread_id ("user2") has completely independent state
#
#   This is how production chatbots maintain per-user conversations.
#   InMemorySaver is for development; in production you'd use
#   SqliteSaver or PostgresSaver for durable persistence.
#
# EXPECTED BEHAVIOR:
#   Turn 1 (thread_id="user1"): "My name is Alice" -> model acknowledges
#   Turn 2 (thread_id="user1"): "What is my name?" -> model says "Alice"
#   Turn 1 (thread_id="user2"): "What is my name?" -> model doesn't know
#
# HINT:
#   The config dict for invoke looks like:
#     config = {"configurable": {"thread_id": "user1"}}
#   Pass it as: graph.invoke({"messages": [HumanMessage(content="...")]}, config)


def build_conversational_graph():
    def conversation_step(state: ConversationState) -> dict:
        return {"messages": [llm.invoke(state["messages"])]}

    builder = StateGraph(ConversationState)
    memory = InMemorySaver()
    builder.add_node("chatbot", conversation_step)
    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)
    return builder.compile(checkpointer=memory)


# ===================================================================
# EXERCISE 2: Human-in-the-Loop Approval Gate
# ===================================================================


class ApprovalState(TypedDict):
    """State for the HITL approval workflow.

    - request: what the user asked the agent to do
    - proposed_action: the agent's planned action (before approval)
    - approval: the human's decision ("approved" or "rejected")
    - result: final outcome message
    """

    request: str
    proposed_action: str
    approval: str
    result: str


# ---------------------------------------------------------------------------
# TODO(human) #2: Build the HITL approval gate graph
# ---------------------------------------------------------------------------
#
# WHAT TO IMPLEMENT:
#   Three nodes: propose, approve (with interrupt), and execute/abort.
#
# Node 1 — propose_action(state: ApprovalState) -> dict:
#   - Read state["request"]
#   - Use the LLM to determine what action to take:
#     Prompt: "The user wants: {request}
#              Propose a specific action to fulfill this request.
#              Be concrete (e.g., 'Send email to john@example.com with subject...')
#              Reply with ONLY the proposed action, nothing else."
#   - Return {"proposed_action": response.content}
#
# Node 2 — human_approval(state: ApprovalState) -> dict:
#   - Print the proposed action so the human can see it
#   - Call interrupt() to pause execution:
#       approval = interrupt(
#           f"Agent proposes: {state['proposed_action']}\n"
#           f"Reply 'approved' or 'rejected'."
#       )
#   - interrupt() PAUSES the graph here. Execution stops.
#     The checkpointer saves the current state.
#     When the user resumes with Command(resume="approved"),
#     interrupt() RETURNS the resume value ("approved").
#   - Return {"approval": approval}
#
# Node 3 — execute_or_abort(state: ApprovalState) -> dict:
#   - If state["approval"] == "approved":
#       return {"result": f"Executed: {state['proposed_action']}"}
#   - Else:
#       return {"result": f"Aborted: user rejected the proposed action."}
#
# Graph wiring:
#   builder = StateGraph(ApprovalState)
#   Add nodes: "propose", "approve", "execute"
#   Edges: START -> "propose" -> "approve" -> "execute" -> END
#   Compile with InMemorySaver checkpointer (REQUIRED for interrupt)
#
# WHY THIS MATTERS:
#   interrupt() is LangGraph's mechanism for human-in-the-loop workflows.
#   In production, this is used for:
#   - Approving financial transactions before execution
#   - Reviewing agent-generated emails before sending
#   - Confirming destructive operations (delete, deploy, etc.)
#   - Any high-stakes action that needs human oversight
#
#   The flow is:
#   1. First invoke() runs propose -> approve (hits interrupt, pauses)
#   2. Application shows proposed action to human
#   3. Human decides (approve/reject)
#   4. Second invoke() with Command(resume="approved") resumes from interrupt
#   5. execute_or_abort runs with the human's decision
#
# EXPECTED BEHAVIOR:
#   First invoke: graph runs propose_action, then hits interrupt in approve.
#     Returns with the graph in "interrupted" state.
#   Resume with Command(resume="approved"): approve node completes,
#     execute_or_abort sees approval="approved", returns success.
#   Resume with Command(resume="rejected"): execute_or_abort aborts.
#
# HINT:
#   To resume an interrupted graph:
#     graph.invoke(Command(resume="approved"), config)
#   The config MUST use the same thread_id as the original invoke.


def build_approval_graph():
    def propose_action(state: ApprovalState) -> dict:
        return {
            "proposed_action": llm.invoke(
                [
                    f"""The user wants {state["request"]}
                    Propose a specific action to fulfill this request.
                    Be concrete (e.g., 'Send email to X@Y')
                    Reply with ONLY the proposed action, nothing else
                """
                ]
            ).content
        }

    def human_approval(state: ApprovalState) -> dict:
        return {
            "approval": interrupt(
                f"Agent proposes: {state['proposed_action']}\nReply 'approved' or 'rejected'."
            )
        }

    def execute_or_abort(state: ApprovalState) -> dict:
        if state["approval"] == "approved":
            return {"result": f"Executed: {state['proposed_action']}"}
        else:
            return {"result": "Aborted: user rejected the proposed action."}

    builder = StateGraph(ApprovalState)
    memory = InMemorySaver()

    builder.add_node("propose_action", propose_action)
    builder.add_node("human_approval", human_approval)
    builder.add_node("execute_or_abort", execute_or_abort)

    builder.add_edge(START, "propose_action")
    builder.add_edge("propose_action", "human_approval")
    builder.add_edge("human_approval", "execute_or_abort")
    builder.add_edge("execute_or_abort", END)

    return builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Exercise 1: Conversational Memory ---
    print("=" * 60)
    print("EXERCISE 1: Conversational Memory with Checkpointing")
    print("=" * 60)

    conv_graph = build_conversational_graph()

    config_user1 = {"configurable": {"thread_id": "user1"}}
    config_user2 = {"configurable": {"thread_id": "user2"}}

    # Turn 1: Tell the model a name (user1)
    print("\n[user1] Turn 1: 'My name is Alice and I work at AutoScheduler.AI'")
    result1 = conv_graph.invoke(
        {
            "messages": [
                HumanMessage(content="My name is Alice and I work at AutoScheduler.AI")
            ]
        },
        config_user1,
    )
    print(f"Assistant: {result1['messages'][-1].content}\n")

    # Turn 2: Ask if it remembers (user1, same thread)
    print("[user1] Turn 2: 'What is my name and where do I work?'")
    result2 = conv_graph.invoke(
        {"messages": [HumanMessage(content="What is my name and where do I work?")]},
        config_user1,
    )
    print(f"Assistant: {result2['messages'][-1].content}\n")

    # Turn 1: Different thread — should NOT remember
    print("[user2] Turn 1: 'What is my name?'")
    result3 = conv_graph.invoke(
        {"messages": [HumanMessage(content="What is my name?")]},
        config_user2,
    )
    print(f"Assistant: {result3['messages'][-1].content}\n")

    # --- Exercise 2: HITL Approval Gate ---
    print("=" * 60)
    print("EXERCISE 2: Human-in-the-Loop Approval Gate")
    print("=" * 60)

    approval_graph = build_approval_graph()
    config_hitl = {"configurable": {"thread_id": "hitl-demo"}}

    # Step 1: Start the workflow — will pause at interrupt()
    print("\nStarting approval workflow...")
    result = approval_graph.invoke(
        {
            "request": "Send a meeting invitation to the team for Friday at 3pm",
            "proposed_action": "",
            "approval": "",
            "result": "",
        },
        config_hitl,
    )
    print(f"Proposed action: {result.get('proposed_action', '(interrupted)')}")
    print("Graph is now paused, waiting for human approval.\n")

    # Step 2: Resume with approval
    print("Resuming with approval='approved'...")
    result = approval_graph.invoke(
        Command(resume="approved"),
        config_hitl,
    )
    print(f"Result: {result['result']}\n")

    print("Phase 3 complete.")
