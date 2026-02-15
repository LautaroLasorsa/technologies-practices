# Practice 029b — LangGraph: Stateful Agent Graphs

## Technologies

- **LangGraph** — Graph-based framework for stateful, multi-actor agent orchestration
- **StateGraph** — Core abstraction for defining agent workflows as directed graphs
- **Checkpointers** — Persistence layer for graph state (InMemorySaver, SqliteSaver)
- **LangChain** — Underlying primitives (ChatModels, Tools, Messages)

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### What LangGraph Is & The Problem It Solves

LangGraph models LLM workflows as **stateful directed graphs** — nodes are computation steps (LLM calls, tool executions, arbitrary logic), and edges define transitions between them (including conditional routing). Unlike basic LangChain chains, which are linear pipelines (prompt -> model -> output parser), LangGraph supports **cycles** (iterative refinement loops), **branching** (conditional routing to different nodes), **parallel execution** (fan-out to multiple nodes simultaneously), and **persistent state** across multiple invocations.

The core problem: real-world agent workflows are not linear. An agent might need to:
- Loop back to retry after a failed tool call
- Route to different specialists based on query type
- Fan out to multiple workers in parallel, then aggregate results
- Pause for human approval before executing a dangerous action
- Remember conversation history across multiple interactions

LangChain's basic agent executor (`AgentExecutor`) handles simple ReAct loops (think -> act -> observe -> repeat), but falls apart when you need fine-grained control over the execution flow. LangGraph replaces it with a declarative graph where every transition is explicit and inspectable.

LangGraph is the production standard for agent development in 2025-2026. LangChain itself now builds its recommended agent patterns **on top of** LangGraph for durable, controllable execution.

### Core Architecture

LangGraph's architecture has four pillars:

**State** — A shared data structure (TypedDict, Pydantic model, or dataclass) that flows through the graph. Every node reads from state and returns partial updates. The key innovation is **reducers**: annotated fields that control *how* updates are merged. For example, `Annotated[list, operator.add]` means "append to the list" instead of "replace it." This is what makes parallel execution work — multiple worker nodes can each return `{"results": [my_result]}` and the reducer concatenates them all.

**Nodes** — Python functions that receive the current state and return a dict of state updates. Nodes can be sync or async. Two special nodes exist: `START` (the graph entry point) and `END` (terminal node that stops execution). Any Python function with signature `(state: State) -> dict` can be a node.

**Edges** — Define transitions between nodes. Three types:
1. **Normal edges** — `add_edge("A", "B")`: A always transitions to B
2. **Conditional edges** — `add_conditional_edges("A", routing_fn)`: the routing function examines state and returns the name of the next node (or list of nodes)
3. **Send API** — `Send("node_name", custom_state)`: dynamic fan-out from a conditional edge. Returns a list of `Send` objects, each dispatching to a node with its own input state. This is the map-reduce primitive.

**Compilation** — `builder.compile()` transforms the builder into an executable graph. You can pass a `checkpointer` parameter to enable state persistence. The compiled graph supports `invoke()`, `stream()`, `ainvoke()`, and `astream()`.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **StateGraph** | Builder class for defining graph structure. Parametrized by state schema (TypedDict or Pydantic). |
| **State Schema** | TypedDict, dataclass, or Pydantic model defining shared state fields. Supports reducer annotations. |
| **Reducer** | Function that merges a node's output with existing state. E.g., `Annotated[list, operator.add]` appends to list instead of replacing. Without a reducer, the latest value wins. |
| **Node** | Python function `(state) -> state_update`. Can be sync or async. Added via `builder.add_node("name", fn)`. |
| **Conditional Edge** | `add_conditional_edges(source, routing_fn)` — `routing_fn` receives state, returns node name(s). Enables branching and cycles. |
| **Checkpointer** | Saves full state at every superstep. Enables resumption, HITL, multi-turn memory, and fault-tolerance. `InMemorySaver` for dev, `SqliteSaver`/`PostgresSaver` for prod. |
| **Thread** | Identified by `thread_id` in config. Isolates state between conversations/users. Same graph, different threads = independent state. |
| **Superstep** | One complete pass through all nodes that are ready to execute. Parallel nodes run in the same superstep. State is checkpointed after each superstep. |
| **Human-in-the-Loop (HITL)** | `interrupt()` pauses execution mid-node. Checkpointer preserves state. `Command(resume=value)` continues from where it stopped. Requires a checkpointer. |
| **Send API** | `Send(node_name, state)` — dynamic parallel routing. A conditional edge function returns a list of `Send` objects, each dispatching to a node with custom input state. This is LangGraph's map-reduce primitive. |
| **Subgraph** | A compiled graph nested as a node in a parent graph. If schemas match, pass compiled graph directly to `add_node`. If schemas differ, wrap in a function that transforms state in/out. |

### How Execution Works

1. Graph starts at `START`, follows edges to the first node(s)
2. Each node receives the current state, processes it, returns a dict of state updates
3. Reducers merge the updates into the shared state (append for lists, replace for scalars, etc.)
4. If the next edge is conditional, the routing function evaluates and determines the destination
5. Process repeats until `END` is reached or `interrupt()` is called
6. The checkpointer (if configured) saves state after each superstep, enabling resumption

This cycle naturally supports loops: a conditional edge can route back to a previous node, creating iterative refinement until a quality threshold is met.

### LangGraph vs Basic LangChain Agents

| Aspect | LangChain AgentExecutor | LangGraph |
|--------|------------------------|-----------|
| Flow control | Linear ReAct loop (think -> act -> observe) | Arbitrary directed graph with cycles, branches, parallel paths |
| State | Implicit (agent scratchpad) | Explicit typed state with reducers |
| Persistence | None built-in | Checkpointers (memory, SQLite, Postgres) |
| Human-in-the-loop | Not supported | `interrupt()` + `Command(resume=)` |
| Parallel execution | Not supported | `Send` API for map-reduce |
| Debugging | Opaque loop | Each node/edge is explicit and inspectable |
| Production readiness | Deprecated for complex workflows | Recommended approach since 2024 |

LangChain's own documentation now recommends LangGraph for any agent that goes beyond a simple single-tool ReAct loop.

### Ecosystem Context

LangGraph is to agents what React is to UIs — a declarative way to define complex, stateful workflows with explicit state management and composable components.

| Framework | Level | Control | Persistence | Best For |
|-----------|-------|---------|-------------|----------|
| **LangGraph** | Low (graph primitives) | Full | Built-in checkpointers | Production agents, complex flows |
| **CrewAI** | High (role-based agents) | Limited | External | Quick multi-agent prototypes |
| **AutoGen** | Medium (conversational) | Moderate | External | Multi-agent conversations |
| **Custom state machines** | Lowest | Full | Manual | When you don't want a framework |

Choose LangGraph when you need fine-grained control over execution flow, persistent state, human-in-the-loop, or when building production systems that must be debuggable and testable.

**Sources:**
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangChain Blog — interrupt() for HITL](https://blog.langchain.com/making-it-easier-to-build-human-in-the-loop-agents-with-interrupt/)

## Description

Build progressively complex agent workflows using LangGraph's graph-based state machine. You'll implement state management with reducers, conditional routing, persistent checkpointing, human-in-the-loop approval, parallel execution, and multi-node agent coordination.

### What you'll learn

- StateGraph construction with typed state and reducers
- Conditional edge routing for dynamic workflows
- Checkpointing for persistence and fault-tolerance
- Human-in-the-loop with interrupt/resume
- Parallel execution with Send API (map-reduce)
- Supervisor pattern for multi-agent coordination
- Subgraphs for modular composition

## Instructions

### Phase 1: Setup & Basic Graph (~15 min)

1. Start Docker services: `docker compose up -d`
2. Pull model: `docker exec ollama ollama pull qwen2.5:7b`
3. Install dependencies: `uv sync`
4. Run the verification script (`src/00_verify_setup.py`) to confirm Ollama connectivity and LangGraph imports work

**Exercise — Basic 2-Node Graph (`src/01_basic_graph.py`):**

The simplest LangGraph workflow: two nodes connected in sequence. The "generate" node creates content from a prompt using the LLM, and the "refine" node improves it. This teaches the fundamental pattern: define state -> add nodes -> add edges -> compile -> invoke.

You'll define a `TypedDict` state, create two node functions, wire them with `START -> generate -> refine -> END`, compile, and invoke. This is the "hello world" of LangGraph — every more complex pattern builds on this foundation.

### Phase 2: Conditional Routing & Cycles (~20 min)

Conditional edges are what make LangGraph a **graph** instead of a linear chain. A routing function examines the current state and decides which node to transition to next. Combined with cycles (edges that point backward), this enables iterative refinement loops — a pattern used in virtually every production agent.

**Exercise 1 — Quality-Check Loop (`src/02_conditional_routing.py`, TODO #1):**

Build a generate-evaluate-loop pattern: the "generate" node produces text, the "evaluate" node scores its quality (1-10 via LLM), and a conditional edge either loops back to "generate" (if score < 7) or proceeds to END. A max-iterations guard prevents infinite loops. This teaches: conditional edges, cycles, and convergence guards.

**Exercise 2 — Query Router (`src/02_conditional_routing.py`, TODO #2):**

Build a routing pattern: a single entry node examines the user's query and routes to one of three specialist nodes ("math_node", "creative_node", "general_node") based on content. Each specialist has a different system prompt. This teaches: branching with conditional edges and the router pattern.

### Phase 3: Checkpointing & Human-in-the-Loop (~20 min)

Checkpointers save the full graph state after every superstep. This enables: (1) multi-turn conversations that remember history, (2) fault-tolerant execution that can resume after crashes, (3) human-in-the-loop workflows where execution pauses for approval. The `thread_id` in config isolates state between different conversations.

**Exercise 1 — Conversational Memory (`src/03_checkpointing_hitl.py`, TODO #1):**

Add `InMemorySaver` as a checkpointer to a conversational graph. Test persistence by invoking with `thread_id="user1"`, sending a message, then invoking again with the same thread_id and verifying the model remembers the previous context. Then invoke with `thread_id="user2"` to confirm thread isolation. This teaches: checkpointer setup, thread-based state isolation, and multi-turn memory.

**Exercise 2 — HITL Approval Gate (`src/03_checkpointing_hitl.py`, TODO #2):**

Build an approval workflow: the agent proposes an action (e.g., "send email to X"), `interrupt()` pauses execution for user review, then `Command(resume="approved")` or `Command(resume="rejected")` continues or aborts. This teaches: `interrupt()`, `Command(resume=)`, and the approval gate pattern used in production agents.

### Phase 4: Parallel Execution — Send API (~20 min)

The `Send` API enables dynamic fan-out: a conditional edge returns a list of `Send(node_name, custom_state)` objects, each dispatching a node invocation with its own input state. All dispatched nodes run in the same superstep (parallel). Combined with a reducer (`operator.add` on a list field), this implements map-reduce natively in the graph.

**Exercise — Map-Reduce Research (`src/04_parallel_send.py`):**

Build a map-reduce workflow: a "split" node breaks a research question into N subtasks, a conditional edge returns `[Send("worker", {"task": t}) for t in subtasks]`, each "worker" node processes its subtask independently via LLM, and an "aggregate" node collects all results (via `Annotated[list, operator.add]` reducer) and produces a final summary. This teaches: `Send` API, parallel execution, reducers, and the map-reduce pattern.

### Phase 5: Multi-Agent Supervisor (~25 min)

The supervisor pattern is the standard architecture for multi-agent systems in LangGraph: a central "supervisor" node decides which specialist agent to invoke, routes to it via conditional edges, collects the result, and either routes to another specialist or produces a final answer. Each specialist is a node with a focused system prompt.

**Exercise 1 — Supervisor Graph (`src/05_supervisor.py`, TODO #1):**

Build a supervisor that coordinates three specialists: researcher (factual questions), calculator (math), and writer (creative content). The supervisor examines the query, picks the right specialist, collects the result, and synthesizes a final answer. This teaches: the supervisor pattern, multi-agent coordination, and conditional routing with cycles.

**Exercise 2 — Subgraph Extraction (`src/05_supervisor.py`, TODO #2):**

Extract the researcher agent into a self-contained subgraph with its own internal nodes (e.g., "plan_research" -> "execute_research" -> "summarize"). Compile it independently, then nest it as a node in the main supervisor graph. This teaches: subgraph composition, state transformation between parent and child graphs, and modular agent design.

## Motivation

LangGraph is the production standard for building stateful AI agents. Understanding graph-based state machines, checkpointing, and multi-agent coordination is essential for building robust AI systems beyond simple prompt-response patterns. Key reasons:

- **Industry standard**: LangChain (the most widely-used LLM framework) recommends LangGraph for all non-trivial agents
- **Production patterns**: Checkpointing, HITL, and fault-tolerance are requirements in enterprise AI deployments
- **Architecture skill**: Graph-based state machines transfer to any agent framework — the concepts are universal
- **Foundation for advanced practices**: This builds toward DSPy integration (030c) and Agentic AI practices (031a-c)

## Commands

All commands run from `practice_029b_langgraph/`.

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container with persistent volume |
| | `docker exec ollama ollama pull qwen2.5:7b` | Download the Qwen 2.5 7B model into Ollama |
| | `docker compose down` | Stop and remove the Ollama container |
| | `docker compose logs -f ollama` | Stream Ollama container logs |
| **Setup** | `uv sync` | Install Python dependencies from pyproject.toml |
| | `uv run python src/00_verify_setup.py` | Verify Ollama connectivity and LangGraph imports |
| **Phase 1** | `uv run python src/01_basic_graph.py` | Run basic 2-node generate-refine graph |
| **Phase 2** | `uv run python src/02_conditional_routing.py` | Run conditional routing: quality loop + query router |
| **Phase 3** | `uv run python src/03_checkpointing_hitl.py` | Run checkpointing and human-in-the-loop exercises |
| **Phase 4** | `uv run python src/04_parallel_send.py` | Run parallel map-reduce with Send API |
| **Phase 5** | `uv run python src/05_supervisor.py` | Run multi-agent supervisor with subgraph |

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [LangGraph HITL Guide](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/)
- [LangGraph Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [LangGraph Map-Reduce (Send API)](https://docs.langchain.com/oss/python/langgraph/graph-api#map-reduce-and-the-send-api)
- [LangGraph GitHub Repository](https://github.com/langchain-ai/langgraph)
- [LangChain Ollama Integration](https://docs.langchain.com/oss/python/integrations/providers/ollama)

## State

`completed`
