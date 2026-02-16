# Practice 054b — Google ADK: Multi-Agent Orchestration

## Technologies

- **Google ADK** — multi-agent orchestration with SequentialAgent, ParallelAgent, LoopAgent
- **LiteLLM** — model proxy for Ollama
- **Ollama** — local LLM inference

## Stack

- Python 3.11+
- Docker (Ollama)
- `google-adk`, `litellm`

## Theoretical Context

### Multi-Agent Systems in ADK

While a single agent can handle many tasks, complex workflows benefit from **decomposition** — splitting work among specialized agents. ADK provides first-class support for multi-agent systems through its hierarchical agent architecture.

**The problem it solves:** A single agent with 20 tools becomes confused about which tool to use. Splitting into specialized agents (each with 3-5 tools) improves reliability — each agent has a clear role and fewer choices to make. This mirrors microservices: monolith → decomposed services.

### Agent Hierarchy — The Tree Model

ADK organizes agents in a **parent-child tree**:

```
RootAgent (orchestrator)
├── ResearchAgent (tools: search, summarize)
├── WriterAgent (tools: draft, edit)
└── ReviewerAgent (tools: check_facts, score)
```

**Key rules:**
- A parent can only delegate to its **direct children**
- Children share state via the parent's `InvocationContext`
- The parent decides which child to invoke based on LLM reasoning or workflow rules
- `parent_agent` attribute is auto-set during composition

### Workflow Agents (Non-LLM Orchestrators)

ADK provides three workflow agent types that orchestrate without using an LLM — they follow fixed patterns:

#### SequentialAgent (Pipeline)
Executes children **in order**, passing results forward. Like a Unix pipeline.

```
Input → Agent1 → Agent2 → Agent3 → Output
```

**Use case:** Research → Write → Review pipeline. Each stage gets the accumulated context from prior stages.

#### ParallelAgent (Fan-out)
Executes children **concurrently**, collecting all results. Like `asyncio.gather()`.

```
        ┌→ Agent1 →┐
Input → ├→ Agent2 →├→ Merged Output
        └→ Agent3 →┘
```

**Use case:** Fetch data from 3 sources simultaneously. Results are merged into shared state.

#### LoopAgent (Iteration)
Repeats a child agent until a condition is met or max iterations reached. Like a `while` loop.

```
Input → Agent → Check → Agent → Check → ... → Output
```

**Use case:** Iterative refinement — write draft, review, improve, review again until quality score > threshold.

### LLM-Based Delegation vs Workflow Agents

| Approach | How routing works | Best for |
|----------|-------------------|----------|
| **LLM delegation** | Parent LLM decides which child to call based on the conversation | Dynamic routing, unclear paths |
| **SequentialAgent** | Fixed order, no LLM reasoning | Known pipelines, assembly lines |
| **ParallelAgent** | All children simultaneously | Independent data gathering |
| **LoopAgent** | Repeat until condition | Iterative refinement, retries |

You can **nest** these: a SequentialAgent whose first stage is a ParallelAgent and whose second stage is a LoopAgent.

### Communication Between Agents

Agents in ADK communicate through **shared state**, not direct messaging:

1. **State propagation**: Child writes to `context.state["result"]`, parent reads it
2. **Session context**: All agents in the tree share the same session's message history
3. **Explicit handoff**: Parent uses `sub_agents=[child1, child2]` and the LLM picks which to delegate to

**Compared to LangGraph:** LangGraph uses explicit edges and conditional routing on a graph. ADK uses a tree hierarchy where the parent LLM decides delegation. LangGraph is more flexible for complex branching; ADK is simpler for straightforward hierarchies.

### Agent-as-Tool Pattern

An agent can be registered as a **tool** in another agent. The caller invokes it like any other function, and the callee runs its full reasoning loop:

```python
# research_agent is an Agent with its own tools
parent = Agent(
    name="Coordinator",
    tools=[research_agent.as_tool()],  # Agent wrapped as a tool
    ...
)
```

This enables **horizontal composition** — agents can collaborate without strict parent-child hierarchy.

### State Sharing Patterns

| Pattern | Mechanism | Example |
|---------|-----------|---------|
| **Parent → Child** | Set state before delegation | `state["task"] = "summarize"` |
| **Child → Parent** | Write to shared state | `state["summary"] = result` |
| **Sibling → Sibling** | Via shared session state | Agent1 writes `state["data"]`, Agent2 reads it |
| **Global** | `app:` prefixed state | `state["app:config"]` visible to all agents |

### Design Principles for Multi-Agent Systems

1. **Single Responsibility**: Each agent has one clear job with 3-5 tools max
2. **Explicit contracts**: Document what state keys each agent reads/writes
3. **Fail gracefully**: If a child agent fails, the parent should handle it
4. **Minimize coupling**: Agents should communicate through state, not assumptions about internal behavior

## Description

This practice explores ADK's multi-agent capabilities by building progressively complex agent systems:

1. LLM-based delegation (parent agent routing to specialized children)
2. SequentialAgent pipelines for ordered workflows
3. ParallelAgent for concurrent data gathering
4. LoopAgent for iterative refinement
5. Nested orchestration combining multiple patterns
6. Agent-as-tool for horizontal composition

## Instructions

### Exercise 1: LLM-Based Delegation — Specialist Router

**What you'll learn:** How a parent LLM agent dynamically routes tasks to specialized child agents.

This is ADK's most natural multi-agent pattern: give a parent agent several `sub_agents`, and the LLM decides which one to invoke based on the user's request.

**Steps:**
1. Create 2-3 specialist agents (e.g., math helper, trivia expert, translator)
2. Create a router agent with `sub_agents=[...]` that delegates based on query type
3. Test that the router correctly identifies which specialist to call
4. Observe delegation in the Trace tab — see how the parent's LLM reasons about routing

### Exercise 2: SequentialAgent — Research Pipeline

**What you'll learn:** How to build ordered workflows where each stage builds on the previous stage's output.

SequentialAgent doesn't use an LLM — it just runs children in order. State written by Agent1 is readable by Agent2, creating a natural pipeline.

**Steps:**
1. Build a 3-stage pipeline: Gather → Process → Format
2. Use shared state to pass data between stages
3. Define clear state contracts (what each agent reads and writes)
4. Verify the full pipeline executes in order and produces the expected output

### Exercise 3: ParallelAgent — Concurrent Gathering

**What you'll learn:** How to fan out work to multiple agents simultaneously and merge their results.

ParallelAgent runs all children at once, which is useful when agents perform independent work (e.g., fetching data from different sources).

**Steps:**
1. Create 2-3 agents that each gather different types of information
2. Wrap them in a ParallelAgent
3. Create a summarizer agent that reads all results from shared state
4. Combine into a SequentialAgent: [ParallelGatherer, Summarizer]

### Exercise 4: LoopAgent — Iterative Refinement

**What you'll learn:** How to use LoopAgent for retry/refinement patterns, and how to define termination conditions.

LoopAgent repeats its child agent until either a condition function returns False or max iterations are reached. This models iterative improvement.

**Steps:**
1. Create a writer agent that produces text and self-evaluates with a quality score
2. Store the score in state
3. Create a LoopAgent with a condition function that checks if score > threshold
4. Set `max_iterations` as a safety bound
5. Observe how the agent improves its output across iterations

### Exercise 5: Nested Orchestration — Full Workflow

**What you'll learn:** Combining Sequential, Parallel, and Loop patterns into a real-world workflow.

**Steps:**
1. Design a multi-stage workflow that uses at least 2 orchestration patterns
2. Implement it with nested workflow agents
3. Use the ADK Web UI to trace execution across all levels
4. Document the state contract — which keys are written/read at each stage

## Motivation

- **Real-world pattern**: Multi-agent systems are the dominant architecture for complex AI applications (customer support, content generation, data analysis)
- **Architectural thinking**: Designing agent hierarchies exercises the same decomposition skills as microservices
- **ADK vs LangGraph**: Understanding ADK's tree-based approach complements your LangGraph graph-based knowledge — know both, pick the right one per problem
- **Production readiness**: These orchestration patterns are what Google uses internally and recommends for Vertex AI deployments

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `docker compose up -d` | Start Ollama container |
| Setup | `docker exec ollama ollama pull qwen2.5:7b` | Pull local model |
| Setup | `uv sync` | Install Python dependencies |
| Dev | `uv run adk web .` | Launch ADK Web UI for multi-agent debugging |
| Dev | `uv run python main.py` | Run orchestration programmatically |
| Teardown | `docker compose down -v` | Stop and clean up containers |
