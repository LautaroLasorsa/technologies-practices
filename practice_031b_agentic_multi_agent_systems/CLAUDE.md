# Practice 031b — Agentic AI: Multi-Agent Systems

## Technologies

- **LangGraph Supervisor**: Central orchestrator routing to specialized agents
- **LangGraph Swarm**: Peer-to-peer agent handoffs without central coordinator
- **CrewAI**: Role-based multi-agent teams (researcher, analyst, writer)
- **Agent-as-Tool**: Wrapping agents as callable tools for composition

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### Why Multi-Agent

Single agents struggle with complex tasks requiring diverse expertise. Multi-agent systems decompose problems among specialists, each with focused capabilities. Analogous to a team of experts vs one generalist — a research team with a biologist, statistician, and technical writer produces better scientific papers than one person doing all three roles.

The key insight: by giving each agent a narrow role, focused system prompt, and specialized tools, you get better results than a single agent with a massive prompt trying to be everything at once.

### Three Multi-Agent Architectures

#### 1. Supervisor (Centralized)

A central supervisor agent routes tasks to specialized workers. The supervisor sees all communication, decides which agent should handle the next step, and collects/synthesizes results.

**LangGraph implementation**: StateGraph with a supervisor node that uses an LLM to decide routing. Conditional edges connect the supervisor to worker nodes, and workers return results back to the supervisor for final synthesis or further delegation.

```
User Query → [Supervisor] → decides → [Researcher] → result → [Supervisor] → decides → [Writer] → result → [Supervisor] → final answer → User
```

**Pros**: Clear control flow, easy debugging (all decisions visible in supervisor), good for well-defined task decomposition, central point for logging/monitoring.

**Cons**: Supervisor is a bottleneck (all context flows through one agent), extra LLM calls for routing decisions, single point of failure if supervisor makes bad routing choices.

#### 2. Swarm (Decentralized)

Agents hand off control to each other dynamically using handoff tools. No central coordinator — each agent decides which peer should handle the next step. Inspired by OpenAI's Swarm framework.

**LangGraph Swarm implementation**: Each agent is a `create_react_agent` with `create_handoff_tool` tools. `create_swarm([agent_a, agent_b, agent_c])` wires them into a graph where any agent can transfer control to any other.

```
User Query → [Triage Agent] → handoff → [Math Agent] → handoff → [Writer Agent] → final answer → User
```

**Pros**: Natural for customer service flows (triage -> specialist), no bottleneck, agents can hand off mid-task, lower latency (no supervisor overhead per step).

**Cons**: Harder to debug (no central view of decisions), can get stuck in handoff loops, harder to ensure task completion, requires each agent to know when to hand off.

#### 3. Crew/Team (Role-Based)

Agents defined by role, goal, and backstory collaborate on tasks with dependencies. CrewAI framework provides Agent + Task + Crew abstractions.

**CrewAI implementation**: Define `Agent` objects with roles and goals, `Task` objects with descriptions and expected outputs (tasks can depend on other tasks), and a `Crew` that orchestrates execution.

**Processes**:
- **Sequential** (A -> B -> C): Each task runs in order, output feeds the next
- **Hierarchical**: A manager agent delegates to workers
- **Consensus**: Agents discuss and agree (experimental)

**Pros**: Very intuitive (mirrors real teams), minimal code to set up, natural for content workflows, built-in role/goal/backstory prompting.

**Cons**: Less fine-grained control than LangGraph, harder to implement complex conditional routing, framework makes many decisions for you.

### Agent-as-Tool Pattern

Wrap a specialized agent as a callable tool for another agent. The outer agent can "call" the inner agent like any other tool function. This enables hierarchical composition and reuse.

Example: A supervisor has a `research_agent_tool` in its tool list. When the supervisor calls it, the tool internally compiles and invokes a full LangGraph research subgraph, then returns the result as a string. The supervisor doesn't know (or care) that the tool is itself an agent.

### Agent Communication Strategies

| Strategy | Mechanism | Token Cost | Isolation | Use Case |
|----------|-----------|------------|-----------|----------|
| **Message passing** | Shared message list (LangGraph `MessagesState`) | High (full history) | Low | Rich context needed between agents |
| **Shared state** | Common `TypedDict` fields agents read/write | Low (only structured fields) | Medium | Structured data pipeline |
| **Tool-based** | Agent calls another agent as a tool, receives result | Medium (result only) | High | Hierarchical composition |

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Supervisor** | Central agent that routes tasks and aggregates results |
| **Worker/Specialist** | Agent with focused role and tools (math, research, writing) |
| **Swarm** | Decentralized agents with dynamic handoffs |
| **Handoff Tool** | Special tool that transfers control to another agent |
| **Crew** | CrewAI abstraction — collection of role-based agents + tasks |
| **Agent-as-Tool** | Wrap agent as callable tool for hierarchical composition |
| **Task Dependency** | CrewAI concept — one task's output feeds another's input |
| **Process** | Execution strategy (sequential, hierarchical, consensus) |
| **MessagesState** | LangGraph built-in state schema with a `messages` list |
| **StateGraph** | LangGraph graph builder — add nodes, edges, compile to runnable |

### LangGraph vs CrewAI Comparison

| Dimension | LangGraph | CrewAI |
|-----------|-----------|--------|
| **Control** | Full graph control, explicit edges | Framework manages flow |
| **Code** | More code, more flexibility | Less code, role-based DSL |
| **Debugging** | Excellent (graph visualization, step-by-step) | Limited (framework internals) |
| **Routing** | Custom conditional edges, any topology | Sequential / hierarchical / consensus |
| **Local LLMs** | ChatOllama (native LangChain integration) | LLM class via LiteLLM (ollama/model_name) |
| **Best for** | Complex routing, custom state, production systems | Rapid prototyping, team-based workflows |

**Sources:**
- [LangGraph Multi-Agent Concepts](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph Supervisor Library](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph Swarm Library](https://github.com/langchain-ai/langgraph-swarm-py)
- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI LLM Connections](https://docs.crewai.com/en/learn/llm-connections)

## Description

Build multi-agent systems using three architectures: centralized supervisor, decentralized swarm, and role-based crew. You'll implement the same task with different approaches to understand trade-offs, then explore agent-as-tool composition.

### What you'll learn

1. **Supervisor pattern** with LangGraph (centralized routing via StateGraph)
2. **Swarm pattern** with dynamic handoffs (decentralized peer-to-peer)
3. **CrewAI role-based teams** (intuitive collaboration with Agent/Task/Crew)
4. **Agent-as-tool composition** for hierarchical agents
5. **Framework comparison** — same task, different approaches, measured trade-offs
6. **Agent communication strategies** — message passing vs shared state vs tool-based

## Instructions

### Phase 1: Setup & Supervisor Pattern (~25 min)

1. Start Docker, pull model, install dependencies with `uv sync`
2. Run `00_verify_setup.py` to confirm LangGraph, CrewAI, and Ollama connectivity
3. Open `01_supervisor.py` — the file defines the state schema (`SupervisorState`), specialist system prompts, and ChatOllama configuration
4. **TODO(human) #1**: Implement `supervisor_node` — the supervisor receives the user query and the list of available specialists, uses the LLM to decide which specialist should handle it, and returns a routing decision. This is the core of the centralized pattern: one agent making all routing decisions based on the conversation so far.
5. **TODO(human) #2**: Implement three specialist nodes (`researcher_node`, `mathematician_node`, `writer_node`) — each takes the state, calls ChatOllama with a specialized system prompt, and appends its response to the messages. These demonstrate the "focused agent" principle: narrow system prompts produce better results than one omniscient agent.
6. **TODO(human) #3**: Wire the supervisor graph — add all nodes, create a conditional edge from the supervisor node that routes to the correct specialist based on the supervisor's decision, add edges from each specialist back to the supervisor for synthesis, and define the terminal condition. This teaches LangGraph's core graph-building API.
7. Run and test with the provided sample queries.

### Phase 2: Swarm Pattern (~20 min)

1. Open `02_swarm.py` — the file provides handoff tool utilities, agent configurations, and ChatOllama setup
2. **TODO(human) #1**: Create three peer agents using `create_react_agent`, each equipped with `create_handoff_tool` tools pointing to the other agents. The triage agent gets handoff tools to both specialists; each specialist gets a handoff tool back to the triage agent. This teaches decentralized control: each agent autonomously decides when to hand off.
3. **TODO(human) #2**: Build the swarm using `create_swarm()` with the agents list and a default active agent, compile with a checkpointer, and invoke with the same test queries from Phase 1. This demonstrates how the same task flows differently without a central coordinator.
4. Compare: observe the handoff sequence vs supervisor's routing decisions.

### Phase 3: CrewAI Comparison (~20 min)

1. Open `03_crewai.py` — the file provides CrewAI imports and Ollama LLM configuration
2. **TODO(human) #1**: Define three CrewAI `Agent` objects with role, goal, backstory, and the Ollama LLM. Roles should mirror the LangGraph specialists (researcher, mathematician, writer). The backstory is unique to CrewAI — it gives the agent a "persona" that shapes its reasoning style.
3. **TODO(human) #2**: Define `Task` objects with descriptions and expected outputs. Set up a dependency chain: the research task produces facts, the analysis task depends on research output, and the writing task depends on analysis. This teaches CrewAI's task dependency graph.
4. **TODO(human) #3**: Create a `Crew` with sequential process, assign the agents and tasks, run the same queries from Phases 1-2. Compare code complexity, output structure, and ease of setup vs LangGraph.
5. Discussion: when would you choose CrewAI over LangGraph?

### Phase 4: Agent-as-Tool (~15 min)

1. Open `04_agent_as_tool.py` — the file imports components from previous exercises
2. **TODO(human)**: Wrap the researcher specialist as a callable tool function. The function should compile and invoke the researcher subgraph (or the full supervisor graph), extract the result, and return it as a string. Then create a new supervisor agent that has this researcher-tool in its tool list alongside direct specialist nodes. This demonstrates hierarchical composition: agents calling agents as tools.
3. Test with queries that require research before other tasks.

### Phase 5: Communication & Comparison (~20 min)

1. Open `05_comparison.py` — the file provides a benchmark harness with 5 test queries
2. **TODO(human) #1**: Import and invoke all three architectures (supervisor, swarm, CrewAI) with the same 5 queries. Collect results in a structured format.
3. **TODO(human) #2**: Measure and compare metrics: number of LLM calls (count), total tokens used (from LLM response metadata if available), wall-clock time, and output quality (your subjective 1-5 rating). Print a comparison table. This teaches framework-agnostic evaluation — essential for choosing the right architecture in production.
4. Discussion: which architecture won, and why?

## Motivation

Multi-agent systems are the dominant paradigm for complex AI applications — Gartner projects 40% of enterprise apps will use agent-based architectures by end of 2026. Understanding the trade-offs between centralized (supervisor), decentralized (swarm), and team-based (crew) patterns is essential for designing effective AI systems. This practice builds directly on 031a (single-agent ReAct) by scaling from one agent to coordinated teams.

## Commands

All commands run from `practice_031b_agentic_multi_agent_systems/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama container with health check |
| `docker exec ollama ollama pull qwen2.5:7b` | Download the Qwen 2.5 7B model (if model-pull didn't run) |
| `docker compose down` | Stop and remove containers |
| `docker compose logs -f ollama` | Stream Ollama logs for debugging |

### Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from pyproject.toml |
| `uv run python src/00_verify_setup.py` | Verify LangGraph + CrewAI + Ollama connectivity |

### Phase 1: Supervisor

| Command | Description |
|---------|-------------|
| `uv run python src/01_supervisor.py` | Run centralized supervisor multi-agent pattern |

### Phase 2: Swarm

| Command | Description |
|---------|-------------|
| `uv run python src/02_swarm.py` | Run decentralized swarm pattern with handoffs |

### Phase 3: CrewAI

| Command | Description |
|---------|-------------|
| `uv run python src/03_crewai.py` | Run CrewAI role-based team comparison |

### Phase 4: Agent-as-Tool

| Command | Description |
|---------|-------------|
| `uv run python src/04_agent_as_tool.py` | Run agent-as-tool hierarchical composition |

### Phase 5: Comparison

| Command | Description |
|---------|-------------|
| `uv run python src/05_comparison.py` | Run architecture comparison benchmark |

## References

- [LangGraph Multi-Agent Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/)
- [LangGraph Supervisor Library](https://github.com/langchain-ai/langgraph-supervisor-py)
- [LangGraph Swarm Library](https://github.com/langchain-ai/langgraph-swarm-py)
- [CrewAI Documentation](https://docs.crewai.com/)
- [CrewAI LLM Connections (Ollama)](https://docs.crewai.com/en/learn/llm-connections)
- [Multi-Agent Benchmarks (LangChain blog)](https://blog.langchain.com/benchmarking-multi-agent-architectures/)
- [LangGraph StateGraph Reference](https://reference.langchain.com/python/langgraph/graphs/)

## State

`not-started`
