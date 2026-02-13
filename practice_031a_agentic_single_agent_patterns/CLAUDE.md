# Practice 031a — Agentic AI: Single-Agent Design Patterns

## Technologies

- **LangGraph** — Graph-based agent orchestration (StateGraph, conditional edges, cycles)
- **ReAct Pattern** — Reasoning + Acting in iterative loops
- **Plan-and-Execute** — Upfront planning with step-by-step execution
- **Reflection Pattern** — Self-critique and iterative refinement
- **ReWOO** — Reason Without Observation — batch tool execution for cost efficiency

## Stack

- Python 3.12+ (uv)
- Docker (Ollama)

## Theoretical Context

### What Makes an "Agent"

An agent is an LLM that can take actions (call tools, retrieve data, execute code) in a loop, observing results and deciding next steps autonomously. The key distinction from a simple chain: an agent has a **decision loop** — it chooses what to do next based on intermediate results.

### Four Foundational Single-Agent Patterns

**1. ReAct (Reasoning + Acting)** — The most common pattern:
- Loop: Thought → Action → Observation → Thought → ...
- LLM reasons about current state, selects a tool, executes it, observes result, continues
- Implemented in LangGraph as a cycle: agent_node → tool_node → agent_node → ...
- Pros: Simple, general-purpose, works well for tool-heavy tasks
- Cons: Can loop indefinitely, each iteration costs an LLM call, may forget early context

**2. Plan-and-Execute** — For complex multi-step tasks:
- Two phases: (1) Planner creates a step-by-step plan upfront, (2) Executor runs each step
- Optional replanner adjusts plan based on intermediate results
- Pros: More structured than ReAct, easier to audit/debug, can parallelize independent steps
- Cons: Upfront planning may be wrong, replanning adds cost
- Best for: research tasks, data pipelines, multi-tool workflows

**3. Reflection / Self-Critique** — For quality-sensitive outputs:
- Generate → Evaluate → Refine → Iterate until satisfactory
- The evaluator can be the same LLM (self-critique) or a different one (cross-critique)
- Implemented as a LangGraph cycle with conditional edge checking quality
- Pros: Improves output quality significantly, catches errors
- Cons: Multiple LLM calls per iteration, may over-refine
- Best for: code generation, writing, complex reasoning

**4. ReWOO (Reason Without Observation)** — For cost efficiency:
- Phase 1: Worker LLM creates complete plan with placeholders (#E1, #E2...) for tool results
- Phase 2: Execute ALL tools in batch (no LLM calls between)
- Phase 3: Solver LLM integrates actual results into final answer
- Pros: Minimal LLM calls (only 2: plan + solve), tools can run in parallel
- Cons: Can't adapt plan based on intermediate results
- Best for: predictable tool sequences, cost-sensitive applications

### Agent Memory Types

| Type | Description |
|------|-------------|
| **Short-term** | Messages in current context window (automatic) |
| **Long-term** | Vector store for persistent knowledge across sessions |
| **Episodic** | Records of past agent runs (what worked, what failed) |
| **Working memory** | Scratchpad for intermediate reasoning (state fields in LangGraph) |

### Dynamic Tool Creation

Agent generates new tools at runtime by writing Python code. The tool becomes available for subsequent iterations. Advanced pattern — enables agents to extend their own capabilities.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **ReAct Loop** | Thought→Action→Observation cycle. Standard agent execution pattern. |
| **Plan-and-Execute** | Upfront planning → step execution → optional replanning. |
| **Reflection** | Generate→Critique→Refine loop for quality improvement. |
| **ReWOO** | Plan with placeholders → batch execute → integrate results. |
| **Tool Node** | LangGraph node that executes tools selected by agent. |
| **Max Iterations** | Safety limit on agent loops to prevent infinite execution. |
| **Agent State** | LangGraph TypedDict tracking messages, plan, intermediate results. |
| **Working Memory** | State fields used as scratchpad during agent execution. |

### Ecosystem Context

These patterns are framework-agnostic concepts. We implement them in LangGraph because it provides the best control over execution flow. Same patterns can be built in CrewAI, AutoGen, or custom code.

**Sources:**
- [ReAct Paper — Yao et al. (2022)](https://arxiv.org/abs/2210.03629)
- [ReWOO Paper — Xu et al. (2023)](https://arxiv.org/abs/2305.18323)
- [LangGraph Agent Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [Agentic Design Patterns — Andrew Ng / DeepLearning.AI](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)

## Description

Implement four foundational single-agent design patterns: ReAct, Plan-and-Execute, Reflection, and ReWOO. Each pattern is built as a LangGraph state machine, demonstrating how different architectures suit different tasks. You'll also explore agent memory and dynamic tool creation.

### What you'll learn

1. **ReAct pattern** with tool calling and observation loops
2. **Plan-and-Execute** for structured multi-step tasks
3. **Reflection/self-critique** for quality improvement
4. **ReWOO** for cost-efficient tool execution
5. **Agent memory patterns** (working memory, scratchpad)
6. **Dynamic tool creation** at runtime
7. **When to choose which pattern**

## Instructions

### Phase 1: Setup & ReAct Deep Dive (~20 min)

1. Start Docker Compose and pull the `qwen2.5:7b` model into Ollama
2. Initialize project with `uv sync`, verify connectivity with `src/00_verify_setup.py`
3. **User implements:** ReAct agent from scratch (not using prebuilt `create_react_agent`) — `agent_node` (LLM decides action via `bind_tools`), `tool_node` (executes selected tools), conditional edge `should_continue` (continue if tool calls present AND iterations < max, else END), state with `messages` + `iteration_count`
4. Key question: Why does the ReAct loop need a max iteration limit? What happens without one?

### Phase 2: Plan-and-Execute (~25 min)

1. Understand planner → executor → replanner architecture
2. **User implements:** `planner_node` (LLM creates numbered step list), `executor_node` (runs each step via LLM call or tool), `replanner_node` (LLM reviews results, adjusts remaining plan)
3. Test with a multi-step research question
4. Key question: How does Plan-and-Execute handle a step that fails? Compare with ReAct's approach.

### Phase 3: Reflection / Self-Critique (~20 min)

1. Understand generate → evaluate → refine cycle
2. **User implements:** `generator_node` (produce code/text, or improve based on critique), `evaluator_node` (LLM critiques output, scores 1-10, provides specific feedback), conditional edge (if score < 7 → loop back with critique; else → END)
3. Test: generate a Python function, observe how reflection improves it across iterations

### Phase 4: ReWOO (~20 min)

1. Understand plan-with-placeholders → batch-execute → solve architecture
2. **User implements:** `worker_node` (creates plan with #E1, #E2 placeholders), `executor_node` (parses plan for tool calls, executes all tools, stores results in evidence dict), `solver_node` (substitutes evidence into plan, LLM produces final answer)
3. Compare: same question with ReAct vs ReWOO — count LLM calls for each

### Phase 5: Memory & Dynamic Tools (~15 min)

1. Understand working memory (state fields) vs long-term memory
2. **User implements:** Agent with scratchpad state field — agent writes intermediate notes to `state["scratchpad"]`, references them in later iterations
3. **User implements:** Dynamic tool creation — agent writes a Python function as string, `exec()` it to create a callable, add it to tools dict for use in subsequent iterations

## Motivation

Understanding agent design patterns is fundamental to building effective AI systems. Each pattern has distinct trade-offs (quality vs cost vs control), and choosing the right one for a given task is a critical skill. These patterns form the building blocks for multi-agent systems (031b) and production agent development (031c).

## Commands

All commands run from `practice_031a_agentic_single_agent_patterns/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama container |
| `docker exec ollama ollama pull qwen2.5:7b` | Download the Qwen 2.5 7B model for tool-calling support |
| `docker compose down` | Stop and remove the Ollama container |
| `docker compose logs -f ollama` | Stream Ollama logs (useful for checking model loading) |
| `docker exec ollama ollama list` | List downloaded models inside the container |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` into `.venv` |
| `uv run python src/00_verify_setup.py` | Verify Ollama connectivity and model availability |

### Phase 1: ReAct Agent

| Command | Description |
|---------|-------------|
| `uv run python src/01_react_agent.py` | Run hand-built ReAct agent with tool-calling loop |

### Phase 2: Plan-and-Execute

| Command | Description |
|---------|-------------|
| `uv run python src/02_plan_execute.py` | Run Plan-and-Execute agent with planner/executor/replanner |

### Phase 3: Reflection / Self-Critique

| Command | Description |
|---------|-------------|
| `uv run python src/03_reflection.py` | Run Reflection agent with generate/evaluate/refine cycle |

### Phase 4: ReWOO

| Command | Description |
|---------|-------------|
| `uv run python src/04_rewoo.py` | Run ReWOO agent with batch tool execution |

### Phase 5: Memory & Dynamic Tools

| Command | Description |
|---------|-------------|
| `uv run python src/05_memory_tools.py` | Run agent with scratchpad memory and dynamic tool creation |

## References

- [ReAct Paper — Yao et al. (2022)](https://arxiv.org/abs/2210.03629)
- [ReWOO Paper — Xu et al. (2023)](https://arxiv.org/abs/2305.18323)
- [LangGraph Agent Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)
- [Agentic Design Patterns — Andrew Ng](https://www.deeplearning.ai/the-batch/agentic-design-patterns-part-2-reflection/)
- [LangGraph Quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [ChatOllama + bind_tools](https://docs.langchain.com/oss/python/integrations/chat/ollama)
- [LangGraph StateGraph API](https://langchain-ai.github.io/langgraph/reference/graphs/)

## State

`not-started`
