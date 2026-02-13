# Practice 030c — DSPy + LangGraph Integration

## Technologies
- **DSPy modules as LangGraph nodes**: Optimized LM calls within graph workflows
- **dspy.ReAct**: Built-in agent with automatic tool calling
- **Tool.from_langchain()**: Convert LangChain tools for DSPy
- **dspy.streamify()**: Streaming support for DSPy modules

## Stack
Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### Why Integrate DSPy + LangGraph

Each framework solves a different problem:

- **LangGraph** = control flow + state machines. Excels at: conditional routing, parallel execution, checkpointing, human-in-the-loop (HITL). But LLM calls within nodes use manual prompts — no automatic optimization.
- **DSPy** = optimized LLM programming. Excels at: automatic prompt optimization, few-shot selection, quality constraints via assertions. But has limited orchestration capabilities — no branching, no parallel nodes, no state machines.
- **Together**: LangGraph orchestrates the workflow (decides *what* to call and *when*), DSPy optimizes each LLM call within nodes (decides *how* to prompt). Best of both worlds.

### Integration Patterns

1. **DSPy module as LangGraph node**: Define a LangGraph node function that internally calls a DSPy module. The node receives graph state, extracts relevant inputs, calls the DSPy module, and returns state updates. This is the fundamental pattern — everything else builds on it.

2. **LangChain tools in DSPy**: Use `Tool.from_langchain(lc_tool)` to wrap existing LangChain tools for use in `dspy.ReAct` agents. This bridges the LangChain ecosystem (hundreds of pre-built tools) into DSPy's optimized agent framework. Requires the `dspy[langchain]` extra.

3. **Conditional routing via DSPy**: Use a DSPy classifier module (e.g., `ChainOfThought("question -> category")`) to determine which LangGraph node to route to next. The classifier output drives `add_conditional_edges()`.

4. **End-to-end optimization**: Optimize DSPy modules within LangGraph nodes using the graph's overall metric. Train each module independently with `BootstrapFewShot`, then compose them into the graph and evaluate the full pipeline.

### dspy.ReAct Agent

Fully-managed ReAct (Reason + Act) agent. Define tools as Python functions with docstrings and type hints — DSPy auto-extracts tool schemas from these.

```python
agent = dspy.ReAct("question -> answer", tools=[fn1, fn2], max_iters=5)
```

Handles the reasoning -> tool selection -> execution -> observation loop automatically. The agent decides when to stop iterating (when it has enough information to answer).

### Streaming

`dspy.streamify(module)` returns an async generator that yields tokens incrementally as the LLM generates them. Use `StreamListener` to capture specific output fields during streaming.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **DSPy node** | LangGraph node function that internally calls a DSPy module |
| **Tool.from_langchain()** | Converts a LangChain tool to a DSPy-compatible tool |
| **dspy.ReAct** | Built-in agent module with automatic tool calling loop |
| **dspy.Tool** | Wraps a Python function as a DSPy tool (auto-extracts schema from type hints + docstring) |
| **dspy.streamify()** | Converts any DSPy module to an async streaming generator |
| **StreamListener** | Captures specific output fields during streaming |
| **Graph-level metric** | Evaluates entire LangGraph + DSPy pipeline end-to-end |
| **Node-level optimization** | Optimizes DSPy modules within individual nodes independently |

### Performance

DSPy overhead is approximately 3.5ms per call, LangGraph approximately 14ms per call. Combined overhead is negligible compared to LLM inference time (typically 500ms-5s depending on model and prompt length).

### Ecosystem Context

This integration pattern is emerging as a best practice in 2025-2026. LangGraph provides the "railway tracks" (control flow, state management, checkpointing), DSPy provides the "engine optimization" (prompt quality, few-shot selection, constraint enforcement). Together they enable production systems that are both controllable and self-optimizing.

## Description

Combine DSPy's optimization with LangGraph's orchestration. You'll build ReAct agents with tool calling, integrate LangChain tools into DSPy, use DSPy modules as LangGraph nodes, implement conditional routing based on DSPy classifiers, and stream results in real-time.

**What you'll learn:**
- dspy.ReAct agent with custom Python tools
- Converting LangChain tools for DSPy via Tool.from_langchain()
- DSPy modules as LangGraph graph nodes
- Conditional routing driven by DSPy classifiers
- Streaming DSPy outputs through LangGraph
- End-to-end query routing system with optimization

## Instructions (~90 min)

### Phase 1: Setup & ReAct Agent (~15 min)

1. Start Docker and pull the model:
   ```
   docker compose up -d
   docker exec ollama ollama pull qwen2.5:7b
   ```
2. Install dependencies: `uv sync`
3. Verify everything works: `uv run python src/00_verify_setup.py`
4. Open `src/01_react_agent.py`:
   - **Exercise 1 — Define Python tools**: Create 3 tool functions (`calculator`, `word_count`, `text_reverse`) with proper docstrings and type hints. DSPy extracts tool schemas from these annotations automatically — this teaches how DSPy's tool interface works without manual schema definitions.
   - **Exercise 2 — Build ReAct agent**: Instantiate `dspy.ReAct` with a signature and your tools. Test with questions that require tool usage. This teaches how DSPy's built-in agent handles the reason-act-observe loop.

### Phase 2: LangChain Tool Integration (~15 min)

5. Open `src/02_langchain_tools.py`:
   - **Exercise 3 — Bridge LangChain tools to DSPy**: Create a LangChain `@tool`, convert it with `Tool.from_langchain()`, and use it in a ReAct agent alongside native DSPy tools. This teaches the interop bridge between the two ecosystems — important because LangChain has hundreds of pre-built tools you can reuse.

### Phase 3: DSPy Modules as LangGraph Nodes (~25 min)

6. Open `src/03_dspy_langgraph_nodes.py`:
   - **Exercise 4 — DSPy classifier node**: Build a `ChainOfThought` module that categorizes questions into `math`, `factual`, or `creative`. This module will drive LangGraph's conditional routing — teaching how DSPy modules produce structured outputs that control graph flow.
   - **Exercise 5 — Specialized DSPy nodes**: Build 3 LangGraph node functions, each wrapping a specialized `ChainOfThought` DSPy module. Each node reads from graph state, calls its DSPy module, and writes back to state. This is the core integration pattern.
   - **Exercise 6 — Wire the LangGraph**: Connect: START -> classify -> conditional_edge (route by category) -> specialist_node -> END. This teaches the full integration: DSPy handles LLM quality, LangGraph handles control flow.

### Phase 4: Streaming & End-to-End System (~20 min)

7. Open `src/04_streaming_e2e.py`:
   - **Exercise 7 — Stream DSPy output**: Wrap a DSPy module with `dspy.streamify()` and iterate tokens with `async for`. This teaches how to get incremental output from optimized DSPy modules.
   - **Exercise 8 — End-to-end query system**: Build a full pipeline: query -> parse intent -> route -> expert answer -> format response. Stream the final output. This combines all previous concepts into a production-like system.

### Phase 5: Optimization Within Graph (~15 min)

8. Open `src/05_optimize_graph.py`:
   - **Exercise 9 — Optimize individual DSPy modules**: Use `BootstrapFewShot` to optimize the classifier, math QA, factual QA, and creative QA modules independently with small training sets. This teaches that DSPy modules can be optimized in isolation then composed.
   - **Exercise 10 — Evaluate full pipeline**: Plug optimized modules into LangGraph nodes, run the full graph, compare accuracy against the unoptimized baseline. This teaches graph-level evaluation of the integrated system.

## Motivation

The DSPy + LangGraph combination represents the state of the art for building optimized, controllable LLM systems. This practice bridges the gap between optimization (DSPy) and orchestration (LangGraph) — a skill set that's increasingly valued for production AI development. Understanding how to compose these frameworks is essential for building systems that are both reliable (LangGraph's state machines) and high-quality (DSPy's automatic optimization).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container |
| | `docker exec ollama ollama pull qwen2.5:7b` | Download Qwen 2.5 7B model |
| **Setup** | `uv sync` | Install Python dependencies |
| | `uv run python src/00_verify_setup.py` | Verify DSPy + LangGraph + Ollama connectivity |
| **Phase 1** | `uv run python src/01_react_agent.py` | Run ReAct agent with custom Python tools |
| **Phase 2** | `uv run python src/02_langchain_tools.py` | Test LangChain tool integration with DSPy |
| **Phase 3** | `uv run python src/03_dspy_langgraph_nodes.py` | Run DSPy modules as LangGraph nodes with routing |
| **Phase 4** | `uv run python src/04_streaming_e2e.py` | Run streaming end-to-end query system |
| **Phase 5** | `uv run python src/05_optimize_graph.py` | Optimize DSPy modules and evaluate full pipeline |

## References
- DSPy Tools: https://dspy.ai/learn/programming/tools/
- DSPy ReAct: https://dspy.ai/api/modules/ReAct/
- DSPy Streaming: https://dspy.ai/tutorials/streaming/
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
