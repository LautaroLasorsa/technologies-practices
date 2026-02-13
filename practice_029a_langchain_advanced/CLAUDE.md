# Practice 029a — LangChain Advanced: LCEL, Tools & Structured Output

## Technologies

- **LangChain** (v0.3+) — Framework for LLM application development
- **LangChain Expression Language (LCEL)** — Declarative chain composition via pipe operator
- **langchain-ollama** — Local LLM integration (ChatOllama)
- **ChromaDB** — In-memory vector store for RAG exercises
- **Pydantic** — Structured output validation and schema definition

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### What LangChain Is & The Problem It Solves

LangChain is a framework for building applications powered by language models. It provides standardized interfaces for LLM interactions, tool usage, retrieval-augmented generation, and chain composition. Without it, developers manually handle prompt templates, output parsing, error handling, retry logic, and multi-step orchestration — each team reinventing these patterns. LangChain standardizes them into composable, reusable components.

The core insight: most LLM applications follow the same pattern — **prepare input → call model → parse output → repeat**. LangChain abstracts this into a pipeline of composable "Runnables" that snap together like LEGO blocks.

### Architecture: Modular Package Structure

LangChain v0.3 is split into focused packages to avoid monolithic dependencies:

- **`langchain-core`**: Base abstractions that everything builds on. Contains `Runnable` (the universal interface), `PromptTemplate`, `OutputParser`, `ChatModel`, `Tool`, and `CallbackHandler`. Every component implements the `Runnable` protocol, which defines `invoke()`, `stream()`, `batch()`, `ainvoke()`, `astream()`, and `abatch()`. This is the only mandatory dependency.

- **`langchain`**: Higher-level chains, agents, and retrieval logic built on `langchain-core`. Contains `ConversationChain`, `RetrievalQA`, agent executors, and memory wrappers. Think of it as "batteries included" orchestration.

- **Partner packages** (`langchain-ollama`, `langchain-openai`, `langchain-anthropic`): Provider-specific integrations. Each implements `BaseChatModel` for its provider. Installed separately — you only pull in what you use.

- **`langchain-community`**: Community-maintained integrations (vector stores, document loaders, etc.). Broader ecosystem support.

### LCEL Deep Dive

LCEL (LangChain Expression Language) is the **pipe-based composition system** for building chains. The mental model:

```
prompt_template | llm | output_parser
```

Each component is a `Runnable` that transforms `input → output`. The pipe operator (`|`) connects them into a `RunnableSequence` where each component's output becomes the next component's input. Data flows left-to-right, like Unix pipes.

**Key advantage**: Because every component implements the `Runnable` protocol, you get **streaming**, **batching**, and **async** for free across the entire chain. When you call `.stream()` on a chain, tokens propagate through every component as they arrive — the output parser processes tokens incrementally, not waiting for the full response.

**Core LCEL primitives:**

| Primitive | Purpose |
|-----------|---------|
| `RunnableSequence` | Chain components in series (created by `\|` operator) |
| `RunnableParallel` | Fan-out: run multiple chains on the same input concurrently, merge results into a dict |
| `RunnablePassthrough` | Pass input through unchanged (useful for forwarding data alongside transformations) |
| `RunnableLambda` | Wrap any Python function as a Runnable (escape hatch for custom logic) |

**Data flow example:**
```
Input: {"topic": "recursion"}
  → ChatPromptTemplate: formats into [HumanMessage("Explain recursion")]
    → ChatOllama: generates AIMessage("Recursion is when...")
      → StrOutputParser: extracts "Recursion is when..."
Output: "Recursion is when..."
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Runnable** | Base protocol — anything with `invoke`/`stream`/`batch`. All LangChain components implement it. The universal building block. |
| **LCEL** | LangChain Expression Language — pipe-based composition of Runnables (`prompt \| llm \| parser`). Creates `RunnableSequence` automatically. |
| **PromptTemplate / ChatPromptTemplate** | Templating for LLM inputs with variable interpolation. `ChatPromptTemplate` produces message lists (system + human + AI). |
| **OutputParser** | Converts LLM text output to structured data. `StrOutputParser` extracts text, `PydanticOutputParser` validates against a schema, `JsonOutputParser` parses raw JSON. |
| **Tool** | Function callable by an LLM, with name, description, and JSON schema for arguments. Created via `@tool` decorator or `BaseTool` subclass. |
| **Structured Output** | `.with_structured_output(schema)` — forces LLM to return data matching a Pydantic model. Uses native JSON schema support when available. |
| **Callbacks** | Hook into chain execution (`on_llm_start`, `on_llm_new_token`, `on_llm_end`, etc.) for logging, tracing, streaming, and debugging. |
| **Fallbacks** | `.with_fallbacks([backup_chain])` — automatic failover when primary chain errors or returns invalid output. |
| **RunnableWithMessageHistory** | Modern memory pattern — wraps a chain to automatically load/save conversation history per session. Replaces deprecated `ConversationBufferMemory`. |

### Structured Output Strategies

Three approaches for getting structured data from LLMs, from best to most manual:

1. **`.with_structured_output(PydanticModel)`** — Best approach. Tells the model to output JSON matching the schema. Uses native tool-calling/JSON mode when the model supports it. Returns a validated Pydantic instance directly. Use this when your model supports it (Ollama 0.5+, OpenAI, Anthropic).

2. **`PydanticOutputParser` in chain** — The chain's prompt includes format instructions generated from the Pydantic schema. The LLM must follow the instructions (no enforcement). Then the parser validates the output. Use this as a fallback when `.with_structured_output()` isn't available.

3. **`JsonOutputParser`** — Generic JSON extraction without Pydantic validation. Useful when you want flexible JSON without strict schema enforcement, or for streaming partial JSON objects.

### Ecosystem Context

**LangChain vs LlamaIndex**: LangChain is a general-purpose LLM orchestration framework (chains, agents, tools, memory). LlamaIndex is RAG-focused (document indexing, retrieval, query engines). Choose LangChain for complex multi-step pipelines with tool usage; choose LlamaIndex for pure document Q&A.

**LangChain vs direct API calls**: LangChain adds ~15-25% overhead per call but provides standardized abstractions, automatic retries, streaming propagation, and composability. For a single LLM call, use the API directly. For multi-step chains, tool-using agents, or structured output pipelines, LangChain pays for itself in reduced boilerplate.

**When to use LangChain**: Complex chains, multi-step pipelines, tool-using agents, structured extraction, RAG with multiple retrievers, conversation management.

**When NOT to use LangChain**: Simple single-shot LLM calls, latency-critical paths where every millisecond matters, or when you need full control over the API request/response cycle.

## Description

Hands-on practice with LangChain's advanced composition patterns. You'll build progressively complex chains using LCEL, create custom tools with Pydantic schemas, extract structured data from LLM outputs, implement streaming for real-time responses, and add conversation memory using the modern `RunnableWithMessageHistory` pattern.

### What you'll learn

1. **LCEL pipe composition** — building modular, reusable chains with the `|` operator
2. **Runnable primitives** — `RunnableParallel`, `RunnablePassthrough`, `RunnableLambda` for data transformation
3. **Custom tool creation** — `@tool` decorator with Pydantic schemas and error handling
4. **Structured output extraction** — three strategies and when to use each
5. **Streaming patterns** — token-by-token output for responsive user experiences
6. **Callback system** — logging, debugging, and tracing chain execution
7. **Conversation memory** — `RunnableWithMessageHistory` for multi-turn conversations
8. **Fallback chains** — resilient LLM applications that degrade gracefully

## Instructions

### Phase 1: Setup & First Chain (~15 min)

1. Start Docker services with `docker compose up -d`, then pull the Ollama model (`qwen2.5:7b`)
2. Initialize the Python project with `uv sync`
3. Run `src/00_verify_setup.py` to confirm Ollama connectivity
4. **Exercise (01_first_chain.py):** Build your first LCEL chain. This teaches the fundamental `prompt | llm | parser` pattern — the building block for everything else in LangChain. You'll see how three independent components snap together via the pipe operator to form a complete pipeline.

### Phase 2: LCEL Composition (~20 min)

1. Open `src/02_lcel_composition.py` and review the pre-built prompt templates
2. **Exercise 1 — Multi-step chain:** Build a chain that summarizes text, then translates to Spanish, then formats as bullet points. This teaches sequential composition — how to build complex behavior from simple, single-purpose steps. Each step is independently testable and reusable.
3. **Exercise 2 — RunnableParallel fan-out:** Build a parallel chain that simultaneously summarizes, extracts keywords, and determines sentiment for the same input text. This teaches concurrent composition — how LCEL runs independent branches in parallel and merges results into a dict.
4. **Exercise 3 — RunnableLambda transformation:** Insert custom Python logic into an LCEL chain using `RunnableLambda`. This teaches the "escape hatch" pattern — when you need arbitrary data transformation between Runnables.

### Phase 3: Custom Tools & Structured Output (~25 min)

1. Open `src/03_tools_structured.py` and review the `@tool` decorator examples
2. **Exercise 1 — Custom tools:** Create three tools (calculator, word counter, text reverser) using the `@tool` decorator with proper type hints and docstrings. This teaches how LLMs discover and call tools — the schema is auto-generated from your function signature and docstring.
3. **Exercise 2 — Structured output extraction:** Use `.with_structured_output()` to extract a structured `Recipe` from unstructured cooking text. This teaches the most important production pattern — getting reliable structured data from free-form LLM output.
4. **Exercise 3 — Fallback chain:** Build a chain that tries structured extraction first, then falls back to text parsing if it fails. This teaches resilient chain design — production systems must handle LLM non-determinism gracefully.

### Phase 4: Streaming & Callbacks (~15 min)

1. Open `src/04_streaming_callbacks.py` and review the streaming architecture
2. **Exercise 1 — Streaming output:** Implement a streaming chain that prints tokens as they arrive. This teaches how LCEL propagates streaming through the entire chain — a critical pattern for responsive UIs.
3. **Exercise 2 — Custom callback handler:** Create a `BaseCallbackHandler` subclass that logs LLM calls, counts tokens, and measures latency. This teaches the observability pattern — understanding what happens inside your chains without modifying them.

### Phase 5: Memory & Conversation (~15 min)

1. Open `src/05_memory_conversation.py` and review the modern memory pattern
2. **Exercise 1 — Conversation with buffer memory:** Build a conversational chain using `RunnableWithMessageHistory` + `InMemoryChatMessageHistory`. This teaches the modern LangChain memory pattern — how to maintain conversation state across multiple turns without deprecated APIs.
3. **Exercise 2 — Windowed memory:** Implement conversation memory that only keeps the last N exchanges. This teaches memory management — unbounded history causes context window overflow and degrades response quality.
4. Test both exercises with multi-turn conversations that reference earlier context.

## Motivation

- **Industry standard**: LangChain is the most widely adopted LLM orchestration framework (40k+ GitHub stars, used by thousands of production applications)
- **Composition model**: LCEL's pipe-based composition is a transferable pattern — understanding it makes LangGraph, DSPy, and custom frameworks easier to learn
- **Production patterns**: Structured output, fallbacks, streaming, and callbacks are non-negotiable in production LLM apps — this practice covers all of them
- **Foundation for 029b**: This practice builds the LCEL and tool foundation needed for LangGraph agents (practice 029b) and agentic AI patterns

## Commands

All commands run from `practice_029a_langchain_advanced/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama container (port 11434, persistent volume) |
| `docker compose down` | Stop and remove Ollama container |
| `docker compose logs -f` | Stream Ollama container logs |
| `docker exec ollama ollama pull qwen2.5:7b` | Download 7B chat model for exercises |
| `docker exec ollama ollama pull nomic-embed-text` | Download embedding model (for optional RAG exercises) |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` |
| `uv run python src/00_verify_setup.py` | Verify Ollama connection and model availability |

### Phase 1: First Chain

| Command | Description |
|---------|-------------|
| `uv run python src/01_first_chain.py` | Run basic LCEL chain exercise |

### Phase 2: LCEL Composition

| Command | Description |
|---------|-------------|
| `uv run python src/02_lcel_composition.py` | Run multi-step composition, parallel, and lambda exercises |

### Phase 3: Tools & Structured Output

| Command | Description |
|---------|-------------|
| `uv run python src/03_tools_structured.py` | Run custom tools, structured output, and fallback exercises |

### Phase 4: Streaming & Callbacks

| Command | Description |
|---------|-------------|
| `uv run python src/04_streaming_callbacks.py` | Run streaming output and callback handler exercises |

### Phase 5: Memory & Conversation

| Command | Description |
|---------|-------------|
| `uv run python src/05_memory_conversation.py` | Run conversation memory exercises |

## References

- [LangChain Python Docs](https://python.langchain.com/docs/)
- [LCEL Conceptual Guide](https://python.langchain.com/docs/concepts/lcel/)
- [How to create custom tools](https://python.langchain.com/docs/how_to/custom_tools/)
- [Structured Output Guide](https://python.langchain.com/docs/how_to/structured_output/)
- [Streaming Guide](https://python.langchain.com/docs/how_to/streaming/)
- [Callback Handlers](https://python.langchain.com/docs/how_to/custom_callbacks/)
- [Migrating off ConversationBufferMemory](https://python.langchain.com/docs/versions/migrating_memory/conversation_buffer_memory/)
- [langchain-ollama PyPI](https://pypi.org/project/langchain-ollama/)
- [Runnable Interface Reference](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.Runnable.html)

## State

`not-started`
