# Practice 031c — Agentic AI: Production Patterns

## Technologies

- **PydanticAI** — Type-safe agent framework with Pydantic validation
- **Langfuse** — Open-source LLM observability (tracing, debugging, evaluation)
- **Qdrant** — Vector database for agent long-term memory
- **Guardrails** — Input/output validation, sandboxed execution
- **tenacity** — Retry library with exponential backoff

## Stack

Python 3.12+ (uv), Docker (Ollama, Qdrant, Langfuse, PostgreSQL)

## Theoretical Context

### Production vs Prototype Agents

Prototypes work in demos. Production agents must handle: failures (LLM errors, tool timeouts), safety (prompt injection, hallucination), observability (what went wrong?), and persistence (memory across sessions). This practice covers the patterns that bridge that gap.

### PydanticAI — Type-Safe Agents

PydanticAI is built by the Pydantic team (the most-downloaded Python package). It brings full type safety to agent development: IDE autocomplete, type checking for agent inputs/outputs, and structured outputs validated automatically.

Core API:
- `Agent(model, result_type=PydanticModel)` — structured outputs validated against a Pydantic model
- `@agent.tool` decorator for type-safe tool definitions
- Model-agnostic: OpenAI, Anthropic, Ollama, etc.
- Production features: durable execution, MCP integration, Agent2Agent protocol

Why PydanticAI over raw LangChain: when correctness matters more than flexibility (financial, medical, legal). The type system catches errors at development time rather than in production.

Docs: https://ai.pydantic.dev/

### Safety & Guardrails — Defense-in-Depth

Three layers (Aegis framework):
1. **Input validation**: Sanitize user inputs, detect prompt injection attempts
2. **Processing guardrails**: Constrain tool usage, limit iterations, monitor decisions
3. **Output validation**: Check for hallucination, verify against context, enforce format

Practical patterns:
- Input sanitization: regex check for injection patterns, length limits
- Tool access control: only allow certain tools based on user role/context
- Output validation: LLM-as-judge checks output quality before returning to user
- Sandboxed execution: run agent-generated code in isolated environment

### Human-in-the-Loop (HITL)

- `interrupt()` in LangGraph pauses execution, checkpointer saves state
- `Command(resume=value)` continues with user's input
- Use for: high-risk actions (send email, modify data), ambiguous requests, quality gates
- Escalation patterns: auto-approve low-risk, require approval for medium-risk, block high-risk

### Agent Memory Systems

- **Short-term**: Conversation messages (in LangGraph state)
- **Long-term semantic**: Vector store (Qdrant) — embed and retrieve past knowledge
- **Episodic**: Records of past agent runs — what queries were asked, what tools were used, outcomes
- Pattern: Before answering, retrieve relevant past experiences from vector store

### Observability with Langfuse

- Open-source (MIT license), self-hostable
- Traces: Structured logs of every LLM call (prompt, response, tokens, latency, cost)
- Debugging: Visualize agent chains, see decision points, identify failures
- Evaluation: Run test suites, compare model versions, track quality over time
- Integration: `langfuse.callback_handler()` for LangChain/LangGraph

Docs: https://langfuse.com/docs/

### Error Recovery Patterns

- **Retry with exponential backoff**: `tenacity.retry(wait=wait_exponential(), stop=stop_after_attempt(3))`
- **Circuit breaker**: If service fails N times, stop trying for a cooldown period
- **Fallback agents**: Primary agent fails -> try simpler fallback -> return graceful error
- **Dead letter queue**: Log failed requests for later inspection

### Agent Evaluation

- Custom eval datasets: list of (input, expected_output) pairs
- Metrics: exact match, semantic similarity, LLM-as-judge
- LLM-as-judge: Use LLM to evaluate another LLM's output (rate 1-5 on relevance, accuracy, helpfulness)
- Regression testing: run eval suite after changes to ensure no degradation

### Key Concepts

| Concept | Description |
|---------|-------------|
| **PydanticAI Agent** | Type-safe agent with validated inputs/outputs |
| **Guardrail** | Validation layer (input, processing, or output) that constrains agent behavior |
| **Prompt Injection** | Adversarial input trying to override agent's instructions |
| **HITL** | Human-in-the-Loop — pause for human approval on risky actions |
| **Langfuse Trace** | Structured log of an agent's full execution chain |
| **Vector Memory** | Qdrant-backed long-term memory for agent knowledge |
| **Episodic Memory** | Records of past agent interactions and outcomes |
| **Circuit Breaker** | Pattern that stops retrying after repeated failures |
| **LLM-as-Judge** | Using an LLM to evaluate another LLM's output quality |
| **Eval Dataset** | Collection of test cases for measuring agent performance |

## Description

Bridge the gap from prototype to production agent. You'll build type-safe agents with PydanticAI, implement safety guardrails, add human-in-the-loop approval, persist agent memory in Qdrant, trace execution with Langfuse, implement error recovery patterns, and evaluate agent quality with custom benchmarks.

**What you'll learn:**
- PydanticAI for type-safe agent development
- Three-layer guardrail architecture (input, processing, output)
- Human-in-the-loop approval workflows
- Long-term agent memory with vector stores
- Observability with Langfuse (tracing, debugging)
- Error recovery (retry, circuit breaker, fallback)
- Agent evaluation with LLM-as-judge

## Instructions

### Phase 1: Setup & PydanticAI Basics (~15 min)

1. Start Docker services (Ollama, Qdrant, Langfuse, PostgreSQL), pull model
2. Install Python dependencies with `uv sync`
3. Run `src/00_verify_setup.py` to verify all services are connected
4. **User implements:** PydanticAI agent with a Pydantic result model (`answer`, `confidence`, `sources`) and two `@agent.tool` decorated functions

**Exercise 1 — Pydantic Result Model (`src/01_pydantic_agent.py`, TODO #1):**
PydanticAI's core value proposition is structured, validated outputs. Instead of parsing free-text LLM responses, you define a Pydantic model and PydanticAI guarantees the agent's output conforms to it. This eliminates an entire class of production bugs where downstream code receives unexpected formats.

**Exercise 2 — PydanticAI Agent with Tools (`src/01_pydantic_agent.py`, TODO #2):**
The `Agent` class ties together the model, result type, and tools. The `@agent.tool` decorator makes functions available to the LLM with full type information — the LLM sees parameter names, types, and docstrings. This is how PydanticAI achieves type-safe tool calling.

### Phase 2: Safety & Guardrails (~20 min)

5. Understand the three-layer defense-in-depth model for agent safety
6. **User implements:** Input validator that detects prompt injection patterns and enforces length limits
7. **User implements:** Output validator using LLM-as-judge to check for hallucination
8. **User implements:** Wire validators into the agent pipeline (validate -> run -> validate -> return)

**Exercise 3 — Input Validator (`src/02_guardrails.py`, TODO #1):**
The first defense layer. Prompt injection is the #1 security risk for LLM applications (OWASP Top 10 for LLMs). Checking for known injection patterns (e.g., "ignore previous instructions") and enforcing input length limits catches the most common attack vectors before they reach the model.

**Exercise 4 — Output Validator (`src/02_guardrails.py`, TODO #2):**
The third defense layer. Even with good inputs, LLMs hallucinate. Using a second LLM call as a "judge" to verify the response is grounded in the provided context is a production pattern used by companies like Anthropic and OpenAI in their safety pipelines.

**Exercise 5 — Guardrail Pipeline (`src/02_guardrails.py`, TODO #3):**
Wiring the validators into a pipeline (input check -> agent -> output check) creates a reusable pattern. If input validation fails, the agent never runs (saving cost). If output validation fails, the agent retries or returns a safe fallback.

### Phase 3: Human-in-the-Loop (~15 min)

9. Understand HITL patterns and risk classification
10. **User implements:** Risk classifier that categorizes agent actions as low/medium/high risk
11. **User implements:** LangGraph workflow with HITL — agent proposes action, classifies risk, high-risk triggers `interrupt()`

**Exercise 6 — Risk Classifier (`src/03_hitl.py`, TODO #1):**
Not all agent actions carry the same risk. Classifying actions lets you auto-approve safe operations (read data) while requiring human approval for dangerous ones (delete data, send email). This is the foundation of proportional trust in agent systems.

**Exercise 7 — HITL LangGraph Workflow (`src/03_hitl.py`, TODO #2):**
LangGraph's `interrupt()` function pauses the graph, saves state to a checkpointer, and returns control to the caller. When the human approves, `Command(resume=value)` resumes from exactly where it paused. This is how production agents implement approval workflows without losing context.

### Phase 4: Agent Memory (~20 min)

12. Understand semantic memory architecture (short-term vs long-term vs episodic)
13. **User implements:** Store agent Q&A interactions in Qdrant (embed query+response, store as vector)
14. **User implements:** Retrieve relevant past interactions before answering new queries
15. **User implements:** Wire memory into agent (retrieve history -> include as context -> answer)

**Exercise 8 — Store Interaction (`src/04_memory.py`, TODO #1):**
Long-term memory transforms a stateless agent into one that learns from experience. By embedding each Q&A pair and storing it in Qdrant, you create a searchable knowledge base of past interactions. The embedding captures semantic meaning, so "What's the capital of France?" matches "Tell me about Paris" even though the words differ.

**Exercise 9 — Retrieve Relevant History (`src/04_memory.py`, TODO #2):**
Retrieval is where vector search shines. Given a new query, you embed it and find the k most similar past interactions. This is the same pattern as RAG (Retrieval-Augmented Generation) but applied to the agent's own memory rather than external documents.

**Exercise 10 — Memory-Augmented Agent (`src/04_memory.py`, TODO #3):**
The final step connects retrieval to generation. By prepending relevant past interactions to the agent's context, you give it "memory" — it can reference previous conversations, remember user preferences, and build on past answers.

### Phase 5: Observability with Langfuse (~20 min)

16. Understand Langfuse tracing architecture
17. **User implements:** Initialize Langfuse callback handler and integrate with LangGraph agent
18. **User implements:** Run test queries, then view traces in Langfuse UI (http://localhost:3000)

**Exercise 11 — Langfuse Integration (`src/05_observability.py`, TODO #1):**
Observability is non-negotiable in production. Langfuse's callback handler intercepts every LLM call, tool invocation, and chain step, recording prompts, responses, latency, and token counts. This data is essential for debugging failures, optimizing costs, and understanding agent behavior.

**Exercise 12 — Trace Analysis (`src/05_observability.py`, TODO #2):**
Running test queries and analyzing traces teaches you to use Langfuse as a debugging tool. In production, when an agent gives a bad answer, you trace back through the chain to find which step went wrong — was it a bad retrieval, a hallucinated tool call, or a malformed prompt?

### Phase 6: Error Recovery & Evaluation (~20 min)

19. Understand retry, circuit breaker, and fallback patterns
20. **User implements:** Retry decorator with exponential backoff for flaky tool calls
21. **User implements:** Fallback chain — primary agent -> simpler agent -> error message
22. **User implements:** Eval dataset (10 test cases) with LLM-as-judge metric

**Exercise 13 — Retry with Backoff (`src/06_recovery_eval.py`, TODO #1):**
LLM APIs and external tools fail. The `tenacity` library provides battle-tested retry logic with exponential backoff (wait 1s, 2s, 4s...) to avoid hammering a failing service. This is the simplest and most common production resilience pattern.

**Exercise 14 — Fallback Chain (`src/06_recovery_eval.py`, TODO #2):**
When retries are exhausted, fallback to a simpler strategy. A three-tier fallback (primary model -> smaller/cheaper model -> static error message) ensures the user always gets a response. This is how production systems maintain availability during partial outages.

**Exercise 15 — Evaluation with LLM-as-Judge (`src/06_recovery_eval.py`, TODO #3):**
You can't improve what you don't measure. Creating an eval dataset with expected outputs and using an LLM to judge response quality (1-5 scale) gives you a quantitative baseline. Run this after every change to catch regressions — the same way you'd run unit tests.

## Motivation

Production AI systems require safety, observability, and resilience beyond what prototypes provide. This practice covers the engineering patterns that make agents deployable: type safety (PydanticAI), guardrails, HITL, memory, tracing (Langfuse), and evaluation. These skills are directly applicable to building AI agents at scale.

## Commands

All commands run from `practice_031c_agentic_production_patterns/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start all services (Ollama, Qdrant, Langfuse, PostgreSQL) |
| `docker compose down` | Stop and remove all containers |
| `docker compose logs -f` | Stream logs from all services |
| `docker exec ollama ollama pull qwen2.5:7b` | Download the Qwen 2.5 7B model into Ollama |
| Open `http://localhost:3000` | Access Langfuse dashboard (sign up on first visit) |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` |
| `uv run python src/00_verify_setup.py` | Verify Ollama, Qdrant, and Langfuse connections |

### Phase 1: PydanticAI Basics

| Command | Description |
|---------|-------------|
| `uv run python src/01_pydantic_agent.py` | Run PydanticAI agent with typed result model and tools |

### Phase 2: Guardrails

| Command | Description |
|---------|-------------|
| `uv run python src/02_guardrails.py` | Test input/output guardrails and pipeline |

### Phase 3: Human-in-the-Loop

| Command | Description |
|---------|-------------|
| `uv run python src/03_hitl.py` | Test HITL approval workflow with risk classification |

### Phase 4: Agent Memory

| Command | Description |
|---------|-------------|
| `uv run python src/04_memory.py` | Test long-term memory storage and retrieval |

### Phase 5: Observability

| Command | Description |
|---------|-------------|
| `uv run python src/05_observability.py` | Run agent with Langfuse tracing, then view in UI |

### Phase 6: Error Recovery & Evaluation

| Command | Description |
|---------|-------------|
| `uv run python src/06_recovery_eval.py` | Test retry, fallback chain, and LLM-as-judge evaluation |

## References

- [PydanticAI Docs](https://ai.pydantic.dev/)
- [Langfuse Docs](https://langfuse.com/docs/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [tenacity Docs](https://tenacity.readthedocs.io/)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [LangGraph Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)

## State

`not-started`
