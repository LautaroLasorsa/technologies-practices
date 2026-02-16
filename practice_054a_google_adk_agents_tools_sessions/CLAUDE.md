# Practice 054a — Google ADK: Agents, Tools & Sessions

## Technologies

- **Google ADK** (Agent Development Kit) — open-source, code-first framework for building AI agent systems
- **LiteLLM** — model-agnostic proxy enabling ADK to use Ollama local models
- **Ollama** — local LLM inference server

## Stack

- Python 3.11+
- Docker (Ollama)
- `google-adk`, `litellm`

## Theoretical Context

### What is Google ADK?

Google Agent Development Kit (ADK) is an **open-source, code-first Python framework** for building, evaluating, and deploying AI agent systems. Released at Google Cloud NEXT 2025 under Apache 2.0, it provides a structured way to create agents that use LLMs for reasoning and tools for action.

**The problem it solves:** Building production-grade AI agents requires managing conversation state, tool execution, multi-step reasoning, and evaluation — ADK provides opinionated patterns for all of these, similar to how web frameworks (Django, FastAPI) structure web applications.

### How ADK Works Internally

ADK follows a **hierarchical agent architecture**:

1. **Agent receives a prompt** → sends it to the LLM with tool schemas
2. **LLM decides** → either respond with text or call a tool
3. **Tool executes** → result fed back to LLM for next decision
4. **Loop continues** until LLM produces a final text response

The agent's behavior is defined by:
- `model`: Which LLM to use (Gemini, Ollama via LiteLLM, OpenAI, etc.)
- `instructions`: System prompt guiding behavior
- `tools`: Python functions the agent can call (type hints + docstrings become the schema)
- `description`: Used when this agent is composed into multi-agent systems

### Key Concepts

| Concept | Definition |
|---------|-----------|
| **Agent** | An entity that uses an LLM to reason and tools to act. Three types: LLM agents, Workflow agents, Custom agents |
| **Tool** | A Python function with type annotations + docstring that the LLM can invoke. ADK auto-generates the JSON schema from the signature |
| **Session** | Tracks a conversation (list of messages + events). Identified by `user_id` + `session_id`. Managed by a `SessionService` |
| **State** | Key-value store scoped to a session. Magic prefixes: `user:` (persists across sessions for same user), `app:` (shared across all users), no prefix (session-only) |
| **Runner** | Orchestrates agent execution: manages the LLM call loop, tool dispatch, and event streaming |
| **InvocationContext** | Runtime context passed to tools — contains session, state, and agent metadata |

### State Scoping — The `user:`/`app:` Prefix System

ADK uses a clever naming convention for state persistence:
```python
session.state["temp_data"] = "gone on session end"     # session-scoped
session.state["user:name"] = "Alice"                    # persists for this user across sessions
session.state["app:config"] = {"max_retries": 3}       # shared globally across all users
```

This is similar to browser storage: session-scoped = sessionStorage, `user:` = localStorage, `app:` = server config.

### Tool Schema Generation

ADK converts Python function signatures into JSON Schema for the LLM:

```python
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name (e.g., "London")
        units: Temperature units — "celsius" or "fahrenheit"
    """
    ...
```

The LLM sees this as a structured tool it can call with `{"city": "London", "units": "celsius"}`. **Type hints are mandatory** — they define the schema. **Docstrings are mandatory** — they tell the LLM when and how to use the tool.

### Session Services

| Service | Use Case | Persistence |
|---------|----------|-------------|
| `InMemorySessionService` | Local development / prototyping | Lost on restart |
| `DatabaseSessionService` | Production (SQLite, PostgreSQL) | Persistent |
| `VertexAISessionService` | Google Cloud managed | Fully managed |

### ADK Web UI

ADK ships with a built-in debugging interface (`adk web`) that provides:
- Interactive chat with the agent
- **Trace tab** with 4 views: Event, Request (sent to LLM), Response (from LLM), Graph (tool call flow)
- Session inspection and state viewer
- Real-time event streaming

### Where ADK Fits in the Ecosystem

| Framework | Philosophy | Best For |
|-----------|-----------|----------|
| **Google ADK** | Code-first, hierarchical agents | Production apps, Google Cloud users |
| **LangGraph** | Graph-based FSM | Complex conditional workflows |
| **CrewAI** | Role-based teams | Rapid prototyping |
| **DSPy** | Prompt optimization | ML pipelines |

ADK differentiates itself with: native Google Cloud integration, multi-language support (Python/TS/Go/Java), built-in eval tools, and one-command deployment (`adk deploy cloud_run`).

### Using ADK with Ollama (Local Models)

ADK is model-agnostic via LiteLLM. For local development:
- Run Ollama in Docker (serves models on port 11434)
- Use the `ollama_chat/` prefix for model names (NOT `ollama/` — causes infinite tool loops)
- Example: `model="ollama_chat/qwen2.5:7b"`

**Critical caveat:** Local models (7B-14B) have weaker tool-calling ability than Gemini/GPT-4. Exercises are designed to work with smaller models, but tool-calling reliability depends on model quality.

## Description

This practice introduces the fundamentals of Google ADK by building a single-agent system with custom tools and session management. You'll:

1. Set up ADK with a local Ollama model (no cloud APIs needed)
2. Create custom tools using Python type hints + docstrings
3. Understand the agent execution loop (prompt → LLM → tool call → result → LLM → response)
4. Manage conversation state with session-scoped, user-scoped, and app-scoped variables
5. Debug agent behavior using the ADK Web UI trace viewer
6. Build a practical assistant agent that maintains state across interactions

## Instructions

### Exercise 1: Environment Setup & First Agent

**What you'll learn:** ADK project structure, agent creation, and local model integration.

ADK expects a specific project layout:
```
my_agent/
  __init__.py      # Must export `root_agent`
  agent.py         # Agent definition
```

**Steps:**
1. Start the Ollama Docker container
2. Pull a model suitable for tool calling (qwen2.5:7b recommended)
3. Create the ADK project structure with `__init__.py` exporting `root_agent`
4. Define a minimal agent with `name`, `model`, `instructions`, and `description`
5. Test it via `adk web` — verify it responds to basic prompts
6. Inspect the Trace tab to understand the request/response cycle

### Exercise 2: Custom Tool Definition

**What you'll learn:** How ADK converts Python functions into LLM-callable tools, and how the LLM decides when to use them.

ADK's tool system is deceptively simple: write a normal Python function with type hints and a Google-style docstring, and ADK generates the JSON schema the LLM uses to decide whether and how to call it.

**Steps:**
1. Create 2-3 tools for a task manager agent (e.g., add task, list tasks, mark complete)
2. Register tools with the agent via the `tools=` parameter
3. Test that the LLM correctly invokes tools based on natural language
4. Observe in the Trace tab how tool schemas are sent to the LLM and how it formats the call

### Exercise 3: Session & State Management

**What you'll learn:** How ADK tracks conversation history and persists state across different scopes.

State management is where ADK shines over raw LLM calls. The `user:` prefix pattern lets you build agents that "remember" user preferences across conversations without a separate database.

**Steps:**
1. Set up `InMemorySessionService` with user/session creation
2. Use `Runner` to execute the agent within a session context
3. Implement state reading/writing in tools using `tool_context.state`
4. Test the three state scopes: session-only, `user:` prefixed, `app:` prefixed
5. Verify that `user:` state persists when creating a new session for the same user

### Exercise 4: Putting It Together — Stateful Assistant

**What you'll learn:** Combining tools and state into a practical agent that demonstrates the full ADK execution model.

**Steps:**
1. Build a complete task-management assistant that:
   - Uses tools to add/list/complete tasks
   - Persists tasks in `user:` state (survives across sessions)
   - Tracks app-wide statistics in `app:` state
2. Test multi-turn conversations where the agent uses context from previous turns
3. Create a new session for the same user and verify task persistence
4. Use the ADK Web UI to trace the full execution flow

## Motivation

- **Industry adoption**: Google ADK is gaining traction as the "FastAPI of AI agents" — production-ready, well-documented, backed by Google
- **Complements existing skills**: You've done LangChain/LangGraph (029a/b) — ADK offers a different architectural perspective (hierarchical vs graph-based)
- **Production relevance**: ADK's state management and session system directly apply to building AI features at AutoScheduler.AI
- **Model-agnostic**: Works with any LLM, so the patterns transfer regardless of which model provider you use

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `docker compose up -d` | Start Ollama container |
| Setup | `docker exec ollama ollama pull qwen2.5:7b` | Pull local model for tool calling |
| Setup | `uv sync` | Install Python dependencies |
| Dev | `uv run adk web .` | Launch ADK Web UI for interactive testing |
| Dev | `uv run python main.py` | Run agent programmatically (CLI mode) |
| Teardown | `docker compose down -v` | Stop and clean up containers |
