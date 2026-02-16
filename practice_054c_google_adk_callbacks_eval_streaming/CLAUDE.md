# Practice 054c — Google ADK: Callbacks, Evaluation & Streaming

## Technologies

- **Google ADK** — callbacks, evaluation framework, streaming, and memory service
- **LiteLLM** — model proxy for Ollama
- **Ollama** — local LLM inference

## Stack

- Python 3.11+
- Docker (Ollama)
- `google-adk`, `litellm`

## Theoretical Context

### Callbacks — Hooks Into Agent Execution

Callbacks let you observe, modify, or block agent behavior at key points in the execution lifecycle. They're ADK's equivalent of middleware in web frameworks.

**The problem they solve:** Production agents need logging, content moderation, rate limiting, and dynamic data injection — without cluttering the agent's core logic. Callbacks cleanly separate cross-cutting concerns.

#### Callback Types

| Callback | When it fires | Use cases |
|----------|--------------|-----------|
| `before_model_callback` | Before sending request to LLM | Prompt modification, logging, cost tracking |
| `after_model_callback` | After receiving LLM response | Content filtering, response transformation |
| `before_tool_callback` | Before executing a tool | Permission checks, argument validation, logging |
| `after_tool_callback` | After tool returns result | Result transformation, caching, audit logging |

#### Callback Signatures

```python
def before_tool_callback(
    callback_context: CallbackContext,   # session, state access
    tool_name: str,                      # which tool is being called
    args: dict,                          # arguments the LLM chose
) -> Optional[dict]:
    # Return None to proceed normally
    # Return a dict to short-circuit (skip tool, use this as result)
    pass

def after_tool_callback(
    callback_context: CallbackContext,
    tool_name: str,
    args: dict,
    result: dict,                        # what the tool returned
) -> Optional[dict]:
    # Return None to use original result
    # Return a dict to replace the result
    pass
```

**Key insight:** Returning a non-None value from `before_tool_callback` **skips the tool entirely** and uses the returned value as the result. This enables mocking, caching, and permission denial.

### Evaluation — Testing Agent Quality

Agents are non-deterministic — the same prompt may produce different tool call sequences or responses. Traditional unit tests don't work well. ADK provides a structured evaluation system.

#### Golden Datasets

A "golden dataset" is a collection of verified correct interactions:

```json
{
  "prompt": "What's the weather in London?",
  "expected_tool_calls": [
    {"tool": "get_weather", "args": {"city": "London"}}
  ],
  "expected_response_contains": ["London", "temperature"]
}
```

Golden datasets serve as regression tests — when you change prompts, tools, or models, you verify the agent still handles known cases correctly.

#### Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Tool use accuracy** | Did the agent call the right tools with correct arguments? |
| **Response quality** | Does the response contain expected information? (substring/regex match) |
| **Rubric evaluation** | LLM-as-judge scores the response against criteria |
| **Trajectory match** | Did the agent follow the expected sequence of actions? |

#### ADK Web UI for Evaluation

The Web UI's Trace tab shows:
- **Event view**: Raw events (tool calls, LLM responses)
- **Request view**: Exact payload sent to the LLM (including tool schemas)
- **Response view**: Raw LLM output
- **Graph view**: Visual flow of tool calls and decisions

This is invaluable for understanding *why* an agent made a specific decision.

### Streaming — Real-Time Responses

By default, ADK waits for the full LLM response before returning. Streaming delivers tokens incrementally as they're generated.

```python
async for event in runner.run_async(
    user_id="user1",
    session_id="session1",
    new_message=Content(parts=[Part(text="Hello")])
):
    if event.text:
        print(event.text, end="", flush=True)
    if event.tool_call:
        print(f"\n[Calling {event.tool_call.name}]")
```

**Streaming + Callbacks caveat:** There's a known issue where grounding metadata may not be fully populated when `after_model_callback` fires in streaming mode. Non-streaming callbacks work correctly.

### Memory Service — Long-Term Knowledge

While **State** persists key-value pairs, **Memory** provides semantic search over past interactions:

```python
from google.adk.memory import InMemoryMemoryService

memory_service = InMemoryMemoryService()
# Memory accumulates from completed sessions
# Agent can search: "What did the user say about their preferences?"
```

Memory enables agents to recall information from past conversations without explicit state management — useful for personalization and context continuity.

| Feature | State | Memory |
|---------|-------|--------|
| **Structure** | Key-value pairs | Free-text, semantically searchable |
| **Access** | Direct key lookup | Search query |
| **Scope** | Session / User / App | Cross-session |
| **Use case** | Structured data (preferences, configs) | Unstructured recall ("what did we discuss?") |

### MCP (Model Context Protocol) Integration

ADK can connect to MCP servers — external services that provide tools and context to agents:

```python
from google.adk.tools.mcp_tool import MCPTool

mcp_tools = MCPTool.from_server("http://localhost:8080")
agent = Agent(tools=[*mcp_tools], ...)
```

MCP enables a plugin ecosystem — agents can dynamically discover and use tools from external servers without hardcoding them.

## Description

This practice covers ADK's production-readiness features:

1. Before/after callbacks for logging, content moderation, and dynamic injection
2. Callback-based tool caching and permission control
3. Golden dataset creation for agent evaluation
4. Evaluation metrics (tool accuracy, response quality)
5. Streaming responses for real-time UX
6. Memory service for cross-session knowledge

## Instructions

### Exercise 1: Before/After Tool Callbacks

**What you'll learn:** How to inject cross-cutting concerns (logging, validation, timing) without modifying agent or tool code.

Callbacks are the cleanest way to add observability to agents. They separate concerns: the tool does its job, the callback handles everything around it.

**Steps:**
1. Start with the task-management agent from 054a
2. Add a `before_tool_callback` that logs tool name, arguments, and timestamp
3. Add an `after_tool_callback` that logs results and execution duration
4. Test and observe the callback output in the console

### Exercise 2: Callback-Based Caching & Permission Control

**What you'll learn:** How returning a value from `before_tool_callback` short-circuits tool execution — enabling caching and access control.

This is one of ADK's most powerful patterns: the callback can prevent a tool from running and return a cached/denied result instead.

**Steps:**
1. Implement a caching callback that stores tool results by (tool_name, args) key
2. On cache hit, return the cached result (skipping the actual tool)
3. Implement a permission callback that blocks certain tools based on a `user:role` state
4. Test both: verify cached calls skip execution, verify blocked tools return denial messages

### Exercise 3: Golden Dataset & Evaluation

**What you'll learn:** How to create ground-truth test cases and evaluate agent quality systematically.

Agent evaluation is critical for production — you need to know if a prompt change broke existing behavior. Golden datasets are ADK's answer to this.

**Steps:**
1. Define 5-8 test cases as a golden dataset (prompt → expected tool calls → expected response)
2. Run the evaluation suite against your agent
3. Analyze results: which cases pass/fail and why
4. Use the Trace tab to debug failing cases
5. Iterate on agent instructions to improve evaluation scores

### Exercise 4: Streaming Responses

**What you'll learn:** How to stream agent responses token-by-token for a real-time user experience.

Streaming matters for UX — users see responses appear immediately instead of waiting for the full generation. It also enables progress indicators during tool execution.

**Steps:**
1. Implement async streaming using `runner.run_async()`
2. Handle different event types: text deltas, tool call start/end, errors
3. Build a simple CLI that displays streaming output with tool call indicators
4. Compare the UX difference between streaming and non-streaming modes

### Exercise 5: Memory Service

**What you'll learn:** How ADK's Memory service enables agents to recall information from past sessions semantically.

Memory complements State: State stores structured data by key, Memory stores unstructured knowledge searchable by meaning.

**Steps:**
1. Set up `InMemoryMemoryService`
2. Run several conversations that mention user preferences
3. Start a new session and ask the agent to recall preferences
4. Verify the agent retrieves relevant information from memory
5. Compare the behavior with and without memory service enabled

## Motivation

- **Production readiness**: Callbacks, evaluation, and streaming are what separate prototypes from production agents
- **Testing non-deterministic systems**: Golden datasets teach you how to test LLM-based systems — a skill gap most developers have
- **Observability**: Callback patterns apply to any agent framework, not just ADK
- **Real-time UX**: Streaming is table-stakes for AI products — users expect immediate feedback

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| Setup | `docker compose up -d` | Start Ollama container |
| Setup | `docker exec ollama ollama pull qwen2.5:7b` | Pull local model |
| Setup | `uv sync` | Install Python dependencies |
| Dev | `uv run adk web .` | Launch ADK Web UI with trace inspection |
| Dev | `uv run python main.py` | Run agent with callbacks and streaming |
| Test | `uv run python evaluate.py` | Run golden dataset evaluation |
| Teardown | `docker compose down -v` | Stop and clean up containers |
