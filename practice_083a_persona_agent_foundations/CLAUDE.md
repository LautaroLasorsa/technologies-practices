# Practice 083a -- Self-Evolving Persona Agent: Foundations

## Technologies

- **LangGraph** -- Graph-based framework for stateful agent workflows with conditional routing and state persistence
- **instructor** -- Structured LLM output extraction via Pydantic models, patching OpenAI-compatible clients
- **Ollama** -- Local LLM inference server with OpenAI-compatible API endpoint
- **Pydantic** -- Data validation and schema definition for all agent state

## Stack

Python 3.12, Docker (Ollama with `qwen2.5:3b`)

## Theoretical Context

### What Makes an AI Agent Feel Human

Human-likeness in conversational AI is not about intelligence -- it is about **behavioral texture**. Humans are inconsistent, emotional, forgetful, and opinionated. An agent that responds perfectly, instantly, and uniformly reads as mechanical regardless of how sophisticated its language model is. The gap between "capable AI" and "believable persona" lies in four dimensions:

1. **Imperfection signals**: Humans hedge ("hm, let me think..."), start sentences and restart them, occasionally use lowercase, and vary their verbosity based on mood. Perfect grammar and exhaustive answers are the strongest bot tells.
2. **Timing variance**: The single most reliable bot detector is **consistent response latency**. Humans take longer to answer complex questions and reply instantly to simple ones. A bot that takes 2 seconds for both is immediately suspicious. The "Human or Not" social Turing game (AI21 Labs, 2023) -- where 10M+ conversations were played -- found that players who focused on response timing achieved the highest detection accuracy. Consistent timing, perfect grammar, and over-explanation were the top three tells.
3. **Persona consistency**: A human personality has quirks, opinions, contradictions, and recurring phrases. An agent that shifts register or knowledge-domain mid-conversation breaks immersion. Consistency comes from a detailed **persona card** that constrains generation.
4. **Memory recall**: Humans remember what you told them. An agent that forgets your name between sessions, or fails to reference something you said 5 minutes ago, signals "stateless machine." Persistent memory -- even basic fact extraction -- dramatically increases perceived humanness.

**Source**: [Human or Not? A Social Turing Game](https://humanornot.so/blog/human-or-bot-ai-conversation-patterns) -- AI21 Labs, 2023. 10M+ games, 67% overall accuracy. Key finding: timing consistency and over-helpfulness are the strongest bot signals.

### Persona Design: Static Cards vs Dynamic Evolution

A **persona card** is a structured document defining who the agent is: name, backstory, speech patterns, emotional baseline, opinions, and forbidden behaviors. This is injected as the system prompt for every LLM call.

**Static personas** (this practice) define the card once and render it into every prompt. The persona doesn't change -- but the agent's *emotional state* and *memory* do, creating the illusion of a living character. This is sufficient for single-user assistants and companions.

**Dynamic personas** (practice 083b) allow the agent to modify its own persona card based on experience -- developing new opinions, softening traits, or acquiring new speech patterns. This requires a "Reflector" that periodically reviews conversation logs and proposes persona updates.

The key insight from persona design is that **specificity beats generality**. A persona described as "friendly and helpful" produces generic output. A persona described as "uses 'honestly' as a verbal tic, gets annoyed by vague questions, loves obscure history facts, types in lowercase when tired" produces memorable, consistent personality.

### Structured Introspection: Think -> Feel -> Respond

Standard LLM chat generates a response directly from the conversation. Structured introspection forces the model through an intermediate reasoning step: before producing the user-visible response, the model must output:

- **inner_thought**: What the agent is "thinking" about the user's message (never shown to user)
- **emotion**: The agent's current emotional state as a structured enum
- **response**: The actual user-facing text

This is implemented via the `instructor` library, which patches an OpenAI-compatible client (Ollama's `/v1` endpoint) to return Pydantic models instead of raw text. The model is constrained to output valid JSON matching the schema, and `instructor` handles validation and retries.

**Why this matters**: Making emotional state *explicit and structured* (rather than implicit in the prompt) means the agent's internal state is inspectable, loggable, and can drive downstream logic (like the emotional state machine). It also improves consistency -- the model must commit to an emotion *before* generating the response, rather than improvising mood mid-sentence.

**Source**: [instructor documentation](https://python.useinstructor.com/integrations/ollama/) -- structured output extraction over Ollama's OpenAI-compatible endpoint.

### Emotional State Machines

In most chatbots, "emotion" is a prompt afterthought: "respond empathetically" or "be cheerful." This produces shallow, inconsistent affect. A better architecture treats emotion as a **first-class state variable** that:

- **Persists across turns**: The agent doesn't reset to neutral after each response
- **Has transition rules**: Certain user behaviors (rudeness, curiosity, vulnerability) shift the agent's emotion in predictable directions
- **Influences generation**: Different emotions produce different response *shapes* -- not just different words. An annoyed agent gives shorter responses, skips pleasantries, and uses flat punctuation. An excited agent uses more exclamation marks, asks follow-up questions, and elaborates.

This is implemented as a LangGraph node that runs before the response generator. The `update_emotion` node examines the user's message and the agent's current emotion, then outputs a new emotion + intensity. Conditional edges route to different response templates based on the emotional state.

The emotion is *not* a sentiment analysis of the user -- it's the agent's *own* emotional reaction, filtered through its persona. A sarcastic persona might react to a compliment with suspicion; a warm persona might react to criticism with concern rather than defensiveness.

### Two-Tier Memory: In-Context vs Persistent

LLM agents have two memory systems:

1. **In-context memory** (conversation messages): Everything in the current chat window. Dies when the process ends. Limited by context window size (32k tokens for Qwen 2.5 3B).
2. **Persistent memory** (cross-session facts): Extracted facts about the user stored externally (in this practice, local JSON files). Survives restarts. Loaded at session start and prepended to the system prompt.

The key operation is **fact extraction**: after each user turn, the model is asked "What new facts about the user can be extracted from this message?" The extracted facts are merged into the persistent store. At session start, accumulated facts are formatted into a "What you know about this person" block in the system prompt.

This is deliberately simple -- no vector database, no embeddings, no RAG. For a single-user agent with a manageable fact set, a flat JSON file is sufficient, human-readable, and inspectable. The limitation (which motivates practice 083b) is that facts accumulate without curation -- the Reflector in 083b will prune, merge, and prioritize stored facts.

**Why file-based persistence**: For a single-user local agent, there is no concurrent access, no need for query optimization, and maximum value in human-readability. You can open `data/memory/user_facts.json` in any editor and see exactly what the agent "knows." This transparency is both a debugging tool and a learning aid.

### Small Model Capabilities: Qwen 2.5 3B

Qwen 2.5 3B (Alibaba, September 2024) is chosen for this practice because:

- **System prompt resilience**: Qwen 2.5 models are specifically hardened for diverse system prompts, making them reliable for role-play and persona implementation. The model follows character constraints more consistently than many larger models.
- **Structured output**: Even at 3B parameters, Qwen 2.5 produces reliable JSON output when constrained by `instructor`'s schema enforcement. The model was explicitly trained on structured generation tasks.
- **Context and generation**: 32k token context window, up to 8k token generation. Sufficient for multi-turn conversations with persona + memory prompt overhead.
- **Speed**: At 3B parameters, inference is fast enough for a responsive chat experience on CPU (Ollama), making timing-based humanization practical.
- **Resource-friendly**: Runs comfortably in Docker with ~4GB RAM, no GPU required.

**Source**: [Qwen 2.5 announcement](https://qwenlm.github.io/blog/qwen2.5/) -- "models are generally more resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots."

### File-Based Persistence Architecture

All agent state lives in local files under `data/`:

| File | Purpose | Format |
|------|---------|--------|
| `data/persona.json` | Persona card (identity, traits, speech patterns) | JSON object |
| `data/memory/user_facts.json` | Cross-session facts about the user | JSON array of fact objects |
| `data/logs/conversation_*.json` | Conversation transcripts | JSON array of turn objects |
| `data/config.json` | Model settings, timing parameters | JSON object |

This is sufficient for a single-user agent because: (1) no concurrent access -- only one process reads/writes, (2) human-readable -- you can inspect and edit agent state directly, (3) no setup required -- no database, no migrations, no Docker volumes beyond Ollama, (4) debuggable -- every piece of agent state is a file you can `cat`.

The limitation is scalability -- multiple users, concurrent sessions, or large fact stores would require a proper database. Practice 083b may introduce SQLite if needed.

## Description

Build a conversational AI agent with consistent personality, emotional state tracking, and persistent memory -- all running on a local 3B model. The agent demonstrates human-likeness through structured introspection (think -> feel -> respond), an emotional state machine that influences response style, two-tier memory (in-context + persistent file-based), and deliberate imperfection in response rendering.

### What you'll build

1. **File-based state management** -- A `FileStore` class handling all persistence: persona loading, user fact storage, conversation logging
2. **Persona system** -- JSON persona card rendered into dynamic system prompts with memory and emotional context
3. **Structured introspection** -- `instructor` over Ollama extracting `{emotion, inner_thought, response}` Pydantic models
4. **Emotional state machine** -- LangGraph graph where emotion is a state variable with transition rules and conditional response routing
5. **Persistent memory** -- Per-turn fact extraction about the user, stored in JSON files, loaded at session start
6. **Humanization layer** -- Post-processor adding timing delays, hesitation markers, and imperfection signals

### What you'll learn

- Persona design as character writing -- specificity over generality
- Structured generation as a tool for making agent internals explicit and inspectable
- Emotion as a first-class state variable that shapes response generation
- The difference between in-context memory and persistent memory
- Why imperfection signals (timing, hesitation, truncation) matter for human-likeness

## Instructions

### Setup

1. Start Ollama: `docker compose up -d`
2. Pull the model: `docker compose exec ollama ollama pull qwen2.5:3b`
3. Install dependencies: `uv sync`
4. Copy the example persona: `cp data/persona_example.json data/persona.json`
5. Run the verification: `uv run python -m src.main --verify`

### Exercise 1 -- File Architecture & Data Schemas (~20 min)

**File:** `src/file_store.py`

Design the complete file architecture for the persona agent: directory layout, file naming conventions, and read/write strategies. Then implement the `FileStore` class that encapsulates all file I/O.

This is a **design-first exercise** -- before writing code, you'll make architectural decisions about how agent state is organized on disk. The TODO(human) block at the top of `file_store.py` walks you through three design steps.

**What you'll design:**
- **Directory tree**: Flat vs nested, by-entity vs by-type, committed vs gitignored
- **File naming**: How conversation logs and user fact files are named (UUID? timestamp? hybrid?)
- **Read/write strategy**: How to handle missing files, whether to use full JSON rewrite or append-only JSONL, atomic writes vs direct writes

**What you'll implement:**
- Directory layout constants (`DATA_DIR`, `MEMORY_DIR`, `LOGS_DIR` — or your own structure)
- `FileStore` methods: `load_user_facts()`, `save_user_facts()`, `start_new_conversation()`, `append_conversation_turn()`
- The provided `load_persona()`, `save_persona()`, `load_config()` methods show the expected pattern

**Why it matters:** File organization is the first architectural decision in any stateful system. A good layout is debuggable (you can `ls data/logs/` and understand what happened), extensible (083b's Reflector needs to scan conversation logs), and self-documenting (directory names explain purpose). Bad layouts lead to "where does this file go?" confusion that compounds over time. The same principles apply whether you're organizing agent state, ML experiment artifacts, or microservice configs.

### Exercise 2 -- Persona Card Architecture (~15 min)

**File:** `src/persona.py`

Define a persona in `data/persona.json` and write the function that renders it into a dynamic system prompt. The system prompt is not just the persona card -- it includes current emotional state, known user facts, and behavioral constraints.

**What you'll implement:**
- `render_system_prompt(persona, emotion, user_facts)` -- Assembles the complete system prompt from persona card + emotional context + memory context

**Why it matters:** System prompt design is character writing. A vague persona ("be friendly") produces generic responses. A specific one (quirks, contradictions, opinions, verbal tics) produces memorable personality. The rendering function must weave together static identity (persona card), dynamic state (current emotion), and accumulated knowledge (user facts) into a coherent prompt that constrains the model without over-specifying.

### Exercise 3 -- Structured Introspection with `instructor` (~20 min)

**File:** `src/introspection.py`

Wire `instructor` over Ollama's OpenAI-compatible endpoint to produce structured `AgentTurn` responses. The model outputs `{emotion, inner_thought, response}` as validated Pydantic -- only `response` is shown to the user.

**What you'll implement:**
- `generate_agent_turn(client, model, system_prompt, messages)` -- Calls the instructor-patched client with the conversation history and returns a validated `AgentTurn`

**Why it matters:** Structured generation makes the agent's internal state explicit and inspectable. Instead of guessing what the model "feels," you can log the emotion and inner_thought for every turn. This is the foundation for the emotional state machine (Exercise 4) and conversation analysis (083b). The key design decision: should the model determine its own emotion, or should emotion be computed externally and fed back? (This exercise does the former; Exercise 4 adds the latter as a correction mechanism.)

### Exercise 4 -- Emotional State Machine in LangGraph (~25 min)

**File:** `src/emotional_fsm.py`

Build a LangGraph `StateGraph` where emotion is a first-class state variable. The `update_emotion` node examines the user's message and current emotional state to determine a new emotion. Conditional edges route to different response behaviors based on mood.

**What you'll implement:**
- `update_emotion(state)` -- Given last user message + current emotion, determine new emotion and intensity
- Conditional edge routing function that selects response behavior based on emotional state
- The complete LangGraph graph: `START -> update_emotion -> [response_engaged | response_annoyed | response_reflective | response_default] -> END`

**Why it matters:** Emotion as a state variable means the agent's mood persists across turns and influences generation *structurally* (different nodes, different prompts) rather than just lexically (different words in the same prompt). An annoyed agent doesn't just say angry things -- it gives shorter responses, drops pleasantries, and uses flat punctuation. An engaged agent elaborates, asks follow-ups, and uses more expressive language. The conditional routing makes these behavioral differences explicit and testable.

### Exercise 5 -- Two-Tier Memory (File-Based) (~20 min)

**File:** `src/memory.py`

Implement per-turn fact extraction and persistent storage. After each user message, the model extracts facts about the user (name, preferences, context). Facts are stored in JSON files and loaded at session start.

**What you'll implement:**
- `extract_user_facts(client, model, message)` -- Calls the model to extract structured facts from a user message
- `merge_facts(existing, new_facts)` -- Merges new facts into the existing fact store, handling updates and deduplication

**Why it matters:** This is the difference between a chatbot and a companion. Without persistent memory, every session starts from zero. With it, the agent accumulates knowledge about the user over time: "You mentioned you work in supply chain optimization," "Last time you were frustrated about a deployment." The extraction step teaches the model to identify *memorable* information, and the merge step teaches you to handle fact evolution (the user's job title might change).

### Exercise 6 -- Humanization Layer: Timing & Imperfection (~10 min)

**File:** `src/humanizer.py`

Post-process the agent's response to add human-like imperfection signals: typing delay proportional to length and emotion, truncation of over-explanation, and probabilistic injection of hesitation markers.

**What you'll implement:**
- `humanize_response(text, emotion, complexity)` -- Applies timing delay, truncation, and hesitation injection based on emotional state and message complexity

**Why it matters:** Human-likeness is partly a rendering concern. A perfectly formatted, instantly delivered response reads as mechanical regardless of content. Adding variable delays (longer when "thinking," shorter when "excited"), occasional hesitation markers ("hm," "well,"), and truncation of over-explanation creates behavioral texture that passes the "gut feel" test. This is the cheapest intervention with the highest impact on perceived humanness.

## Motivation

Understanding conversational AI design beyond task completion is increasingly important as AI companions, tutors, and assistants become mainstream products. The techniques in this practice -- structured introspection, emotional state machines, persistent memory, humanization -- transfer to any product involving sustained human-AI interaction.

Specific value:

- **Product design**: Character AI, Replika, Pi (Inflection) -- the companion/persona market is growing rapidly. Understanding the engineering behind believable agents is a differentiator.
- **Agent architecture**: LangGraph emotional FSM + instructor structured output is a production pattern for any agent that needs inspectable internal state.
- **Memory systems**: The two-tier memory pattern (in-context + persistent) appears in every serious agent framework (LangGraph checkpointers, MemGPT, Letta).
- **Small model engineering**: Making a 3B model behave convincingly teaches constraints-driven design -- you learn what matters (specificity, structure) vs what doesn't (parameter count).

## Commands

All commands run from `practice_083a_persona_agent_foundations/`.

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container with persistent volume |
| | `docker compose exec ollama ollama pull qwen2.5:3b` | Download the Qwen 2.5 3B model into Ollama |
| | `docker compose down` | Stop and remove the Ollama container |
| | `docker compose logs -f ollama` | Stream Ollama container logs |
| **Setup** | `uv sync` | Install Python dependencies from pyproject.toml |
| | `cp data/persona_example.json data/persona.json` | Copy example persona card as starting point |
| | `uv run python -m src.main --verify` | Verify Ollama connectivity, instructor patching, and file structure |
| **Exercises** | `uv run python -m src.main` | Start the interactive chat loop (uses all exercises) |
| | `uv run python -m src.main --user-id alice` | Start chat with a specific user ID for memory isolation |
| | `uv run python -m src.main --show-internals` | Chat with visible inner_thought and emotion (debug mode) |
| **Testing** | `uv run python -m src.file_store` | Test file store operations in isolation |
| | `uv run python -m src.persona` | Test persona loading and system prompt rendering |
| | `uv run python -m src.introspection` | Test structured introspection with a single message |
| | `uv run python -m src.emotional_fsm` | Test emotional state machine transitions |
| | `uv run python -m src.memory` | Test fact extraction and persistence |
| | `uv run python -m src.humanizer` | Test humanization post-processing |

## LLM Configuration

The agent supports multiple LLM providers via environment variables. Copy `.env.example` to `.env` and set the relevant vars before running.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | Provider name: `ollama`, `lmstudio`, `openai`, `anthropic` |
| `LLM_MODEL` | `qwen3:8b` | Model identifier for the chosen provider |
| `LLM_BASE_URL` | _(provider default)_ | Override the provider's API base URL |
| `LLM_API_KEY` | _(empty)_ | API key — required for `openai` and `anthropic` |

**Provider defaults:**

| Provider | Default base URL | Requires key |
|----------|-----------------|-------------|
| `ollama` | `http://localhost:11434/v1` | No |
| `lmstudio` | `http://localhost:1234/v1` | No |
| `openai` | _(SDK default)_ | Yes |
| `anthropic` | _(SDK default)_ | Yes — also needs `uv add anthropic` |

All wiring lives in `src/llm_config.py`. With no env vars set, the agent behaves exactly as before (Ollama on localhost).

## State

`not-started`
