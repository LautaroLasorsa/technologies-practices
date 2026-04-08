# Practice 083b -- Self-Evolving Persona Agent: ACE Loop & Self-Improvement

## Technologies

- **LangGraph** -- Graph-based framework for stateful agent workflows with conditional routing and state persistence
- **instructor** -- Structured LLM output extraction via Pydantic models, patching OpenAI-compatible clients
- **sentence-transformers** -- Lightweight embedding models for semantic similarity (deduplication of playbook entries)
- **Ollama** -- Local LLM inference server with OpenAI-compatible API endpoint
- **Pydantic** -- Data validation and schema definition for all agent and playbook state

## Stack

Python 3.12, Docker (Ollama with `qwen2.5:3b`)

## Theoretical Context

### The Problem: Why Agents Plateau

Practice 083a built a persona agent with static behavior -- the persona card, emotional state machine, and memory system never evolve based on conversational experience. If the agent's persona feels wooden after 50 conversations, it stays wooden forever. The only path to improvement is manual editing by the developer.

This is the general problem of **context adaptation**: how do you make an agent improve its own instructions and strategies based on experience, without retraining the underlying model?

Naive approaches fail in predictable ways. The two most common failure modes are **brevity bias** and **context collapse**, both identified and formalized in the ACE paper.

**Source**: [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618) -- Zhang et al., ICLR 2026.

### Brevity Bias: The Optimizer's Trap

When you ask an LLM to "optimize" or "improve" a prompt, it reliably gravitates toward shorter, more generic instructions. This is brevity bias -- the tendency for iterative prompt optimization to collapse toward concise platitudes that lose domain-specific detail.

Example from the ACE paper: iterative optimization methods produced near-identical generic instructions like *"Create unit tests to ensure methods behave as expected"* -- erasing task-specific strategies that had accumulated over dozens of iterations. The optimizer sacrifices diversity and domain knowledge for apparent "clarity."

Why this happens: LLMs trained on instruction-following data learn that shorter, clearer instructions are generally "better." When asked to rewrite a long, detailed playbook, the model applies this bias -- stripping out specific, hard-won strategies in favor of tidy generalizations.

For persona agents specifically: a rich persona description with quirks, contradictions, and specific behavioral rules will collapse into "be friendly and engaging" after a few optimization rounds. The very specificity that makes a persona feel human is what the optimizer removes first.

### Context Collapse: The Catastrophic Forgetting of Prompts

Context collapse is the more severe failure mode: when an LLM is asked to repeatedly rewrite its entire accumulated context, it can suffer catastrophic information loss in a single step.

The ACE paper documents a striking example on the AppWorld benchmark: at iteration 60, the accumulated context had grown to **18,282 tokens** with **66.7% task accuracy**. At iteration 61, a monolithic rewrite collapsed this to just **122 tokens** with **57.1% accuracy** -- worse than the **63.7% baseline** with no adaptation at all. Sixty iterations of accumulated knowledge, destroyed in one rewrite.

This is not gradual degradation -- it is cliff-edge failure. The model decides that its accumulated context is "too long" and aggressively summarizes, losing specific strategies, edge-case handling, and domain knowledge that took dozens of iterations to discover.

For persona agents: imagine 50 conversations worth of learned emotional patterns and behavioral strategies, wiped out because the model decided to "clean up" the playbook into three bullet points.

### ACE: Agentic Context Engineering

ACE (Agentic Context Engineering) is a framework introduced at ICLR 2026 that solves both brevity bias and context collapse through a simple architectural insight: **never let the LLM rewrite the entire context**. Instead, treat the context as a structured document (the "playbook") that evolves through small, incremental delta updates.

ACE achieves this via a three-role architecture with strict separation of concerns:

1. **Generator**: Executes tasks using the current playbook as guidance. Produces reasoning trajectories -- the raw experience data. The Generator never modifies the playbook directly.

2. **Reflector**: Analyzes completed tasks (successes and failures) to extract concrete lessons. Given pairs of traces -- one successful, one failed -- the Reflector identifies what differed and outputs structured delta lessons. The Reflector never modifies the playbook directly.

3. **Curator**: The only role that touches the playbook. Takes delta lessons from the Reflector and integrates them through deterministic operations: add new entries, update counters on existing entries, merge semantically similar entries, and prune entries that have proven harmful. The Curator never generates new strategies -- it only organizes what the Reflector extracted.

This separation prevents any single LLM call from seeing (and rewriting) the entire playbook. The Generator sees the playbook as read-only context. The Reflector sees conversation traces, not the playbook. The Curator sees individual delta lessons and the playbook section they target, never the full playbook at once.

**Performance**: ACE consistently outperforms baselines: +10.6% on agent benchmarks and +8.6% on finance tasks. On the AppWorld leaderboard, ACE matches the top-ranked production agent and surpasses it on harder challenge splits, despite using a smaller open-source model.

**Source**: [ACE paper](https://arxiv.org/abs/2510.04618), [ACE GitHub repository](https://github.com/ace-agent/ace), [VentureBeat coverage](https://venturebeat.com/ai/ace-prevents-context-collapse-with-evolving-playbooks-for-self-improving-ai).

### The Playbook: Structured Text as Knowledge Base

The playbook is ACE's central data structure -- a structured text document organized into sections, where each section contains itemized entries (bullets) with metadata. The format is deliberately human-readable (markdown with conventions) rather than JSON or a database, because:

1. **Inspectable**: You can open the playbook in any text editor and immediately understand what the agent has learned
2. **Diffable**: Git diff shows exactly what changed between iterations
3. **Editable**: A human can manually add, remove, or modify entries
4. **LLM-friendly**: Markdown is a natural format for LLM consumption and generation

Each entry has:
- **Unique identifier**: `[strat-00001]`, `[emot-00003]`, `[avoid-00002]` -- enables precise references
- **Counters**: `helpful=N harmful=M` -- lightweight fitness tracking without complex scoring
- **Content**: The actual strategy, pattern, or rule in natural language

Sections group entries by type (strategies, emotional patterns, mistakes to avoid). This structure enables targeted operations: the Curator can update a specific section without touching others.

### Delta Updates vs Monolithic Rewrites

The core mechanism that prevents context collapse is **delta updates**: instead of rewriting the entire playbook, the Curator produces small sets of new or updated entries and merges them deterministically.

**Monolithic rewrite** (what Exercise 2 demonstrates failing):
- Ask the LLM: "Here is the current playbook and 10 new conversations. Rewrite the playbook."
- The LLM sees the full playbook + conversations and produces a complete replacement
- Risk: brevity bias, context collapse, loss of specific strategies

**Delta update** (ACE's approach):
- The Reflector extracts specific lessons: "In conversation 7, transitioning from playful to serious mid-sentence felt jarring"
- The Curator adds this as a new entry: `[emot-00004] helpful=0 harmful=0 :: Transition between playful and serious via a neutral beat, not mid-sentence`
- Existing entries are untouched; only the delta is applied
- The playbook grows monotonically (until pruning removes proven-harmful entries)

This is analogous to the difference between `git rebase --squash` (monolithic) and `git commit` (incremental). Incremental commits preserve history; squashing risks losing context.

### Semantic Deduplication

As the playbook grows through delta updates, it accumulates redundant entries -- different phrasings of the same insight. Without deduplication, the playbook bloats and wastes context window tokens.

ACE uses **embedding-based semantic similarity** to detect near-duplicate entries. The process:
1. Embed all entries in a section using a sentence-transformer model
2. Compute pairwise cosine similarity
3. For pairs above a threshold (e.g., 0.85), merge: combine counters, keep the more specific wording
4. Replace both entries with the merged result

This is more robust than string matching -- "don't use bullet points in responses" and "avoid enumerated lists when speaking" are semantically similar but lexically different. Embedding similarity catches these.

Model choice: `all-MiniLM-L6-v2` -- 22M parameters, produces 384-dimensional embeddings, runs fast on CPU. Sufficient for sentence-level similarity without GPU requirements.

**Source**: [Sentence-Transformers documentation](https://www.sbert.net/docs/pretrained_models.html) -- all-MiniLM-L6-v2 achieves 68.06% on STS benchmark, best quality/speed tradeoff for its size.

### Counter-Based Pruning: Lightweight Fitness Signals

Each playbook entry tracks two counters:
- **helpful**: Incremented when the entry's advice was followed and the conversation scored well
- **harmful**: Incremented when the entry's advice was followed and the conversation scored poorly

Pruning rule: remove entries where `harmful > helpful * 2`. This is deliberately conservative -- an entry needs to be demonstrably harmful (not just unhelpful) to be removed. New entries start at `helpful=0 harmful=0` and accumulate signal over time.

This is a lightweight alternative to complex scoring functions. It requires no gradient computation, no reward model, and no human feedback loop -- just automated comparison between conversation quality and playbook adherence.

### Adaptation for Persona Agents

The original ACE paper targets task completion (coding, API usage). For persona agents, we adapt the framework:

| ACE (Original) | ACE (Persona Adaptation) |
|-----------------|--------------------------|
| Task accuracy (pass/fail) | Human-likeness score (rubric from 083a) |
| Code strategies | Conversation strategies |
| API patterns | Emotional patterns |
| Bug patterns | Mistakes to avoid (bot tells) |
| Offline batch optimization | Online after each conversation batch |

The playbook sections become:
- **STRATEGIES & INSIGHTS**: Conversation techniques that make the persona feel natural
- **EMOTIONAL PATTERNS**: Rules about emotional transitions, timing, intensity
- **MISTAKES TO AVOID**: Specific bot tells and anti-patterns to suppress

The Reflector receives pairs of conversations -- one where the persona felt natural, one where it felt robotic -- and identifies behavioral differences rather than correctness differences.

### Online vs Offline Adaptation

ACE supports two modes:

- **Offline**: Optimize the playbook on a training set of conversations, then evaluate on a held-out test set. Good for initial playbook development.
- **Online**: After each batch of conversations, reflect and curate. The playbook evolves in real-time. Good for continuous improvement during deployment.

This practice implements online adaptation: run N conversations, reflect on the batch, curate the playbook, repeat. Each iteration's playbook is snapshotted for learning curve analysis.

## Description

Extend 083a's persona agent with ACE's (Agentic Context Engineering) self-improvement mechanism. Instead of a static persona that never learns from experience, build a three-role system (Generator, Reflector, Curator) that accumulates behavioral strategies in an evolving playbook. Experience Exercise 2's context collapse firsthand (naive full rewrite destroys accumulated knowledge), then implement ACE's delta-update architecture that prevents it. By the end, the agent demonstrably improves its human-likeness score over successive conversation batches.

### What you'll build

1. **Playbook data structure** -- Markdown-based knowledge store with sections, unique IDs, and helpful/harmful counters
2. **Context collapse baseline** -- Naive full-rewrite updater that demonstrates the failure mode ACE solves
3. **Reflector** -- Conversation analysis role that extracts delta lessons from natural vs robotic conversation pairs
4. **Curator** -- Knowledge management role with merge, semantic dedup, counter update, and prune operations
5. **ACE loop** -- Full Generator + Reflector + Curator iteration with snapshots and learning curve plotting

### What you'll learn

- Why monolithic context rewrites fail (brevity bias, context collapse)
- The ACE three-role architecture and its separation of concerns
- Structured text (markdown with conventions) as a lightweight knowledge base
- Embedding-based semantic deduplication for knowledge management
- Counter-based fitness tracking without reward models
- Online self-improvement through incremental context evolution

## Instructions

### Prerequisites

This practice builds on **083a** (Self-Evolving Persona Agent: Foundations). You need the concepts from 083a (persona cards, structured introspection, emotional state machine, memory) but NOT a running 083a implementation -- this practice is self-contained.

### Setup

1. Start Ollama: `docker compose up -d`
2. Pull the model: `docker compose exec ollama ollama pull qwen2.5:3b`
3. Install dependencies: `uv sync`
4. Verify setup: `uv run python -m src.main --verify`

### Exercise 1 -- Playbook Data Structure & Format (~15 min)

**File:** `src/playbook.py`

Design and implement the `Playbook` class that parses and serializes a structured markdown playbook. The playbook is the central knowledge store for ACE -- every strategy, pattern, and mistake the agent learns is stored here as a uniquely-identified entry with fitness counters.

**What you'll implement:**
- `Playbook.parse(text)` -- Parse markdown text into structured sections and entries
- `Playbook.serialize()` -- Render the playbook back to markdown format
- `Playbook.add_entry(section, content)` -- Add a new entry with auto-generated ID
- `Playbook.update_counters(entry_id, helpful_delta, harmful_delta)` -- Update fitness counters
- `Playbook.query_section(section)` -- Retrieve all entries in a section

**Why it matters:** The playbook format is deliberately text-based (not JSON, not a database) because it must be: (1) human-readable for inspection and manual editing, (2) diffable via git to see what changed between iterations, (3) directly consumable as LLM context without serialization. The parse/serialize round-trip must be lossless -- parsing a playbook and re-serializing it should produce identical text. This teaches structured text as a data format -- a pattern used in many knowledge management systems.

### Exercise 2 -- Witness Context Collapse (~15 min)

**File:** `src/baseline.py`

Implement a naive "full rewrite" context updater: after each batch of conversations, ask the model to rewrite the entire playbook from scratch. Run 3 iterations and observe the playbook shrink from rich, specific strategies to generic platitudes.

**What you'll implement:**
- `NaiveRewriter.rewrite(current_playbook, conversations)` -- Ask the LLM to rewrite the entire playbook given new conversation data

**Why it matters:** This exercise exists to make the problem visceral. The ACE paper shows that at iteration 60 on AppWorld, a monolithic rewrite collapsed 18,282 tokens of accumulated knowledge to 122 tokens, dropping accuracy from 66.7% to 57.1% (below the 63.7% no-adaptation baseline). You'll witness a scaled-down version: a playbook with 5-10 specific persona strategies collapsing to 2-3 generic bullets after just 3 rewrites. This is the fundamental problem ACE solves -- and experiencing it firsthand makes the solution meaningful.

### Exercise 3 -- Reflector: Conversation Analysis (~25 min)

**File:** `src/reflector.py`

Implement the Reflector role: given conversation transcripts, prompt the model to extract delta lessons about persona behavior. The Reflector receives pairs of conversations -- one where the persona felt natural, one where it felt robotic -- and identifies what differed.

**What you'll implement:**
- `Reflector.extract_lessons(natural_conversation, robotic_conversation)` -- Analyze a pair of conversations and extract delta lessons about what made the natural one work and the robotic one fail

**Why it matters:** The Reflector embodies ACE's separation of concerns -- it only extracts insights, never modifies the playbook. This constraint is critical: if the Reflector could edit the playbook directly, it would face the same full-rewrite temptation that causes context collapse. By restricting it to producing `DeltaLesson` objects (section, content, confidence), the architecture ensures that knowledge extraction is decoupled from knowledge management. The prompt engineering here is key: you must guide the model to produce specific, actionable lessons ("transition between emotions via a neutral beat") rather than vague observations ("the natural one was better").

### Exercise 4 -- Curator: Merge, Dedup & Prune (~25 min)

**File:** `src/curator.py`

Implement the Curator role -- the only component that modifies the playbook. The Curator performs four operations: merge (add new entries), semantic dedup (find and combine similar entries using embeddings), counter update (track entry fitness), and prune (remove harmful entries).

**What you'll implement:**
- `Curator.merge_lessons(playbook, lessons)` -- Add delta lessons as new playbook entries
- `Curator.deduplicate(playbook, threshold)` -- Find semantically similar entries and merge them
- `Curator.update_counters(playbook, conversation_scores)` -- Increment helpful/harmful based on conversation quality
- `Curator.prune(playbook)` -- Remove entries where harmful > helpful * 2

**Why it matters:** The Curator is where ACE's knowledge management happens. Each operation addresses a specific failure mode: merge prevents knowledge loss (new insights are always added, never replacing old ones), dedup prevents bloat (semantically redundant entries are combined), counter update provides fitness signal (which entries actually help?), and prune removes proven-bad advice. The semantic dedup step uses sentence-transformer embeddings to detect near-duplicates that string matching would miss -- "don't use bullet points" and "avoid enumerated lists" are the same advice phrased differently. The 0.85 cosine similarity threshold balances aggressive dedup (might merge distinct strategies) against conservative dedup (allows bloat).

### Exercise 5 -- Online Self-Improvement Loop (~10 min)

**File:** `src/ace_loop.py`

Wire Generator + Reflector + Curator into a complete iteration loop: run N conversations, reflect on the batch, curate the playbook, snapshot, and repeat. After all iterations, plot the learning curve comparing ACE against the Exercise 2 baseline.

**What you'll implement:**
- `ACELoop.run_iteration(playbook)` -- Execute one full ACE cycle: generate conversations, reflect, curate, snapshot
- `ACELoop.run(num_iterations)` -- Run multiple iterations and collect metrics for the learning curve

**Why it matters:** This exercise wires the three roles together and demonstrates online self-improvement. After each iteration, the playbook is snapshotted (saved to `data/snapshots/`) and conversations are scored on the human-likeness rubric. Plotting the learning curve shows whether ACE's incremental approach actually improves the agent over time -- and comparing against the Exercise 2 baseline shows the cost of monolithic rewrites. The key insight: ACE's playbook should grow monotonically in quality (with occasional dips from bad lessons, corrected by pruning), while the baseline oscillates and eventually collapses.

## Motivation

ACE (Agentic Context Engineering) is a cutting-edge technique from ICLR 2026 for making agents self-improve without retraining. It addresses a fundamental limitation of current agent architectures: they don't learn from experience at the prompt level. Applied to persona agents, this practice teaches:

- **Structured knowledge management**: How to accumulate, organize, and prune domain knowledge in a format that's both human-readable and LLM-consumable
- **Incremental context engineering**: Why delta updates preserve knowledge while monolithic rewrites destroy it -- a lesson applicable to any system that maintains evolving prompts or instructions
- **Separation of concerns in agent architectures**: The Generator/Reflector/Curator split is a general pattern for any system where experience should feed back into behavior
- **Embedding-based deduplication**: A practical application of sentence embeddings beyond RAG -- using similarity for knowledge management rather than retrieval

These techniques are relevant to any system where agents need to accumulate domain knowledge over time: customer support bots that learn from resolved tickets, coding assistants that learn project conventions, tutoring agents that adapt to student patterns.

## Commands

All commands run from `practice_083b_persona_agent_ace/`.

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container with persistent volume |
| | `docker compose exec ollama ollama pull qwen2.5:3b` | Download the Qwen 2.5 3B model into Ollama |
| | `docker compose down` | Stop and remove the Ollama container |
| | `docker compose logs -f ollama` | Stream Ollama container logs |
| **Setup** | `uv sync` | Install Python dependencies from pyproject.toml |
| | `uv run python -m src.main --verify` | Verify Ollama connectivity, embedding model, and file structure |
| **ACE Loop** | `uv run python -m src.main --iterations 3` | Run 3 ACE iterations (generate, reflect, curate per iteration) |
| | `uv run python -m src.main --iterations 5 --batch-size 4` | Run 5 iterations with 4 conversations per batch |
| | `uv run python -m src.main --baseline --iterations 3` | Run 3 iterations using the naive baseline (Exercise 2) for comparison |
| **Inspection** | `uv run python -m src.playbook` | Display current playbook contents with entry counts and token stats |
| | `uv run python -m src.playbook --seed` | Reset playbook to initial seed state |
| | `uv run python -m src.main --plot` | Plot learning curve from saved snapshots and scores |
| **Testing** | `uv run python -m src.playbook` | Test playbook parse/serialize round-trip (Exercise 1) |
| | `uv run python -m src.baseline` | Run naive rewrite baseline and show collapse (Exercise 2) |
| | `uv run python -m src.reflector` | Test reflector on sample conversation pairs (Exercise 3) |
| | `uv run python -m src.curator` | Test curator merge/dedup/prune operations (Exercise 4) |
| | `uv run python -m src.ace_loop` | Run a single ACE iteration for testing (Exercise 5) |
| | `uv run python -m src.evaluator` | Score sample conversations with the human-likeness rubric |
| **Cleanup** | `python clean.py` | Remove generated data, caches, and Docker volumes |

## LLM Configuration

By default the practice uses a local **Ollama** instance (started via `docker compose up -d`). You can switch to any OpenAI-compatible provider — or Anthropic — without touching any source file.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` |
| `LLM_MODEL` | `qwen2.5:3b` | Model name passed to the provider |
| `LLM_BASE_URL` | *(provider default)* | Override the API base URL |
| `LLM_API_KEY` | *(empty)* | Required for cloud providers |

### Quick-start for alternative providers

```bash
# Copy the example and edit
cp .env.example .env

# OpenAI
LLM_PROVIDER=openai LLM_MODEL=gpt-4o LLM_API_KEY=sk-... uv run python -m src.main --verify

# Anthropic (requires: uv add anthropic)
LLM_PROVIDER=anthropic LLM_MODEL=claude-3-5-sonnet-20241022 LLM_API_KEY=sk-ant-... uv run python -m src.main --verify

# LM Studio (local)
LLM_PROVIDER=lmstudio LLM_MODEL=<model-name> uv run python -m src.main --verify
```

The factory lives in `src/llm_config.py`. All source files import `get_openai_client()` / `get_instructor_client()` from there rather than constructing clients directly.

## State

`not-started`
