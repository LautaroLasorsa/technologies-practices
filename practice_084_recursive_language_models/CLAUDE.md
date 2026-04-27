# Practice 084 — Recursive Language Models

## Technologies

- **Recursive Language Models (RLMs)** — Inference paradigm proposed by Zhang, Kraska & Khattab (MIT CSAIL, Oct 2025)
- **Ollama** — Local LLM inference (chat completions, OpenAI-compatible)
- **OpenAI Python client** — Provider-neutral chat-completions wrapper used to talk to Ollama / LM Studio / OpenAI / Groq
- **Pydantic** — Structured config / message types

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### What an RLM Is & The Problem It Solves

A **Recursive Language Model (RLM)** is an inference-time strategy in which an LM, faced with a long or complex prompt, treats that prompt as **an external environment** instead of stuffing it into its own context window. The model is given a Python REPL where the long context is already bound to a variable, plus a tool `query(text, question)` that spawns a *fresh sub-LM* on a snippet of the context. The root LM writes code that **peeks, greps, partitions, and recursively queries** the context until it has enough information to answer — then emits `FINAL(answer)`.

The original blog post (Alex L. Zhang, MIT CSAIL, October 2025) and the companion paper *Recursive Language Models* (arXiv:2512.24601, Dec 2025) make the case that this beats vanilla long-context prompting because:

1. **Long-context attention degrades**. Even models advertised at 1M-token windows suffer from "context rot" and "lost-in-the-middle" effects: the answer hidden somewhere in 800k tokens is hard to surface in one forward pass.
2. **Each sub-LM gets fresh attention over a small slice**. A `query()` over a 5k-token chunk doesn't dilute attention across the other 795k tokens.
3. **The LM itself decides the decomposition**, on the fly, in code. This is the key contrast with agentic loops or RAG, where the *human* designs the decomposition strategy upfront.

Reported headline numbers:
- **OOLONG (132k tokens)**: RLM-wrapped GPT-5-mini scores >2× vanilla GPT-5 on the hardest split, at comparable cost.
- **BrowseComp-Plus (6–11M tokens)**: vanilla base models score ~0%; RLM(GPT-5) achieves ~91%.
- **Inputs up to 100× beyond the model's native context window** become tractable.

### How It Works Internally

Formally, an RLM receives a query *q* and a context *C = [c₁, c₂, …, cₘ]*. The root LM is exposed to:

| Capability | Mechanism |
|------------|-----------|
| **Read context** | `context` is a Python variable. `len(context)`, `context[:1000]`, `re.findall(...)` are all available. |
| **Recurse** | `query(text, question)` invokes a *new* RLM (or sub-LM at depth ≥ max_depth) on a slice of context with a focused sub-question. Returns a string. |
| **Answer** | Output protocol: emit `FINAL(answer)` (or `FINAL_VAR(name)` to return a REPL variable). |

The root LM operates inside a multi-turn loop. Each turn the LM emits *either* a Python code block (the harness `exec`s it and feeds stdout back) *or* `FINAL(...)` (the loop terminates). Common emergent strategies the paper documents:

- **Peeking** — read the first/last few hundred chars to learn structure.
- **Grepping** — `re.findall` to narrow the search before invoking sub-LMs.
- **Partition + map** — split the context into chunks, `query()` each, aggregate.
- **Summarize** — recurse with "summarize this chunk" sub-questions for hierarchical compression.

**Bounded recursion**: the original experiments fix `max_depth=1` (root may call sub-LMs; sub-LMs may not recurse further). Without that bound the call tree can explode.

**Cost reality**: each `query()` is blocking and re-encodes its prompt — there is no prefix caching across sub-calls. So total cost grows roughly linearly with the number of sub-calls. The trade-off versus vanilla long-context: many small focused calls vs. one huge dilute call.

### Where RLMs Sit Among Alternatives

| Approach | What decides the decomposition | Where the long context lives |
|----------|--------------------------------|------------------------------|
| **Vanilla long-context** | Nothing — model attends over everything | Entirely inside the prompt |
| **RAG** | A *retriever* (engineered) picks chunks before the LM sees them | A vector DB outside the prompt |
| **Agentic loops (ReAct)** | A *human-designed plan* in the system prompt | Tool outputs, partly in the prompt |
| **Tree-of-thought / self-ask** | A *fixed branching template* the LM follows | Entirely inside the prompt |
| **RLMs** | The *LM itself*, by writing Python code | A variable in a REPL — never directly in the prompt |

RLMs are also distinct from "code interpreter" tools: in code-interpreter, the LM uses Python to compute over data; in an RLM, the LM uses Python to **route language to other LMs**.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Root LM** | The "outer" model that writes REPL code and decides recursion structure. |
| **Sub-LM** | A fresh LM invocation triggered by `query()`, with no memory of root context. |
| **REPL state** | A persistent Python namespace shared across the root LM's turns. |
| **Depth bound** | Max recursion depth — `max_depth=1` matches the original paper. |
| **Output protocol** | `FINAL(answer)` ends the session; `FINAL_VAR(name)` returns a REPL variable. |
| **NOT_FOUND sentinel** | Sub-LM contract: reply with `NOT_FOUND` if the snippet doesn't contain the answer. Lets the root distinguish "missing" from "hallucinated". |
| **Context rot** | Empirical degradation of long-context attention as input grows. The motivating problem. |

### References

- Original blog post: [Recursive Language Models — Alex L. Zhang](https://alexzhang13.github.io/blog/2025/rlm/) (MIT CSAIL, Oct 2025)
- Paper: [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab — Dec 2025)
- Reference implementation: [ysz/recursive-llm](https://github.com/ysz/recursive-llm)
- Prime Intellect's RLMEnv: [Recursive Language Models: the paradigm of 2026](https://www.primeintellect.ai/blog/rlm)
- VentureBeat coverage: [MIT's recursive framework lets LLMs process 10M tokens without context rot](https://venturebeat.com/orchestration/mits-new-recursive-framework-lets-llms-process-10-million-tokens-without)

## Description

Build a tiny RLM from scratch and compare it against vanilla long-context prompting on a synthetic needle-in-a-haystack benchmark. You'll implement the recursive `query()` tool with depth and cost bounds, the REPL-style root agent loop, and the comparison harness. The goal is to *feel* the trade-off — wall-clock latency, token cost, and accuracy — that the paper reports at frontier scale.

### What you'll learn

1. The **context-as-variable** trick that distinguishes RLMs from RAG and ReAct.
2. How to expose a **fresh-attention sub-LM** as a tool to the root LM.
3. The **bounded-recursion** discipline (`max_depth`, cost tracking) that keeps an RLM from melting your laptop.
4. A REPL-style root agent loop that interleaves `python` code and `FINAL()` outputs.
5. Where RLMs win and lose against vanilla long-context, on a benchmark you can grade in seconds.

## Instructions

### Phase 0: Setup & Verification (~10 min)

1. `docker compose up -d` to start Ollama, then pull the model:
   `docker exec rlm_ollama ollama pull qwen2.5:7b`.
2. `uv sync` to install dependencies.
3. `uv run python -m src._00_verify_setup` — smoke-tests the OpenAI-compatible client against your local Ollama.

### Phase 1: Vanilla Long-Context Baseline (~15 min) — `src/_01_baseline_longctx.py`

The control experiment: stuff the haystack into a single prompt and ask. Without this, you can't measure RLM's improvement.

1. **TODO #1 — `build_baseline_messages(haystack, question)`**: assemble a `system` + `user` message pair that pins down a strict QA persona and embeds the haystack and question. This is the prompt the RLM is supposed to *replace* — getting it right matters because every Phase 4 result is measured against it.
2. **TODO #2 — `is_correct(prediction, gold)`**: lowercase + strip both strings and substring-match. The gold answers are short and unambiguous (e.g. `ALPHA-9831`), so a substring grader is enough and avoids burning a second LM call per evaluation.

### Phase 2: The Recursive `query()` Tool (~20 min) — `src/_02_recursive_query.py`

The fundamental RLM primitive — invoke a fresh sub-LM on a snippet, with depth and cost guards.

3. **TODO #1 — `_sub_lm_answer(cfg, text, question)`**: build sub-LM `messages` with the focused-reading-assistant system prompt (including the `NOT_FOUND` contract), then call `chat(...)` and return the stripped string. The `NOT_FOUND` sentinel is what lets the root LM distinguish "this chunk doesn't have it" from "the sub-LM hallucinated".
4. **TODO #2 — `_check_depth(current_depth, max_depth)`**: raise `DepthExceeded` if `current_depth >= max_depth`. The original paper fixes this at 1 — without a bound the call tree can recurse forever and bury you in API calls.
5. **TODO #3 — `query(...)`**: wire the depth guard, timing, sub-LM call, and `tracker.record(...)` together in the right order. This is the function the *root* LM will call from inside its REPL — making it observable here is what makes the cost/latency table in Phase 4 possible.

### Phase 3: REPL-Style Root Agent (~30 min) — `src/_03_rlm_root_agent.py`

Now the interesting part: the root LM that writes Python, executes it, and decides when to recurse.

6. **TODO #1 — `build_root_system_prompt()`**: write the root system prompt. Cover the role, the REPL environment (`context` variable, `query()` tool), the strict output protocol (```python``` block OR `FINAL(...)`), and a one-line example. The protocol is load-bearing — without it the LM will just print the entire context back at you.
7. **TODO #2 — `execute_python(code, state)`**: `exec` the code into the *persistent* `state.namespace`, capture stdout via `redirect_stdout`, and turn exceptions into `ERROR: <traceback>` strings (so the LM can recover on the next turn). Sharing the namespace across turns is what lets the LM build up `chunks = ...` once and reference it later.
8. **TODO #3 — `run_root_agent(question, context, ...)`**: drive the round loop. Each round: call `chat(...)`; check `_extract_final` (return if hit); else `_extract_code` and `execute_python` (feed truncated stdout back as a `user` message); else nudge the LM toward the protocol. This is the inference-time-scaling axis — more rounds = more compute = (potentially) better answer.

### Phase 4: Comparison Harness (~15 min) — `src/_04_compare_rlm_vs_baseline.py`

Tabulate accuracy, sub-calls, characters-to-LM, and wall-clock time across haystack sizes.

9. **TODO — `normalize_final_answer(raw)`**: strip quotes, common prefixes (`Answer: `, `The answer is `), and trailing punctuation from the root LM's `FINAL(...)` payload. Local 7B models are noisy; this is the single point where you tune the grader to stop emitting false negatives. Inspect a few raw Phase-3 outputs first.

### What to look for in the results

- At small haystack sizes the baseline often wins on latency (one call vs. many).
- As `n_filler` grows, baseline accuracy drops; RLM accuracy holds longer.
- RLM `chars_to_lm` is dominated by the *aggregate* of all sub-LM inputs — usually larger than the baseline's, because of overlap between chunks. The point is *not* fewer tokens; it's *better-attended* tokens.

## Motivation

- **Frontier-relevant**: RLMs were popularized in late 2025 and are an active area at MIT and Prime Intellect. Understanding the pattern is currency for any AI engineering interview in 2026.
- **Complementary to 029a/029b/030a-c (LangChain, LangGraph, DSPy)**: those teach orchestration; RLMs teach *recursive* orchestration where the LM picks the structure.
- **Architectural intuition for long-context systems**: even if you never deploy an RLM, the lesson — "don't make one model do attention over everything; route language between models" — generalises to any production RAG / agent design.

## LLM Configuration

By default the practice runs against local Ollama (`qwen2.5:7b`). To switch providers or use a smaller sub-LM (mirroring the GPT-5 / GPT-5-mini split from the paper), copy `.env.example` to `.env` and set the variables below before running any script.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` \| `mistral` \| `groq` |
| `LLM_MODEL` | `qwen2.5:7b` | Root model |
| `LLM_SUB_MODEL` | _(falls back to `LLM_MODEL`)_ | Sub-LM model (set to e.g. `qwen2.5:3b` or `mistral-small-latest` to use a cheaper sub-LM) |
| `LLM_BASE_URL` | _(provider default)_ | Override the API base URL |
| `LLM_API_KEY` | _(empty)_ | Required for cloud providers |

All provider routing lives in `src/llm_config.py` and is dispatched through LiteLLM, so the recursive `query()` plumbing in `_02_recursive_query.py` works the same regardless of provider.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container (port 11434) |
| | `docker exec rlm_ollama ollama pull qwen2.5:7b` | Download the chat model |
| | `docker compose down` | Stop the Ollama container |
| **Setup** | `uv sync` | Install Python dependencies |
| | `uv run python -m src._00_verify_setup` | Smoke-test root + sub LM connections |
| **Phase 1** | `uv run python -m src._01_baseline_longctx` | Vanilla long-context QA baseline |
| **Phase 2** | `uv run python -m src._02_recursive_query` | Sanity-check the `query()` tool in isolation |
| **Phase 3** | `uv run python -m src._03_rlm_root_agent` | Run the full REPL-style root agent on the haystack problems |
| **Phase 4** | `uv run python -m src._04_compare_rlm_vs_baseline` | Tabulated comparison across haystack sizes |
| **Cleanup** | `python clean.py` | Remove caches, venv, Docker volumes, generated data |

## State

`completed`
