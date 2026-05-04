# Practice 090 — Chain-of-Verification: Decomposed Self-Consistency

## Technologies

- **Chain-of-Verification (CoVe)** — Dhuliawala et al., ACL 2024 (https://arxiv.org/abs/2309.11495). A prompting protocol that reduces hallucinations by *decomposing* an answer into independent verifications answered in *isolated* contexts.
- **LangChain** — provider-neutral chat-model abstraction. `ChatOllama`, `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenerativeAI`, `ChatMistralAI`, `ChatGroq`. Used here in particular for `model.with_structured_output(PydanticSchema)`, which replaces the `instructor` dependency from earlier practices.
- **Pydantic v2** — schema-validated structured data flowing between CoVe stages.
- **Ollama** — local LLM inference (default backend; `qwen2.5:7b`).

## Stack

Python (LangChain, Pydantic), Docker (Ollama).

## Theoretical Context

### What CoVe Solves and Why It Works

LLMs are *autoregressive*: each token conditions on every prior token. Once the model has written a hallucinated fact into its own context, every subsequent token — including any "self-check" the model might perform — is conditioned on the wrong claim. The model is much more likely to *defend* its own previous output than to flag it. Self-consistency techniques like CoT-SC (Wang et al., 2022) work around this by sampling multiple full answers and majority-voting, but they do **not** decompose the question; if the model is biased toward a particular hallucination, every sample inherits the bias.

**Chain-of-Verification (CoVe)** breaks the chain in a different way: it decomposes the answer into a small number of *atomic* factual sub-questions, then answers each sub-question in a *fresh* context that does not contain the original draft. Because the verifier never sees the contaminated text, it cannot confabulate consistency with it. Decomposition + isolation is the algorithm.

### The Four Stages

```
USER PROMPT
    │
    ▼   (1) BASELINE          one ordinary LLM call, no scaffolding
BaselineDraft
    │
    ▼   (2) PLAN              ask the LLM for short, independent verification questions
VerificationPlan
    │
    ▼   (3) EXECUTE           answer EACH question in its own fresh context
                               — without the original prompt, without the draft
list[VerificationAnswer]
    │
    ▼   (4) REFINE            condition the final answer on (prompt + draft + verifications)
RefinedAnswer
```

**Worked example.** User asks "Name 5 actors born in Brooklyn." Baseline lists Mae West, Adam Sandler, Mel Brooks, Barbra Streisand, **Eddie Murphy**. Plan emits five questions: "Was Mae West born in Brooklyn?", …, "Was Eddie Murphy born in Brooklyn?". The verifier, in a fresh context, answers the last question correctly: "No — Eddie Murphy was born in Brooklyn, NY in 1961" (or for some models, "No — Eddie Murphy was born in Brooklyn but raised in Roosevelt", which on a different prompt the model would have garbled). The refinement stage receives all five (Q, A, verdict) triples plus the draft and either confirms or replaces the offending item, listing what changed in `corrections`.

### The Four Execution Modes (paper §3)

| Mode | Stages share context? | Key property | This practice |
|------|----------------------|--------------|---------------|
| **Joint** | All 4 in one transcript | Verifications inherit the draft's mistakes — *worst* | Implemented as the comparison baseline |
| **2-step** | (1+2+3) in one, (4) in another | Partial decontamination | Not implemented |
| **Factored** | Each verification in its OWN context | Strongest single setting | **Implemented as the focus** |
| **Factor + Revise** | Factored + an explicit "compare answers" pre-refinement step | Best overall in the paper | Not implemented (extension idea) |

The practice has you implement *factored* (the principal contribution) and contrast it with *joint* head-to-head on hallucination-prone prompts. You will *feel* the contamination effect in your fingers when you write the joint variant in `_05_pipeline.cove_joint`.

### Failure Modes and Trade-offs

- **Question quality is load-bearing.** If the planning stage produces vague or non-atomic questions ("is this list good?"), the rest of the pipeline cannot help. The `with_structured_output(VerificationPlan)` call in stage 2 is the single most important LLM call in the system.
- **Verifications themselves can hallucinate.** CoVe is not magic; it just gives the model an easier sub-problem ("Was X born in Y?") than the original ("Name five Xs born in Y"). For obscure entities both will fail. Combine with retrieval (RAG) to ground them.
- **Cost & latency multiply.** Factored CoVe makes ~`1 + 1 + N + 1` LLM calls where `N` is the number of verifications. That's ~5×–8× the latency of a single shot. The trade-off is correctness vs throughput; CoVe is for the offline / high-stakes path, not the real-time chat completion.
- **Over-correction.** A confidently wrong verifier can *introduce* mistakes into a previously correct draft. Logging `corrections` lets you audit this.

### Where CoVe Sits Among Related Techniques

| Technique | Decomposes the question? | Breaks contamination? | Adds external grounding? |
|-----------|--------------------------|------------------------|---------------------------|
| Chain-of-Thought (CoT) | No | No (CoT *is* contamination) | No |
| Self-Consistency CoT (CoT-SC) | No | No (votes among contaminated samples) | No |
| Reflexion | Partial (self-critique loop) | Partial (memory across attempts) | No |
| RAG | No (retrieval is orthogonal) | N/A | Yes |
| **CoVe (factored)** | **Yes** | **Yes** | No (combines well with RAG) |

CoVe is *complementary* to RAG: RAG gives you correct facts to verify against; CoVe is the discipline that makes use of them. The practice is intentionally RAG-free so the algorithmic insight stands on its own.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Baseline draft** | The naive single-shot answer; the control condition. |
| **Verification question** | An atomic, self-contained factual question whose answer would help check one item in the draft. |
| **Isolation / factored execution** | Answering each verification in a fresh context that does not include the draft or the original prompt. |
| **Refinement** | The final stage where the draft is rewritten conditional on the verifications. |
| **Structured output** | LLM reply coerced into a typed Pydantic schema (`with_structured_output`); used here for the plan, the per-question verdict, and the final answer. |

### References

- Dhuliawala et al., *Chain-of-Verification Reduces Hallucination in Large Language Models*, ACL 2024 — <https://arxiv.org/abs/2309.11495>
- Min et al., *FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation* — <https://arxiv.org/abs/2305.14251> (related decomposition-for-evaluation idea)
- Wang et al., *Self-Consistency Improves Chain of Thought Reasoning* — <https://arxiv.org/abs/2203.11171>
- Shinn et al., *Reflexion: Language Agents with Verbal Reinforcement Learning* — <https://arxiv.org/abs/2303.11366>
- LangChain structured-output guide — <https://python.langchain.com/docs/how_to/structured_output/>
- LangChain `init_chat_model` reference — <https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html>

## Description

Replicate the core kernel of CoVe on a small fact-list benchmark and compare *factored* against *joint* execution head-to-head. You'll implement the prompts and structured-output calls for each of the four stages, plus the joint contamination baseline, and run them side by side on 8 hallucination-prone prompts.

### What you'll learn

1. The CoVe four-stage pipeline (baseline → plan → execute → refine) and the *contamination chain* it breaks.
2. Why **isolating** verification calls is the algorithmic insight — and what it costs in latency.
3. How to use **LangChain `with_structured_output(PydanticSchema)`** to force the LLM into a typed list/verdict instead of free-form prose. This is the LangChain replacement for `instructor`.
4. A reusable **Joint vs Factored** comparison harness that you can drop onto any future hallucination-mitigation idea.

## Instructions

### Phase 0: Setup & Verification (~10 min)

1. `docker compose up -d` to start Ollama, then pull the model:
   `docker exec cove_ollama ollama pull qwen2.5:7b`
2. `uv sync` to install dependencies. (For non-Ollama providers, e.g. `uv sync --extra openai` or `--extra all`.)
3. `uv run python -m src._00_verify_setup` — pings the LM via LangChain and runs one `with_structured_output` call against a 2-field Pydantic model.

### Phase 1: Baseline Draft (~10 min) — `src/_01_baseline.py`

The control condition: an ordinary LLM call with no scaffolding. The trickiest part is restraint — *don't* tell the model to be careful here.

1. **TODO #1 — `generate_baseline(...)`**: build `[SystemMessage(BASELINE_SYSTEM_PROMPT), HumanMessage(user_prompt)]`, call `chat(model, messages)`, wrap the reply in `BaselineDraft`. No "verify yourself" instructions.

### Phase 2: Plan Verifications (~25 min) — `src/_02_plan_verifications.py`

The most pedagogically rich step. Quality of the questions determines everything downstream. Uses LangChain's `with_structured_output` so the planner's reply is a typed `VerificationPlan` instead of prose.

2. **TODO #1 — `plan_verifications(...)`**: wrap the chat model with `.with_structured_output(VerificationPlan)`, build a 2-message prompt that includes BOTH the original question and the draft (clearly labelled), invoke, return the plan.

### Phase 3: Factored Execution (~15 min) — `src/_03_execute_factored.py`

The principal contribution of CoVe. Each verification is answered in a *fresh* context that does not include the original prompt or the baseline draft.

3. **TODO #1 — `answer_question_isolated(...)`**: call the model with `[SystemMessage(VERIFIER_SYSTEM_PROMPT), HumanMessage(question.question)]` only — no prompt, no draft. Return `VerificationAnswer` via structured output. The fan-out loop `execute_factored(...)` is fully scaffolded.

### Phase 4: Refinement (~10 min) — `src/_04_refine.py`

Combine baseline + verifications into a final answer. Contamination is fine here; the verifications were already computed cleanly.

4. **TODO #1 — `refine(...)`**: build the human-message body labelling the original prompt, the draft, and the formatted verifications (helper `_format_verifications` is provided); invoke the model wrapped with `.with_structured_output(RefinedAnswer)`.

### Phase 5: Joint Pipeline (the contamination baseline) (~10 min) — `src/_05_pipeline.py`

The factored runner is fully scaffolded (it just calls the four stages in order). The *joint* runner — all four stages in a single chat call — is the TODO. Writing it yourself is how you internalise why contamination happens.

5. **TODO #1 — `cove_joint(...)`**: send `[SystemMessage(JOINT_SYSTEM_PROMPT), HumanMessage(prompt)]` as a single call, wrap the whole reply in a `RunRecord(mode="joint", ...)`. Keep it under ~15 lines; we don't try to parse the four sections out of the reply — the demo prints them verbatim.

### Phase 6: Demo (no TODO)

Run `uv run python demo.py` to drive both pipelines over the 8 golden prompts and dump JSONL records under `runs/`. Read the comparison output and look for cases where joint silently confirms a baseline mistake but factored catches it.

### What to look for in the results

- On easy prompts (capitals of South American countries) joint and factored should agree. CoVe is not free; it's only a win when the baseline is wrong.
- On hallucination-prone prompts (Brooklyn actors, one-term presidents) factored should produce more `corrections` than joint, and the corrections should be defensible.
- The factored mode makes ~5×–8× more LLM calls. Notice the wall-clock difference and the structure of the calls in the dumped `runs/*.jsonl`.

## Motivation

- **Senior → Staff differentiator**: most LLM engineers default to "ask the model to double-check itself," which is precisely the contaminated joint mode. Knowing the *isolation* trick is the difference between performative self-check and the protocol that actually works.
- **AutoScheduler.AI relevance**: scheduling assistants frequently produce confidently wrong recommendations ("X employee can cover Y shift" — no, they can't because of qualification Z). Factored CoVe is exactly the discipline to wrap around any high-stakes recommendation step.
- **Generalises**: the (decompose → isolate → refine) pattern transfers to any system where a model's first answer needs to be audited — code review, citation checking, structured-data extraction, recommendation explanations.

## LLM Configuration

By default the practice runs against local Ollama (`qwen2.5:7b`). To switch providers, copy `.env.example` to `.env` and set the variables below.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` \| `mistral` \| `groq` |
| `LLM_MODEL` | `qwen2.5:7b` | Model name (no provider prefix) |
| `LLM_BASE_URL` | _(provider default)_ | Override the API base URL |
| `LLM_API_KEY` | _(empty)_ | Required for cloud providers |

All provider routing lives in `src/llm_config.py`. The provider-specific LangChain integration is imported lazily inside its branch, so `uv sync` works without all SDKs; install only what you need (`uv sync --extra openai`, `--extra anthropic`, `--extra all`, …).

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container (port 11434). |
| | `docker exec cove_ollama ollama pull qwen2.5:7b` | Download the chat model. |
| | `docker compose down` | Stop the Ollama container. |
| **Setup** | `uv sync` | Install Python dependencies (Ollama-only by default). |
| | `uv sync --extra openai` | Add the OpenAI LangChain integration. |
| | `uv sync --extra anthropic` | Add the Anthropic LangChain integration. |
| | `uv sync --extra google` | Add the Google Gemini LangChain integration. |
| | `uv sync --extra mistral` | Add the Mistral LangChain integration. |
| | `uv sync --extra groq` | Add the Groq LangChain integration. |
| | `uv sync --extra all` | Add every provider integration at once. |
| | `cp .env.example .env` | Configure provider (optional — defaults to local Ollama). |
| | `uv run python -m src._00_verify_setup` | Ping the LM and verify `with_structured_output`. |
| **Phase 1** | `uv run python -m src._01_baseline` | Generate one baseline draft for a sample prompt. |
| **Phase 2** | `uv run python -m src._02_plan_verifications` | Plan verifications against a stub baseline. |
| **Phase 3** | `uv run python -m src._03_execute_factored` | Execute three example verifications in isolation. |
| **Phase 4** | `uv run python -m src._04_refine` | Refine a stub baseline against stub verifications. |
| **Phase 5** | `uv run python -m src._05_pipeline` | Run factored and joint side by side on one prompt. |
| **End-to-end** | `uv run python demo.py` | Run both modes over all golden cases; write `runs/*.jsonl`. |
| **Cleanup** | `python clean.py` | Remove caches, venv, Docker volumes, generated outputs. |

## Notes

_(populated during the practice — observations, surprises, and cross-domain connections discovered while implementing.)_

## State

`not-started`
