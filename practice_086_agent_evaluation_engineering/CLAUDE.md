# Practice 086 — Agent Evaluation Engineering: tau-bench, LLM-as-Judge & Bootstrap CIs

## Technologies

- **LangChain** (>=0.3) — provider-neutral chat models, tool binding, and `with_structured_output(...)` for typed verdicts; replaces both LiteLLM (transport) and instructor (structured output) in one library.
- **Pydantic v2** — schema-validated structured data (Task, Trajectory, JudgeVerdict, EvalReport).
- **scipy.stats.bootstrap** — bundled bootstrap routine; used in the verify-setup smoke test. The paired-bootstrap CI in `_05_bootstrap.py` is hand-rolled on purpose.
- **pytest** — present in deps so the eval can be run as parametrised tests if you want; the harness here is plain Python.
- **Ollama** — local LLM inference (default backend).

## Stack

Python 3.11+ (uv), Docker (Ollama).

## Theoretical Context

### Three Papers, One Workflow

This practice fuses three foundational evaluation papers into the canonical Staff-level workflow for measuring agent quality. None of them is enough on its own; together they answer the three questions every agent eval must answer:

1. **What does "the agent works" even mean?** — *tau-bench* (Yao et al., 2024 — <https://arxiv.org/abs/2406.12045>) introduced the **simulated-user** harness: instead of single-turn prompts, the agent talks to an LLM-driven user with goals over multiple turns inside a domain (retail, airline) backed by real tools. Their headline metric — **pass^k** — is the probability that *all* k independent rollouts of the same task pass. This kills the "average ability" mirage that pass@1 papers cultivate.
2. **Who scores it?** — *LLM-as-a-Judge / MT-Bench* (Zheng et al., 2023 — <https://arxiv.org/abs/2306.05685>) showed that strong LLMs agree with human raters ~80% of the time on pairwise judging — but only after you control for **position bias**, **verbosity bias**, and **self-enhancement bias**. The paper's mitigations (random ordering, chain-of-thought judging, ensembles) are now standard.
3. **Is the difference real?** — *Paired Bootstrap CIs* (Du et al., 2025 — <https://arxiv.org/abs/2511.19794>) is the most recent of the three: when comparing two systems on the *same* prompts, the unpaired bootstrap is statistically wrong (it ignores per-prompt correlation and overstates variance). The paired bootstrap on per-prompt deltas is the correct estimator and gives defensible "A beats B" intervals.

### pass^k With a Concrete Example

Define `p_i = P[rollout passes | task i]`. Under independence, the per-task `pass^k_i = p_i^k`, and the dataset-level `pass^k = mean_i pass^k_i`.

A model with `p_i = 0.80` everywhere has:
- `pass@1 = 0.80`
- `pass^3 = 0.80^3 = 0.512`
- `pass^5 = 0.80^5 = 0.328`

So a "80% accurate" agent fails *more than half the time* if you require 3-out-of-3 reliability — the metric a customer actually feels. The empirical pass^k is often even lower than the independence formula predicts because failures are *correlated* (the same task fails for the same systematic reason every time). That gap between empirical pass^k and `mean(p_i^k)` is one of the most informative diagnostics in agent evaluation.

### LLM-as-Judge: Biases and Mitigations

| Bias | What it is | Mitigation |
|------|-----------|------------|
| **Position bias** | In pairwise judging, the first-listed answer wins ~10–20% more often. | Randomise order at the call site; or grade single-answer instead of pairwise. |
| **Verbosity bias** | Judges reward longer, more confident-sounding answers regardless of correctness. | Explicit "ignore length, grade only against the rubric" instruction; or constrain output length. |
| **Self-enhancement bias** | Judges over-rate outputs from their own model family (~10% on MT-Bench). | Use a *different* judge model than the agent. We expose `JUDGE_PROVIDER` / `JUDGE_MODEL` for exactly this. |
| **Calibration drift** | A judge calibrated against GPT-4 outputs may mis-score a smaller model's outputs. | Audit judge agreement against a small human-labelled holdout periodically. |

This practice uses **single-answer 0/1 grading** (cheaper, more interpretable, no position bias) but `models.py` ships a `PairwiseVerdict` schema if you want to swap modes.

### Paired Bootstrap CIs: Why "Paired" Matters

When systems A and B are scored on the same prompts, their per-prompt scores are correlated: easy prompts inflate both, hard prompts depress both. Treating them as independent samples (the unpaired bootstrap) double-counts that shared variance and gives wider — but wrong — CIs. The correct estimator works on *per-prompt deltas* `d_i = score_A_i - score_B_i`:

```
for b in 1..B:
    idx = sample N task indices with replacement
    boot_mean[b] = mean(d[idx])
lo, hi = percentile(boot_means, [2.5, 97.5])
```

If `[lo, hi]` excludes 0, the improvement is significant at the 95% level. We implement the **percentile method** here for clarity. **BCa** (bias-corrected and accelerated) is more accurate when the bootstrap distribution is skewed but adds ~30 lines and isn't necessary at the typical N≈30–500 of an agent eval. See `scipy.stats.bootstrap` for a one-liner BCa implementation if you want to swap in.

### Where This Fits

tau-bench is *one* style of agent evaluation — domain-grounded, simulated-user, tool-using. Other styles: SWE-bench (real GitHub issues, no simulated user), AgentBench (pure tool-use without conversation), HELM (capability suites). The harness shape we build here — golden cases → simulated user → agent rollout → judge → pass^k → paired-bootstrap CI — generalises to any of them. Swap the tool set for the new domain, swap the judge rubric, keep the metrics.

This is also where Process Reward Models (practice 087) plug in: instead of judging the *final* trajectory, you score *each step* and use that signal for both eval and training.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Trajectory** | The full agent rollout for one task: tool calls + final message + meta. |
| **pass^k** | P(all k independent rollouts of the same task pass), averaged across tasks. |
| **Simulated user** | An LLM with a goal-encoding system prompt that drives the agent through multi-turn conversation. |
| **LLM-as-Judge** | An LLM scoring agent outputs against a rubric and gold answer; "as a service" replacement for human eval. |
| **Paired bootstrap CI** | Bootstrap CI computed on per-prompt score *deltas* between two systems — the correct estimator when both systems see the same prompts. |
| **Position / verbosity / self-enhancement bias** | The three classical LLM-judge biases (Zheng et al., 2023). |

### References

- tau-bench paper: <https://arxiv.org/abs/2406.12045>
- tau-bench code: <https://github.com/sierra-research/tau-bench>
- MT-Bench / LLM-as-a-Judge: <https://arxiv.org/abs/2306.05685>
- Paired Bootstrap CIs for LLM Eval: <https://arxiv.org/abs/2511.19794>
- LangChain `with_structured_output`: <https://python.langchain.com/docs/concepts/structured_outputs/>
- LangChain tool calling: <https://python.langchain.com/docs/concepts/tool_calling/>
- scipy `bootstrap`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html>

## Description

Build a tau-bench-style evaluation harness end-to-end on a tiny mini-airline domain (~5 tools, ~8 hand-built tasks). Two prompt variants of the same agent (`SYSTEM_PROMPT_A`, `SYSTEM_PROMPT_B`) are run k times per task; an LLM judge scores each rollout; the harness reports **pass^k** for each system plus a **paired-bootstrap 95% CI** on the difference. By the end you'll have an opinionated, defensible answer to the question "is system A actually better than B?" — and a reusable template for any future agent-eval work.

### What you'll learn

1. The **simulated-user** pattern: how multi-turn rollouts surface failure modes single-turn evals miss.
2. **pass^k** vs pass@1: why "average ability" is the wrong metric for production reliability.
3. **LLM-as-Judge** with structured output: schema-validated verdicts, bias mitigations, agent-vs-judge model decoupling.
4. **Paired bootstrap CI**: the right statistical procedure for "system A beats B" claims when both are scored on the same prompts.
5. LangChain tool-calling end-to-end: `bind_tools`, `ToolMessage`, `with_structured_output` — replacing the older LiteLLM + instructor stack.

## Instructions

### Phase 0: Setup & Verification (~10 min)

1. `docker compose up -d` to start Ollama, then pull the model:
   `docker exec aee_ollama ollama pull qwen2.5:7b`
2. `uv sync` to install dependencies.
3. `uv run python -m src._00_verify_setup` — pings the LM, runs a scipy bootstrap smoke test, asserts pytest is importable.

### Phase 1: Tools & Simulated User (~15 min) — `src/_01_tools_and_simulated_user.py`

The mini-airline tools (`search_flights`, `book`, `cancel`, `lookup_reservation`, `get_user_info`, `submit`) are fully scaffolded as LangChain `StructuredTool`s closing over a per-rollout `AirlineState`. The teaching content is the **simulated-user system prompt** — the part of tau-bench that determines whether your eval actually measures anything.

1. **TODO #1 — `simulated_user_system_prompt()`**: write the system-prompt *template* (a string with one `{user_goal}` placeholder) that turns a chat model into a tau-bench-style simulated user. Must encode persona, knowledge boundaries, brevity, and the `###STOP###` termination token. Read the TODO comment block — it summarises tau-bench section 3.

### Phase 2: Agent Loop (~20 min) — `src/_02_agent_loop.py`

A ReAct-style loop: simulated-user message → agent → tool call(s) → tool messages → repeat. Two prompt variants (`SYSTEM_PROMPT_A`, `SYSTEM_PROMPT_B`) live as constants. Loop wiring, tool dispatch, error capture, and trajectory recording are all scaffolded.

2. **TODO #1 — `is_terminated(state, last_user_message, last_agent_text)`**: the rollout's exit predicate. Combines (a) explicit `submit()` call, (b) `###STOP###` from the simulated user, (c) plain-text final agent message with no tool call. ~5 lines.

### Phase 3: Judge (~15 min) — `src/_03_judge.py`

LLM-as-Judge using `model.with_structured_output(JudgeVerdict)`. Trajectory rendering, judge call wiring, and structured-output normalisation are all scaffolded.

3. **TODO #1 — `judge_prompt()`**: design the (system, user-template) prompt pair that turns a (task, trajectory) into a faithful 0/1 verdict. Must address verbosity bias and tell the judge exactly what counts as score=1 vs 0. Use `{task_goal}`, `{gold_answer}`, `{transcript}`, `{final_message}` as placeholders (the call site formats them).

### Phase 4: pass^k (~10 min) — `src/_04_pass_at_k.py`

The headline metric. Two small functions:

4. **TODO #1 — `pass_at_k(scores)`**: empirical pass^k = fraction of tasks whose every rollout passed.
5. **TODO #2 — `expected_pass_at_k_under_independence(scores)`**: mean over tasks of `p_i^k`. The gap to empirical pass^k diagnoses rollout-level correlation.

(Two functions, ~6–10 lines combined — they're naturally paired so they live in one TODO block.)

### Phase 5: Paired Bootstrap CI (~15 min) — `src/_05_bootstrap.py`

The statistical core. The convenience wrapper that pairs scores by task ID is scaffolded; you implement the resampling loop.

6. **TODO #1 — `paired_bootstrap_percentile(deltas, n_boot, confidence, rng)`**: hand-rolled paired-bootstrap percentile CI. Sample with replacement, recompute means, percentile-cut. Do NOT call `scipy.stats.bootstrap` — implementing the loop is the point.

### Phase 6: End-to-End Demo (no TODO)

Run `uv run python demo.py` to drive the whole pipeline on 5 golden tasks with k=3 rollouts. The harness writes per-rollout `Trajectory` JSON files and an `EvalReport` to `runs/<timestamp>/`. Then play with the prompts: tweak `SYSTEM_PROMPT_B`, rerun, watch the CI move.

### What to look for in the results

- The empirical `pass^k` is almost always lower than the under-independence prediction. That gap is the eval telling you "your failures are systematic, not noise." Worth investigating.
- A small, hand-built 8-task eval set is usually too small for the paired-bootstrap CI to exclude 0 — that's a *feature*. It correctly tells you "I can't distinguish A from B yet, scale your eval set or re-design the test cases."
- Switching `JUDGE_MODEL` to a different family (e.g. agent on Ollama, judge on OpenAI) often moves scores by a few percentage points. That's the self-enhancement bias in action.

## Motivation

- **AutoScheduler.AI relevance**: every agent feature shipped to customers needs a defensible "is the new prompt actually better than the old one?" answer. This practice gives you the canonical workflow.
- **Senior → Staff differentiator**: most AI engineers compute pass@1 and stop. Knowing the pass^k → simulated-user → paired-bootstrap chain — and being able to defend it against statistically literate stakeholders — is genuinely uncommon.
- **Generalises beyond tau-bench**: swap the tool set for any domain (scheduling, routing, code review) and the harness still applies. This is reusable infra.

## LLM Configuration

By default the practice runs against local Ollama (`qwen2.5:7b`). To switch providers, copy `.env.example` to `.env` and set the variables below.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` \| `mistral` \| `groq` |
| `LLM_MODEL` | `qwen2.5:7b` | Model name (no provider prefix) |
| `LLM_BASE_URL` | _(provider default)_ | Override the API base URL |
| `LLM_API_KEY` | _(empty)_ | Required for cloud providers |
| `JUDGE_PROVIDER` | _(falls back to `LLM_PROVIDER`)_ | Judge provider — set to a different family to mitigate self-enhancement bias. |
| `JUDGE_MODEL` | _(falls back to `LLM_MODEL`)_ | Judge model. |

All provider routing lives in `src/llm_config.py` and uses LangChain's per-provider chat-model classes. Provider imports are lazy so `uv sync` works without every SDK installed.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container (port 11434). |
| | `docker exec aee_ollama ollama pull qwen2.5:7b` | Download the chat model. |
| | `docker compose down` | Stop the Ollama container. |
| **Setup** | `uv sync` | Install Python dependencies. |
| | `cp .env.example .env` | Configure provider (optional — defaults to local Ollama). |
| | `uv run python -m src._00_verify_setup` | Ping the LM, scipy, pytest. |
| **Phase 1** | `uv run python -m src._01_tools_and_simulated_user` | Sanity-check tools + one simulated-user reply. |
| **Phase 2** | `uv run python -m src._02_agent_loop` | Run one agent rollout on the first golden task. |
| **Phase 3** | `uv run python -m src._03_judge` | Judge a hand-built fake trajectory. |
| **Phase 4** | `uv run python -m src._04_pass_at_k` | Compute empirical and expected pass^k on a fake score matrix. |
| **Phase 5** | `uv run python -m src._05_bootstrap` | Compute a paired-bootstrap CI on a fake score matrix. |
| **Phase 6** | `uv run python -m src._06_run_eval` | Run the full eval on the first 2 tasks with k=1 (smoke). |
| **End-to-end** | `uv run python demo.py` | Full eval: 5 tasks, k=3, A vs B, paired-bootstrap 95% CI. |
| **Cleanup** | `python clean.py` | Remove caches, venv, Docker volumes, generated outputs. |

## Notes

_(populated during the practice — observations, surprises, and cross-domain connections discovered while implementing.)_

## State

`not-started`
