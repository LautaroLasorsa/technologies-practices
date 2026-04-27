# Practice 085 — Hybrid Symbolic-Neural Agents: LLM + CP-SAT for Scheduling

## Technologies

- **OR-Tools CP-SAT** — Google's open-source constraint-programming SAT solver, the workhorse behind production scheduling systems at Google and many enterprises.
- **Pydantic v2** — schema-validated structured data passed between LLM and solver stages.
- **instructor** — wraps an LLM client to coerce its output into a Pydantic model (with retries on validation failure).
- **LiteLLM** — provider-neutral chat-completions wrapper. Speaks Ollama, LM Studio, OpenAI, Anthropic, Gemini, Mistral, and Groq through one API.
- **Ollama** — local LLM inference (default backend).

## Stack

Python 3.11+ (uv), Docker (Ollama).

## Theoretical Context

### What a Hybrid Symbolic-Neural Agent Is, and the Problem It Solves

LLMs are extraordinary at *language* — extracting structure from messy text, generating natural-language explanations, deciding what to do next from context. They are **terrible** at *combinatorial search*. Asking GPT-5 to schedule 30 employees over 200 shifts will produce something plausible-looking that violates constraints, misses optima, and silently hallucinates assignments. The math is unforgiving: an n-employee × m-shift assignment problem has 2^(nm) Boolean configurations. No autoregressive sampler can navigate that.

A **hybrid symbolic-neural agent** splits the work along its grain:

```
natural language ──► [LLM]      ──► structured problem
structured problem ──► [Solver] ──► structured solution / proof of infeasibility
structured output ──► [LLM]     ──► natural-language explanation
```

The LLM **orchestrates** (parses intent, picks the tool, formats the answer). The **solver decides** (combinatorial search, optimality, infeasibility proofs). The LLM **explains** (turns the structured result into something a human will read). This is a Staff-level architectural pattern for production agents in scheduling, routing, resource allocation, and configuration domains — where decisions must be *correct*, *explainable*, and *auditable*.

The pattern is increasingly common in industry: Microsoft's TaskWeaver routes user intents to code-execution backends; SAP's Joule pairs LLM front-ends with classical optimisation back-ends for supply-chain decisions; Google's NL2KB layer in their internal scheduling tools follows the same shape.

### How CP-SAT Works Internally

CP-SAT is a hybrid CP/SAT/LP solver: it combines three solver families that historically lived in different camps.

| Mechanism | What it does |
|-----------|--------------|
| **Constraint propagation** | Each variable assignment narrows the domains of related variables. E.g. once `x_alice_mon_morning = 1` and the shift demand is 1, every other `x_*_mon_morning` is forced to 0. Cheap, repeated to a fixed point after every decision. |
| **CDCL search** (Conflict-Driven Clause Learning) | Modern SAT solvers don't blindly backtrack. When a branch hits a contradiction, they analyse the implication graph, learn a *no-good* clause, and add it to the database — preventing every future visit to the same dead end. |
| **LP relaxation** | For integer-linear constraints CP-SAT runs a continuous relaxation in parallel and uses its bounds to prune the discrete search. |
| **Search heuristics** | Variable/value ordering (e.g. activity-based, "fail-first") chosen automatically. CP-SAT's defaults are very good. |

The result is a complete solver — if it returns INFEASIBLE, the problem really has no solution. That guarantee is what an LLM cannot offer and what makes the hybrid pattern valuable.

### Assumption Literals and Explainable Infeasibility

A naive solver replies "INFEASIBLE." with no further information — useless to a user. The trick that makes this pattern work in production is **assumption literals**: each hard constraint is gated by a Boolean literal that the model marks as an "assumption." If the model is infeasible, the solver can return a *minimum subset* of those assumptions that already proves infeasibility:

```python
solver.SufficientAssumptionsForInfeasibility() -> list[int]
```

That subset is a Minimum (or near-minimum) Unsatisfiable Subset (MUS) — equivalent to MILP's Irreducible Inconsistent Subsystem (IIS). It's small (often just 2–3 constraints out of dozens), focused, and exactly what the LLM explainer needs to produce a useful "the schedule is impossible because X conflicts with Y" message.

This is the *killer feature* of hybrid systems and the part most LLM engineers don't know about. Building a system that says "you need 12 shift-coverage slots but your two part-timers can only cover 6, raise their `max_shifts` cap or hire" beats one that says "I tried but couldn't make a schedule" by orders of magnitude.

### Where Hybrid Agents Sit Among Alternatives

| Approach | Combinatorial guarantees | Natural-language interface | Explainable infeasibility |
|----------|--------------------------|----------------------------|---------------------------|
| **Pure LLM ReAct** | None. Hallucinates. | Yes. | No. |
| **Pure OR (CP-SAT/MILP)** | Optimal or proven infeasible. | No — needs a structured input form. | Yes via IIS/MUS. |
| **MILP + LLM front-end** | Yes (linear-numeric domains). | Yes. | Yes (IIS). |
| **CP-SAT + LLM front-end** *(this practice)* | Yes (logical/Boolean domains). | Yes. | Yes (MUS via assumptions). |

CP-SAT vs. MILP rule of thumb: CP-SAT shines for problems whose structure is mostly Boolean / logical (assignment, scheduling, sequencing, packing); MILP shines when the structure is mostly linear arithmetic (flows, blending, allocation with continuous quantities). Schedule-style problems sit firmly in CP-SAT territory.

### Key Concepts

| Concept | Definition |
|---------|------------|
| **Decision variable** | A solver variable representing one degree of freedom (e.g. `x[employee, shift]` ∈ {0, 1}). |
| **Hard constraint** | Must hold in every feasible solution. |
| **Soft constraint** | A weighted preference contributing to the objective. Never makes a model infeasible. |
| **Assumption literal** | A Boolean literal that gates a constraint; the solver can identify which assumptions are "in conflict." |
| **MUS / IIS** | Minimum Unsatisfiable Subset / Irreducible Inconsistent Subsystem — a smallest subset of constraints that is already infeasible. |
| **Structured output** | LLM reply coerced into a typed schema (here: Pydantic) instead of free-form prose. |

### References

- OR-Tools CP-SAT solver guide: <https://developers.google.com/optimization/cp/cp_solver>
- OR-Tools assumption literals: <https://github.com/google/or-tools/blob/stable/ortools/sat/docs/cp_solver.md>
- Pydantic v2 docs: <https://docs.pydantic.dev/latest/>
- instructor library: <https://python.useinstructor.com/>
- LiteLLM provider matrix: <https://docs.litellm.ai/docs/providers>
- Background on MUS / unsatisfiable cores: <https://en.wikipedia.org/wiki/Unsatisfiable_core>
- Microsoft TaskWeaver paper (LLM-orchestrated tools): <https://arxiv.org/abs/2311.17541>

## Description

Build a small employee-shift scheduling agent end-to-end. A user types a free-form scheduling request; the agent extracts a typed `ScheduleRequest`, runs CP-SAT, and either returns a feasible schedule with an LLM-written summary or — if the problem is infeasible — extracts a minimum unsatisfiable subset and asks the LLM to explain *why* and recommend a relaxation.

### What you'll learn

1. The **LLM-orchestrates → solver-decides → LLM-explains** pattern, the bedrock of production hybrid agents.
2. How to use **structured output** (instructor + Pydantic) to keep the LLM on a tight schema.
3. How to model an assignment problem in **CP-SAT** with Boolean decision variables and assumption-gated constraints.
4. How to use **assumption literals** to extract a Minimum Unsatisfiable Subset and turn "INFEASIBLE" into a useful, actionable explanation.
5. How to choose between a **hand-rolled state machine** and a **graph-based agent framework** for orchestration — and why "linear-with-one-branch" pipelines don't need LangGraph.

## Instructions

### Phase 0: Setup & Verification (~10 min)

1. `docker compose up -d` to start Ollama, then pull the model:
   `docker exec hsna_ollama ollama pull qwen2.5:7b`
2. `uv sync` to install dependencies.
3. `uv run python -m src._00_verify_setup` — pings the LM, runs a 2-variable CP-SAT model.

### Phase 1: Constraint Extraction (~15 min) — `src/_01_constraint_extraction.py`

The LLM as a typed-output extractor. The Pydantic schema is the contract; instructor enforces it with retries.

1. **TODO #1 — `extract_schedule_request(...)`**: call the instructor-patched LiteLLM client with `response_model=ScheduleRequest` and return the result. Do not parse JSON manually — that's instructor's job.

### Phase 2: CP-SAT Solver (~30 min) — `src/_02_cpsat_solver.py`

The symbolic core. No LLMs. This is where you learn the canonical CP-SAT pattern.

2. **TODO #1 — `_add_hard_constraints(...)`**: encode each hard constraint as one or more `model.Add(...)` clauses, each gated by `.OnlyEnforceIf(assumption_lit)`. Append every assumption literal to the returned list in the same order as the input. The encoding cookbook is in the docstring.

### Phase 3: Infeasibility Analyser (~15 min) — `src/_03_infeasibility_analyzer.py`

The killer feature. Turn an `InfeasibleResult` into a structured `Conflict` carrying just the constraints that participate in the contradiction.

3. **TODO #1 — `analyse(...)`**: rebuild the model, call `solver.SufficientAssumptionsForInfeasibility()`, and map the returned indices back to the original `HardConstraint` objects.

### Phase 4: LLM Explainer (~10 min) — `src/_04_explainer.py`

The LLM as a *prose generator*, not a decision-maker. One function with two branches, both calling the same chat helper so prompt caching applies.

4. **TODO #1 — `explain(...)`**: format the `ScheduleSolution` or `Conflict` into a user message body, prepend the stable `EXPLAINER_SYSTEM_PROMPT`, call `chat(...)`, return the stripped reply.

### Phase 5: Orchestrator (~10 min) — `src/_05_orchestrator.py`

Tie it all together. Read the design discussion in the module docstring before implementing — there's an explicit hand-rolled-vs-LangGraph choice to argue.

5. **TODO #1 — `run_agent(...)`**: extract → solve → branch (feasible: explain solution; infeasible: analyse + explain conflict) → return `AgentResult`.

### Phase 6: Demo (no TODO)

Run `uv run python demo.py` to drive the whole pipeline on the example coffee-shop problem. Then play with the input text — drop a manager, set everyone's `max_shifts` to 0, see the conflict explanation change.

### What to look for in the results

- The CP-SAT solve is *milliseconds*. The LLM stages dominate latency. That's the hybrid trade-off in microcosm.
- Removing or weakening a single hard constraint usually flips the problem from infeasible to feasible — the MUS tells you which one to pick.
- The same LLM that struggles to schedule 5 employees succeeds at *describing* a schedule produced by CP-SAT. Use each model for what it's good at.

## Motivation

- **AutoScheduler.AI relevance**: scheduling is the company's core domain. This is the architectural pattern any AI feature in that product line should follow.
- **Senior → Staff differentiator**: most LLM engineers don't know how to bridge LLMs with classical OR solvers. Knowing this pattern (and the assumption-literal trick for explainable infeasibility) is genuinely uncommon and high-leverage.
- **Generalises beyond scheduling**: routing, resource allocation, configuration validation, planning under constraints — anywhere "the LLM made a hard combinatorial decision" is a code smell, this pattern replaces it.

## LLM Configuration

By default the practice runs against local Ollama (`qwen2.5:7b`). To switch providers, copy `.env.example` to `.env` and set the variables below.

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `ollama` | `ollama` \| `lmstudio` \| `openai` \| `anthropic` \| `google` \| `mistral` \| `groq` |
| `LLM_MODEL` | `qwen2.5:7b` | Model name (no provider prefix) |
| `LLM_BASE_URL` | _(provider default)_ | Override the API base URL |
| `LLM_API_KEY` | _(empty)_ | Required for cloud providers |

All provider routing lives in `src/llm_config.py` and is dispatched through LiteLLM, so the extraction/explanation plumbing is provider-agnostic.

## Commands

| Phase | Command | Description |
|-------|---------|-------------|
| **Infrastructure** | `docker compose up -d` | Start Ollama container (port 11434). |
| | `docker exec hsna_ollama ollama pull qwen2.5:7b` | Download the chat model. |
| | `docker compose down` | Stop the Ollama container. |
| **Setup** | `uv sync` | Install Python dependencies. |
| | `cp .env.example .env` | Configure provider (optional — defaults to local Ollama). |
| | `uv run python -m src._00_verify_setup` | Ping the LM and CP-SAT. |
| **Phase 1** | `uv run python -m src._01_constraint_extraction` | Sanity-check NL → ScheduleRequest extraction in isolation. |
| **Phase 2** | `uv run python -m src._02_cpsat_solver` | Solve `golden_cases.simple_feasible()` with CP-SAT. |
| **Phase 3** | `uv run python -m src._03_infeasibility_analyzer` | Analyse `golden_cases.infeasible_overdemand()`. |
| **Phase 4** | `uv run python -m src._04_explainer` | Generate prose for both a solution and a conflict. |
| **Phase 5** | `uv run python -m src._05_orchestrator` | Run the full agent on a hard-coded example. |
| **End-to-end** | `uv run python demo.py` | Drive the whole pipeline on the coffee-shop example. |
| **Cleanup** | `python clean.py` | Remove caches, venv, Docker volumes, generated outputs. |

## Notes

_(populated during the practice — observations, surprises, and cross-domain connections discovered while implementing.)_

## State

`not-started`
