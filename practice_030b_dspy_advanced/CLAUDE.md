# Practice 030b — DSPy Advanced: RAG, Assertions & MIPROv2

## Technologies

- **DSPy RAG modules**: `dspy.Retrieve` for retrieval-augmented generation, custom retriever backends
- **Qdrant**: Vector database for document storage and semantic search
- **DSPy Output Refinement**: `dspy.Refine` (iterative self-refinement with feedback) and `dspy.BestOfN` (best-of-N sampling) for output quality constraints
- **MIPROv2**: Advanced Bayesian optimizer for joint instruction + demonstration optimization
- **Typed Predictors**: Type-enforced modules with Pydantic validation via typed signatures
- **sentence-transformers**: Local embedding model for document vectorization

## Stack

Python 3.12+ (uv), Docker (Ollama, Qdrant)

## Theoretical Context

### RAG in DSPy

Unlike LangChain where you manually build retrieval chains, DSPy treats RAG as a **composable module**. `dspy.Retrieve(k=N)` fetches passages, which you pipe into `ChainOfThought` for answer generation. The key advantage: **optimizers improve both retrieval queries and answer generation jointly** — something manual prompt engineering cannot achieve.

The pattern is straightforward:

```python
class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)
```

DSPy uses a configured retrieval model (`dspy.configure(rm=...)`) as the backend. For custom vector databases like Qdrant, you either use the `dspy-qdrant` integration (`QdrantRM`) or implement a custom retriever that queries the database and returns `dspy.Prediction(passages=[...])`.

### Multi-Hop Reasoning

Complex questions often need multiple retrieval steps. The pattern: retrieve -> extract entities -> refine query -> retrieve again. DSPy's modular design makes this natural — chain multiple Retrieve + ChainOfThought modules, each handling one "hop." The optimizer can then tune each hop independently.

Example: "What award did the director of Inception win in 2024?" requires:
1. **Hop 1**: Retrieve "Inception director" -> learn it's Christopher Nolan
2. **Hop 2**: Retrieve "Christopher Nolan awards 2024" -> find the answer

This is inherently a multi-module DSPy program where each hop is a separate Retrieve + ChainOfThought pair.

### Output Refinement: Refine & BestOfN

DSPy 2.6+ replaces the deprecated `dspy.Assert`/`dspy.Suggest` with two refinement modules:

**`dspy.BestOfN(module, N, reward_fn, threshold)`**: Runs the module up to N times with different rollout IDs (temperature=1.0 for diversity). Returns the first prediction that exceeds the threshold, or the highest-scoring one if none do. No feedback between attempts — just parallel sampling.

**`dspy.Refine(module, N, reward_fn, threshold)`**: Extends BestOfN with an **automatic feedback loop**. After each unsuccessful attempt (except the last), DSPy generates detailed feedback about *why* the output failed and injects it as hints for the next attempt. This creates an iterative self-correction cycle:

```
Attempt 1 -> reward_fn evaluates -> feedback generated -> Attempt 2 (with hints) -> ...
```

**Key parameters:**
- `module`: The DSPy module to refine (e.g., `dspy.ChainOfThought(...)`)
- `N`: Maximum number of attempts
- `reward_fn(args, pred) -> float`: Evaluates prediction quality (0.0 = bad, 1.0 = perfect)
- `threshold`: Minimum reward to accept (early stopping)
- `fail_count`: Max failures before raising error (defaults to N)

**Use cases**: format validation, citation checking, length constraints, factual consistency, structured output enforcement.

**When to use which:**
- `BestOfN`: Independent attempts are sufficient (e.g., generating diverse creative outputs)
- `Refine`: Feedback helps self-correction (e.g., fixing citation errors, format issues)

### MIPROv2 Optimizer

MIPROv2 (Multiprompt Instruction PRoposal Optimizer v2) jointly optimizes **instructions + few-shot demonstrations** using Bayesian Optimization. It operates in three stages:

1. **Bootstrap demos**: Sample training examples, run through program, keep traces that pass the metric
2. **Propose instructions**: Generate candidate instructions grounded in data summaries, program code, and example traces
3. **Bayesian search**: Use Bayesian Optimization to find the best combination of instructions + demos for each predictor

**Auto modes** control optimization intensity:
- `'light'`: Fast, few trials — good for quick iteration
- `'medium'`: Balanced search — recommended starting point
- `'heavy'`: Thorough exploration — best quality, highest compute cost

**Data requirements**: 50-500+ training examples for best results. Supports `minibatch=True` for efficient search on large datasets.

**vs BootstrapFewShot**: BootstrapFewShot only optimizes demonstrations. MIPROv2 optimizes both instructions AND demonstrations, often achieving significantly better results on complex multi-step programs.

### Typed Predictors

DSPy signatures support full Python type annotations, including Pydantic models:

```python
class StructuredAnswer(pydantic.BaseModel):
    answer: str
    confidence: float
    sources: list[str]

class QASignature(dspy.Signature):
    """Answer the question based on the context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    result: StructuredAnswer = dspy.OutputField()
```

The `JSONAdapter` (default in DSPy 2.5+) automatically handles serialization — the LM produces JSON matching the Pydantic schema, and DSPy validates + parses it. Supports `list[str]`, `dict`, `Literal`, `Optional`, nested models, etc.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **dspy.Retrieve** | Retrieval module. Returns passages from a configured retriever backend (`dspy.configure(rm=...)`). |
| **QdrantRM** | Qdrant-powered retriever from `dspy-qdrant`. Wraps Qdrant client for use as DSPy retrieval model. |
| **Custom Retriever** | Subclass or wrap `dspy.Retrieve`, implement `forward()` to query your own vector DB. |
| **Multi-Hop RAG** | Iterative retrieval — retrieve -> extract entities -> refine query -> retrieve again. Each hop is a DSPy module. |
| **dspy.Refine** | Iterative refinement with feedback. Runs module up to N times, generates feedback between attempts. |
| **dspy.BestOfN** | Best-of-N sampling. Runs module N times independently, returns highest-scoring result. |
| **reward_fn** | `Callable[[dict, Prediction], float]` — evaluates prediction quality for Refine/BestOfN. |
| **MIPROv2** | Advanced optimizer using Bayesian Optimization over instructions + demonstrations jointly. |
| **auto mode** | MIPROv2 parameter: `'light'`, `'medium'`, or `'heavy'` controlling optimization intensity. |
| **Typed Signature** | Signature with Python type annotations (Pydantic models, `list[str]`, `Literal`, etc.). |
| **Program Serialization** | `program.save(path)` / `program.load(path)` for persistence of compiled programs. |

### Ecosystem Context

| Aspect | DSPy RAG | LangChain RAG |
|--------|----------|---------------|
| **Retrieval** | `dspy.Retrieve` — module-level, optimizer-aware | Manual chain assembly with retrievers |
| **Optimization** | End-to-end: optimizer tunes retrieval queries + generation jointly | Manual prompt tuning per component |
| **Quality control** | `dspy.Refine` with reward functions | Custom validation chains, output parsers |
| **Multi-hop** | Natural via module composition, each hop independently optimizable | Requires manual agent/chain orchestration |

DSPy's Refine/BestOfN approach is unique — no equivalent exists in LangChain or LlamaIndex for automatic iterative self-correction with reward-driven feedback.

**Sources:**
- [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- [DSPy Assertions (legacy) -> Output Refinement](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)
- [DSPy Cheatsheet](https://dspy.ai/cheatsheet/)
- [MIPROv2 API](https://dspy.ai/api/optimizers/MIPROv2/)
- [Qdrant DSPy Integration](https://qdrant.tech/documentation/frameworks/dspy/)
- [dspy-qdrant GitHub](https://github.com/qdrant/dspy-qdrant)

## Description

Advanced DSPy features for building robust LLM pipelines. You'll implement RAG with a custom Qdrant retriever, build multi-hop reasoning chains, add quality constraints with Refine/BestOfN, optimize with MIPROv2, and use typed predictors for structured outputs.

**What you'll learn:**
- RAG pipeline design in DSPy (Retrieve + Generate)
- Custom retriever implementation with Qdrant vector DB
- Multi-hop reasoning for complex questions
- Output refinement for quality enforcement (Refine + BestOfN)
- MIPROv2 optimization (Bayesian approach)
- Typed predictors with Pydantic validation
- Saving and loading compiled programs

## Instructions

Source files use a leading underscore (`_NN_*.py`) so they form valid Python identifiers — run them as modules from the practice root:

```
uv run python -m src._01_ingest_documents
```

Each TODO is a small, focused piece (~5–25 lines). Comments inside `TODO` blocks contain critical learning material — do not skim past them.

### Phase 1: Setup & Document Ingestion (~15 min)

Start Docker (Ollama + Qdrant), pull the model, run `uv sync`, run `_00_verify_setup`.

1. **`_01_ingest_documents.create_collection`** — call `client.recreate_collection(...)` with `VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)`. Teaches that vector size MUST match the embedder's output dimension and distance MUST match what the retriever uses at query time.
2. **`_01_ingest_documents.embed_texts`** — extract `text` fields and call `embedder.encode(texts)`. Teaches that the same embedder must be used for documents and queries so they live in the same semantic space.
3. **`_01_ingest_documents.upsert_points`** — wrap each `(id, vector, payload)` triple in a `PointStruct` and `client.upsert(...)`. Teaches the payload-as-passage idiom (the retriever returns the payload's `text`).

### Phase 2: Basic RAG Pipeline (~20 min)

4. **`_02_basic_rag.QdrantRetriever.forward`** — embed the query, call `client.query_points(...)`, return `dspy.Prediction(passages=[hit.payload["text"] for hit in results.points])`. Teaches the bridge between DSPy modules and any external vector store.
5. **`_02_basic_rag.BasicRAG.forward`** — call `self.retrieve(query=question)` then `self.generate(context=..., question=...)`. Teaches the canonical Retrieve-then-Generate composition that DSPy compilers can optimize end-to-end.

### Phase 3: Multi-Hop Reasoning (~20 min)

6. **`_03_multihop_rag.MultiHopRAG.hop1_retrieve`** — initial retrieval against the original question. Teaches the entry point of iterative retrieval.
7. **`_03_multihop_rag.MultiHopRAG.extract_and_refine`** — call `self.extract(context=hop1_passages, question=question)` to identify entities and propose a refined query. Teaches that the LM, not the developer, formulates the next query — and that ChainOfThought helps it reason about what's missing.
8. **`_03_multihop_rag.MultiHopRAG.hop2_and_synthesize`** — second retrieval with the refined query, dedup-merge passages, then `self.synthesize(...)`. Teaches passage merging + final synthesis across both hops.

### Phase 4: Output Refinement (~20 min)

9. **`_04_refinement.citation_and_length_reward`** — score 0.0/0.5/1.0 based on a non-trivial-length floor and a 100-word cap. Teaches how to encode quality criteria as a numeric signal that `dspy.Refine` can drive.
10. **`_04_refinement.wrap_with_refine`** — wrap a fresh RAG pipeline with `dspy.Refine(module=..., N=3, reward_fn=..., threshold=1.0)`. Teaches the retry-with-feedback loop on top of an existing module.
11. **`_04_refinement.no_hallucination_reward`** — return `1.0` iff the answer contains an "I don't know" / uncertainty phrase. Teaches encoding *honesty* as a reward signal — directly attacking the #1 RAG failure mode (hallucination).
12. **`_04_refinement.wrap_honest_rag`** — second `dspy.Refine` wrapper using `no_hallucination_reward`. Teaches that the same Refine machinery generalizes to different quality contracts just by swapping the reward.

### Phase 5: MIPROv2 Optimization & Typed Predictors (~25 min)

13. **`_05_mipro_typed.bootstrap_compile`** — `BootstrapFewShot(metric=..., max_bootstrapped_demos=2, max_labeled_demos=2).compile(...)`. Teaches the simpler optimizer that only learns demonstrations.
14. **`_05_mipro_typed.mipro_compile`** — `MIPROv2(metric=..., auto="light", num_threads=1, verbose=True).compile(program, trainset=..., valset=...)`. Teaches Bayesian joint optimization of instructions + demonstrations and why it beats BootstrapFewShot.
15. **`_05_mipro_typed.build_typed_rag`** — define `TypedRAG(dspy.Module)` using `dspy.ChainOfThought(TypedQA)`, where `TypedQA` outputs the `StructuredAnswer` Pydantic model. Teaches how the JSONAdapter enforces a schema as the contract between LM and downstream code.
16. **`_05_mipro_typed.save_program`** — call `program.save(str(SAVED_PROGRAM_PATH))`. Teaches that only learned parameters (prompts + demos) are persisted, not module structure.
17. **`_05_mipro_typed.load_program`** — build a fresh pipeline via `create_rag_pipeline()` and call `.load(path=...)`. Teaches the compile-once / deploy-many separation that makes optimized programs versionable artifacts.

## Motivation

RAG is the most common LLM application pattern in production. DSPy's output refinement modules (Refine/BestOfN) provide unique quality guarantees — automatic self-correction driven by reward functions, with no equivalent in other frameworks. MIPROv2 represents state-of-the-art prompt optimization via Bayesian search. These skills are essential for building production LLM systems that are reliable, measurable, and improvable. Prerequisite for DSPy+LangGraph integration (030c).

## LLM Configuration

The practice supports multiple LLM providers via environment variables. Default behavior (no configuration) uses local Ollama with `qwen2.5:7b` — identical to the original setup.

### Selecting a provider

Copy `.env.example` to `.env` and set the variables:

```bash
# Local Ollama (default — no .env needed)
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b

# LM Studio
LLM_PROVIDER=lmstudio
LLM_MODEL=<model-name-in-lmstudio>

# OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-5-haiku-20241022
LLM_API_KEY=sk-ant-...

# Google Gemini
LLM_PROVIDER=google
LLM_MODEL=gemini-1.5-flash
LLM_API_KEY=AIza...
```

The configuration lives in `src/llm_config.py`. All source files call `configure_lm()` from that module — no per-file changes are needed when switching providers.

**Embedding model** (sentence-transformers `all-MiniLM-L6-v2`) always runs locally and is independent of the LLM provider.

## Commands

All commands run from `practice_030b_dspy_advanced/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama + Qdrant containers |
| `docker compose down` | Stop and remove containers |
| `docker exec ollama ollama pull qwen2.5:7b` | Download Qwen 2.5 7B model for exercises |
| `docker compose logs -f ollama` | Stream Ollama logs (check model loading) |
| `docker compose logs -f qdrant` | Stream Qdrant logs (check storage init) |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from pyproject.toml |
| `uv run python -m src._00_verify_setup` | Verify DSPy + Qdrant connection |

### Phase 1: Document Ingestion

| Command | Description |
|---------|-------------|
| `uv run python -m src._01_ingest_documents` | Run TODOs 1-3: create collection, embed texts, upsert points |

### Phase 2: Basic RAG

| Command | Description |
|---------|-------------|
| `uv run python -m src._02_basic_rag` | Run TODOs 4-5: QdrantRetriever.forward + BasicRAG.forward |

### Phase 3: Multi-Hop Reasoning

| Command | Description |
|---------|-------------|
| `uv run python -m src._03_multihop_rag` | Run TODOs 6-8: hop1, extract+refine, hop2+synthesize |

### Phase 4: Output Refinement

| Command | Description |
|---------|-------------|
| `uv run python -m src._04_refinement` | Run TODOs 9-12: citation/length reward + Refine, no-hallucination reward + Refine |

### Phase 5: MIPROv2 & Typed Predictors

| Command | Description |
|---------|-------------|
| `uv run python -m src._05_mipro_typed` | Run TODOs 13-17: BootstrapFewShot + MIPROv2 compile, TypedRAG, save & load |

## References

- [DSPy RAG Tutorial](https://dspy.ai/tutorials/rag/)
- [DSPy Output Refinement (BestOfN & Refine)](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/)
- [DSPy Refine API](https://dspy.ai/api/modules/Refine/)
- [MIPROv2 API](https://dspy.ai/api/optimizers/MIPROv2/)
- [DSPy Signatures (Typed Fields)](https://dspy.ai/learn/programming/signatures/)
- [DSPy Cheatsheet](https://dspy.ai/cheatsheet/)
- [Qdrant DSPy Integration](https://qdrant.tech/documentation/frameworks/dspy/)
- [dspy-qdrant GitHub](https://github.com/qdrant/dspy-qdrant)
- [DSPy Optimizers Overview](https://dspy.ai/learn/optimization/optimizers/)
