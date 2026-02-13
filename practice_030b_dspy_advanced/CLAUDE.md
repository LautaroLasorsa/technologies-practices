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

### Phase 1: Setup & Document Ingestion (~15 min)

Start Docker (Ollama + Qdrant), pull model, init project.

**Exercise 1 — Ingest Documents into Qdrant** (`src/01_ingest_documents.py`, TODO):
Loading documents into a vector database is the foundation of any RAG system. This exercise teaches the full ingestion pipeline: read documents, generate embeddings with sentence-transformers, and upsert into Qdrant with proper payloads. Understanding this process is essential because the quality of your embeddings and payload structure directly determines retrieval quality downstream.

### Phase 2: Basic RAG Pipeline (~20 min)

**Exercise 2 — Custom QdrantRetriever** (`src/02_basic_rag.py`, TODO #1):
DSPy's `dspy.Retrieve` expects a configured retrieval model backend, but for custom vector databases you need to build your own retriever. This exercise teaches how to query Qdrant with a sentence-transformers embedding, convert results into `dspy.Prediction(passages=[...])`, and integrate it into a DSPy program. This is the bridge between DSPy's module system and any external vector store.

**Exercise 3 — BasicRAG Module** (`src/02_basic_rag.py`, TODO #2):
Composing retrieval with generation is the core RAG pattern. This exercise teaches how to chain your custom retriever with `ChainOfThought` in a single `dspy.Module`. The `forward()` method defines the pipeline: retrieve context -> generate answer. Once composed as a module, the entire pipeline becomes optimizable by DSPy's compilers.

### Phase 3: Multi-Hop Reasoning (~20 min)

**Exercise 4 — MultiHopRAG Module** (`src/03_multihop_rag.py`, TODO):
Single-hop retrieval fails when the answer requires synthesizing information from multiple documents. This exercise teaches the multi-hop pattern: hop 1 retrieves initial context and extracts key entities, hop 2 builds a refined query from those entities and retrieves additional passages, then a final ChainOfThought synthesizes all context into an answer. This is how production RAG systems handle complex questions.

### Phase 4: Output Refinement (~20 min)

**Exercise 5 — Reward Functions & Refine** (`src/04_refinement.py`, TODO #1):
Reward functions define what "good output" means programmatically. This exercise teaches how to write reward functions that check citation quality, answer length, and factual grounding — then wrap your RAG module with `dspy.Refine` so it automatically self-corrects when the reward is below threshold. Understanding this pattern is key to building LLM pipelines with quality guarantees.

**Exercise 6 — Handling Unanswerable Questions** (`src/04_refinement.py`, TODO #2):
Real RAG systems must gracefully handle questions that fall outside the knowledge base. This exercise tests your refinement pipeline with intentionally unanswerable questions — observing how the reward function detects hallucination and how Refine's feedback loop steers the model toward honest "I don't know" responses.

### Phase 5: MIPROv2 Optimization & Typed Predictors (~25 min)

**Exercise 7 — MIPROv2 Optimization** (`src/05_mipro_typed.py`, TODO #1):
MIPROv2 represents state-of-the-art prompt optimization. This exercise teaches how to optimize your RAG pipeline with Bayesian search over instructions + demonstrations jointly. Comparing MIPROv2 results with BootstrapFewShot reveals the power of joint optimization.

**Exercise 8 — Typed Predictor with Pydantic** (`src/05_mipro_typed.py`, TODO #2):
Typed predictors enforce structured output at the signature level. This exercise teaches how to define a Pydantic model as an output field, so DSPy validates the LM's JSON output against your schema. This is essential for production systems where downstream code expects specific types.

**Exercise 9 — Save & Load Compiled Program** (`src/05_mipro_typed.py`, TODO #3):
Compiled programs contain optimized prompts and demonstrations. This exercise teaches program serialization — saving a compiled program to disk and loading it back. In production, you compile once (expensive) and deploy the saved program (cheap).

## Motivation

RAG is the most common LLM application pattern in production. DSPy's output refinement modules (Refine/BestOfN) provide unique quality guarantees — automatic self-correction driven by reward functions, with no equivalent in other frameworks. MIPROv2 represents state-of-the-art prompt optimization via Bayesian search. These skills are essential for building production LLM systems that are reliable, measurable, and improvable. Prerequisite for DSPy+LangGraph integration (030c).

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
| `uv run python src/00_verify_setup.py` | Verify DSPy + Qdrant connection |

### Phase 1: Document Ingestion

| Command | Description |
|---------|-------------|
| `uv run python src/01_ingest_documents.py` | Load documents into Qdrant with embeddings |

### Phase 2: Basic RAG

| Command | Description |
|---------|-------------|
| `uv run python src/02_basic_rag.py` | Run basic RAG pipeline with custom Qdrant retriever |

### Phase 3: Multi-Hop Reasoning

| Command | Description |
|---------|-------------|
| `uv run python src/03_multihop_rag.py` | Run multi-hop reasoning with iterative retrieval |

### Phase 4: Output Refinement

| Command | Description |
|---------|-------------|
| `uv run python src/04_refinement.py` | Test Refine/BestOfN with reward functions |

### Phase 5: MIPROv2 & Typed Predictors

| Command | Description |
|---------|-------------|
| `uv run python src/05_mipro_typed.py` | Run MIPROv2 optimization and typed predictors |

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
