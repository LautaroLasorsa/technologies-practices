# Practice 067 — GraphRAG: Knowledge Graph-Enhanced RAG

## Technologies

- **Microsoft GraphRAG** (v3.0+) — Graph-based retrieval-augmented generation pipeline
- **Ollama** — Local LLM inference (chat + embeddings)
- **NetworkX** — Graph analysis and metrics
- **Pandas / PyArrow** — Parquet output inspection
- **LanceDB** — Default vector store (built into GraphRAG)

## Stack

Python 3.12+ (uv), Docker (Ollama)

## Theoretical Context

### What GraphRAG Is & The Problem It Solves

Traditional RAG retrieves the top-k semantically similar text chunks to answer a question. This works well for **point queries** ("What is X?") but fails for two critical question classes:

1. **Synthesis questions** — requiring information scattered across many disconnected documents ("How do organizations A, B, and C collaborate?")
2. **Holistic/thematic questions** — about the entire corpus ("What are the major themes in this dataset?")

Traditional RAG fails here because no single chunk contains the answer — the answer emerges from **connections between** chunks.

GraphRAG solves this by building a **knowledge graph** (entities + relationships) from the corpus during indexing, then clustering that graph into **communities** using the Leiden algorithm, and generating **LLM-written summary reports** for each community. At query time, different search modes traverse this structure:

- **Local Search**: entity-centric graph traversal → detailed, focused answers
- **Global Search**: map-reduce over community summaries → thematic, corpus-wide answers

**Original paper**: [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130) (Microsoft Research, 2024)

### How the Indexing Pipeline Works

The pipeline has 6 sequential phases:

```
Documents → TextUnits → Graph Extraction → Graph Augmentation → Community Summarization → Embeddings
```

**Phase 1 — Compose TextUnits**: Input documents are chunked into overlapping TextUnits (default 1200 tokens, 100 overlap). TextUnits are the atomic unit — all extracted knowledge traces back to them.

**Phase 2 — Document Processing**: Documents are linked to their constituent TextUnits for provenance tracking.

**Phase 3 — Graph Extraction** (LLM-intensive, 3 sub-stages):
- **Entity & Relationship Extraction**: The LLM processes each TextUnit to extract entities (title, type, description) and relationships (source, target, description, weight). Subgraphs from overlapping chunks are merged.
- **Entity & Relationship Summarization**: Multiple descriptions per entity/relationship across chunks are consolidated into single summaries by the LLM.
- **Claim Extraction** (optional, disabled by default): Extracts time-bound factual statements (covariates) about entities.

**Phase 4 — Graph Augmentation**: The **Hierarchical Leiden Algorithm** is applied recursively to cluster entities into communities at multiple granularity levels (level 0 = most granular, higher = broader).

**Phase 5 — Community Summarization**: The LLM generates executive summaries and reports for each community at each hierarchical level. These reports are what Global Search queries against.

**Phase 6 — Text Embedding**: Vector embeddings are generated for TextUnits, entity descriptions, and community report content. Written to the vector store (LanceDB by default).

### Key Concepts

| Concept | Description |
|---------|-------------|
| **TextUnit** | Overlapping chunk of source text (default 1200 tokens). Smallest unit processed by LLMs. All extracted knowledge traces back to TextUnits. |
| **Entity** | A named thing extracted from text — person, organization, location, event. Has: title, type, description, and a vector embedding. |
| **Relationship** | A directed connection between two entities with a description and `combined_degree` (sum of endpoint degrees = importance weight). |
| **Community** | A cluster of densely-connected entities produced by Leiden. Communities exist at multiple hierarchical levels. |
| **CommunityReport** | An LLM-generated executive summary for a community. Contains: title, summary, findings, rating. These power Global Search. |
| **Covariate/Claim** | Optional time-bound factual statement about an entity (e.g., "Company X raised $Y in Q3"). Disabled by default. |

### Query Modes

**Local Search** — Entry point: embed the query → find nearest entities via vector similarity → traverse the graph pulling connected entities, relationships, community reports, and original TextUnits → build a rich context window → LLM generates the answer. Best for entity-specific questions.

**Global Search** — Map phase: the question is sent to every community report (or sampled subset), each producing a partial answer with a score. Reduce phase: partial answers are aggregated into a final synthesis. Best for corpus-wide thematic questions. Expensive — processes potentially hundreds of reports per query.

**DRIFT Search** — Hybrid: starts like local (entity-centric) but incorporates community context and generates follow-up sub-questions to broaden scope. Best quality, highest cost.

**Basic Search** — Standard vector similarity over TextUnits. Naive RAG baseline for comparison.

### Ecosystem Context

**GraphRAG vs traditional RAG**: Traditional RAG retrieves chunks independently — no awareness of entity relationships or corpus-level themes. GraphRAG trades indexing cost (LLM calls during graph construction) for dramatically better multi-hop and thematic query answering.

**GraphRAG vs LangGraph/LangChain**: Not competing technologies. LangChain/LangGraph are orchestration frameworks for building LLM applications. GraphRAG is a specific RAG architecture. You could use LangChain to build a pipeline that uses GraphRAG's output artifacts.

**Key trade-off**: GraphRAG's indexing is expensive — every TextUnit requires LLM calls for entity extraction + summarization. With GPT-4, indexing a modest corpus costs significant API spend. With local models (Ollama), quality degrades but cost is zero. This practice uses Ollama to understand the architecture without API costs.

## Description

Build and query a knowledge graph from a small corpus of interconnected articles about a fictional tech ecosystem. You'll run the full GraphRAG indexing pipeline with a local LLM, then analyze the extracted knowledge graph structure, community hierarchy, and compare query modes.

### What you'll learn

1. **GraphRAG configuration** — setting up the pipeline for local LLMs via Ollama
2. **Indexing pipeline** — understanding each phase and what artifacts it produces
3. **Knowledge graph analysis** — loading entities/relationships, computing graph metrics with NetworkX
4. **Community structure** — how Leiden clustering organizes entities hierarchically
5. **Query mode comparison** — when local vs global vs basic search excels

## Instructions

### Phase 1: Setup & Configuration (~15 min)

1. Start Docker services with `docker compose up -d`, then pull models:
   - `docker exec graphrag_ollama ollama pull qwen2.5:7b` (chat model)
   - `docker exec graphrag_ollama ollama pull nomic-embed-text` (embedding model)
2. Install Python dependencies with `uv sync`
3. Run `uv run python src/00_verify_setup.py` to confirm Ollama connectivity
4. Initialize the GraphRAG project: `uv run graphrag init --root .`
   This creates `settings.yaml` and `.env`. The input corpus is already provided.
5. **Configure `settings.yaml` for Ollama.** This is the critical setup step — you need to point GraphRAG at your local Ollama instance instead of OpenAI. Edit the generated `settings.yaml`:
   - Under `models` → `default_chat_model`: set `type: openai_chat`, `model: qwen2.5:7b`, `api_base: http://localhost:11434/v1`, `api_key: ollama`
   - Under `models` → `default_embedding_model`: set `type: openai_embedding`, `model: nomic-embed-text`, `api_base: http://localhost:11434/api`, `api_key: ollama`
   - Under `chunks`: set `size: 600` and `overlap: 100` (smaller chunks work better with 7B models)
   - Under `entity_extraction`: set `entity_types: [person, organization, location, event, technology]`

   **Note:** The exact YAML structure depends on your GraphRAG version. Run `graphrag init` first, then adapt the generated template. Consult the [GraphRAG config docs](https://microsoft.github.io/graphrag/config/yaml/) if field names differ.

### Phase 2: Indexing (~20 min)

1. Run the indexing pipeline: `uv run graphrag index --root .`
   This will take 10-20 minutes with a local 7B model. Watch the output — it shows each pipeline phase executing.
2. While indexing runs, review the input corpus in `input/` to understand the entities and relationships you expect the pipeline to extract. The corpus describes a fictional tech ecosystem in "Meridian City" with companies, people, partnerships, and events.
3. After indexing completes, verify the output: `ls output/` should show Parquet files (entities, relationships, communities, community_reports, text_units, documents) and a `lancedb/` directory.

### Phase 3: Knowledge Graph Inspection (~25 min)

1. **Exercise (`src/01_inspect_knowledge_graph.py`):** Load the extracted entities and relationships from Parquet, build a NetworkX directed graph, and compute structural metrics. This teaches you what the LLM-based extraction pipeline actually produces — you'll see which entities it found, how they connect, and which are the "hubs" of the knowledge graph. Understanding graph structure is essential for knowing whether your extraction configuration (entity types, chunk size, model choice) is working well.

### Phase 4: Community Analysis (~20 min)

1. **Exercise (`src/02_community_analysis.py`):** Analyze the community hierarchy and inspect community reports. This teaches the key innovation behind Global Search — how the Leiden algorithm clusters entities and how LLM-generated summaries capture the themes of each cluster. You'll see the hierarchical structure (fine-grained to broad communities) that enables answering questions at different levels of abstraction.

### Phase 5: Query Mode Comparison (~20 min)

1. **Exercise (`src/03_query_comparison.py`):** Run a set of questions through local, global, and basic search modes and compare results side-by-side. This is the payoff exercise — you'll see concretely how graph-based retrieval (local/global) outperforms naive chunk retrieval (basic) for different question types, and build intuition for when to use each mode.

## Motivation

- **Production RAG limitation**: Naive RAG is the default pattern, but it fails on multi-hop and thematic queries — GraphRAG addresses the most common production RAG failure modes
- **Microsoft-backed**: GraphRAG is actively developed by Microsoft Research, used in Azure AI, and increasingly adopted in enterprise applications
- **Knowledge graphs + LLMs**: The intersection of structured knowledge and generative AI is a rapidly growing area — understanding GraphRAG teaches transferable concepts (entity extraction, community detection, graph-augmented retrieval)
- **Complements practices 029a/029b**: Understanding GraphRAG's architecture deepens your RAG knowledge beyond the basic retrieval patterns covered in the LangChain/LangGraph practices

## Commands

All commands run from `practice_067_graphrag/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama container (port 11434, persistent volume) |
| `docker compose down` | Stop and remove Ollama container |
| `docker exec graphrag_ollama ollama pull qwen2.5:7b` | Download 7B chat model for entity extraction and querying |
| `docker exec graphrag_ollama ollama pull nomic-embed-text` | Download embedding model for vector search |

### Project Setup

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from `pyproject.toml` |
| `uv run python src/00_verify_setup.py` | Verify Ollama connection and model availability |
| `uv run graphrag init --root .` | Initialize GraphRAG project (creates settings.yaml, .env) |

### Indexing

| Command | Description |
|---------|-------------|
| `uv run graphrag index --root .` | Run the full indexing pipeline (entity extraction → communities → embeddings) |
| `uv run graphrag index --root . --verbose` | Run indexing with verbose logging to see each phase |

### Querying (CLI)

| Command | Description |
|---------|-------------|
| `uv run graphrag query --root . --method local --query "Who is Dr. James Chen?"` | Local search — entity-centric graph traversal |
| `uv run graphrag query --root . --method global --query "What are the major themes?"` | Global search — map-reduce over community summaries |
| `uv run graphrag query --root . --method basic --query "What companies are in Meridian?"` | Basic search — naive vector RAG baseline |
| `uv run graphrag query --root . --method drift --query "How do companies collaborate?"` | DRIFT search — hybrid local+global with follow-up sub-questions |

### Exercises

| Command | Description |
|---------|-------------|
| `uv run python src/00_verify_setup.py` | Verify Ollama setup before starting |
| `uv run python src/01_inspect_knowledge_graph.py` | Exercise 1: Load and analyze extracted entities/relationships |
| `uv run python src/02_community_analysis.py` | Exercise 2: Analyze community hierarchy and reports |
| `uv run python src/03_query_comparison.py` | Exercise 3: Compare local vs global vs basic search |

## References

- [Microsoft GraphRAG GitHub](https://github.com/microsoft/graphrag)
- [GraphRAG Official Docs](https://microsoft.github.io/graphrag/)
- [GraphRAG Default Dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/)
- [GraphRAG Configuration Reference](https://microsoft.github.io/graphrag/config/yaml/)
- [GraphRAG Query Overview](https://microsoft.github.io/graphrag/query/overview/)
- [Original Paper: From Local to Global (arXiv:2404.16130)](https://arxiv.org/abs/2404.16130)
- [DRIFT Search: Combining Global and Local Methods](https://www.microsoft.com/en-us/research/blog/introducing-drift-search-combining-global-and-local-search-methods-to-improve-quality-and-efficiency/)
- [Leiden Algorithm (community detection)](https://www.nature.com/articles/s41598-019-41695-z)

## State

`not-started`
