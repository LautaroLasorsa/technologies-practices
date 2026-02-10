# Practice 008: Vector Databases

## Technologies

- **Qdrant** -- Rust-based open-source vector search engine with HNSW indexing
- **qdrant-client** -- Official Python client with sync/async support
- **NumPy** -- Vector generation and manipulation
- **Docker Compose** -- Local Qdrant instance

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose

## Description

Build a **document similarity search system** that demonstrates core vector database operations: creating collections with different distance metrics, upserting points with rich payloads, performing similarity search with filters, scrolling through results, tuning HNSW parameters, and benchmarking search strategies.

The sample domain is a **technical article corpus** -- each article has a title, category, publication year, word count, and a simulated embedding vector. This mirrors real-world RAG (Retrieval-Augmented Generation) pipelines where you store document embeddings and retrieve the most relevant ones given a query.

### What you'll learn

1. **Vector database fundamentals** -- why specialized vector stores exist vs. brute-force search or traditional DB extensions (pgvector)
2. **Collections & distance metrics** -- Cosine, Dot Product, Euclidean; when each is appropriate
3. **Points, vectors & payloads** -- the core data model (id + vector + metadata)
4. **HNSW indexing** -- how Hierarchical Navigable Small World graphs enable fast ANN search; tuning `m` and `ef_construct`
5. **Similarity search** -- basic nearest-neighbor queries with score thresholds
6. **Filtered search** -- combining vector similarity with payload conditions (must/should/must_not)
7. **Scroll API** -- paginated retrieval for bulk operations
8. **Payload indexing** -- creating indices on payload fields for efficient filtering
9. **Benchmarking** -- measuring latency/recall tradeoffs across distance metrics and HNSW configurations

### Vector database landscape (context)

| Database | Type | Key strength | Trade-off |
|----------|------|-------------|-----------|
| **Qdrant** | Open-source, self-hosted or cloud | Rust performance, rich filtering, payload indexing | Smaller ecosystem than Milvus |
| **Pinecone** | Managed cloud-only | Zero-ops, serverless scale | No self-hosting, vendor lock-in |
| **Weaviate** | Open-source, self-hosted or cloud | GraphQL API, built-in vectorizers | Higher memory at very large scale |
| **Milvus** | Open-source, self-hosted or cloud | Billion-scale, industrial track record | Complex deployment (etcd, MinIO) |
| **FAISS** | Library (not a database) | Raw speed for in-process search | No persistence, no filtering, no API |
| **Chroma** | Open-source, lightweight | Simple API, great for prototyping | Not designed for production scale |

We use Qdrant because it offers the best balance of performance, filtering, and ease of local setup (single Docker container).

## Instructions

### Phase 1: Setup & Concepts (~10 min)

1. Start Qdrant via `docker compose up -d`, verify at `http://localhost:6333/dashboard`
2. Run `app/setup.py` to create the collection and payload indices
3. Run `app/generate_data.py` to produce sample articles with synthetic embeddings
4. Key question: Why can't a regular B-tree or hash index efficiently search high-dimensional vectors?

### Phase 2: Ingesting Vectors (~20 min)

1. Open `app/ingest.py` and read the structure
2. **User implements:** `upsert_batch()` -- upload points in batches using `client.upsert()` with `PointStruct`
3. **User implements:** `upsert_single()` -- upload a single point (useful for real-time updates)
4. Run `python app/ingest.py` to populate the collection
5. Verify: `client.get_collection("articles")` should report the correct point count
6. Key question: Why batch upserts instead of one-by-one? What's the network overhead difference?

### Phase 3: Similarity Search & Filtering (~25 min)

1. Open `app/search.py` and read the structure
2. **User implements:** `basic_search()` -- find top-k similar articles given a query vector
3. **User implements:** `filtered_search()` -- search with payload filters (e.g., category="AI", year >= 2023)
4. **User implements:** `scroll_all()` -- paginated retrieval using `client.scroll()`
5. **User implements:** `search_with_score_threshold()` -- only return results above a similarity threshold
6. Run `python app/search.py` to test all search modes
7. Key question: How does Qdrant combine HNSW traversal with payload filtering? (Hint: filtered HNSW edges)

### Phase 4: Benchmarking & Tuning (~20 min)

1. Open `app/benchmark.py` and read the structure
2. **User implements:** `benchmark_distance_metrics()` -- create collections with Cosine/Dot/Euclidean, measure search latency
3. **User implements:** `benchmark_hnsw_params()` -- vary `m` and `ef_construct`, compare build time and search quality
4. Run `python app/benchmark.py` and analyze the results table
5. Key question: Why does higher `m` improve recall but slow down indexing? What's the memory trade-off?

### Phase 5: Discussion (~10 min)

1. When would you use Qdrant vs. pgvector (Postgres extension)?
2. How do vector databases fit into a RAG pipeline?
3. What happens when your dataset outgrows a single node? (Sharding, replication)
4. Cosine vs. Dot Product: if your embeddings are already L2-normalized, does the choice matter?

## Motivation

- **RAG pipelines**: Vector databases are the retrieval backbone of modern LLM applications -- storing and querying document embeddings at scale
- **Market demand**: Every AI/ML team needs vector search; Qdrant, Pinecone, and Weaviate appear in most AI infrastructure job postings
- **Beyond brute-force**: Understanding HNSW, payload indexing, and distance metrics enables informed architecture decisions (not just "plug in Pinecone")
- **Complements existing skills**: Pairs naturally with embedding models, LLM orchestration frameworks (LangChain, LlamaIndex), and data pipelines

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://python-client.qdrant.tech/)
- [Qdrant Quickstart](https://qdrant.tech/documentation/quickstart/)
- [HNSW Indexing Fundamentals (Qdrant course)](https://qdrant.tech/course/essentials/day-2/what-is-hnsw/)
- [Distance Metrics (Qdrant course)](https://qdrant.tech/course/essentials/day-1/distance-metrics/)
- [Filtering in Qdrant](https://qdrant.tech/documentation/concepts/filtering/)
- [Qdrant Collections](https://qdrant.tech/documentation/concepts/collections/)
- [Vector DB Comparison 2025](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)

## Commands

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start the Qdrant container in detached mode |
| `docker compose down` | Stop and remove the Qdrant container |
| `docker compose down -v` | Stop Qdrant and delete the persistent volume (clean slate) |
| `docker compose logs -f qdrant` | Follow Qdrant container logs |

### Python environment (run from `app/`)

| Command | Description |
|---------|-------------|
| `uv sync` | Install Python dependencies from the lockfile |

### Phase 1: Setup & Data Generation (run from `app/`)

| Command | Description |
|---------|-------------|
| `uv run python setup.py` | Create the Qdrant collection with HNSW config and payload indices |
| `uv run python generate_data.py` | Generate 500 synthetic articles with embeddings to `articles.json` |

### Phase 2: Ingestion (run from `app/`)

| Command | Description |
|---------|-------------|
| `uv run python ingest.py` | Upsert all generated articles into Qdrant (batch + single demo) |

### Phase 3: Search (run from `app/`)

| Command | Description |
|---------|-------------|
| `uv run python search.py` | Run all search modes: basic, filtered, scroll, and score threshold |

### Phase 4: Benchmarking (run from `app/`)

| Command | Description |
|---------|-------------|
| `uv run python benchmark.py` | Benchmark distance metrics and HNSW parameter configurations |

### Verification

| Command | Description |
|---------|-------------|
| Open `http://localhost:6333/dashboard` | Qdrant web dashboard for inspecting collections and points |

## State

`not-started`
