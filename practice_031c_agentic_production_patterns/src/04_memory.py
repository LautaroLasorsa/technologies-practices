"""Phase 4 — Agent Memory: Long-Term Semantic Memory with Qdrant.

Demonstrates vector-based long-term memory: embed Q&A interactions, store in Qdrant,
and retrieve relevant past interactions to augment future responses.

Run:
    uv run python src/04_memory.py
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# ── Configuration ────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
EMBEDDING_MODEL = "qwen2.5:7b"  # Ollama can generate embeddings too

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "agent_memory"

# Embedding dimension depends on the model. For qwen2.5:7b via Ollama,
# the embedding dimension is typically 3584. We'll detect it dynamically.


# ── Clients ──────────────────────────────────────────────────────────

qdrant = QdrantClient(url=QDRANT_URL)
embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBEDDING_MODEL)
llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=MODEL_NAME)


# ── Collection Setup ─────────────────────────────────────────────────

def ensure_collection(dimension: int) -> None:
    """Create the Qdrant collection if it doesn't exist."""
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )
        print(f"  Created Qdrant collection '{COLLECTION_NAME}' (dim={dimension})")
    else:
        print(f"  Collection '{COLLECTION_NAME}' already exists")


def detect_embedding_dimension() -> int:
    """Embed a test string to determine the embedding dimension."""
    test_embedding = embeddings.embed_query("test")
    return len(test_embedding)


# ── TODO(human) #1: Store Interaction ─────────────────────────────────

async def store_interaction(query: str, response: str) -> str:
    """Embed a Q&A pair and store it in Qdrant.

    TODO(human): Implement this function.

    This is how the agent builds long-term memory. Each interaction (query + response)
    is embedded into a vector and stored in Qdrant. The embedding captures the
    semantic meaning of the interaction — similar questions will have similar vectors,
    enabling retrieval by meaning rather than exact keyword match.

    Steps:
      1. Combine query and response into a single text for embedding:
         text = f"Question: {query}\nAnswer: {response}"
      2. Embed the text using embeddings.embed_query(text)
         This returns a list of floats (the vector representation).
      3. Generate a unique ID: point_id = str(uuid.uuid4())
      4. Store in Qdrant using qdrant.upsert():
         qdrant.upsert(
             collection_name=COLLECTION_NAME,
             points=[PointStruct(
                 id=point_id,
                 vector=embedding_vector,
                 payload={"query": query, "response": response},
             )]
         )
         The payload stores the original text for retrieval — the vector is for
         searching, the payload is what you get back.
      5. Return the point_id

    Returns:
        The Qdrant point ID (string UUID).
    """
    raise NotImplementedError


# ── TODO(human) #2: Retrieve Relevant History ─────────────────────────

async def retrieve_relevant_history(query: str, k: int = 3) -> list[dict[str, str]]:
    """Search Qdrant for past interactions similar to the current query.

    TODO(human): Implement this function.

    This is the retrieval half of the memory system. Given a new query, embed it
    and find the k most similar past interactions. This is the same mechanism as
    RAG (Retrieval-Augmented Generation), but applied to the agent's own past
    conversations instead of external documents.

    Steps:
      1. Embed the query: query_vector = embeddings.embed_query(query)
      2. Search Qdrant for the k nearest neighbors:
         results = qdrant.query_points(
             collection_name=COLLECTION_NAME,
             query=query_vector,
             limit=k,
         )
      3. Extract the payload from each result point:
         history = []
         for point in results.points:
             history.append({
                 "query": point.payload["query"],
                 "response": point.payload["response"],
                 "score": point.score,  # cosine similarity (0.0 to 1.0)
             })
      4. Return the history list (already sorted by score, highest first)

    The score threshold matters in production: a score of 0.9 means very similar,
    0.5 means loosely related. You might filter out results below 0.6 to avoid
    injecting irrelevant context.

    Returns:
        List of dicts with keys: query, response, score.
    """
    raise NotImplementedError


# ── TODO(human) #3: Memory-Augmented Query ────────────────────────────

async def memory_augmented_query(query: str) -> str:
    """Answer a query using the LLM with relevant past interactions as context.

    TODO(human): Implement this function.

    This wires retrieval into generation — the key integration step. By prepending
    relevant past interactions to the prompt, the agent gains "memory" of previous
    conversations. It can reference past answers, remember user preferences, and
    build on prior knowledge without being explicitly told.

    Steps:
      1. Call await retrieve_relevant_history(query, k=3)
      2. If history is found, format it into a context string:
         "Previous relevant interactions:\n"
         For each item: "- Q: {item['query']}\n  A: {item['response']}\n"
      3. Build the prompt:
         f"{history_context}\nBased on any relevant past interactions above, "
         f"answer the following question:\n{query}"
         If no history, just use the query directly.
      4. Call await llm.ainvoke(prompt)
      5. Get the response text from .content
      6. Store this new interaction: await store_interaction(query, response)
         This ensures future queries can find this interaction too —
         the memory grows with each use.
      7. Return the response text

    Returns:
        The LLM's response string.
    """
    raise NotImplementedError


# ── Seed data ────────────────────────────────────────────────────────

SEED_INTERACTIONS = [
    ("What is my preferred programming language?", "Based on our conversations, you prefer Python for rapid prototyping and Rust for systems programming."),
    ("How should I structure a FastAPI project?", "Use the domain-driven layout: routers/, services/, models/, repositories/. Keep business logic in services, not in route handlers."),
    ("What's the best way to handle errors in Python?", "Use explicit exception types, not bare except. Create a hierarchy: AppError > ValidationError, NotFoundError. Return Result types for expected failures."),
    ("What database should I use for vector search?", "Qdrant is excellent for pure vector search. If you need hybrid (vector + keyword), consider Weaviate. For embedded use, try ChromaDB."),
    ("How do I deploy a Docker Compose stack?", "For development: docker compose up -d. For production: use Docker Swarm or migrate to Kubernetes. Always pin image versions in production."),
]

TEST_QUERIES = [
    "What language should I use for my next prototype?",
    "How should I organize my new API project?",
    "I need to add vector search to my application. What are my options?",
    "What's the recommended error handling pattern?",
]


# ── Orchestration ────────────────────────────────────────────────────

async def seed_memory() -> None:
    """Populate Qdrant with seed interactions."""
    print("\nSeeding memory with past interactions...")
    for query, response in SEED_INTERACTIONS:
        point_id = await store_interaction(query, response)
        print(f"  Stored: {query[:50]}... (id={point_id[:8]}...)")
    print(f"  Seeded {len(SEED_INTERACTIONS)} interactions")


async def test_retrieval() -> None:
    """Test that retrieval finds relevant past interactions."""
    print("\nTesting retrieval...")
    for query in TEST_QUERIES:
        print(f"\n  Query: {query}")
        history = await retrieve_relevant_history(query, k=2)
        for item in history:
            print(f"    [{item['score']:.3f}] Q: {item['query'][:60]}...")


async def test_augmented_query() -> None:
    """Test the full memory-augmented query pipeline."""
    print("\nTesting memory-augmented queries...")
    for query in TEST_QUERIES[:2]:
        print(f"\n{'─' * 60}")
        print(f"Query: {query}")
        print(f"{'─' * 60}")
        response = await memory_augmented_query(query)
        print(f"Response: {response[:300]}")


async def main() -> None:
    print("=" * 60)
    print("Phase 4 — Agent Memory")
    print("=" * 60)

    # Detect embedding dimension and ensure collection
    print("\nInitializing...")
    dim = detect_embedding_dimension()
    print(f"  Embedding dimension: {dim}")
    ensure_collection(dim)

    await seed_memory()
    await test_retrieval()
    await test_augmented_query()

    print(f"\n{'=' * 60}")
    print("Phase 4 complete. Agent now has persistent memory across sessions.")
    print(f"Collection '{COLLECTION_NAME}' persists in Qdrant volume.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
