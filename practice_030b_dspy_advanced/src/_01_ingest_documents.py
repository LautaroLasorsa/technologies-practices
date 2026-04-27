"""Phase 1 -- Document Ingestion: Load Solar System docs into Qdrant.

The ingestion pipeline has three stages, each isolated as a small TODO:
  1. Create (or recreate) a Qdrant collection with the right vector config.
  2. Generate embeddings for each document text using sentence-transformers.
  3. Upsert the documents as points (vector + payload) into the collection.

Loaders, the orchestrator, and the verification print are scaffolded.

Run:
    uv run python -m src._01_ingest_documents

Prereq:
    docker compose up -d   (Qdrant must be running)
"""

from __future__ import annotations

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


# -- Configuration ----------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "solar_system"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
DATA_PATH = Path(__file__).parent / "data" / "documents.json"


# -- Scaffolded helpers -----------------------------------------------------


def load_documents() -> list[dict]:
    """Load documents from the JSON data file."""
    with open(DATA_PATH) as f:
        return json.load(f)


def create_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def create_embedder() -> SentenceTransformer:
    """Instantiate the sentence-transformers model used for ingestion + retrieval."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# -- TODO 1 -----------------------------------------------------------------
# Creating (or recreating) a Qdrant collection with the right vector config
# is the first step of any ingestion pipeline.  The vector size MUST match
# the embedding model's output dimension, and the distance metric MUST match
# what your retriever will use at query time (cosine for normalized
# sentence-transformers embeddings).
#
# What to do:
#   - Use client.recreate_collection(collection_name=..., vectors_config=...)
#   - vectors_config = VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
#   - recreate_collection drops and recreates if it already exists, ensuring
#     a clean state on each run of the practice.
# ---------------------------------------------------------------------------


def create_collection(client: QdrantClient) -> None:
    """Recreate the Qdrant collection with the correct vector params.

    Use COSINE distance and EMBEDDING_DIM as the vector size.
    """
    raise NotImplementedError("TODO(human): create the Qdrant collection")


# -- TODO 2 -----------------------------------------------------------------
# sentence-transformers' encode() takes a list of strings and returns a
# numpy array of shape (num_docs, EMBEDDING_DIM).  Each row is the dense
# vector for one document.  The same embedder MUST be used at query time
# (in _02_basic_rag.py) so that document and query vectors live in the
# same semantic space.
#
# What to do:
#   - Pull the "text" field from each document.
#   - Call embedder.encode(texts) and return the resulting numpy array.
# ---------------------------------------------------------------------------


def embed_texts(embedder: SentenceTransformer, documents: list[dict]):
    """Encode every document's text into a dense vector array.

    Returns a numpy array of shape (len(documents), EMBEDDING_DIM).
    """
    raise NotImplementedError("TODO(human): embed the document texts")


# -- TODO 3 -----------------------------------------------------------------
# Upserting in Qdrant means insert-or-update.  Each point carries:
#   - id: integer or UUID (use the index)
#   - vector: list[float] (call .tolist() on a numpy row)
#   - payload: dict of metadata returned alongside search hits
#
# The payload stores the original text and metadata — this is exactly what
# your retriever will later return as `passages` to the RAG pipeline, so
# storing the full text matters even though it is not part of the vector.
#
# What to do:
#   - Build PointStruct(id=i, vector=vec.tolist(),
#       payload={"doc_id": doc["id"], "title": doc["title"], "text": doc["text"]})
#     for each (i, doc, vec) triple.
#   - Call client.upsert(collection_name=COLLECTION_NAME, points=points).
# ---------------------------------------------------------------------------


def upsert_points(client: QdrantClient, documents: list[dict], embeddings) -> None:
    """Wrap each (document, embedding) pair as a PointStruct and upsert."""
    raise NotImplementedError("TODO(human): build PointStructs and upsert them")


# -- Orchestrator + verification (scaffolded) -------------------------------


def ingest_documents() -> None:
    """Drive the full ingestion pipeline: collection -> embeddings -> upsert."""
    client = create_qdrant_client()
    documents = load_documents()
    embedder = create_embedder()

    create_collection(client)
    embeddings = embed_texts(embedder, documents)
    upsert_points(client, documents, embeddings)

    info = client.get_collection(COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}': {info.points_count} points")


def main() -> None:
    print("=" * 60)
    print("Phase 1: Document Ingestion into Qdrant")
    print("=" * 60)
    ingest_documents()


if __name__ == "__main__":
    main()
