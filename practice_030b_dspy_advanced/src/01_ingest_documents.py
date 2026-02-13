"""
Phase 1 -- Document Ingestion: Load Documents into Qdrant
==========================================================
This script loads factual documents about the Solar System into a Qdrant
vector database. Each document is embedded using sentence-transformers and
stored with its metadata as payload.

The ingested collection will be used by the RAG exercises in subsequent phases.

Run: uv run python src/01_ingest_documents.py
Prereq: docker compose up -d (Qdrant must be running)
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


# -- Configuration -----------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "solar_system"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors
DATA_PATH = Path(__file__).parent / "data" / "documents.json"


# -- Setup: load data and initialize clients ---------------------------------

def load_documents() -> list[dict]:
    """Load documents from the JSON data file."""
    with open(DATA_PATH) as f:
        return json.load(f)


def create_qdrant_client() -> QdrantClient:
    """Create and return a Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# ---------------------------------------------------------------------------
# TODO(human) -- Ingest documents into Qdrant
# ---------------------------------------------------------------------------
# This is the foundational step for any RAG system: getting your documents
# into a vector database where they can be retrieved by semantic similarity.
#
# The ingestion pipeline has three stages:
#   1. Create (or recreate) a Qdrant collection with the right vector config
#   2. Generate embeddings for each document's text using sentence-transformers
#   3. Upsert the documents as points into the collection
#
# What to do:
#   1. Import SentenceTransformer from sentence_transformers and instantiate
#      it with EMBEDDING_MODEL_NAME. This model converts text into dense
#      vectors suitable for semantic similarity search.
#
#   2. Call create_qdrant_client() and load_documents() to get the client
#      and document list.
#
#   3. Recreate the collection:
#      - client.recreate_collection(
#            collection_name=COLLECTION_NAME,
#            vectors_config=VectorParams(
#                size=EMBEDDING_DIM,
#                distance=Distance.COSINE
#            )
#        )
#      - recreate_collection drops and recreates if it already exists,
#        ensuring a clean state each run.
#
#   4. Generate embeddings for all document texts:
#      - texts = [doc["text"] for doc in documents]
#      - embeddings = model.encode(texts)
#      - sentence-transformers' encode() returns a numpy array of shape
#        (num_docs, EMBEDDING_DIM). Each row is a dense vector.
#
#   5. Build PointStruct objects and upsert:
#      - For each document and its corresponding embedding, create:
#        PointStruct(
#            id=index,         # integer ID (0, 1, 2, ...)
#            vector=embedding.tolist(),  # convert numpy to list
#            payload={
#                "doc_id": doc["id"],
#                "title": doc["title"],
#                "text": doc["text"]
#            }
#        )
#      - The payload stores the original text and metadata — this is what
#        your retriever will return as "passages" in the RAG pipeline.
#      - client.upsert(collection_name=COLLECTION_NAME, points=points)
#
#   6. Verify by printing the collection info:
#      - info = client.get_collection(COLLECTION_NAME)
#      - print(f"Collection '{COLLECTION_NAME}': {info.points_count} points")
#
# After this exercise, you'll have a Qdrant collection with 20 documents
# about the Solar System, each stored as a dense vector + text payload.
# The next exercise will build a DSPy retriever that queries this collection.
# ---------------------------------------------------------------------------
def ingest_documents() -> None:
    raise NotImplementedError("TODO(human): Ingest documents into Qdrant")


def main() -> None:
    print("=" * 60)
    print("Phase 1: Document Ingestion into Qdrant")
    print("=" * 60)
    ingest_documents()


if __name__ == "__main__":
    main()
