"""
Phase 1: Create the Qdrant collection and payload indices.

This file creates the collection and indices needed before ingesting data.

Key concepts:
  - A **collection** is a named container for vectors + payloads (like a table).
  - **VectorParams** defines the vector size and distance metric.
  - **HnswConfigDiff** tunes the HNSW graph (m = edges per node, ef_construct = build quality).
  - **Payload index** accelerates filtered search on specific payload fields.

Run: python app/setup.py
"""

from qdrant_client import models

from client import create_client
from config import COLLECTION_NAME, VECTOR_DIMENSION


# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches how vector database collections and indices are configured.
# Understanding distance metrics, HNSW parameters, and payload indexing is crucial for
# designing vector search systems that balance speed, recall, and filtering capabilities.

def create_collection(client) -> None:
    """Create the articles collection with Cosine distance and tuned HNSW.

    TODO(human): Implement this function.

    Steps:
      1. Check if the collection already exists and delete it if so (idempotent setup):
         if client.collection_exists(COLLECTION_NAME):
             client.delete_collection(COLLECTION_NAME)
             print(f"  Deleted existing collection '{COLLECTION_NAME}'")

      2. Create the collection using client.create_collection() with:
         - collection_name=COLLECTION_NAME
         - vectors_config=models.VectorParams(
               size=VECTOR_DIMENSION,
               distance=models.Distance.COSINE,
           )
         - hnsw_config=models.HnswConfigDiff(
               m=16,              # Each node connects to up to 16 neighbors
               ef_construct=100,  # Candidates evaluated during graph build
           )

      Higher m/ef_construct values give better recall but use more memory and slower indexing.
      Cosine distance is ideal for normalized embeddings (most neural network outputs).

      3. Print a confirmation message.

    Docs: https://qdrant.tech/documentation/concepts/collections/
    """
    raise NotImplementedError("Implement create_collection — see TODO above")


# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches payload indexing, which enables fast filtered vector search.
# Without indices, filtered search must scan all payloads; with indices, Qdrant extends
# the HNSW graph to make filtered search nearly as fast as unfiltered search.

def create_payload_indices(client) -> None:
    """Create payload indices on fields used in filtered search.

    TODO(human): Implement this function.

    Steps:
      1. Create a KEYWORD index on 'category' field for exact match filters:
         client.create_payload_index(
             collection_name=COLLECTION_NAME,
             field_name="category",
             field_schema=models.PayloadSchemaType.KEYWORD,
         )
         print("  Created KEYWORD index on 'category'")

      2. Create an INTEGER index on 'year' field for range filters (year >= 2023):
         client.create_payload_index(
             collection_name=COLLECTION_NAME,
             field_name="year",
             field_schema=models.PayloadSchemaType.INTEGER,
         )
         print("  Created INTEGER index on 'year'")

      3. Create an INTEGER index on 'word_count' field for range filters:
         client.create_payload_index(
             collection_name=COLLECTION_NAME,
             field_name="word_count",
             field_schema=models.PayloadSchemaType.INTEGER,
         )
         print("  Created INTEGER index on 'word_count'")

    Best practice: create indices right after collection creation, before ingesting data.
    Creating them after ingestion blocks updates temporarily while building the index.

    Docs: https://qdrant.tech/documentation/concepts/indexing/#payload-index
    """
    raise NotImplementedError("Implement create_payload_indices — see TODO above")


def verify_collection(client) -> None:
    """Print collection info to verify setup."""
    info = client.get_collection(COLLECTION_NAME)
    print(f"\n  Collection info:")
    print(f"    Status:          {info.status}")
    print(f"    Vectors count:   {info.vectors_count}")
    print(f"    Points count:    {info.points_count}")
    print(f"    Vector size:     {info.config.params.vectors.size}")
    print(f"    Distance metric: {info.config.params.vectors.distance}")


def main() -> None:
    print("=== Phase 1: Setting up Qdrant collection ===\n")

    client = create_client()

    print("[1/3] Creating collection...")
    create_collection(client)

    print("[2/3] Creating payload indices...")
    create_payload_indices(client)

    print("[3/3] Verifying collection...")
    verify_collection(client)

    print("\nSetup complete. Next: run 'python app/generate_data.py' then 'python app/ingest.py'")


if __name__ == "__main__":
    main()
