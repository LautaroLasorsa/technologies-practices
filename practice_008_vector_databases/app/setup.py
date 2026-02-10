"""
Phase 1: Create the Qdrant collection and payload indices.

This file is FULLY IMPLEMENTED -- run it to set up the collection before
ingesting data. Read through it to understand how collections, distance
metrics, HNSW configuration, and payload indexing work in Qdrant.

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


def create_collection(client) -> None:
    """Create the articles collection with Cosine distance and tuned HNSW."""
    # Delete if it already exists (idempotent setup)
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"  Deleted existing collection '{COLLECTION_NAME}'")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE,
        ),
        # HNSW tuning:
        #   m=16         -> each node connects to up to 16 neighbors (default 16)
        #   ef_construct=100 -> candidates evaluated during graph build (default 100)
        # Higher values = better recall but slower indexing and more memory.
        hnsw_config=models.HnswConfigDiff(
            m=16,
            ef_construct=100,
        ),
    )
    print(f"  Created collection '{COLLECTION_NAME}' (dim={VECTOR_DIMENSION}, Cosine, HNSW m=16)")


def create_payload_indices(client) -> None:
    """Create payload indices on fields used in filtered search.

    Without payload indices, Qdrant must scan all payloads during filtered
    search. With indices, it extends the HNSW graph with extra edges for
    the indexed fields, making filtered search nearly as fast as unfiltered.

    Best practice: create indices right after collection creation, before
    ingesting data. Creating them after ingestion blocks updates temporarily.
    """
    # Keyword index on 'category' -- for exact match filters
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="category",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
    print("  Created KEYWORD index on 'category'")

    # Integer index on 'year' -- for range filters (year >= 2023)
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="year",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    print("  Created INTEGER index on 'year'")

    # Integer index on 'word_count' -- for range filters
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="word_count",
        field_schema=models.PayloadSchemaType.INTEGER,
    )
    print("  Created INTEGER index on 'word_count'")


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
