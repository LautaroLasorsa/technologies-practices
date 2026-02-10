"""Qdrant client factory -- single place to create the client connection."""

from qdrant_client import QdrantClient

from config import QDRANT_HOST, QDRANT_PORT


def create_client() -> QdrantClient:
    """Create and return a QdrantClient connected to the local Qdrant instance."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
