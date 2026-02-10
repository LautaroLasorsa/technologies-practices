"""Shared configuration constants for the vector database practice."""

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

COLLECTION_NAME = "articles"
VECTOR_DIMENSION = 128
DEFAULT_BATCH_SIZE = 64

# Categories for synthetic article data
CATEGORIES = [
    "AI",
    "Systems",
    "Security",
    "Databases",
    "Web",
    "Cloud",
    "DevOps",
    "Networking",
]

# Year range for synthetic articles
YEAR_MIN = 2018
YEAR_MAX = 2025
