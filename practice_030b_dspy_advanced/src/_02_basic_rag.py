"""Phase 2 -- Basic RAG Pipeline: Custom Qdrant Retriever + Generate.

Two small TODOs in this file:
  1. QdrantRetriever.forward — query Qdrant and return passages as a Prediction.
  2. BasicRAG.forward       — chain retrieval into a ChainOfThought generator.

The signature, module skeletons, the demo loop, and printing are scaffolded.

Run:
    uv run python -m src._02_basic_rag

Prereq:
    uv run python -m src._01_ingest_documents   (documents ingested)
"""

from __future__ import annotations

import dspy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from .llm_config import configure_lm


# -- Configuration ----------------------------------------------------------

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "solar_system"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# -- Setup helpers ----------------------------------------------------------


def configure_dspy() -> None:
    """Configure DSPy with the LM selected via environment variables."""
    configure_lm()


def create_clients() -> tuple[QdrantClient, SentenceTransformer]:
    """Create a Qdrant client and a sentence-transformers embedder."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return client, embedder


# -- Signature for answer generation ---------------------------------------


class AnswerWithContext(dspy.Signature):
    """Answer the question based on the provided context passages."""

    context: list[str] = dspy.InputField(desc="retrieved passages from the knowledge base")
    question: str = dspy.InputField(desc="the user's question")
    answer: str = dspy.OutputField(desc="a concise, factual answer based on the context")


# -- TODO 1 -----------------------------------------------------------------
# DSPy modules communicate via dspy.Prediction objects.  Your retriever must:
#   1. Accept a query string.
#   2. Embed the query using the same sentence-transformers model used during
#      ingestion (so query and document vectors live in the same space).
#   3. Search the Qdrant collection for the top-k most similar documents.
#   4. Return a dspy.Prediction(passages=[...]) where each passage is a
#      string containing the document text.
#
# What to do (inside forward):
#   - vector = self.embedder.encode(query).tolist()
#   - results = self.qdrant_client.query_points(
#         collection_name=self.collection_name,
#         query=vector,
#         limit=self.k,
#         with_payload=True,
#     )
#   - passages = [hit.payload["text"] for hit in results.points]
#   - return dspy.Prediction(passages=passages)
#
# Why this matters: this is the bridge between DSPy's composable module
# system and any external vector store (Elasticsearch, Pinecone, Weaviate,
# Qdrant…).  By returning a dspy.Prediction, your retriever integrates
# cleanly with DSPy optimizers and downstream modules.
# ---------------------------------------------------------------------------


class QdrantRetriever(dspy.Module):
    def __init__(
        self,
        collection_name: str,
        qdrant_client: QdrantClient,
        embedder: SentenceTransformer,
        k: int = 3,
    ):
        super().__init__()
        self.collection_name = collection_name
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.k = k

    def forward(self, query: str) -> dspy.Prediction:
        """Embed `query`, search Qdrant, return Prediction(passages=[...])."""
        raise NotImplementedError("TODO(human): implement QdrantRetriever.forward")


# -- TODO 2 -----------------------------------------------------------------
# A RAG module composes retrieval with generation in a single dspy.Module.
# The forward method drives the pipeline:
#     retrieve context  ->  generate answer using context
# Once the whole pipeline lives inside a dspy.Module, DSPy's compilers can
# optimize BOTH the retrieval query formulation AND the answer generation
# end-to-end — something manual prompt engineering cannot achieve.
#
# What to do (inside forward):
#   - retrieved = self.retrieve(query=question)
#   - context = retrieved.passages
#   - return self.generate(context=context, question=question)
#     (a dspy.Prediction with .answer and .reasoning, since we use ChainOfThought)
# ---------------------------------------------------------------------------


class BasicRAG(dspy.Module):
    def __init__(self, retriever: QdrantRetriever):
        super().__init__()
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought(AnswerWithContext)

    def forward(self, question: str) -> dspy.Prediction:
        """Retrieve context for `question`, then call self.generate."""
        raise NotImplementedError("TODO(human): implement BasicRAG.forward")


# -- Demo loop (scaffolded) -------------------------------------------------


TEST_QUESTIONS = [
    "What is the largest volcano in the Solar System?",
    "Why is Europa considered a candidate for extraterrestrial life?",
    "How was Neptune discovered?",
    "What is the surface temperature of Venus?",
    "When did Voyager 1 enter interstellar space?",
]


def test_rag_pipeline(rag: BasicRAG) -> None:
    """Run test questions through the RAG pipeline and display results."""
    for question in TEST_QUESTIONS:
        print(f"\nQ: {question}")
        result = rag(question=question)
        print(f"A: {result.answer}")
        print(f"   Reasoning: {result.reasoning[:120]}...")
        print("-" * 60)


def main() -> None:
    configure_dspy()
    client, embedder = create_clients()

    print("=" * 60)
    print("Phase 2: Basic RAG Pipeline")
    print("=" * 60)

    retriever = QdrantRetriever(
        collection_name=COLLECTION_NAME,
        qdrant_client=client,
        embedder=embedder,
        k=3,
    )
    rag = BasicRAG(retriever=retriever)
    test_rag_pipeline(rag)


if __name__ == "__main__":
    main()
