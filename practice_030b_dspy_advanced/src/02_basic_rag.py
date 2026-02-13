"""
Phase 2 -- Basic RAG Pipeline: Custom Qdrant Retriever + Generate
==================================================================
This script implements a basic RAG pipeline in DSPy:
  1. A custom retriever that queries Qdrant for relevant passages
  2. A RAG module that chains retrieval with answer generation

The custom retriever bridges DSPy's module system with the Qdrant vector
database populated in Phase 1.

Run: uv run python src/02_basic_rag.py
Prereq: uv run python src/01_ingest_documents.py (documents ingested)
"""

import dspy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


# -- Configuration -----------------------------------------------------------

OLLAMA_BASE = "http://localhost:11434"
MODEL_ID = "ollama_chat/qwen2.5:7b"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "solar_system"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# -- Setup: DSPy + clients --------------------------------------------------

def configure_dspy() -> None:
    """Configure DSPy with the local Ollama LM."""
    lm = dspy.LM(MODEL_ID, api_base=OLLAMA_BASE, api_key="")
    dspy.configure(lm=lm)


def create_clients() -> tuple[QdrantClient, SentenceTransformer]:
    """Create and return Qdrant client and embedding model."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return client, embedder


# -- Signature for answer generation ----------------------------------------

class AnswerWithContext(dspy.Signature):
    """Answer the question based on the provided context passages."""

    context: list[str] = dspy.InputField(desc="retrieved passages from the knowledge base")
    question: str = dspy.InputField(desc="the user's question")
    answer: str = dspy.OutputField(desc="a concise, factual answer based on the context")


# ---------------------------------------------------------------------------
# TODO(human) #1 -- Implement QdrantRetriever
# ---------------------------------------------------------------------------
# DSPy modules communicate via dspy.Prediction objects. Your retriever must:
#   1. Accept a query string
#   2. Embed the query using the same sentence-transformers model used during
#      ingestion (this ensures the vectors are in the same space)
#   3. Search the Qdrant collection for the top-k most similar documents
#   4. Return a dspy.Prediction(passages=[...]) where each passage is a
#      string containing the document text
#
# What to do:
#   1. Define a class QdrantRetriever(dspy.Module):
#      - __init__(self, collection_name, qdrant_client, embedder, k=3):
#        Store all parameters as instance attributes and call super().__init__()
#
#   2. Implement forward(self, query: str) -> dspy.Prediction:
#      - Embed the query: vector = self.embedder.encode(query).tolist()
#      - Search Qdrant:
#          results = self.qdrant_client.query_points(
#              collection_name=self.collection_name,
#              query=vector,
#              limit=self.k,
#              with_payload=True,
#          )
#      - Extract passages from results:
#          passages = [hit.payload["text"] for hit in results.points]
#      - Return dspy.Prediction(passages=passages)
#
# Why this matters: This is the bridge between DSPy's composable module
# system and any external data source. The same pattern works for
# Elasticsearch, Pinecone, Weaviate, or any vector store — you just change
# the query logic inside forward(). By returning dspy.Prediction, your
# retriever integrates seamlessly with DSPy optimizers.
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
        raise NotImplementedError("TODO(human): Implement QdrantRetriever.forward")


# ---------------------------------------------------------------------------
# TODO(human) #2 -- Implement BasicRAG module
# ---------------------------------------------------------------------------
# A RAG module composes retrieval with generation in a single dspy.Module.
# This is the standard DSPy pattern for any retrieval-augmented task:
#   retrieve context -> generate answer using context
#
# What to do:
#   1. Define a class BasicRAG(dspy.Module):
#      - __init__(self, retriever):
#        - Call super().__init__()
#        - Store the retriever: self.retrieve = retriever
#        - Create the generator: self.generate = dspy.ChainOfThought(AnswerWithContext)
#          ChainOfThought adds a "reasoning" step before producing the answer,
#          which improves quality for factual questions by forcing the model
#          to think through the context before answering.
#
#   2. Implement forward(self, question: str) -> dspy.Prediction:
#      - Step 1: Retrieve context
#          retrieved = self.retrieve(query=question)
#          context = retrieved.passages
#      - Step 2: Generate answer using context
#          result = self.generate(context=context, question=question)
#      - Step 3: Return the result (a dspy.Prediction with .answer and .reasoning)
#          return result
#
# Why this matters: Once composed as a dspy.Module, the entire RAG pipeline
# becomes a single unit that DSPy optimizers can compile. The optimizer can
# find few-shot demonstrations that improve BOTH the retrieval query
# formulation AND the answer generation — end-to-end optimization that
# manual prompt engineering cannot achieve.
# ---------------------------------------------------------------------------
class BasicRAG(dspy.Module):
    def __init__(self, retriever: QdrantRetriever):
        super().__init__()
        self.retrieve = retriever
        self.generate = dspy.ChainOfThought(AnswerWithContext)

    def forward(self, question: str) -> dspy.Prediction:
        raise NotImplementedError("TODO(human): Implement BasicRAG.forward")


# -- Test questions ----------------------------------------------------------

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
