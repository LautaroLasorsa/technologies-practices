"""
Phase 3 -- Multi-Hop RAG: Iterative Retrieval for Complex Questions
====================================================================
Single-hop retrieval fails when the answer requires synthesizing information
from multiple documents. This script implements a multi-hop RAG module that
performs iterative retrieval: hop 1 retrieves initial context and extracts
key entities, hop 2 builds a refined query and retrieves additional passages,
then a final step synthesizes all context into an answer.

Run: uv run python src/03_multihop_rag.py
Prereq: uv run python src/01_ingest_documents.py (documents ingested)
"""

import importlib
import sys
from pathlib import Path

import dspy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Import from 02_basic_rag.py (numeric prefix requires importlib)
sys.path.insert(0, str(Path(__file__).parent))
_basic_rag = importlib.import_module("02_basic_rag")

COLLECTION_NAME = _basic_rag.COLLECTION_NAME
EMBEDDING_MODEL_NAME = _basic_rag.EMBEDDING_MODEL_NAME
MODEL_ID = _basic_rag.MODEL_ID
OLLAMA_BASE = _basic_rag.OLLAMA_BASE
QDRANT_HOST = _basic_rag.QDRANT_HOST
QDRANT_PORT = _basic_rag.QDRANT_PORT
AnswerWithContext = _basic_rag.AnswerWithContext
QdrantRetriever = _basic_rag.QdrantRetriever


# -- Setup -------------------------------------------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(MODEL_ID, api_base=OLLAMA_BASE, api_key="")
    dspy.configure(lm=lm)


def create_clients() -> tuple[QdrantClient, SentenceTransformer]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return client, embedder


# -- Signatures for multi-hop steps -----------------------------------------

class ExtractEntities(dspy.Signature):
    """Extract key entities and concepts from the context that could help answer the question."""

    context: list[str] = dspy.InputField(desc="retrieved passages")
    question: str = dspy.InputField(desc="the original question")
    entities: str = dspy.OutputField(desc="comma-separated list of key entities and concepts found")
    refined_query: str = dspy.OutputField(desc="a refined search query to find additional relevant information")


class SynthesizeAnswer(dspy.Signature):
    """Synthesize a comprehensive answer from all retrieved context passages."""

    all_context: list[str] = dspy.InputField(desc="all passages retrieved across multiple hops")
    question: str = dspy.InputField(desc="the original question")
    answer: str = dspy.OutputField(desc="a comprehensive answer synthesizing information from all context")


# ---------------------------------------------------------------------------
# TODO(human) -- Implement MultiHopRAG module
# ---------------------------------------------------------------------------
# Multi-hop reasoning is essential when a question requires information
# scattered across multiple documents. The pattern is:
#
#   Hop 1: Initial retrieval -> extract entities -> formulate refined query
#   Hop 2: Retrieve with refined query -> get additional passages
#   Final: Synthesize all passages into a comprehensive answer
#
# This mirrors how humans research: you start with a broad search, learn
# key terms, then search again with more specific queries.
#
# What to do:
#   1. Define class MultiHopRAG(dspy.Module):
#      - __init__(self, retriever):
#        - super().__init__()
#        - self.retrieve = retriever
#        - self.extract = dspy.ChainOfThought(ExtractEntities)
#          This module takes the initial retrieved passages and the question,
#          then identifies key entities and formulates a refined search query.
#          ChainOfThought here helps the LM reason about what additional
#          information is needed before producing the refined query.
#        - self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)
#          This module takes ALL passages (from both hops) and produces
#          the final answer, reasoning through the combined context.
#
#   2. Implement forward(self, question: str) -> dspy.Prediction:
#      - Hop 1: Initial retrieval
#          hop1_result = self.retrieve(query=question)
#          hop1_passages = hop1_result.passages
#
#      - Entity extraction + query refinement
#          extraction = self.extract(
#              context=hop1_passages, question=question
#          )
#          Print the extracted entities and refined query for visibility:
#            print(f"  Entities: {extraction.entities}")
#            print(f"  Refined:  {extraction.refined_query}")
#
#      - Hop 2: Retrieve with the refined query
#          hop2_result = self.retrieve(query=extraction.refined_query)
#          hop2_passages = hop2_result.passages
#
#      - Combine all passages (deduplicate if needed):
#          all_passages = list(set(hop1_passages + hop2_passages))
#          Deduplication is important because the refined query may
#          retrieve some of the same documents as the initial query.
#
#      - Final synthesis:
#          result = self.synthesize(
#              all_context=all_passages, question=question
#          )
#          return result
#
# Why multi-hop matters: Consider "What geysers exist on moons orbiting
# ice giant planets?" Hop 1 retrieves info about ice giants (Uranus,
# Neptune) and their moons. The entity extractor identifies "Triton" as
# a key entity. Hop 2 retrieves Triton-specific info mentioning nitrogen
# geysers. Neither hop alone has the complete answer.
# ---------------------------------------------------------------------------
class MultiHopRAG(dspy.Module):
    def __init__(self, retriever: QdrantRetriever):
        super().__init__()
        self.retrieve = retriever
        self.extract = dspy.ChainOfThought(ExtractEntities)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        raise NotImplementedError("TODO(human): Implement MultiHopRAG.forward")


# -- Test questions that require multi-hop reasoning -------------------------

MULTIHOP_QUESTIONS = [
    # Requires: Galilean moons -> Europa -> Europa ocean details
    "What evidence suggests that one of Galileo's discovered moons might harbor life?",

    # Requires: Retrograde orbit moons -> Triton -> Triton geysers
    "Which moon with a retrograde orbit has active geysers?",

    # Requires: Ice giants -> Uranus moons -> Shakespeare naming
    "What naming convention is used for the moons of the planet that rotates on its side?",

    # Requires: Largest moon info -> Ganymede -> compare with Mercury
    "How does the largest moon in the Solar System compare in size to the smallest planet?",

    # Requires: Formation -> rocky vs gas planets -> specific examples
    "Why did rocky planets form closer to the Sun while gas giants formed farther away?",
]


def test_multihop_pipeline(rag: MultiHopRAG) -> None:
    """Run multi-hop test questions and display intermediate steps."""
    for question in MULTIHOP_QUESTIONS:
        print(f"\nQ: {question}")
        print("  [Multi-hop retrieval in progress...]")
        result = rag(question=question)
        print(f"A: {result.answer}")
        print("=" * 60)


def main() -> None:
    configure_dspy()
    client, embedder = create_clients()

    print("=" * 60)
    print("Phase 3: Multi-Hop RAG")
    print("=" * 60)

    retriever = QdrantRetriever(
        collection_name=COLLECTION_NAME,
        qdrant_client=client,
        embedder=embedder,
        k=3,
    )
    rag = MultiHopRAG(retriever=retriever)
    test_multihop_pipeline(rag)


if __name__ == "__main__":
    main()
