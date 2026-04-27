"""Phase 3 -- Multi-Hop RAG: Iterative Retrieval for Complex Questions.

Single-hop retrieval fails when the answer requires synthesizing information
from multiple documents.  This module performs two retrieval hops with an
entity-extraction step in between, then synthesizes a final answer.

Three small TODOs split the multi-hop forward pass:
  1. hop1_retrieve              — initial retrieval for the original question.
  2. extract_and_refine         — extract entities + propose a refined query.
  3. hop2_and_synthesize        — second retrieval + final synthesis.

The signatures, sub-modules, demo loop and printing are scaffolded.

Run:
    uv run python -m src._03_multihop_rag

Prereq:
    uv run python -m src._01_ingest_documents
"""

from __future__ import annotations

import dspy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from ._02_basic_rag import (
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    QdrantRetriever,
)
from .llm_config import configure_lm


# -- Setup helpers ----------------------------------------------------------


def configure_dspy() -> None:
    configure_lm()


def create_clients() -> tuple[QdrantClient, SentenceTransformer]:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return client, embedder


# -- Signatures for the multi-hop steps ------------------------------------


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


# -- TODO 1 -----------------------------------------------------------------
# The first hop is just a vanilla retrieval against the original question.
# Use the retriever stored on the module (`self.retrieve`) and return the
# list of passages from the resulting Prediction.
#
# What to do:
#   - hop1_result = self.retrieve(query=question)
#   - return hop1_result.passages
# ---------------------------------------------------------------------------


class MultiHopRAG(dspy.Module):
    def __init__(self, retriever: QdrantRetriever):
        super().__init__()
        self.retrieve = retriever
        self.extract = dspy.ChainOfThought(ExtractEntities)
        self.synthesize = dspy.ChainOfThought(SynthesizeAnswer)

    def hop1_retrieve(self, question: str) -> list[str]:
        """First hop: retrieve passages for the original question."""
        raise NotImplementedError("TODO(human): retrieve passages for `question`")

    # -- TODO 2 ---------------------------------------------------------------
    # Given the hop-1 passages, ask the extract sub-module to identify key
    # entities and propose a refined search query.  ChainOfThought helps the
    # LM reason about what's missing from hop 1 before producing the new
    # query.  Return the full Prediction so the caller can read both
    # `.entities` (printed for visibility) and `.refined_query`.
    #
    # What to do:
    #   - return self.extract(context=hop1_passages, question=question)
    # -------------------------------------------------------------------------

    def extract_and_refine(
        self, hop1_passages: list[str], question: str
    ) -> dspy.Prediction:
        """Extract entities from hop-1 passages and propose a refined query."""
        raise NotImplementedError(
            "TODO(human): call self.extract on (context, question)"
        )

    # -- TODO 3 ---------------------------------------------------------------
    # Use the refined query to perform the second retrieval, merge passages
    # from both hops (deduplicate — the refined query often retrieves some
    # of the same docs), and synthesize the final answer.
    #
    # What to do:
    #   - hop2_passages = self.retrieve(query=refined_query).passages
    #   - all_passages = list(set(hop1_passages + hop2_passages))
    #   - return self.synthesize(all_context=all_passages, question=question)
    # -------------------------------------------------------------------------

    def hop2_and_synthesize(
        self,
        hop1_passages: list[str],
        refined_query: str,
        question: str,
    ) -> dspy.Prediction:
        """Second hop + final synthesis."""
        raise NotImplementedError(
            "TODO(human): retrieve with refined_query, merge passages, synthesize"
        )

    # -- Orchestrator (scaffolded) -------------------------------------------

    def forward(self, question: str) -> dspy.Prediction:
        hop1_passages = self.hop1_retrieve(question)
        extraction = self.extract_and_refine(hop1_passages, question)
        print(f"  Entities: {extraction.entities}")
        print(f"  Refined:  {extraction.refined_query}")
        return self.hop2_and_synthesize(
            hop1_passages, extraction.refined_query, question
        )


# -- Test questions that require multi-hop reasoning ------------------------

MULTIHOP_QUESTIONS = [
    # Galilean moons -> Europa -> Europa ocean details
    "What evidence suggests that one of Galileo's discovered moons might harbor life?",
    # Retrograde-orbit moons -> Triton -> Triton geysers
    "Which moon with a retrograde orbit has active geysers?",
    # Ice giants -> Uranus moons -> Shakespeare naming
    "What naming convention is used for the moons of the planet that rotates on its side?",
    # Largest moon -> Ganymede -> compare with Mercury
    "How does the largest moon in the Solar System compare in size to the smallest planet?",
    # Formation -> rocky vs gas planets -> specific examples
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
