"""Phase 4 -- Output Refinement: Reward Functions, Refine & BestOfN.

dspy.Refine wraps a module and re-runs it (with feedback hints) until a
reward function clears a threshold.  Reward functions encode "good output"
programmatically — citation grounding, length, honesty about unanswerable
questions, etc.

Four small TODOs:
  1. citation_and_length_reward — write the reward fn for in-domain answers.
  2. wrap_with_refine          — wrap the RAG pipeline with dspy.Refine.
  3. no_hallucination_reward    — reward fn that prefers "I don't know".
  4. wrap_honest_rag           — Refine wrapper using the no-hallucination reward.

The pipeline factory, the question lists, the demo loops and printing
are scaffolded.

Run:
    uv run python -m src._04_refinement

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
    BasicRAG,
    QdrantRetriever,
)
from .llm_config import configure_lm


# -- Setup ------------------------------------------------------------------


def configure_dspy() -> None:
    configure_lm()


def create_rag_pipeline() -> BasicRAG:
    """Create the full RAG pipeline (retriever + generator)."""
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    retriever = QdrantRetriever(
        collection_name=COLLECTION_NAME,
        qdrant_client=client,
        embedder=embedder,
        k=3,
    )
    return BasicRAG(retriever=retriever)


# -- TODO 1 -----------------------------------------------------------------
# Reward functions are the heart of DSPy's quality control system.  They
# evaluate the LM's output and return a float score (0.0 = terrible,
# 1.0 = perfect).  dspy.Refine uses this score to decide whether to accept
# the output or retry with feedback.
#
# The reward function signature is:
#     def reward_fn(args: dict, pred: dspy.Prediction) -> float
# where:
#   - args: the original input arguments (e.g. {"question": "..."})
#   - pred: the module's prediction (e.g. pred.answer, pred.reasoning)
#
# What to do — score two simple criteria summing to at most 1.0:
#   a) Non-empty / non-trivial answer (0.5 points):
#        score = 0.5 if len(pred.answer.strip()) >= 10 else 0.0
#      This catches garbage / empty outputs that share no vocabulary with
#      anything useful.
#   b) Concise (0.5 points): the answer should be under ~100 words.  Long
#      rambling answers usually mean the model is hedging or padding.
#        word_count = len(pred.answer.split())
#        score += 0.5 if word_count <= 100 else 0.0
#   - Return the combined score (0.0, 0.5, or 1.0).
#
# Why reward functions matter: they make quality criteria explicit and
# measurable.  Instead of hoping the LM produces good output, you define
# "good" programmatically and let Refine enforce it with retries +
# feedback — the DSPy equivalent of assertion-driven development.
# ---------------------------------------------------------------------------


def citation_and_length_reward(args: dict, pred: dspy.Prediction) -> float:
    """Reward 0.0 / 0.5 / 1.0 based on length floor + word-count cap."""
    raise NotImplementedError("TODO(human): score length floor + word-count cap")


# -- TODO 2 -----------------------------------------------------------------
# Wrap a freshly-built BasicRAG pipeline with dspy.Refine using the reward
# function from TODO 1.  Refine will re-run the module up to N times,
# generating feedback between attempts, until the reward clears `threshold`.
#
# What to do:
#   - rag = create_rag_pipeline()
#   - return dspy.Refine(
#         module=rag,
#         N=3,                            # up to 3 attempts
#         reward_fn=citation_and_length_reward,
#         threshold=1.0,                  # accept only perfect scores
#     )
# ---------------------------------------------------------------------------


def wrap_with_refine() -> dspy.Module:
    """Build a Refine-wrapped RAG pipeline using the reward fn above."""
    raise NotImplementedError(
        "TODO(human): build a dspy.Refine around create_rag_pipeline()"
    )


# -- TODO 3 -----------------------------------------------------------------
# Real RAG systems must handle questions that fall outside the knowledge
# base.  When the retrieved passages don't contain the answer, the model
# should say "I don't know" rather than hallucinate.
#
# Encode this preference as a reward function: if the answer admits
# uncertainty, give it 1.0; otherwise 0.0.  This penalizes hallucination
# directly — Refine's feedback loop then teaches the model to say "I
# don't know" on the retry.
#
# What to do:
#   uncertainty_phrases = [
#       "don't know", "not enough information", "cannot determine",
#       "not mentioned", "no information", "unable to answer",
#       "not available", "outside", "beyond",
#   ]
#   answer_lower = pred.answer.lower()
#   admits_uncertainty = any(p in answer_lower for p in uncertainty_phrases)
#   return 1.0 if admits_uncertainty else 0.0
#
# Why this matters: hallucination is the #1 failure mode in production
# RAG.  After a hallucinated first attempt, Refine's generated feedback
# will say "your answer is not grounded in the context, try again" — and
# the model learns to be honest.
# ---------------------------------------------------------------------------


def no_hallucination_reward(args: dict, pred: dspy.Prediction) -> float:
    """Reward 1.0 only if the answer admits uncertainty (vs. hallucinates)."""
    raise NotImplementedError(
        "TODO(human): return 1.0 iff pred.answer matches an uncertainty phrase"
    )


# -- TODO 4 -----------------------------------------------------------------
# Build a second Refine wrapper that uses no_hallucination_reward.  Same
# shape as TODO 2, just a different reward_fn.  Used to test out-of-domain
# questions where the knowledge base has nothing useful.
#
# What to do:
#   - return dspy.Refine(
#         module=create_rag_pipeline(),
#         N=3,
#         reward_fn=no_hallucination_reward,
#         threshold=1.0,
#     )
# ---------------------------------------------------------------------------


def wrap_honest_rag() -> dspy.Module:
    """Build a Refine-wrapped RAG using the no_hallucination_reward."""
    raise NotImplementedError(
        "TODO(human): build a dspy.Refine using no_hallucination_reward"
    )


# -- Demo loops (scaffolded) ------------------------------------------------


IN_DOMAIN_QUESTIONS = [
    "What is the largest moon in the Solar System?",
    "How many moons does Saturn have?",
    "What causes the greenhouse effect on Venus?",
]

OUT_OF_DOMAIN_QUESTIONS = [
    "What is the GDP of France?",
    "Who won the 2024 Super Bowl?",
    "What programming language is DSPy written in?",
]


def test_refinement() -> None:
    """Demo: run in-domain questions through the citation+length Refine wrapper."""
    refined_rag = wrap_with_refine()
    for q in IN_DOMAIN_QUESTIONS:
        result = refined_rag(question=q)
        score = citation_and_length_reward({}, result)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        print(f"Score: {score}")
        print("-" * 60)


def test_unanswerable_questions() -> None:
    """Demo: run OOD questions through the no-hallucination Refine wrapper."""
    honest_rag = wrap_honest_rag()
    for q in OUT_OF_DOMAIN_QUESTIONS:
        result = honest_rag(question=q)
        print(f"\nQ: {q}")
        print(f"A: {result.answer}")
        print("-" * 60)


def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 4: Output Refinement")
    print("=" * 60)

    print("\n--- Part 1: Reward Functions & Refine ---")
    test_refinement()

    print("\n--- Part 2: Handling Unanswerable Questions ---")
    test_unanswerable_questions()


if __name__ == "__main__":
    main()
