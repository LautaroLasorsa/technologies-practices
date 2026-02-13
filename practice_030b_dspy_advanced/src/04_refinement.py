"""
Phase 4 -- Output Refinement: Reward Functions, Refine & BestOfN
=================================================================
This script teaches how to enforce output quality constraints on your RAG
pipeline using DSPy's refinement modules: dspy.Refine and dspy.BestOfN.

You'll write reward functions that evaluate answer quality (citation,
length, factual grounding), then wrap your RAG module with Refine so it
automatically self-corrects when the reward is below threshold.

Run: uv run python src/04_refinement.py
Prereq: uv run python src/01_ingest_documents.py (documents ingested)
"""

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dspy
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Import from 02_basic_rag.py (numeric prefix requires importlib)
_basic_rag = importlib.import_module("02_basic_rag")

COLLECTION_NAME = _basic_rag.COLLECTION_NAME
EMBEDDING_MODEL_NAME = _basic_rag.EMBEDDING_MODEL_NAME
MODEL_ID = _basic_rag.MODEL_ID
OLLAMA_BASE = _basic_rag.OLLAMA_BASE
QDRANT_HOST = _basic_rag.QDRANT_HOST
QDRANT_PORT = _basic_rag.QDRANT_PORT
AnswerWithContext = _basic_rag.AnswerWithContext
BasicRAG = _basic_rag.BasicRAG
QdrantRetriever = _basic_rag.QdrantRetriever


# -- Setup -------------------------------------------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(MODEL_ID, api_base=OLLAMA_BASE, api_key="")
    dspy.configure(lm=lm)


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


# ---------------------------------------------------------------------------
# TODO(human) #1 -- Reward functions & Refine
# ---------------------------------------------------------------------------
# Reward functions are the heart of DSPy's quality control system. They
# evaluate the LM's output and return a score (0.0 = terrible, 1.0 = perfect).
# dspy.Refine uses this score to decide whether to accept the output or
# retry with feedback.
#
# The reward function signature is:
#   def reward_fn(args: dict, pred: dspy.Prediction) -> float
# Where:
#   - args: the original input arguments (e.g., {"question": "..."})
#   - pred: the module's prediction (e.g., pred.answer, pred.reasoning)
#
# What to do:
#   1. Write a reward function `citation_and_length_reward(args, pred)`:
#      This function checks two quality criteria:
#
#      a) Citation check (0.5 points): The answer should reference content
#         from the retrieved context. A simple heuristic: check if at least
#         one significant word (>4 characters) from the answer also appears
#         in the context passages. This catches completely hallucinated
#         answers that share no vocabulary with the retrieved documents.
#
#         Implementation hint:
#           - Get the context from the prediction's trace or from args.
#             Since our BasicRAG stores context internally during forward(),
#             you can check pred.answer against common knowledge terms.
#           - A simpler approach: check that the answer is not empty and
#             contains at least 10 characters (filters garbage outputs).
#           score = 0.5 if len(pred.answer.strip()) >= 10 else 0.0
#
#      b) Length check (0.5 points): The answer should be concise — under
#         100 words. Long, rambling answers often indicate the model is
#         hedging or including irrelevant information.
#           word_count = len(pred.answer.split())
#           score += 0.5 if word_count <= 100 else 0.0
#
#      Return the combined score (0.0, 0.5, or 1.0).
#
#   2. Wrap the RAG pipeline with dspy.Refine:
#        rag = create_rag_pipeline()
#        refined_rag = dspy.Refine(
#            module=rag,
#            N=3,                            # up to 3 attempts
#            reward_fn=citation_and_length_reward,
#            threshold=1.0,                  # accept only perfect scores
#        )
#
#   3. Test with questions from the knowledge base:
#        questions = [
#            "What is the largest moon in the Solar System?",
#            "How many moons does Saturn have?",
#            "What causes the greenhouse effect on Venus?",
#        ]
#        for q in questions:
#            result = refined_rag(question=q)
#            score = citation_and_length_reward({}, result)
#            print(f"Q: {q}")
#            print(f"A: {result.answer}")
#            print(f"Score: {score}")
#
# Why reward functions matter: They make quality criteria explicit and
# measurable. Instead of hoping the LM produces good output, you define
# "good" programmatically and let Refine enforce it with retries + feedback.
# This is the DSPy equivalent of assertion-driven development.
# ---------------------------------------------------------------------------
def test_refinement() -> None:
    raise NotImplementedError("TODO(human): Implement reward functions and Refine")


# ---------------------------------------------------------------------------
# TODO(human) #2 -- Handle unanswerable questions
# ---------------------------------------------------------------------------
# Real RAG systems must handle questions that fall outside the knowledge
# base. When the retrieved passages don't contain the answer, the model
# should say "I don't know" rather than hallucinate.
#
# What to do:
#   1. Write a reward function `no_hallucination_reward(args, pred)`:
#      This function specifically checks for hallucination on out-of-domain
#      questions. The logic:
#
#      a) If the question IS answerable from context (in-domain), reward
#         the answer normally (use citation_and_length_reward).
#
#      b) If the question is NOT answerable from context (out-of-domain),
#         check if the answer admits uncertainty. Look for phrases like:
#         "I don't know", "not enough information", "cannot determine",
#         "not mentioned", "no information available".
#
#         Implementation:
#           uncertainty_phrases = [
#               "don't know", "not enough information", "cannot determine",
#               "not mentioned", "no information", "unable to answer",
#               "not available", "outside", "beyond"
#           ]
#           answer_lower = pred.answer.lower()
#           admits_uncertainty = any(
#               phrase in answer_lower for phrase in uncertainty_phrases
#           )
#           return 1.0 if admits_uncertainty else 0.0
#
#   2. Create a Refine wrapper using no_hallucination_reward:
#        honest_rag = dspy.Refine(
#            module=create_rag_pipeline(),
#            N=3,
#            reward_fn=no_hallucination_reward,
#            threshold=1.0,
#        )
#
#   3. Test with intentionally unanswerable questions:
#        out_of_domain_questions = [
#            "What is the GDP of France?",
#            "Who won the 2024 Super Bowl?",
#            "What programming language is DSPy written in?",
#        ]
#        for q in out_of_domain_questions:
#            result = honest_rag(question=q)
#            print(f"Q: {q}")
#            print(f"A: {result.answer}")
#
#   4. Also test with in-domain questions to verify the model still answers
#      those correctly (the reward function shouldn't penalize good answers
#      to answerable questions).
#
# Why this matters: Hallucination is the #1 failure mode in production RAG.
# By encoding "admit uncertainty" as a reward signal, Refine's feedback
# loop teaches the model to be honest. After a hallucinated first attempt,
# the feedback says "your answer doesn't match the context, try again" --
# and the model learns to say "I don't know" on the retry.
# ---------------------------------------------------------------------------
def test_unanswerable_questions() -> None:
    raise NotImplementedError("TODO(human): Handle unanswerable questions with Refine")


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
