"""Phase 5 -- MIPROv2 Optimization & Typed Predictors.

Three groups of small TODOs:
  Optimization:
    1. bootstrap_compile — compile with BootstrapFewShot (baseline).
    2. mipro_compile     — compile with MIPROv2 (Bayesian, jointly optimizes
                           instructions + demos).
  Typed predictors:
    3. build_typed_rag   — assemble a TypedRAG using the StructuredAnswer
                           Pydantic model + JSONAdapter validation.
  Serialization:
    4. save_program      — `program.save(path)` to persist compiled artifacts.
    5. load_program      — `program.load(path)` into a fresh module instance.

The dataset loaders, evaluator, comparison printing, demo loops and the
typed signature/Pydantic model are scaffolded.

Run:
    uv run python -m src._05_mipro_typed

Prereq:
    uv run python -m src._01_ingest_documents
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy
import pydantic
from dspy.teleprompt import BootstrapFewShot, MIPROv2
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


# -- Configuration ----------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.json"
SAVED_PROGRAM_PATH = Path(__file__).parent / "compiled_rag.json"


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


def load_qa_dataset() -> list[dspy.Example]:
    """Load QA dataset and convert to dspy.Example objects."""
    with open(QA_DATASET_PATH) as f:
        raw = json.load(f)
    return [
        dspy.Example(question=item["question"], answer=item["answer"]).with_inputs(
            "question"
        )
        for item in raw
    ]


def split_dataset(
    examples: list[dspy.Example],
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split examples into train (70%) and validation (30%) sets."""
    split_idx = int(len(examples) * 0.7)
    return examples[:split_idx], examples[split_idx:]


def qa_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Token-overlap metric: significant gold words appearing in the prediction."""
    gold_words = set(example.answer.lower().split())
    pred_words = set(pred.answer.lower().split())
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
        "and", "for", "with", "that", "this",
    }
    gold_significant = {w for w in gold_words if len(w) > 3 and w not in stopwords}
    if not gold_significant:
        return 1.0
    return len(gold_significant & pred_words) / len(gold_significant)


def build_evaluator(valset: list[dspy.Example]) -> dspy.Evaluate:
    """Standard evaluator used for all three programs."""
    return dspy.Evaluate(
        devset=valset,
        metric=qa_metric,
        num_threads=1,
        display_progress=True,
    )


# -- TODO 1 -----------------------------------------------------------------
# BootstrapFewShot is the simpler baseline optimizer: it samples training
# examples, runs them through the program, and keeps traces that pass the
# metric — those traces become few-shot demos.  Only demos are optimized;
# instructions stay as-written.
#
# What to do:
#   - bootstrap = BootstrapFewShot(
#         metric=qa_metric,
#         max_bootstrapped_demos=2,
#         max_labeled_demos=2,
#     )
#   - return bootstrap.compile(create_rag_pipeline(), trainset=trainset)
# ---------------------------------------------------------------------------


def bootstrap_compile(trainset: list[dspy.Example]) -> dspy.Module:
    """Compile the RAG pipeline with BootstrapFewShot (baseline optimizer)."""
    raise NotImplementedError("TODO(human): compile with BootstrapFewShot")


# -- TODO 2 -----------------------------------------------------------------
# MIPROv2 uses Bayesian Optimization to jointly optimize instructions AND
# few-shot demonstrations for each predictor in your program.  The
# optimization runs in three stages:
#   1. Bootstrap demos: sample examples, keep traces that pass the metric.
#   2. Propose instructions: generate candidates from data summaries,
#      program code, and traces.
#   3. Bayesian search: pick the best (instruction, demos) combo per
#      predictor on a validation set.
#
# auto modes: "light" (fast, few trials), "medium", "heavy" (best quality).
#
# What to do:
#   - mipro = MIPROv2(
#         metric=qa_metric,
#         auto="light",         # fast mode for local Ollama
#         num_threads=1,
#         verbose=True,
#     )
#   - return mipro.compile(
#         create_rag_pipeline(),
#         trainset=trainset,
#         valset=valset,        # MIPROv2 needs both to score candidates
#     )
#
# Note: with a small local model and limited data the gain may be modest.
# In production with GPT-4 and 500+ examples MIPROv2 typically gives
# 10–30 % improvements over BootstrapFewShot.
# ---------------------------------------------------------------------------


def mipro_compile(
    trainset: list[dspy.Example], valset: list[dspy.Example]
) -> dspy.Module:
    """Compile the RAG pipeline with MIPROv2 (Bayesian, joint instr+demos)."""
    raise NotImplementedError("TODO(human): compile with MIPROv2 (auto='light')")


# -- Optimization orchestrator (scaffolded) ---------------------------------


def optimize_rag() -> dspy.Module:
    """Drive baseline + Bootstrap + MIPROv2 evaluation and print scores."""
    examples = load_qa_dataset()
    trainset, valset = split_dataset(examples)
    print(f"Train: {len(trainset)}, Val: {len(valset)}")

    evaluator = build_evaluator(valset)

    baseline_rag = create_rag_pipeline()
    baseline_score = evaluator(baseline_rag)
    print(f"Baseline score:        {baseline_score:.2f}")

    bootstrap_rag = bootstrap_compile(trainset)
    bootstrap_score = evaluator(bootstrap_rag)
    print(f"BootstrapFewShot score:{bootstrap_score:.2f}")

    mipro_rag = mipro_compile(trainset, valset)
    mipro_score = evaluator(mipro_rag)
    print(f"MIPROv2 score:         {mipro_score:.2f}")

    print(f"\n{'=' * 40}")
    print(f"Baseline:         {baseline_score:.2f}")
    print(f"BootstrapFewShot: {bootstrap_score:.2f}")
    print(f"MIPROv2:          {mipro_score:.2f}")

    return mipro_rag


# -- Typed Predictor: signature + Pydantic model (scaffolded) --------------


class StructuredAnswer(pydantic.BaseModel):
    """Validated, structured RAG answer — DSPy's JSONAdapter enforces this schema."""

    answer: str = pydantic.Field(
        description="A concise factual answer to the question",
    )
    confidence: float = pydantic.Field(
        ge=0.0, le=1.0,
        description="Confidence score from 0.0 to 1.0",
    )
    sources: list[str] = pydantic.Field(
        description="Key facts from the context that support the answer",
    )


class TypedQA(dspy.Signature):
    """Answer the question and provide confidence + supporting sources."""

    context: list[str] = dspy.InputField(
        desc="retrieved passages from the knowledge base",
    )
    question: str = dspy.InputField(desc="the user's question")
    result: StructuredAnswer = dspy.OutputField()


# -- TODO 3 -----------------------------------------------------------------
# Typed predictors enforce structured output at the signature level.
# Instead of getting a plain string, you get a validated Pydantic model.
# The JSONAdapter (default in DSPy 2.5+) prompts the LM to produce JSON
# matching your schema, then validates and parses it.  Validation
# failures trigger automatic retry with error feedback.
#
# What to do — define `TypedRAG(dspy.Module)`:
#   - __init__(self, retriever):
#       super().__init__()
#       self.retrieve = retriever
#       self.generate = dspy.ChainOfThought(TypedQA)
#   - forward(self, question):
#       retrieved = self.retrieve(query=question)
#       return self.generate(context=retrieved.passages, question=question)
#   - return TypedRAG(retriever=...) instantiated with a fresh
#     QdrantRetriever (use the same client/embedder construction as
#     create_rag_pipeline).
#
# Why typed predictors matter: they eliminate the parsing/validation layer
# you'd normally write yourself — the Pydantic model IS the contract
# between the LM module and downstream code.
# ---------------------------------------------------------------------------


def build_typed_rag() -> dspy.Module:
    """Define and instantiate a TypedRAG module using the TypedQA signature."""
    raise NotImplementedError(
        "TODO(human): define TypedRAG(dspy.Module) and return an instance"
    )


# -- Typed predictor demo loop (scaffolded) ---------------------------------


TYPED_TEST_QUESTIONS = [
    "What is the largest planet?",
    "How old is the Solar System?",
]


def test_typed_predictor() -> None:
    """Demo: run a few questions through the TypedRAG module."""
    typed_rag = build_typed_rag()
    for q in TYPED_TEST_QUESTIONS:
        result = typed_rag(question=q)
        structured: StructuredAnswer = result.result
        print(f"\nQ: {q}")
        print(f"  Answer:     {structured.answer}")
        print(f"  Confidence: {structured.confidence}")
        print(f"  Sources:    {structured.sources}")


# -- TODO 4 -----------------------------------------------------------------
# `program.save(path)` serializes the optimized prompts and few-shot
# demonstrations of every predictor in the program to JSON.  The program
# *structure* (module hierarchy, forward logic) is NOT saved — that comes
# from your Python code.  Only the learned parameters (instructions,
# demos) are persisted.
#
# What to do:
#   - program.save(str(SAVED_PROGRAM_PATH))
# ---------------------------------------------------------------------------


def save_program(program: dspy.Module) -> None:
    """Persist the compiled program's learned parameters to SAVED_PROGRAM_PATH."""
    raise NotImplementedError(
        "TODO(human): call program.save(str(SAVED_PROGRAM_PATH))"
    )


# -- TODO 5 -----------------------------------------------------------------
# `program.load(path)` restores instructions + demos onto a fresh program
# instance.  The fresh instance must have the *same* module structure
# (same submodules, same signatures) — only the parameters are loaded.
#
# What to do:
#   - fresh = create_rag_pipeline()
#   - fresh.load(path=str(SAVED_PROGRAM_PATH))
#   - return fresh
#
# Why serialization matters: it separates compilation (dev time, expensive)
# from inference (production, cheap).  You can version-control compiled
# programs, A/B test compilations, and roll back to previous versions.
# ---------------------------------------------------------------------------


def load_program() -> dspy.Module:
    """Construct a fresh pipeline and `.load()` the saved parameters into it."""
    raise NotImplementedError(
        "TODO(human): build a fresh pipeline and load() the saved JSON"
    )


# -- Save/load demo (scaffolded) --------------------------------------------


def test_save_load() -> None:
    """Demo: optimize, save, load, and compare answers from both copies."""
    optimized_program = optimize_rag()
    save_program(optimized_program)
    print(f"Saved to {SAVED_PROGRAM_PATH}")

    loaded_program = load_program()
    print(f"Loaded from {SAVED_PROGRAM_PATH}")

    test_question = "What is the largest volcano in the Solar System?"
    original_result = optimized_program(question=test_question)
    loaded_result = loaded_program(question=test_question)
    print(f"\nQ: {test_question}")
    print(f"Original: {original_result.answer}")
    print(f"Loaded:   {loaded_result.answer}")


def main() -> None:
    configure_dspy()

    print("=" * 60)
    print("Phase 5: MIPROv2 Optimization & Typed Predictors")
    print("=" * 60)

    print("\n--- Part 1: MIPROv2 Optimization ---")
    optimize_rag()

    print("\n--- Part 2: Typed Predictor with Pydantic ---")
    test_typed_predictor()

    print("\n--- Part 3: Save & Load Compiled Program ---")
    test_save_load()


if __name__ == "__main__":
    main()


# Keep these imports referenced so linters don't strip them — they document
# the optimizer types this exercise uses.
_ = (BootstrapFewShot, MIPROv2)
