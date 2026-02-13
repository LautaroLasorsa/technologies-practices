"""
Phase 5 -- MIPROv2 Optimization & Typed Predictors
====================================================
This script covers three advanced DSPy topics:
  1. MIPROv2: Bayesian optimization over instructions + demonstrations
  2. Typed Predictors: Pydantic-validated structured outputs
  3. Program serialization: saving and loading compiled programs

Run: uv run python src/05_mipro_typed.py
Prereq: uv run python src/01_ingest_documents.py (documents ingested)
"""

import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import dspy
import pydantic
from dspy.teleprompt import BootstrapFewShot, MIPROv2
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
BasicRAG = _basic_rag.BasicRAG
QdrantRetriever = _basic_rag.QdrantRetriever


# -- Configuration -----------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data"
QA_DATASET_PATH = DATA_DIR / "qa_dataset.json"
SAVED_PROGRAM_PATH = Path(__file__).parent / "compiled_rag.json"


# -- Setup -------------------------------------------------------------------

def configure_dspy() -> None:
    lm = dspy.LM(MODEL_ID, api_base=OLLAMA_BASE, api_key="")
    dspy.configure(lm=lm)


def create_rag_pipeline() -> BasicRAG:
    """Create the full RAG pipeline."""
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

    examples = []
    for item in raw:
        example = dspy.Example(
            question=item["question"],
            answer=item["answer"],
        ).with_inputs("question")
        examples.append(example)

    return examples


def split_dataset(
    examples: list[dspy.Example],
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Split examples into train (70%) and validation (30%) sets."""
    split_idx = int(len(examples) * 0.7)
    return examples[:split_idx], examples[split_idx:]


def qa_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Simple QA metric: check if key terms from the gold answer appear in the prediction."""
    gold_words = set(example.answer.lower().split())
    pred_words = set(pred.answer.lower().split())

    # Filter to significant words (>3 chars, not stopwords)
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to", "and", "for", "with", "that", "this"}
    gold_significant = {w for w in gold_words if len(w) > 3 and w not in stopwords}

    if not gold_significant:
        return 1.0  # No significant words to check

    overlap = gold_significant & pred_words
    return len(overlap) / len(gold_significant)


# ---------------------------------------------------------------------------
# TODO(human) #1 -- MIPROv2 Optimization
# ---------------------------------------------------------------------------
# MIPROv2 uses Bayesian Optimization to jointly optimize instructions AND
# few-shot demonstrations for each predictor in your program. This is more
# powerful than BootstrapFewShot (which only optimizes demonstrations).
#
# The optimization process:
#   1. Bootstrap: Generate candidate demonstrations by running the program
#      on training examples and keeping traces that pass the metric.
#   2. Propose: Generate candidate instructions from data summaries,
#      program code, and example traces.
#   3. Search: Use Bayesian Optimization to find the best combination of
#      instructions + demos that maximizes the metric on a validation set.
#
# What to do:
#   1. Load the QA dataset and split into train/val:
#        examples = load_qa_dataset()
#        trainset, valset = split_dataset(examples)
#        print(f"Train: {len(trainset)}, Val: {len(valset)}")
#
#   2. Create a baseline RAG program and evaluate it:
#        baseline_rag = create_rag_pipeline()
#        evaluator = dspy.Evaluate(
#            devset=valset,
#            metric=qa_metric,
#            num_threads=1,       # single thread for local Ollama
#            display_progress=True,
#        )
#        baseline_score = evaluator(baseline_rag)
#        print(f"Baseline score: {baseline_score:.2f}")
#
#   3. Optimize with BootstrapFewShot (for comparison):
#        bootstrap = BootstrapFewShot(
#            metric=qa_metric,
#            max_bootstrapped_demos=2,
#            max_labeled_demos=2,
#        )
#        bootstrap_rag = bootstrap.compile(
#            create_rag_pipeline(),
#            trainset=trainset,
#        )
#        bootstrap_score = evaluator(bootstrap_rag)
#        print(f"BootstrapFewShot score: {bootstrap_score:.2f}")
#
#   4. Optimize with MIPROv2:
#        mipro = MIPROv2(
#            metric=qa_metric,
#            auto="light",         # fast mode — few trials
#            num_threads=1,        # single thread for local Ollama
#            verbose=True,         # see optimization progress
#        )
#        mipro_rag = mipro.compile(
#            create_rag_pipeline(),
#            trainset=trainset,
#            valset=valset,
#            # MIPROv2 needs trainset AND valset to evaluate candidates
#        )
#        mipro_score = evaluator(mipro_rag)
#        print(f"MIPROv2 score: {mipro_score:.2f}")
#
#   5. Compare all three scores and print a summary:
#        print(f"\n{'='*40}")
#        print(f"Baseline:         {baseline_score:.2f}")
#        print(f"BootstrapFewShot: {bootstrap_score:.2f}")
#        print(f"MIPROv2:          {mipro_score:.2f}")
#
# Note: With a small local model and limited data, the improvements may be
# modest. The pattern is what matters — in production with GPT-4 and 500+
# examples, MIPROv2 can yield 10-30% improvements over the baseline.
# ---------------------------------------------------------------------------
def optimize_rag() -> dspy.Module:
    raise NotImplementedError("TODO(human): MIPROv2 optimization")


# ---------------------------------------------------------------------------
# TODO(human) #2 -- Typed Predictor with Pydantic output
# ---------------------------------------------------------------------------
# Typed predictors enforce structured output at the signature level.
# Instead of getting a plain string, you get a validated Pydantic model.
# This is essential for production systems where downstream code expects
# specific types (e.g., an API returning JSON with a fixed schema).
#
# DSPy handles this via the JSONAdapter: it prompts the LM to produce JSON
# matching your Pydantic schema, then validates and parses the output.
# If validation fails, the adapter retries with error feedback.
#
# What to do:
#   1. Define a Pydantic model for structured RAG answers:
#        class StructuredAnswer(pydantic.BaseModel):
#            answer: str = pydantic.Field(
#                description="A concise factual answer to the question"
#            )
#            confidence: float = pydantic.Field(
#                ge=0.0, le=1.0,
#                description="Confidence score from 0.0 to 1.0"
#            )
#            sources: list[str] = pydantic.Field(
#                description="Key facts from the context that support the answer"
#            )
#
#   2. Define a typed signature using this model:
#        class TypedQA(dspy.Signature):
#            """Answer the question and provide confidence + supporting sources."""
#            context: list[str] = dspy.InputField(
#                desc="retrieved passages from the knowledge base"
#            )
#            question: str = dspy.InputField(desc="the user's question")
#            result: StructuredAnswer = dspy.OutputField()
#
#   3. Create a typed RAG module:
#        class TypedRAG(dspy.Module):
#            def __init__(self, retriever):
#                super().__init__()
#                self.retrieve = retriever
#                self.generate = dspy.ChainOfThought(TypedQA)
#
#            def forward(self, question):
#                retrieved = self.retrieve(query=question)
#                result = self.generate(
#                    context=retrieved.passages, question=question
#                )
#                return result
#
#   4. Test with a few questions and verify the output is structured:
#        for q in ["What is the largest planet?", "How old is the Solar System?"]:
#            result = typed_rag(question=q)
#            structured = result.result  # This is a StructuredAnswer instance
#            print(f"Q: {q}")
#            print(f"  Answer:     {structured.answer}")
#            print(f"  Confidence: {structured.confidence}")
#            print(f"  Sources:    {structured.sources}")
#
# Why typed predictors matter: They eliminate the parsing/validation layer
# you'd normally build yourself. The Pydantic model IS the contract between
# your LM module and downstream code. If the LM produces invalid JSON,
# DSPy's adapter handles retry + error correction automatically.
# ---------------------------------------------------------------------------
def test_typed_predictor() -> None:
    raise NotImplementedError("TODO(human): Implement typed predictor with Pydantic")


# ---------------------------------------------------------------------------
# TODO(human) #3 -- Save and load compiled program
# ---------------------------------------------------------------------------
# In production, you compile once (expensive — runs optimizer, many LM calls)
# and deploy the saved program (cheap — just loads optimized prompts).
# DSPy supports saving/loading via JSON serialization.
#
# What to do:
#   1. Save the optimized program from TODO #1:
#        optimized_program = optimize_rag()  # or use a reference if already run
#        optimized_program.save(str(SAVED_PROGRAM_PATH))
#        print(f"Saved to {SAVED_PROGRAM_PATH}")
#
#   2. Load it back into a fresh program instance:
#        loaded_program = create_rag_pipeline()
#        loaded_program.load(path=str(SAVED_PROGRAM_PATH))
#        print(f"Loaded from {SAVED_PROGRAM_PATH}")
#
#   3. Verify the loaded program works identically:
#        test_question = "What is the largest volcano in the Solar System?"
#        original_result = optimized_program(question=test_question)
#        loaded_result = loaded_program(question=test_question)
#        print(f"Original: {original_result.answer}")
#        print(f"Loaded:   {loaded_result.answer}")
#
#   4. (Optional) Inspect the saved JSON to see what was serialized:
#        with open(SAVED_PROGRAM_PATH) as f:
#            saved_data = json.load(f)
#        print(json.dumps(saved_data, indent=2)[:500])
#
# The saved JSON contains the optimized instructions and few-shot
# demonstrations for each predictor in the program. The program structure
# (module hierarchy, forward logic) is NOT saved — that comes from your
# Python code. Only the learned parameters (prompts, demos) are persisted.
#
# Why serialization matters: It separates compilation (dev time) from
# inference (production). You can version-control compiled programs,
# A/B test different compilations, and roll back to previous versions.
# ---------------------------------------------------------------------------
def test_save_load() -> None:
    raise NotImplementedError("TODO(human): Save and load compiled program")


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
