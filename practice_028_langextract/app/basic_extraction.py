"""Phase 1: Basic entity extraction with LangExtract.

Demonstrates:
  - Defining extraction schemas via ExampleData
  - Calling lx.extract() with a local Ollama model
  - Inspecting the AnnotatedDocument result

Run after Docker Compose is up and the model is pulled:
    uv run python basic_extraction.py
"""

import langextract as lx

import config

# ── Sample text to extract from ─────────────────────────────────────

ARTICLE_TEXT = """\
On March 15, 2025, Anthropic announced a partnership with Acme Corp to develop \
AI safety tools. The collaboration, led by CEO Dario Amodei and Acme's CTO \
Sarah Chen, will focus on interpretability research. The project has a budget \
of $50 million and is expected to run for 3 years. Anthropic, based in \
San Francisco, has been a pioneer in AI safety since its founding in 2021."""


# ── TODO(human): Define an ExampleData and run extraction ───────────


def create_example() -> lx.ExampleData:
    """Create a few-shot example that teaches LangExtract what to extract.

    TODO(human): Implement this function.

    You need to build an ExampleData object that demonstrates the extraction
    schema you want. ExampleData takes:
      - text: str — a sample source text (can be ARTICLE_TEXT or a different example)
      - extractions: list[lx.Extraction] — the entities you expect to find

    Each Extraction has:
      - extraction_class: str — category label (e.g., "person", "organization",
        "date", "money", "location")
      - extraction_text: str — MUST be a VERBATIM quote from the text.
        This is critical: LangExtract uses exact string matching to compute
        character offsets (source grounding). If you paraphrase or reword,
        grounding fails silently.

    IMPORTANT: List extractions in the ORDER they appear in the text.
    LangExtract uses positional ordering as an additional signal.

    Example pattern:
        lx.ExampleData(
            text="John works at Google in NYC since 2020.",
            extractions=[
                lx.Extraction(extraction_class="person", extraction_text="John"),
                lx.Extraction(extraction_class="organization", extraction_text="Google"),
                lx.Extraction(extraction_class="location", extraction_text="NYC"),
                lx.Extraction(extraction_class="date", extraction_text="2020"),
            ],
        )

    For this exercise, create an example using ARTICLE_TEXT (or a similar text)
    with extraction classes: "date", "organization", "person", "money", "location".

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement create_example")


def run_extraction(example: lx.ExampleData) -> lx.AnnotatedDocument:
    """Run extraction on ARTICLE_TEXT using the provided example.

    TODO(human): Implement this function.

    Steps:
      1. Call lx.extract() with:
         - text_or_documents=ARTICLE_TEXT
         - prompt_description: a string describing what to extract, e.g.,
           "Extract all named entities: people, organizations, dates, monetary
           amounts, and locations."
         - examples=[example]  (the ExampleData you created)
         - model_id=config.MODEL_ID
         - model_url=config.OLLAMA_URL
         - timeout=config.TIMEOUT
      2. lx.extract() returns a list of AnnotatedDocument objects (one per input text).
         Since we passed a single string, take result[0].
      3. Return the AnnotatedDocument.

    Hint:
        results = lx.extract(
            text_or_documents=ARTICLE_TEXT,
            prompt_description="...",
            examples=[example],
            model_id=config.MODEL_ID,
            model_url=config.OLLAMA_URL,
            timeout=config.TIMEOUT,
        )
        return results[0]

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement run_extraction")


# ── Orchestration ────────────────────────────────────────────────────


def print_results(doc: lx.AnnotatedDocument) -> None:
    """Print all extractions with their grounding info."""
    print("\n=== Extraction Results ===\n")
    for ext in doc.extractions:
        interval = f"[{ext.char_interval[0]}:{ext.char_interval[1]}]" if ext.char_interval else "[no grounding]"
        print(f"  {ext.extraction_class:15s} | {ext.extraction_text:25s} | {interval}")
    print(f"\nTotal: {len(doc.extractions)} extractions")


def main() -> None:
    print("Phase 1: Basic Entity Extraction")
    print(f"Model: {config.MODEL_ID} @ {config.OLLAMA_URL}")
    print(f"Text length: {len(ARTICLE_TEXT)} chars\n")

    example = create_example()
    doc = run_extraction(example)
    print_results(doc)


if __name__ == "__main__":
    main()
