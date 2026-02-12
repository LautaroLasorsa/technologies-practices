"""Phase 2: Source grounding and visualization.

Demonstrates:
  - Character-level source grounding (char_interval)
  - Interactive HTML visualization with lx.visualize()
  - Programmatic grounding for custom output formats

Run after Phase 1:
    uv run python grounded_extraction.py
"""

import langextract as lx

import config

# ── Sample clinical note ─────────────────────────────────────────────

CLINICAL_NOTE = """\
Patient: Maria Rodriguez, 54-year-old female.
Chief Complaint: Persistent chest pain radiating to left arm for 3 days.
History: Diagnosed with Type 2 Diabetes in 2018. Currently on Metformin \
500mg twice daily and Lisinopril 10mg once daily for hypertension.
Vitals: BP 145/92 mmHg, HR 88 bpm, Temp 98.6°F, SpO2 97%.
Assessment: Suspected unstable angina. Recommend ECG, troponin levels, \
and cardiology consult. Continue current medications. Added Aspirin 81mg \
daily and Nitroglycerin 0.4mg sublingual as needed.
Follow-up: 48 hours or sooner if symptoms worsen."""


# ── TODO(human): Extract and visualize ──────────────────────────────


def create_clinical_example() -> lx.ExampleData:
    """Create a few-shot example for clinical entity extraction.

    TODO(human): Implement this function.

    Define an ExampleData using CLINICAL_NOTE (or a similar clinical text)
    with extraction classes that capture medical information:
      - "patient" — patient name and demographics
      - "condition" — diagnoses and complaints
      - "medication" — drug names
      - "dosage" — amounts (e.g., "500mg", "10mg")
      - "vital_sign" — measured values (e.g., "BP 145/92 mmHg")
      - "procedure" — recommended tests or actions

    Remember: extraction_text must be VERBATIM from the source text, in order
    of appearance. You don't need to extract every single entity in the example —
    just enough to teach the model the pattern (5-10 extractions is fine).

    For the clinical domain, source grounding is especially valuable because:
    - Clinicians need to verify AI extractions against the original note
    - Regulatory compliance (HIPAA) requires traceability
    - Extraction errors in healthcare can have serious consequences

    Docs: https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md
    """
    raise NotImplementedError("TODO(human): implement create_clinical_example")


def extract_and_visualize(example: lx.ExampleData) -> lx.AnnotatedDocument:
    """Extract entities and generate an HTML visualization.

    TODO(human): Implement this function.

    Steps:
      1. Call lx.extract() on CLINICAL_NOTE with your example.
         Use prompt_description like: "Extract medical entities: patient info,
         conditions, medications, dosages, vital signs, and procedures."
      2. Call lx.visualize(result) to generate an interactive HTML file.
         This creates a self-contained HTML page with highlighted source regions.
         The file is saved to the current directory.
      3. Print each extraction with its char_interval for verification.
      4. Return the AnnotatedDocument.

    The HTML visualization is LangExtract's killer feature for debugging:
    - Each extraction class gets a distinct color
    - Hovering shows the extraction_class label
    - You can immediately see what the model found vs what it missed

    Hint:
        results = lx.extract(
            text_or_documents=CLINICAL_NOTE,
            prompt_description="...",
            examples=[example],
            model_id=config.MODEL_ID,
            model_url=config.OLLAMA_URL,
            timeout=config.TIMEOUT,
        )
        doc = results[0]
        lx.visualize(doc)
        return doc

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement extract_and_visualize")


# ── Orchestration ────────────────────────────────────────────────────


def highlight_in_text(text: str, doc: lx.AnnotatedDocument) -> None:
    """Print the source text with inline markers around extractions (boilerplate)."""
    print("\n=== Grounded Text (inline markers) ===\n")
    insertions: list[tuple[int, str]] = []
    for ext in doc.extractions:
        if ext.char_interval:
            start, end = ext.char_interval
            insertions.append((start, f"[{ext.extraction_class}:"))
            insertions.append((end, "]"))

    insertions.sort(key=lambda x: (-x[0], x[1].startswith("[")))
    marked = list(text)
    for pos, marker in insertions:
        marked.insert(pos, marker)
    print("".join(marked))


def main() -> None:
    print("Phase 2: Source Grounding & Visualization")
    print(f"Model: {config.MODEL_ID} @ {config.OLLAMA_URL}\n")

    example = create_clinical_example()
    doc = extract_and_visualize(example)
    highlight_in_text(CLINICAL_NOTE, doc)


if __name__ == "__main__":
    main()
