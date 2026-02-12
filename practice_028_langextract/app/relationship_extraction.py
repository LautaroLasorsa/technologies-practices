"""Phase 3: Relationship extraction using attributes.

Demonstrates:
  - Using attributes to link related entities into groups
  - Modeling entity relationships (medication → dosage → frequency)
  - Verifying group consistency

Run after Phase 2:
    uv run python relationship_extraction.py
"""

import langextract as lx

import config

# ── Sample prescription text ─────────────────────────────────────────

PRESCRIPTION_TEXT = """\
Prescriptions for patient John Doe, DOB 1985-03-22:

1. Metformin 500mg, take twice daily with meals, for Type 2 Diabetes.
   Duration: ongoing. Refills: 3.

2. Atorvastatin 20mg, take once daily at bedtime, for hyperlipidemia.
   Duration: 6 months. Refills: 2.

3. Amlodipine 5mg, take once daily in the morning, for hypertension.
   Duration: ongoing. Refills: 3.

Prescribing physician: Dr. Emily Watson, MD. Date: January 10, 2025."""


# ── TODO(human): Define relationship-aware extraction ───────────────


def create_prescription_example() -> lx.ExampleData:
    """Create a few-shot example with attribute-based relationship grouping.

    TODO(human): Implement this function.

    The key concept here is **attributes** — a dict of key-value pairs on each
    Extraction that lets you group related entities. For prescriptions, you want
    to link each medication with its dosage, frequency, condition, and duration.

    Use a "prescription_group" attribute with a numeric ID to create groups:

        lx.Extraction(
            extraction_class="medication",
            extraction_text="Metformin",
            attributes={"prescription_group": "1"},
        ),
        lx.Extraction(
            extraction_class="dosage",
            extraction_text="500mg",
            attributes={"prescription_group": "1"},
        ),
        lx.Extraction(
            extraction_class="frequency",
            extraction_text="twice daily with meals",
            attributes={"prescription_group": "1"},
        ),
        lx.Extraction(
            extraction_class="condition",
            extraction_text="Type 2 Diabetes",
            attributes={"prescription_group": "1"},
        ),

    Then group 2 (Atorvastatin) gets prescription_group="2", etc.

    Also extract non-grouped entities:
      - "patient" for the patient name
      - "physician" for the prescribing doctor
      - "date" for the prescription date

    These don't need the prescription_group attribute.

    Remember: extraction_text must be VERBATIM from PRESCRIPTION_TEXT, in order
    of appearance. You don't need to cover every entity in every group — but
    include at least 2 complete groups to teach the pattern.

    Why attributes matter:
    - Without attributes, you get a flat list of entities with no relationships.
    - With attributes, you can reconstruct structured records:
      {medication: "Metformin", dosage: "500mg", frequency: "twice daily with meals"}
    - This is the difference between NER (entity recognition) and
      relation extraction (understanding how entities connect).

    Docs: https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md
    """
    raise NotImplementedError("TODO(human): implement create_prescription_example")


def extract_prescriptions(example: lx.ExampleData) -> lx.AnnotatedDocument:
    """Extract prescription entities with relationship grouping.

    TODO(human): Implement this function.

    Steps:
      1. Call lx.extract() on PRESCRIPTION_TEXT with your example.
         Use prompt_description like: "Extract prescription information:
         medications, dosages, frequencies, conditions, durations,
         patient name, physician, and date. Group related medication
         details using the prescription_group attribute."
      2. Return results[0].

    Hint: Same pattern as Phase 1 and 2 — just lx.extract() with your example.

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement extract_prescriptions")


# ── Orchestration ────────────────────────────────────────────────────


def print_grouped_results(doc: lx.AnnotatedDocument) -> None:
    """Print extractions grouped by their prescription_group attribute (boilerplate)."""
    print("\n=== Grouped Prescription Extractions ===\n")

    groups: dict[str, list[lx.Extraction]] = {}
    ungrouped: list[lx.Extraction] = []

    for ext in doc.extractions:
        group_id = ext.attributes.get("prescription_group") if ext.attributes else None
        if group_id:
            groups.setdefault(group_id, []).append(ext)
        else:
            ungrouped.append(ext)

    for group_id in sorted(groups.keys()):
        print(f"  Prescription Group {group_id}:")
        for ext in groups[group_id]:
            print(f"    {ext.extraction_class:15s} | {ext.extraction_text}")
        print()

    if ungrouped:
        print("  Ungrouped entities:")
        for ext in ungrouped:
            print(f"    {ext.extraction_class:15s} | {ext.extraction_text}")

    print(f"\nTotal: {len(doc.extractions)} extractions in {len(groups)} groups")


def main() -> None:
    print("Phase 3: Relationship Extraction with Attributes")
    print(f"Model: {config.MODEL_ID} @ {config.OLLAMA_URL}\n")

    example = create_prescription_example()
    doc = extract_prescriptions(example)
    print_grouped_results(doc)


if __name__ == "__main__":
    main()
