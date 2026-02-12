"""Phase 4: Long document processing with chunking and multi-pass extraction.

Demonstrates:
  - Why long documents degrade extraction quality ("needle-in-haystack")
  - Chunking strategies for documents exceeding context windows
  - Multi-pass extraction to catch missed entities
  - Deduplication of extractions across chunks

Run after Phase 3:
    uv run python long_document.py
"""

import langextract as lx

import config

# ── Long document (multi-section report) ─────────────────────────────

QUARTERLY_REPORT = """\
ACME CORPORATION — Q3 2025 QUARTERLY REPORT

EXECUTIVE SUMMARY
Acme Corporation reported revenue of $2.4 billion for Q3 2025, representing \
a 15% year-over-year increase. Net income reached $380 million, up from \
$310 million in Q3 2024. CEO Jane Thompson attributed the growth to strong \
performance in the Cloud Services division and expansion into the European market.

BUSINESS SEGMENTS

Cloud Services Division
Revenue: $1.1 billion (46% of total). Growth rate: 28% YoY. The division \
launched AcmeCloud Pro in July 2025, which attracted 5,000 enterprise customers \
in its first quarter. VP of Cloud Sarah Martinez noted that the AWS and Azure \
migration tools were key drivers. Operating margin improved to 32% from 27%.

Hardware Division
Revenue: $800 million (33% of total). Growth rate: 5% YoY. The AcmeBox 5000 \
server line continued to perform well in data center deployments. CTO Robert \
Kim announced a new partnership with NVIDIA for next-generation AI accelerators, \
expected to launch in Q1 2026. Operating margin: 18%.

Professional Services Division
Revenue: $500 million (21% of total). Growth rate: 8% YoY. The consulting \
team completed 340 enterprise implementations this quarter, led by Managing \
Director Lisa Park. Key clients included Deutsche Bank, Toyota, and Pfizer. \
Operating margin: 22%.

FINANCIAL HIGHLIGHTS
- Total revenue: $2.4 billion
- Gross profit: $1.44 billion (60% margin)
- Operating expenses: $960 million
- Net income: $380 million
- Earnings per share: $3.42
- Free cash flow: $520 million
- Total employees: 45,000 (up 3,000 from Q2)

GEOGRAPHIC BREAKDOWN
North America: $1.56 billion (65%). Europe: $480 million (20%), with \
notable growth in Germany and the UK after opening new offices in Berlin \
and London. Asia-Pacific: $360 million (15%), driven by expansion in \
Japan and South Korea.

OUTLOOK
CFO Michael Chang raised full-year guidance to $9.8 billion in revenue \
(previously $9.5 billion). The company plans to invest $1.2 billion in R&D \
in 2026, focusing on AI infrastructure and quantum computing research. \
Board member Patricia Nguyen emphasized the importance of sustainable growth \
and ESG commitments, including a target of carbon neutrality by 2030.

RISKS
Key risks include increased competition from Microsoft and Google in cloud \
services, semiconductor supply chain constraints affecting the Hardware \
division, and foreign exchange headwinds in European operations. General \
Counsel David Wilson noted ongoing regulatory reviews in the EU regarding \
data privacy compliance."""


# ── TODO(human): Extract from long document with multi-pass ─────────


def create_financial_example() -> lx.ExampleData:
    """Create a few-shot example for financial report extraction.

    TODO(human): Implement this function.

    Define an ExampleData that teaches extraction of financial report entities.
    You can use a short snippet (not the full QUARTERLY_REPORT) as the example text.

    Suggested extraction classes:
      - "person" — named individuals with their roles
      - "organization" — company names
      - "financial_metric" — revenue, profit, margin figures
      - "product" — product or service names
      - "date" — dates and time periods
      - "location" — geographic regions, cities, countries

    Tip: Since this is a long document, you want your example to demonstrate
    the pattern clearly with just a few extractions. The model will generalize
    to extract all matching entities from the full document.

    For a short example text, you could write something like:
        "XYZ Inc reported revenue of $500 million in Q2 2025. CEO Alice
         Brown credited the growth to the XYZ Cloud platform. The company
         expanded into Tokyo and Singapore."

    Then provide extractions for that short text.

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement create_financial_example")


def extract_long_document(
    example: lx.ExampleData,
    extraction_passes: int = 1,
) -> lx.AnnotatedDocument:
    """Extract entities from the full quarterly report.

    TODO(human): Implement this function.

    Steps:
      1. Call lx.extract() on QUARTERLY_REPORT with your example.
      2. Use the extraction_passes parameter to control multi-pass extraction.
         - extraction_passes=1: single pass (default, faster)
         - extraction_passes=2: two passes (catches more entities, slower)
      3. Return results[0].

    Why multi-pass?
      LLMs have a "needle-in-haystack" problem — in long documents, entities
      in the middle tend to be missed more often than entities at the start
      or end. Running multiple extraction passes with different chunking
      offsets catches entities missed in a single pass.

    Trade-off:
      - More passes = better recall (fewer missed entities)
      - More passes = higher cost + latency + potential duplicate extractions
      - LangExtract handles deduplication internally via char_interval overlap

    Hint:
        results = lx.extract(
            text_or_documents=QUARTERLY_REPORT,
            prompt_description="...",
            examples=[example],
            model_id=config.MODEL_ID,
            model_url=config.OLLAMA_URL,
            timeout=config.TIMEOUT,
            extraction_passes=extraction_passes,
        )
        return results[0]

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement extract_long_document")


# ── Orchestration ────────────────────────────────────────────────────


def print_results(doc: lx.AnnotatedDocument, label: str) -> None:
    """Print extraction results with category counts (boilerplate)."""
    print(f"\n=== {label} ===\n")

    by_class: dict[str, list[lx.Extraction]] = {}
    for ext in doc.extractions:
        by_class.setdefault(ext.extraction_class, []).append(ext)

    for cls in sorted(by_class.keys()):
        entities = by_class[cls]
        print(f"  {cls} ({len(entities)}):")
        for ext in entities:
            print(f"    - {ext.extraction_text}")
    print(f"\nTotal: {len(doc.extractions)} extractions")


def main() -> None:
    print("Phase 4: Long Document Processing")
    print(f"Model: {config.MODEL_ID} @ {config.OLLAMA_URL}")
    print(f"Document length: {len(QUARTERLY_REPORT)} chars\n")

    example = create_financial_example()

    # Single pass
    doc_single = extract_long_document(example, extraction_passes=1)
    print_results(doc_single, "Single Pass (extraction_passes=1)")

    # Multi-pass
    doc_multi = extract_long_document(example, extraction_passes=2)
    print_results(doc_multi, "Multi Pass (extraction_passes=2)")

    # Compare
    print(f"\n=== Comparison ===")
    print(f"  Single pass: {len(doc_single.extractions)} extractions")
    print(f"  Multi pass:  {len(doc_multi.extractions)} extractions")
    print(f"  Gain: +{len(doc_multi.extractions) - len(doc_single.extractions)} entities")


if __name__ == "__main__":
    main()
