# Practice 028: LangExtract — Structured Extraction from Unstructured Text

## Technologies

- **LangExtract** — Google open-source Python library for extracting structured information from unstructured text using LLMs
- **Ollama** — Local LLM inference server (Docker container)
- **Gemma 3** — Google's open-weight model family, optimized for instruction following

## Stack

- Python 3.12+ (uv)
- Docker / Docker Compose (Ollama container)

## Theoretical Context

### What Is LangExtract?

LangExtract converts unstructured text (clinical notes, legal documents, financial reports, articles) into structured, schema-enforced data using LLMs. Unlike general-purpose LLM prompting, LangExtract provides three key mechanisms that make extraction reliable and verifiable:

1. **Source Grounding** — Every extracted entity is mapped to exact character offsets in the source text. This means you can trace *where* in the document each extraction came from, enabling visual highlighting and auditability.
2. **Controlled Generation via Few-Shot Examples** — Instead of complex prompt engineering, you provide 1-3 concrete examples of what the extraction should look like. The library constructs the appropriate prompt internally.
3. **Long-Document Chunking** — For documents that exceed the model's context window or suffer from "needle-in-haystack" degradation, LangExtract automatically chunks text and runs parallel extraction passes.

### How It Works Internally

```
Unstructured Text
       │
       ▼
┌─────────────────┐    Few-shot ExampleData
│  Chunking       │◄── (schema + examples)
│  (if needed)    │
└────────┬────────┘
         │  chunks[]
         ▼
┌─────────────────┐
│  LLM Inference  │◄── Ollama / Gemini / OpenAI
│  (per chunk)    │
└────────┬────────┘
         │  raw extractions
         ▼
┌─────────────────┐
│  Grounding      │── Maps each extraction_text to
│  + Validation   │   char_interval in source
└────────┬────────┘
         │
         ▼
   AnnotatedDocument
   (structured + grounded)
```

The core insight: extraction_text in your examples must be **verbatim quotes** from the source text, listed in order of appearance. This enables the grounding engine to find exact character positions automatically.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **ExampleData** | A training example: source text + list of expected Extraction objects. Teaches the model your schema. |
| **Extraction** | A single extracted entity: `extraction_class` (category), `extraction_text` (verbatim quote), optional `attributes` (key-value metadata). |
| **AnnotatedDocument** | Result container. Holds source text + grounded extractions with character intervals. |
| **extraction_class** | Category label for the entity (e.g., "medication", "dosage", "person", "date"). |
| **attributes** | Dict for linking related entities. E.g., `{"medication_group": "1"}` groups a medication with its dosage and frequency. |
| **char_interval** | Auto-computed `[start, end)` character positions mapping extraction to source text. |
| **extraction_passes** | Number of times to re-extract from the same text. Multiple passes catch entities missed in a single pass. |
| **prompt_description** | Natural language description of what to extract. Guides the model alongside few-shot examples. |

### Source Grounding — Why It Matters

Most LLM extraction tools return structured data but provide no *evidence*. LangExtract's grounding lets you:
- **Verify** extractions against the original text (auditability for healthcare, legal, finance)
- **Visualize** extractions with highlighted source regions (interactive HTML output)
- **Debug** extraction quality by seeing what the model matched vs what it missed

This is the key differentiator vs alternatives like Instructor, Marvin, or Outlines (which focus on schema enforcement but not provenance).

### Ecosystem Context

| Library | Focus | Source Grounding | Local LLMs | Schema |
|---------|-------|-----------------|------------|--------|
| **LangExtract** | Document extraction + grounding | Yes (character offsets) | Ollama built-in | Few-shot examples |
| **Instructor** | Structured output (closest to OpenAI SDK) | No | Via providers | Pydantic models |
| **Outlines** | Constrained generation (grammar-guided) | No | Yes (vLLM, etc.) | Pydantic / regex |
| **Marvin** | Simple extraction + classification | No | OpenAI only | Decorators |

Choose LangExtract when you need **provenance/traceability** (where did this extraction come from?). Choose Instructor/Outlines when you need **general structured output** without source mapping.

**Sources:**
- [GitHub — google/langextract](https://github.com/google/langextract)
- [Google Developers Blog — Introducing LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [PyPI — langextract](https://pypi.org/project/langextract/)

## Description

Build a **Document Intelligence Pipeline** that extracts structured data from unstructured text using LangExtract + a local Ollama LLM. No cloud API keys required — everything runs locally via Docker.

### What you'll learn

1. **Few-shot extraction** — Define schemas via examples, not prompt engineering
2. **Source grounding** — Map every extraction to exact character positions in the source
3. **Extraction classes + attributes** — Entity categorization and relationship linking
4. **Long-document processing** — Chunking and multi-pass extraction
5. **Visualization** — Interactive HTML output with highlighted source regions
6. **Local LLM integration** — Running Ollama with Gemma models for private, offline extraction

## Instructions

### Phase 1: Setup & First Extraction (~15 min)

1. Start Docker Compose (Ollama container + model pull)
2. Initialize Python project with `uv`, install `langextract`
3. **User implements:** Basic extraction — define `ExampleData` with extraction classes for a simple text, call `lx.extract()`, inspect the `AnnotatedDocument` result
4. Key question: Why must `extraction_text` be verbatim from the source? What breaks if you paraphrase?

### Phase 2: Source Grounding & Visualization (~20 min)

1. Understand `char_interval` — how LangExtract maps extractions to source positions
2. **User implements:** Extract entities from a clinical note, then generate an interactive HTML visualization with `lx.visualize()`
3. **User implements:** Programmatic grounding — use `char_interval` to highlight extractions in plain text (no HTML)
4. Key question: How does grounding help with extraction quality debugging?

### Phase 3: Relationship Extraction with Attributes (~25 min)

1. Understand `attributes` — how to link related entities (e.g., medication ↔ dosage ↔ frequency)
2. **User implements:** Extract entities from a medication prescription with grouped attributes
3. Verify that related entities share the same attribute group
4. Key question: How would you model a many-to-many relationship with attributes?

### Phase 4: Long Documents & Multi-Pass (~20 min)

1. Understand chunking: why long documents degrade extraction quality ("needle-in-haystack")
2. **User implements:** Extract from a multi-page document using chunking + multiple extraction passes
3. Compare single-pass vs multi-pass extraction results
4. Key question: What trade-offs does chunking introduce? (duplicate extractions at chunk boundaries, cost)

### Phase 5: Custom Domain Extraction (~30 min)

1. Choose a domain: financial reports, legal contracts, job postings, or research papers
2. **User implements:** End-to-end extraction pipeline — define schema, write few-shot examples, extract, visualize
3. Experiment with different models (gemma3:4b vs gemma3:12b) and compare quality
4. Key question: How many few-shot examples are enough? When do more examples stop helping?

## Motivation

- **Document intelligence** is a high-demand skill in enterprise AI (healthcare, legal, finance all need structured extraction from unstructured text)
- **Source grounding** is increasingly required for regulated industries (audit trails, explainability)
- **Local LLM integration** (Ollama) demonstrates private, offline AI workflows — important for sensitive data
- **Few-shot learning** is a transferable pattern across all LLM applications, not just extraction
- **Google ecosystem** — LangExtract integrates with Gemini and Vertex AI, relevant for GCP-based architectures

## Commands

All commands run from `practice_028_langextract/`.

### Infrastructure

| Command | Description |
|---------|-------------|
| `docker compose up -d` | Start Ollama container and pull the Gemma 3 4B model |
| `docker compose down` | Stop and remove the Ollama container |
| `docker compose logs -f ollama` | Stream Ollama logs (useful for checking model loading) |
| `docker exec ollama ollama list` | List downloaded models inside the container |
| `docker exec ollama ollama pull gemma3:12b` | Pull a larger model for better extraction quality |

### Project Setup

| Command | Description |
|---------|-------------|
| `cd app && uv sync` | Install Python dependencies from `pyproject.toml` into `.venv` |

### Phase 1: Basic Extraction

| Command | Description |
|---------|-------------|
| `cd app && uv run python basic_extraction.py` | Run basic entity extraction from a sample text |

### Phase 2: Source Grounding & Visualization

| Command | Description |
|---------|-------------|
| `cd app && uv run python grounded_extraction.py` | Extract with source grounding, generate HTML visualization |

### Phase 3: Relationship Extraction

| Command | Description |
|---------|-------------|
| `cd app && uv run python relationship_extraction.py` | Extract entities with grouped attributes (medication example) |

### Phase 4: Long Document Processing

| Command | Description |
|---------|-------------|
| `cd app && uv run python long_document.py` | Extract from a multi-page document with chunking + multi-pass |

### Phase 5: Custom Domain

| Command | Description |
|---------|-------------|
| `cd app && uv run python custom_extraction.py` | User-defined domain extraction pipeline |

## References

- [GitHub — google/langextract](https://github.com/google/langextract)
- [Google Developers Blog — Introducing LangExtract](https://developers.googleblog.com/en/introducing-langextract-a-gemini-powered-information-extraction-library/)
- [PyPI — langextract 1.1.1](https://pypi.org/project/langextract/)
- [LangExtract Medication Examples](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)
- [LangExtract Ollama Integration](https://github.com/google/langextract/tree/main/examples/ollama)
- [Ollama — Gemma 3 Models](https://ollama.com/library/gemma3)
- [Structured Output Library Comparison](https://simmering.dev/blog/structured_output/)

## State

`not-started`
