"""Phase 5: Custom domain extraction — user's choice.

Demonstrates:
  - Designing an extraction schema from scratch for a new domain
  - Writing effective few-shot examples
  - End-to-end pipeline: define → extract → visualize

Choose a domain and implement the full pipeline.

Run after completing previous phases:
    uv run python custom_extraction.py
"""

import langextract as lx

import config

# ── Choose your domain ──────────────────────────────────────────────
#
# Pick ONE of the following sample texts, or write your own.
# The goal is to design an extraction schema from scratch.

# Option A: Job Posting
JOB_POSTING = """\
Senior Backend Engineer — FinTech Startup (Remote, US)
Salary: $180,000 - $220,000 + equity (0.1% - 0.3%)
Stack: Python, FastAPI, PostgreSQL, Kafka, Kubernetes, AWS
Experience: 5+ years backend development, 2+ years distributed systems

We're building the next generation of payment infrastructure. You'll design \
and implement low-latency transaction processing systems handling 50K TPS. \
Must have experience with event-driven architectures and strong SQL skills. \
Nice to have: Rust, gRPC, experience with PCI-DSS compliance.

Benefits: Unlimited PTO, $5,000 annual learning budget, health/dental/vision, \
401k with 4% match. Reports to VP of Engineering, team size: 8 engineers.
Apply by March 31, 2025."""

# Option B: Legal Contract Clause
CONTRACT_CLAUSE = """\
SECTION 7: LIMITATION OF LIABILITY

7.1 Maximum Liability. The total aggregate liability of Provider under this \
Agreement shall not exceed the fees paid by Client in the twelve (12) months \
preceding the claim, except for breaches of Section 5 (Confidentiality) and \
Section 9 (Indemnification), which are uncapped.

7.2 Exclusion of Damages. In no event shall either party be liable for any \
indirect, incidental, consequential, special, or punitive damages, including \
but not limited to loss of profits, loss of data, or business interruption, \
regardless of the theory of liability.

7.3 Exceptions. The limitations in Sections 7.1 and 7.2 shall not apply to: \
(a) willful misconduct or gross negligence, (b) breach of intellectual property \
rights, (c) obligations under applicable law that cannot be limited by contract.

7.4 Insurance. Provider shall maintain commercial general liability insurance \
with coverage of at least $5,000,000 per occurrence and $10,000,000 aggregate, \
and professional liability (E&O) insurance of at least $2,000,000 per claim."""

# Option C: Research Paper Abstract
RESEARCH_ABSTRACT = """\
Title: Scaling Transformer Inference with Speculative Decoding and KV Cache Compression

Authors: Wei Zhang, Priya Patel, Carlos Rivera (Stanford University), \
and Yuki Tanaka (Google DeepMind)

Abstract: We present FastSpec, a framework that combines speculative decoding \
with dynamic KV cache compression to achieve 3.2x speedup in autoregressive \
transformer inference with less than 0.5% quality degradation on MMLU and \
HumanEval benchmarks. Our approach uses a 125M parameter draft model to \
speculate 5 tokens ahead, verified by the target 70B model. The KV cache \
compression uses learned importance scores to evict low-value entries, \
reducing memory usage by 40%. Experiments on LLaMA-2-70B show latency \
reduction from 45ms/token to 14ms/token on A100 GPUs. We release our code \
at github.com/fastspec/fastspec under the MIT license.

Keywords: transformer inference, speculative decoding, KV cache, LLM optimization
Published: NeurIPS 2025"""


# ── TODO(human): Design and implement a custom extraction pipeline ──


def create_domain_example() -> lx.ExampleData:
    """Design an extraction schema for your chosen domain.

    TODO(human): Implement this function.

    Steps:
      1. Choose one of the sample texts above (or bring your own).
      2. Identify 4-6 extraction classes relevant to that domain. Examples:
         - Job posting: "role", "salary", "technology", "requirement",
           "benefit", "company_info"
         - Legal contract: "clause_type", "liability_limit", "monetary_amount",
           "exception", "insurance_requirement", "section_reference"
         - Research paper: "author", "affiliation", "method", "metric",
           "benchmark", "hardware", "result"
      3. Create an ExampleData with verbatim extraction_text from your chosen text.
      4. Use attributes if entities have relationships (e.g., author → affiliation).

    Design principles:
      - Classes should be mutually exclusive (no entity fits two classes)
      - 5-10 extractions in your example is enough to teach the pattern
      - More specific classes ("salary_range") are better than generic ("number")
      - Test your schema mentally: could someone reconstruct a structured
        record from your extractions?

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement create_domain_example")


def run_pipeline(text: str, example: lx.ExampleData, domain: str) -> None:
    """Run the full extraction pipeline: extract → print → visualize.

    TODO(human): Implement this function.

    Steps:
      1. Call lx.extract() with the chosen text and your example.
         Write a domain-specific prompt_description.
      2. Print the extractions grouped by class.
      3. Call lx.visualize() to generate the HTML visualization.
      4. Print a summary of extraction classes and counts.

    Docs: https://github.com/google/langextract
    """
    raise NotImplementedError("TODO(human): implement run_pipeline")


# ── Orchestration ────────────────────────────────────────────────────


def main() -> None:
    print("Phase 5: Custom Domain Extraction")
    print(f"Model: {config.MODEL_ID} @ {config.OLLAMA_URL}\n")

    # Change this to your chosen text
    chosen_text = JOB_POSTING  # or CONTRACT_CLAUSE or RESEARCH_ABSTRACT
    domain = "job_posting"     # or "legal_contract" or "research_paper"

    example = create_domain_example()
    run_pipeline(chosen_text, example, domain)


if __name__ == "__main__":
    main()
