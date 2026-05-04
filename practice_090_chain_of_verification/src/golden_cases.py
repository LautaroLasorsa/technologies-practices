"""Hand-crafted prompts likely to elicit hallucinations.

These are the prompts you run CoVe against in ``demo.py``.  They follow
the pattern from the original CoVe paper (Dhuliawala et al., ACL 2024,
https://arxiv.org/abs/2309.11495) — short list-style fact-recall
questions where small/medium models tend to invent plausible-but-wrong
items.

The ``hint`` field is what *you* know to be true and is only used by
the demo's sanity-check column — it is **never** shown to the LLM.

Don't add a TODO here — these are reference fixtures, not exercises.
Add or replace prompts freely; the rest of the pipeline doesn't care
about the count.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenCase:
    """One prompt + a private hint string used only for demo-side sanity checks."""
    name: str
    prompt: str
    hint: str = ""


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        name="actors_brooklyn",
        prompt="Name 5 actors who were born in Brooklyn, New York.",
        hint="Common trap: models often list Mae West (Brooklyn yes), "
             "Adam Sandler (Brooklyn yes), but also incorrectly list people "
             "born in Queens or the Bronx. Verify each birthplace independently.",
    ),
    GoldenCase(
        name="olympic_gold_1996",
        prompt="List 5 athletes who won an individual gold medal at the 1996 Atlanta Summer Olympics.",
        hint="Common trap: confusing 1996 with 1992 or 2000; team gold vs individual gold.",
    ),
    GoldenCase(
        name="novels_dostoevsky",
        prompt="List 5 novels written by Fyodor Dostoevsky.",
        hint="Trap: short stories like 'The Gambler' are sometimes labeled novellas; "
             "models occasionally invent plausible-sounding titles.",
    ),
    GoldenCase(
        name="cs_turing_award_2010s",
        prompt="Name 5 winners of the ACM Turing Award between 2010 and 2019 (inclusive).",
        hint="Trap: shifting the decade by one year, or attributing the wrong year to a real winner.",
    ),
    GoldenCase(
        name="capitals_south_america",
        prompt="List the capitals of 5 South American countries.",
        hint="Easy baseline; included as a sanity case where joint and factored should agree.",
    ),
    GoldenCase(
        name="languages_python",
        prompt="Name 5 programming languages whose primary implementation is written in Python itself.",
        hint="Trap: trivia question. Most major languages are implemented in C/C++/Rust, not Python; "
             "models often hallucinate examples here.",
    ),
    GoldenCase(
        name="presidents_one_term",
        prompt="Name 5 US presidents who served exactly one term and were not assassinated.",
        hint="Compound constraint: 'exactly one term' AND 'not assassinated'. "
             "Models routinely violate one of the two clauses.",
    ),
    GoldenCase(
        name="composers_19c_russia",
        prompt="List 5 19th-century Russian composers.",
        hint="Trap: Stravinsky and Shostakovich are 20th century; "
             "models sometimes include them anyway.",
    ),
]
