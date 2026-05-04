"""Phase 0 — Verify the LLM, scipy, and pytest are reachable.

Sends one tiny chat-completion request through the configured backend,
runs a 3-line scipy bootstrap on a synthetic array, and asserts pytest
is importable.  Catches stack issues before you start the real
exercises.

Run: uv run python -m src._00_verify_setup
Prereq: docker compose up -d && docker exec aee_ollama ollama pull qwen2.5:7b
"""

from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap

from .llm_config import build_chat_model, chat, get_lm


def _ping_lm() -> None:
    cfg = get_lm()
    print(f"  [lm]   {cfg.provider} / {cfg.model} @ {cfg.base_url or '(default)'}")
    model = build_chat_model(cfg)
    answer = chat(model, [{"role": "user", "content": "Reply with the single word: pong."}])
    print(f"    -> {answer.strip()!r}")


def _ping_scipy() -> None:
    print("  [scipy] bootstrap on a synthetic array")
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.5, scale=0.2, size=200)
    res = bootstrap((data,), np.mean, n_resamples=1000, confidence_level=0.95, random_state=rng)
    print(
        f"    -> mean={data.mean():.3f}, "
        f"95% CI=[{res.confidence_interval.low:.3f}, {res.confidence_interval.high:.3f}]"
    )


def _ping_pytest() -> None:
    print("  [pytest] import check")
    import pytest  # noqa: F401

    print(f"    -> pytest {__import__('pytest').__version__}")


def main() -> None:
    print("Verifying agent-evaluation stack...")
    _ping_pytest()
    _ping_scipy()
    _ping_lm()
    print("\nSetup verified successfully!")


if __name__ == "__main__":
    main()
