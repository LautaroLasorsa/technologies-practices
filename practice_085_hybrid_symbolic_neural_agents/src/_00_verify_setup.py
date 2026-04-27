"""Phase 0 — Verify the LLM connection + the OR-Tools install.

Sends one tiny chat-completion request through the configured backend
and runs a trivial CP-SAT model so you know both halves of the hybrid
stack are reachable before starting the real exercises.

Run: uv run python -m src._00_verify_setup
Prereq: docker compose up -d && docker exec hsna_ollama ollama pull qwen2.5:7b
"""

from __future__ import annotations

from ortools.sat.python import cp_model

from .llm_config import chat, get_lm


def _ping_lm() -> None:
    cfg = get_lm()
    print(f"  [lm]   {cfg.provider} / {cfg.model} @ {cfg.base_url or '(default)'}")
    answer = chat(
        cfg,
        messages=[{"role": "user", "content": "Reply with the single word: pong."}],
        max_tokens=8,
    )
    print(f"    -> {answer.strip()!r}")


def _ping_solver() -> None:
    print("  [solver] OR-Tools CP-SAT")
    model = cp_model.CpModel()
    x = model.NewBoolVar("x")
    y = model.NewBoolVar("y")
    model.Add(x + y == 1)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    print(f"    -> status={solver.StatusName(status)}, x={solver.Value(x)}, y={solver.Value(y)}")


def main() -> None:
    print("Verifying hybrid symbolic-neural stack...")
    _ping_solver()
    _ping_lm()
    print("\nSetup verified successfully!")


if __name__ == "__main__":
    main()
