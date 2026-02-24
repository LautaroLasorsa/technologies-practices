FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Mirror local layout so path dependency ../shared resolves
COPY shared/ /workspace/shared/
COPY orchestrator/pyproject.toml /workspace/orchestrator/pyproject.toml

# Install deps from service dir (shared resolved via ../shared)
WORKDIR /workspace/orchestrator
RUN uv sync --no-dev

# Copy service source code
COPY orchestrator/ /workspace/orchestrator/

# Run from /workspace so orchestrator/ is importable as a Python package
WORKDIR /workspace
ENV PATH="/workspace/orchestrator/.venv/bin:$PATH"
CMD ["python", "-m", "orchestrator.main"]
