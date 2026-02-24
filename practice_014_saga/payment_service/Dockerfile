FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Mirror local layout so path dependency ../shared resolves
COPY shared/ /workspace/shared/
COPY payment_service/pyproject.toml /workspace/payment_service/pyproject.toml

# Install deps from service dir (shared resolved via ../shared)
WORKDIR /workspace/payment_service
RUN uv sync --no-dev

# Copy service source code
COPY payment_service/ /workspace/payment_service/

# Run from /workspace so payment_service/ is importable as a Python package
WORKDIR /workspace
ENV PATH="/workspace/payment_service/.venv/bin:$PATH"
CMD ["python", "-m", "payment_service.main"]
