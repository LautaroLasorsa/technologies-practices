FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Mirror local layout so path dependency ../shared resolves
COPY shared/ /workspace/shared/
COPY order_service/pyproject.toml /workspace/order_service/pyproject.toml

# Install deps from service dir (shared resolved via ../shared)
WORKDIR /workspace/order_service
RUN uv sync --no-dev

# Copy service source code
COPY order_service/ /workspace/order_service/

# Run from /workspace so order_service/ is importable as a Python package
WORKDIR /workspace
ENV PATH="/workspace/order_service/.venv/bin:$PATH"
CMD ["uvicorn", "order_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
