FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Mirror local layout so path dependency ../shared resolves
COPY shared/ /workspace/shared/
COPY inventory_service/pyproject.toml /workspace/inventory_service/pyproject.toml

# Install deps from service dir (shared resolved via ../shared)
WORKDIR /workspace/inventory_service
RUN uv sync --no-dev

# Copy service source code
COPY inventory_service/ /workspace/inventory_service/

# Run from /workspace so inventory_service/ is importable as a Python package
WORKDIR /workspace
ENV PATH="/workspace/inventory_service/.venv/bin:$PATH"
CMD ["python", "-m", "inventory_service.main"]
