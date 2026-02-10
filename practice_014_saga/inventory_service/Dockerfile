FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy shared library first (changes less often)
COPY shared/ /app/shared/

# Copy service code and install dependencies
COPY inventory_service/pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

# Copy the rest of the service code
COPY inventory_service/ /app/inventory_service/

CMD ["uv", "run", "python", "-m", "inventory_service.main"]
