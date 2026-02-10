FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy shared library first (changes less often)
COPY shared/ /app/shared/

# Copy service code and install dependencies
COPY order_service/pyproject.toml /app/pyproject.toml
RUN uv sync --no-dev

# Copy the rest of the service code
COPY order_service/ /app/order_service/

CMD ["uv", "run", "uvicorn", "order_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
