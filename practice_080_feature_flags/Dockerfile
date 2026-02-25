FROM python:3.12-slim

ARG SERVICE_MODULE=product_api

WORKDIR /app

# Install uv for dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy all source files
COPY src/ ./src/
COPY pyproject.toml ./

# Install dependencies
RUN uv sync --no-dev

# Ensure Python can find modules in src/
ENV PYTHONPATH=/app/src
ENV SERVICE_MODULE=${SERVICE_MODULE}

CMD uv run python src/${SERVICE_MODULE}.py
