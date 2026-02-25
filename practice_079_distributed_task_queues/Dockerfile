FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY src/ src/
COPY ex*.py ./

# Default command (overridden per service in docker-compose.yml)
CMD ["uv", "run", "celery", "-A", "src.celery_app", "worker", "--loglevel=info"]
