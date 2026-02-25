FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml .
COPY .python-version .

# Install dependencies
RUN uv sync --frozen --no-dev 2>/dev/null || uv sync --no-dev

# Copy source code and Lua scripts
COPY src/ src/
COPY lua/ lua/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.06_fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
