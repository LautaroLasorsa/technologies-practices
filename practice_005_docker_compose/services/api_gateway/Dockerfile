# =============================================================================
# Dockerfile for the API Gateway service
#
# TODO(human): Fill in each step below. Hints are provided as comments.
# =============================================================================

# ── Exercise Context ──────────────────────────────────────────────────
# This exercise teaches the multi-stage pattern of Dockerfile construction:
# base image selection, dependency layer caching, and CMD vs ENTRYPOINT.
# Mastering this pattern is critical for building production-ready container images.

# TODO(human) [1/4]: Choose a base image.
# Hint: Use python:3.12-slim for a small image with Python pre-installed.
# Syntax: FROM <image>:<tag>


# TODO(human) [2/4]: Set the working directory inside the container.
# Hint: Use /app as the convention.
# Syntax: WORKDIR <path>


# TODO(human) [3/4]: Copy requirements.txt and install dependencies.
# Hint: Copy requirements.txt first (before source code) so Docker caches
#       the pip install layer -- source code changes won't re-trigger it.
# Syntax:
#   COPY requirements.txt .
#   RUN pip install --no-cache-dir -r requirements.txt


# TODO(human) [4/4]: Copy the application source and define the startup command.
# Hint: Copy main.py (or the whole directory with COPY . .), expose port 8000,
#       and run uvicorn.
# Syntax:
#   COPY . .
#   EXPOSE 8000
#   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
