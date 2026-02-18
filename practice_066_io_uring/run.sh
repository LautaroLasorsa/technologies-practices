#!/usr/bin/env bash
# ============================================================================
# run.sh — Build the Docker image and open a shell for io_uring practice
#
# Usage (from Git Bash on Windows, or any bash shell):
#   bash run.sh          # Build + start + open shell
#   bash run.sh build    # Build only (no shell)
#   bash run.sh down     # Stop and remove container
# ============================================================================

set -euo pipefail

COMPOSE="docker compose"

case "${1:-}" in
    build)
        echo "Building Docker image..."
        $COMPOSE up --build -d
        echo "Container is running. Enter with: docker compose exec dev bash"
        ;;
    down)
        echo "Stopping container..."
        $COMPOSE down
        echo "Done."
        ;;
    *)
        echo "Building Docker image and starting container..."
        $COMPOSE up --build -d
        echo ""
        echo "Entering container shell. Run 'cargo build && cargo run' to start."
        echo "Exit with 'exit' or Ctrl+D."
        echo ""
        $COMPOSE exec dev bash
        ;;
esac
