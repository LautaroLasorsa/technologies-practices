#!/usr/bin/env bash
# ============================================================================
# run.sh — Build and start the eBPF development container
#
# This script:
# 1. Builds the Docker image with the full aya/Rust/bpf-linker toolchain
# 2. Starts a privileged interactive container with all required mounts
#
# Usage:
#   ./run.sh          # Build image and start interactive shell
#   ./run.sh build    # Only build the Docker image
#   ./run.sh shell    # Start shell (assumes image already built)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

case "${1:-}" in
    build)
        echo "[*] Building eBPF development image..."
        docker compose build
        echo "[+] Image built successfully."
        ;;
    shell)
        echo "[*] Starting privileged eBPF container..."
        docker compose run --rm ebpf bash
        ;;
    *)
        echo "[*] Building eBPF development image..."
        docker compose build
        echo "[+] Image built successfully."
        echo ""
        echo "[*] Starting privileged eBPF container..."
        echo "[*] You will be dropped into a bash shell inside the container."
        echo "[*] The project is mounted at /app"
        echo ""
        docker compose run --rm ebpf bash
        ;;
esac
