#!/usr/bin/env bash
# ============================================================================
# Build script for Practice 012a: C++17 Features & abseil-cpp
#
# Prerequisites:
#   - CMake 3.16+ (cmake --version)
#   - A C++17 compiler (GCC 9+, Clang 10+, or MSVC 2019+)
#   - Git (for FetchContent to clone abseil-cpp)
#   - Internet connection (first build only, to fetch abseil-cpp)
#
# Usage:
#   ./build.sh          Build all targets (Release)
#   ./build.sh debug    Build all targets (Debug)
#   ./build.sh clean    Remove build directory
# ============================================================================

set -euo pipefail

BUILD_DIR="build"
BUILD_TYPE="Release"

if [[ "${1:-}" == "debug" ]]; then
    BUILD_TYPE="Debug"
elif [[ "${1:-}" == "clean" ]]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
    echo "Done."
    exit 0
fi

echo ""
echo "=== Practice 012a: C++17 Features & abseil-cpp ==="
echo ""
echo "Build type: $BUILD_TYPE"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"

# Configure (first time fetches abseil-cpp -- may take a few minutes)
echo "[1/2] Configuring with CMake..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

# Build all targets
echo ""
echo "[2/2] Building all phases..."
cmake --build "$BUILD_DIR" --config "$BUILD_TYPE" --target all_phases -j "$(nproc 2>/dev/null || echo 4)"

echo ""
echo "=== Build successful! ==="
echo ""
echo "Run individual phases:"
echo "  ./$BUILD_DIR/phase1_bindings"
echo "  ./$BUILD_DIR/phase2_optional_variant"
echo "  ./$BUILD_DIR/phase3_strings"
echo "  ./$BUILD_DIR/phase4_containers"
echo "  ./$BUILD_DIR/phase5_errors"
echo "  ./$BUILD_DIR/phase6_constexpr"
echo ""
