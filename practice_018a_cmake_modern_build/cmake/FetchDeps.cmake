# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 2: FetchContent — Pulling dependencies from GitHub                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# FetchContent downloads and builds dependencies at CONFIGURE time (when you
# run `cmake -S . -B build`). This is different from ExternalProject_Add which
# downloads at BUILD time (`cmake --build build`).
#
# Key concepts:
#   - FetchContent_Declare: registers a dependency (name, URL, tag) but does NOT fetch it yet
#   - FetchContent_MakeAvailable: fetches + adds to build if not already available
#   - GIT_TAG: pin to a specific release tag (NEVER use a branch name in production!)
#   - GIT_SHALLOW TRUE: clone only the tagged commit, not full history (~10x faster)
#
# Comparison of dependency strategies:
# ┌─────────────────────┬────────────┬──────────────┬────────────────────────┐
# │ Method              │ When       │ Where        │ Best for               │
# ├─────────────────────┼────────────┼──────────────┼────────────────────────┤
# │ FetchContent        │ Configure  │ In-tree      │ Most libraries         │
# │ find_package        │ Configure  │ Pre-installed│ System libs, vcpkg     │
# │ ExternalProject_Add │ Build      │ Out-of-tree  │ Non-CMake projects     │
# └─────────────────────┴────────────┴──────────────┴────────────────────────┘

include(FetchContent)

# ─── nlohmann/json ───────────────────────────────────────────────────────────
# Header-only JSON library. One of the most popular C++ libraries on GitHub.
# After FetchContent_MakeAvailable, the target `nlohmann_json::nlohmann_json`
# becomes available for target_link_libraries.
#
# Docs: https://json.nlohmann.me/
# GitHub: https://github.com/nlohmann/json
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3          # Stable release — pin to exact tag!
    GIT_SHALLOW    TRUE             # Don't clone full history
)

# ─── fmt ─────────────────────────────────────────────────────────────────────
# Fast, safe formatting library. The design behind C++20's std::format.
# After FetchContent_MakeAvailable, the target `fmt::fmt` becomes available.
#
# Docs: https://fmt.dev/
# GitHub: https://github.com/fmtlib/fmt
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.0.2           # Stable release — pin to exact tag!
    GIT_SHALLOW    TRUE
)

# ─── Fetch everything ────────────────────────────────────────────────────────
# FetchContent_MakeAvailable checks if each dependency is already available
# (e.g., from a parent project or system install) before fetching.
# This is idempotent — calling it multiple times for the same dep is fine.
FetchContent_MakeAvailable(nlohmann_json fmt)

# After this line, these targets are available:
#   - nlohmann_json::nlohmann_json  (header-only, INTERFACE library)
#   - fmt::fmt                      (compiled static library)
