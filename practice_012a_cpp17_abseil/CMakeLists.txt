cmake_minimum_required(VERSION 3.16)
project(practice_012a_cpp17_abseil LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Fetch abseil-cpp ─────────────────────────────────────────────────────────
include(FetchContent)

set(ABSL_PROPAGATE_CXX_STD ON)
set(BUILD_TESTING OFF)

FetchContent_Declare(
    abseil-cpp
    GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
    GIT_TAG        20240722.0   # LTS release
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(abseil-cpp)

# ─── Info target ──────────────────────────────────────────────────────────────
add_executable(practice_info src/main.cpp)

# ─── Executables (one per phase) ──────────────────────────────────────────────

# Phase 1: Structured bindings & modern control flow
add_executable(phase1_bindings src/structured_bindings.cpp)
target_link_libraries(phase1_bindings PRIVATE absl::strings absl::flat_hash_map)

# Phase 2: std::optional, std::variant, std::visit
add_executable(phase2_optional_variant src/optional_variant.cpp)
target_link_libraries(phase2_optional_variant PRIVATE absl::strings)

# Phase 3: Abseil string processing
add_executable(phase3_strings src/string_processing.cpp)
target_link_libraries(phase3_strings PRIVATE absl::strings absl::str_format)

# Phase 4: Abseil containers & hashing
add_executable(phase4_containers src/containers.cpp)
target_link_libraries(phase4_containers PRIVATE
    absl::flat_hash_map
    absl::hash
    absl::strings
    absl::time
)

# Phase 5: Error handling with absl::StatusOr
add_executable(phase5_errors src/error_handling.cpp)
target_link_libraries(phase5_errors PRIVATE
    absl::status
    absl::statusor
    absl::strings
)

# Phase 6: if-constexpr & fold expressions
add_executable(phase6_constexpr src/constexpr_folds.cpp)
target_link_libraries(phase6_constexpr PRIVATE absl::strings)

# ─── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_bindings
    phase2_optional_variant
    phase3_strings
    phase4_containers
    phase5_errors
    phase6_constexpr
)
