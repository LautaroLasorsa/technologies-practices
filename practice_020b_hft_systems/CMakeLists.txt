cmake_minimum_required(VERSION 3.16)
project(practice_020b_hft_systems LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── MSVC optimizations ───────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /W4)
else()
    add_compile_options(-O2 -Wall -Wextra)
endif()

# ── Fetch abseil-cpp ─────────────────────────────────────────────────────────
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

# ── Common include path ──────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Abseil libraries used across phases ──────────────────────────────────────
set(ABSL_COMMON_LIBS
    absl::flat_hash_map
    absl::hash
    absl::strings
    absl::str_format
    absl::status
    absl::statusor
)

# ── Phase 1: Limit Order Book ───────────────────────────────────────────────
add_executable(phase1_orderbook src/phase1_orderbook.cpp)
target_link_libraries(phase1_orderbook PRIVATE ${ABSL_COMMON_LIBS})

# ── Phase 2: Matching Engine ────────────────────────────────────────────────
add_executable(phase2_matching src/phase2_matching.cpp)
target_link_libraries(phase2_matching PRIVATE ${ABSL_COMMON_LIBS})

# ── Phase 3: Market Data Feed Handler ───────────────────────────────────────
add_executable(phase3_feed src/phase3_feed.cpp)
target_link_libraries(phase3_feed PRIVATE ${ABSL_COMMON_LIBS})

# ── Phase 4: Signal Generation ──────────────────────────────────────────────
add_executable(phase4_signal src/phase4_signal.cpp)
target_link_libraries(phase4_signal PRIVATE ${ABSL_COMMON_LIBS})

# ── Phase 5: Order Management System ────────────────────────────────────────
add_executable(phase5_oms src/phase5_oms.cpp)
target_link_libraries(phase5_oms PRIVATE ${ABSL_COMMON_LIBS})

# ── Phase 6: End-to-End Simulation ──────────────────────────────────────────
add_executable(phase6_simulation src/phase6_simulation.cpp)
target_link_libraries(phase6_simulation PRIVATE ${ABSL_COMMON_LIBS})

# ── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_orderbook
    phase2_matching
    phase3_feed
    phase4_signal
    phase5_oms
    phase6_simulation
)
