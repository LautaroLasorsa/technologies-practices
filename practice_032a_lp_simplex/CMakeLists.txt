cmake_minimum_required(VERSION 3.16)
project(practice_032a_lp_simplex LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Fetch Eigen (header-only linear algebra) ────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# ─── Executables (one per phase) ──────────────────────────────────────────────

# Phase 1: LP formulation & standard form
add_executable(phase1_formulation src/formulation.cpp)
target_link_libraries(phase1_formulation PRIVATE Eigen3::Eigen)

# Phase 2: Simplex pivoting mechanics
add_executable(phase2_pivoting src/pivoting.cpp)
target_link_libraries(phase2_pivoting PRIVATE Eigen3::Eigen)

# Phase 3: Full simplex solver loop
add_executable(phase3_simplex src/simplex.cpp)
target_link_libraries(phase3_simplex PRIVATE Eigen3::Eigen)

# Phase 4: Two-phase simplex for arbitrary LPs
add_executable(phase4_two_phase src/two_phase.cpp)
target_link_libraries(phase4_two_phase PRIVATE Eigen3::Eigen)

# ─── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_formulation phase2_pivoting phase3_simplex phase4_two_phase
)
