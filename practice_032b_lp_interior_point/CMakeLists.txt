cmake_minimum_required(VERSION 3.16)
project(practice_032b_lp_interior_point LANGUAGES CXX)

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

# Phase 1: Dual construction & duality verification
add_executable(phase1_dual src/dual_construction.cpp)
target_link_libraries(phase1_dual PRIVATE Eigen3::Eigen)

# Phase 2: Complementary slackness & shadow prices
add_executable(phase2_complementary src/complementary_slackness.cpp)
target_link_libraries(phase2_complementary PRIVATE Eigen3::Eigen)

# Phase 3: Sensitivity analysis
add_executable(phase3_sensitivity src/sensitivity.cpp)
target_link_libraries(phase3_sensitivity PRIVATE Eigen3::Eigen)

# Phase 4: Barrier / interior point method
add_executable(phase4_barrier src/barrier_method.cpp)
target_link_libraries(phase4_barrier PRIVATE Eigen3::Eigen)

# ─── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_dual phase2_complementary phase3_sensitivity phase4_barrier
)
