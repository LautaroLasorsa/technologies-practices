cmake_minimum_required(VERSION 3.16)
project(practice_039_stochastic_dp_mdp LANGUAGES CXX)

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

# Phase 1: Value iteration on gridworld
add_executable(phase1_value_iteration src/value_iteration.cpp)
target_link_libraries(phase1_value_iteration PRIVATE Eigen3::Eigen)

# Phase 2: Policy iteration with exact evaluation
add_executable(phase2_policy_iteration src/policy_iteration.cpp)
target_link_libraries(phase2_policy_iteration PRIVATE Eigen3::Eigen)

# Phase 3: Inventory management MDP
add_executable(phase3_inventory src/inventory.cpp)
target_link_libraries(phase3_inventory PRIVATE Eigen3::Eigen)

# ─── Convenience: build all phases at once ────────────────────────────────────
add_custom_target(all_phases DEPENDS
    phase1_value_iteration phase2_policy_iteration phase3_inventory
)
