cmake_minimum_required(VERSION 3.16)
project(practice_033b_convex_second_order LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

add_executable(phase1_proximal_gd src/proximal_gd.cpp)
target_link_libraries(phase1_proximal_gd PRIVATE Eigen3::Eigen)

add_executable(phase2_admm src/admm.cpp)
target_link_libraries(phase2_admm PRIVATE Eigen3::Eigen)

add_executable(phase3_newton src/newton.cpp)
target_link_libraries(phase3_newton PRIVATE Eigen3::Eigen)

add_executable(phase4_lbfgs src/lbfgs.cpp)
target_link_libraries(phase4_lbfgs PRIVATE Eigen3::Eigen)

add_custom_target(all_phases DEPENDS
    phase1_proximal_gd phase2_admm phase3_newton phase4_lbfgs
)
