cmake_minimum_required(VERSION 3.16)
project(practice_033a_convex_first_order LANGUAGES CXX)
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

add_executable(phase1_gradient_descent src/gradient_descent.cpp)
target_link_libraries(phase1_gradient_descent PRIVATE Eigen3::Eigen)

add_executable(phase2_line_search src/line_search.cpp)
target_link_libraries(phase2_line_search PRIVATE Eigen3::Eigen)

add_executable(phase3_projected_gd src/projected_gd.cpp)
target_link_libraries(phase3_projected_gd PRIVATE Eigen3::Eigen)

add_executable(phase4_nesterov src/nesterov.cpp)
target_link_libraries(phase4_nesterov PRIVATE Eigen3::Eigen)

add_custom_target(all_phases DEPENDS
    phase1_gradient_descent phase2_line_search phase3_projected_gd phase4_nesterov
)
