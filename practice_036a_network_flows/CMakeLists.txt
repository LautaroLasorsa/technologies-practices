cmake_minimum_required(VERSION 3.16)
project(practice_036a_network_flows LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(phase1_max_flow src/max_flow.cpp)
add_executable(phase2_min_cost_flow src/min_cost_flow.cpp)
add_executable(phase3_hungarian src/hungarian.cpp)

add_custom_target(all_phases DEPENDS phase1_max_flow phase2_min_cost_flow phase3_hungarian)
