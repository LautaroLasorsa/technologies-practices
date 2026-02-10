cmake_minimum_required(VERSION 3.16)
project(boost_practice LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Find Boost components ---
# If using vcpkg, CMake will find Boost automatically via the toolchain file.
# Components: system (required by asio), graph, program_options, serialization
find_package(Boost REQUIRED COMPONENTS
    system
    graph
    program_options
    serialization
)

# --- Main executable ---
add_executable(boost_practice
    src/main.cpp
    src/cli.cpp
    src/graph_algo.cpp
    src/serialization.cpp
    src/server.cpp
)

target_include_directories(boost_practice PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(boost_practice PRIVATE
    Boost::system
    Boost::graph
    Boost::program_options
    Boost::serialization
    Boost::headers
)

# Windows-specific: link ws2_32 for Asio networking + set Windows target version
if(WIN32)
    target_link_libraries(boost_practice PRIVATE ws2_32 wsock32)
    target_compile_definitions(boost_practice PRIVATE _WIN32_WINNT=0x0601)
endif()
