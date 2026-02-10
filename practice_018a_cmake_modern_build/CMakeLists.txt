# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Practice 018a: Modern CMake — Target-Based Builds & Dependency Management ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# This CMakeLists.txt IS the learning material. Read it top-to-bottom.
# Each section corresponds to a practice phase.
#
# Key principle: MODERN CMAKE IS TARGET-BASED.
# Everything (includes, flags, definitions, dependencies) attaches to a TARGET,
# never globally. If you see include_directories(), link_directories(), or
# add_definitions() in someone else's code, that's legacy CMake.


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: CMake Fundamentals
# ═══════════════════════════════════════════════════════════════════════════════

# cmake_minimum_required:
#   - Sets the minimum CMake version your project supports
#   - Affects policy defaults (newer versions enable stricter, better behavior)
#   - Rule of thumb: use the oldest version that has the features you need
#   - FetchContent requires 3.11+, FetchContent_MakeAvailable requires 3.14+
cmake_minimum_required(VERSION 3.16)

# project():
#   - Declares the project name, version, and languages
#   - VERSION sets PROJECT_VERSION, PROJECT_VERSION_MAJOR/MINOR/PATCH automatically
#   - LANGUAGES CXX means "C++ only" (skip C compiler detection → faster configure)
project(practice_018a
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Modern CMake learning practice"
)

# C++ standard settings — applied per-target via target_compile_features later,
# but these variables set the default for all targets in this project.
#
# CMAKE_CXX_STANDARD: which C++ standard to use (11, 14, 17, 20, 23)
# CMAKE_CXX_STANDARD_REQUIRED: fail if compiler doesn't support it (vs. silently downgrade)
# CMAKE_CXX_EXTENSIONS: OFF means -std=c++17, not -std=gnu++17 (portable)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: FetchContent — Pull dependencies from GitHub
# ═══════════════════════════════════════════════════════════════════════════════

# We keep FetchContent declarations in a separate file for cleanliness.
# This is a common pattern in larger projects: cmake/ directory holds modules.
#
# include() executes another .cmake file in the current scope — like #include in C++.
# After this, targets `nlohmann_json::nlohmann_json` and `fmt::fmt` are available.
include(cmake/FetchDeps.cmake)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Library target — the `greeter` library
# ═══════════════════════════════════════════════════════════════════════════════

# add_library creates a library target. Types:
#   STATIC  — compiled into a .lib/.a, linked at build time (default)
#   SHARED  — compiled into a .dll/.so, linked at runtime
#   INTERFACE — header-only, no compiled code (e.g., nlohmann/json)
#
# We use STATIC here (the default when you don't specify).
add_library(greeter
    src/greeter.cpp
    src/greeter.h       # Headers listed here show up in IDE project views (optional but nice)
)

# target_include_directories:
#   Tells CMake where to find headers for this target.
#
#   PUBLIC    → this target AND anyone who links against it can see these includes
#   PRIVATE   → only this target can see them
#   INTERFACE → only consumers see them (the target itself doesn't use them)
#
#   Generator expressions ($<...>) let you specify DIFFERENT paths for:
#     BUILD_INTERFACE  — when building this project (path in source tree)
#     INSTALL_INTERFACE — when installed and consumed by another project
#
#   This is the MODERN way. Never use the global include_directories() command.
target_include_directories(greeter
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/generated  # For config.h (Phase 4)
)

# target_compile_features:
#   Declares that this target REQUIRES certain compiler features.
#   cxx_std_17 means "this target needs C++17 or later."
#   PUBLIC means consumers also get this requirement propagated.
#
#   This is more precise than setting CMAKE_CXX_STANDARD globally because
#   it's per-target and participates in dependency propagation.
target_compile_features(greeter PUBLIC cxx_std_17)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Custom options, configure_file & compile definitions
# ═══════════════════════════════════════════════════════════════════════════════

# option():
#   Declares a boolean ON/OFF setting that the user can toggle at configure time:
#     cmake -S . -B build -DENABLE_GREETING_EMOJI=ON
#
#   Options appear in cmake-gui and ccmake for interactive configuration.
option(ENABLE_GREETING_EMOJI "Append emoji to greetings" OFF)

# configure_file():
#   Reads a template file (config.h.in), replaces @VAR@ placeholders with
#   CMake variable values, and writes the result to the build directory.
#
#   The output goes to build/generated/config.h — NEVER into the source tree.
#   This keeps the source tree clean and allows different build configurations
#   to coexist (Debug vs Release, emoji ON vs OFF).
#
#   Template: src/config.h.in  →  Output: build/generated/config.h
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/config.h
    @ONLY  # Only replace @VAR@, not ${VAR} (safer — avoids accidental substitution)
)

# target_compile_definitions:
#   Adds preprocessor definitions for a specific target.
#   PRIVATE means only this target sees the definition.
#
#   Alternative to configure_file for simple flags — but configure_file is better
#   when you need structured data (version strings, paths, etc.) in a header.
#
#   Example (not used here, but for reference):
#     target_compile_definitions(greeter PRIVATE DEBUG_GREETER=1)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 + 3 + 4: Main executable
# ═══════════════════════════════════════════════════════════════════════════════

# add_executable: creates an executable target from source files.
add_executable(main_app src/main.cpp)

# target_link_libraries:
#   THE most important command in modern CMake.
#
#   Scope keywords (PUBLIC/PRIVATE/INTERFACE):
#   ┌───────────┬──────────────────────────────────────────────────────────────┐
#   │ PRIVATE   │ Only main_app uses greeter. If something links against      │
#   │           │ main_app, it does NOT get greeter's includes/flags.          │
#   ├───────────┼──────────────────────────────────────────────────────────────┤
#   │ PUBLIC    │ main_app uses greeter AND anyone linking against main_app    │
#   │           │ also gets greeter's includes/flags propagated.               │
#   ├───────────┼──────────────────────────────────────────────────────────────┤
#   │ INTERFACE │ main_app does NOT use greeter itself, but anyone linking     │
#   │           │ against main_app gets greeter's includes/flags.              │
#   │           │ (Rare for executables; common for header-only libraries.)    │
#   └───────────┴──────────────────────────────────────────────────────────────┘
#
#   When you link against `greeter` with PUBLIC scope on greeter's includes,
#   main_app automatically gets greeter's include directories.
#   This is the TRANSITIVE DEPENDENCY magic of modern CMake.
target_link_libraries(main_app PRIVATE greeter)

# main_app also needs access to the generated config.h
target_include_directories(main_app PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: JSON + fmt demo executable
# ═══════════════════════════════════════════════════════════════════════════════

add_executable(json_demo apps/json_demo.cpp)

# This target uses two FetchContent-provided libraries:
#   - nlohmann_json::nlohmann_json (header-only → INTERFACE library, no .lib to link)
#   - fmt::fmt (compiled library → has a .lib that gets linked)
#
# PRIVATE because json_demo is an executable — nothing links against it.
target_link_libraries(json_demo
    PRIVATE
        nlohmann_json::nlohmann_json
        fmt::fmt
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Test executable
# ═══════════════════════════════════════════════════════════════════════════════

# Enable CTest integration — allows `ctest` command to discover and run tests.
# This must be called before add_test().
enable_testing()

add_executable(test_greeter tests/test_greeter.cpp)
target_link_libraries(test_greeter PRIVATE greeter)
target_include_directories(test_greeter PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

# add_test:
#   Registers an executable as a test with CTest.
#   After building, you can run:  ctest --test-dir build -C Release
add_test(
    NAME greeter_tests
    COMMAND test_greeter
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Install rules & export
# ═══════════════════════════════════════════════════════════════════════════════
#
# Install rules define what happens when you run:
#   cmake --install build --prefix /some/path
#
# This is how you make your library CONSUMABLE by other CMake projects via
# find_package(practice_018a). Without install rules, your library only
# exists inside your build tree.

# GNUInstallDirs provides standard directory variables:
#   CMAKE_INSTALL_LIBDIR     → lib/
#   CMAKE_INSTALL_INCLUDEDIR → include/
#   CMAKE_INSTALL_BINDIR     → bin/
include(GNUInstallDirs)

# install(TARGETS ... EXPORT ...):
#   - Copies built artifacts (libraries, executables) to the install prefix
#   - EXPORT registers this target in a named export set for find_package
#   - ARCHIVE DESTINATION: where .lib/.a files go
#   - LIBRARY DESTINATION: where .so/.dylib files go (shared libs)
#   - RUNTIME DESTINATION: where .exe/.dll files go
install(TARGETS greeter
    EXPORT practice_018a-targets          # Name of the export set
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# install(FILES ...):
#   Copies files (headers) to the install prefix.
#   This makes the header available to consumers.
install(FILES src/greeter.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/practice_018a
)

# Also install the generated config.h
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/generated/config.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/practice_018a
)

# install(EXPORT ...):
#   Generates a CMake file that describes the exported targets.
#   This is what find_package() reads to discover your library.
#
#   NAMESPACE: prepends "practice_018a::" to target names, so consumers write:
#     find_package(practice_018a REQUIRED)
#     target_link_libraries(my_app PRIVATE practice_018a::greeter)
install(EXPORT practice_018a-targets
    FILE practice_018a-targets.cmake
    NAMESPACE practice_018a::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/practice_018a
)

# For a complete find_package experience, you'd also generate:
#   - practice_018aConfig.cmake (or use configure_package_config_file)
#   - practice_018aConfigVersion.cmake (via write_basic_package_version_file)
# This is left as a reference — see the CMake packaging docs for full details.
# https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: compile_commands.json & editor tooling
# ═══════════════════════════════════════════════════════════════════════════════

# CMAKE_EXPORT_COMPILE_COMMANDS:
#   When ON, CMake generates a compile_commands.json in the build directory.
#   This file lists every translation unit with its exact compiler invocation.
#   Editors like clangd, VS Code (C/C++ extension), and CLion use it for:
#     - Accurate IntelliSense / code completion
#     - Jump to definition across FetchContent dependencies
#     - Correct include path resolution
#
# NOTE: This only works with Makefile and Ninja generators, NOT with the
# Visual Studio generator. If using MSVC, consider:
#   cmake -S . -B build_clangd -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# and symlink/copy compile_commands.json to the project root.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary of targets
# ═══════════════════════════════════════════════════════════════════════════════
#
# Library targets:
#   greeter           — STATIC library (src/greeter.cpp)
#
# Executable targets:
#   main_app          — Phase 1+3+4: uses greeter + config.h
#   json_demo         — Phase 2: uses nlohmann/json + fmt via FetchContent
#   test_greeter      — Phase 3: tests for greeter library
#
# External (FetchContent) targets:
#   nlohmann_json::nlohmann_json  — header-only JSON library
#   fmt::fmt                      — compiled formatting library
#
# Dependency graph:
#   main_app ──PRIVATE──▶ greeter
#   json_demo ──PRIVATE──▶ nlohmann_json::nlohmann_json
#   json_demo ──PRIVATE──▶ fmt::fmt
#   test_greeter ──PRIVATE──▶ greeter
