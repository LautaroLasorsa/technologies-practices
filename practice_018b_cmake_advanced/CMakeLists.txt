# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Practice 018b: Advanced CMake — Toolchains, Presets & Cross-Compilation  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# This project demonstrates advanced CMake features:
#   Phase 1: CMake Presets (see CMakePresets.json)
#   Phase 2: Toolchain files (see cmake/toolchain-mingw.cmake)
#   Phase 3: Custom Find modules (see cmake/FindMyMath.cmake)
#   Phase 4: CTest (see tests/CMakeLists.txt)
#   Phase 5: Custom commands & code generation (see codegen/CMakeLists.txt)
#   Phase 6: CPack packaging (bottom of this file)
#
# The C++ code is intentionally simple — the CMake files ARE the learning material.


# ═══════════════════════════════════════════════════════════════════════════════
# Project setup
# ═══════════════════════════════════════════════════════════════════════════════

# 3.21 required for CMake presets v3+ and improved preset features.
# Presets v1 requires 3.19, v2 requires 3.20, v3 requires 3.21.
cmake_minimum_required(VERSION 3.21)

project(practice_018b
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Advanced CMake: Toolchains, Presets & Cross-Compilation"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json for editor tooling (Ninja/Makefile generators only)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Append cmake/ directory to module path for FindMyMath.cmake
# ═══════════════════════════════════════════════════════════════════════════════
#
# CMAKE_MODULE_PATH tells find_package() where to look for Find modules.
# Without this, find_package(MyMath) wouldn't find our FindMyMath.cmake.
#
# This is how you make custom Find modules discoverable:
#   1. Put FindXxx.cmake in cmake/ directory
#   2. Add cmake/ to CMAKE_MODULE_PATH
#   3. find_package(Xxx) now works

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")


# ═══════════════════════════════════════════════════════════════════════════════
# Library target: mymath
# ═══════════════════════════════════════════════════════════════════════════════

add_library(mymath
    src/mymath.cpp
    src/mymath.h
)

target_include_directories(mymath
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include/mymath>
)

target_compile_features(mymath PUBLIC cxx_std_17)


# ═══════════════════════════════════════════════════════════════════════════════
# Main executable
# ═══════════════════════════════════════════════════════════════════════════════

add_executable(main_app src/main.cpp)
target_link_libraries(main_app PRIVATE mymath)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: CTest — testing subdirectory
# ═══════════════════════════════════════════════════════════════════════════════
#
# enable_testing() must be called in the ROOT CMakeLists.txt (not in subdirs)
# for CTest to properly discover tests from all subdirectories.
# This is because ctest looks for CTestTestfile.cmake in the build root.

enable_testing()
add_subdirectory(tests)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Code generation subdirectory
# ═══════════════════════════════════════════════════════════════════════════════

add_subdirectory(codegen)


# ═══════════════════════════════════════════════════════════════════════════════
# Install rules — make the mymath library consumable via find_package
# ═══════════════════════════════════════════════════════════════════════════════
#
# After `cmake --install --preset=msvc-release`, the library is installed to
# _install/msvc-release/ with:
#   include/mymath/mymath.h     — public header
#   lib/mymath.lib              — static library
#   lib/cmake/MyMath/           — CMake config files for find_package(MyMath)

include(GNUInstallDirs)

install(TARGETS mymath
    EXPORT MyMathTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES src/mymath.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mymath
)

# Install main_app executable too (for CPack)
install(TARGETS main_app
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Generate and install CMake config files so other projects can do:
#   find_package(MyMath REQUIRED)
#   target_link_libraries(consumer PRIVATE MyMath::mymath)
install(EXPORT MyMathTargets
    FILE MyMathTargets.cmake
    NAMESPACE MyMath::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyMath
)

# Generate MyMathConfig.cmake — the entry point for find_package(MyMath)
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/MyMathConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyMath
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/MyMath
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6: CPack — packaging
# ═══════════════════════════════════════════════════════════════════════════════
#
# CPack uses the install() rules above to determine WHAT goes into the package.
# The CPACK_* variables control HOW the package is created.
#
# After building:
#   cpack --config build/msvc-release/CPackConfig.cmake -G ZIP
#   → Creates practice_018b-1.0.0-win64.zip containing bin/, lib/, include/
#
# Or with the package preset:
#   cmake --build --preset=msvc-release --target package
#
# Docs: https://cmake.org/cmake/help/latest/module/CPack.html

set(CPACK_PACKAGE_NAME "practice_018b")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Advanced CMake practice — mymath library")
set(CPACK_PACKAGE_VENDOR "Practice 018b")

# CPACK_GENERATOR: which package formats to produce.
#   ZIP  — cross-platform archive (always available)
#   NSIS — Windows installer (requires NSIS installed)
#   TGZ  — tar.gz (common on Linux)
set(CPACK_GENERATOR "ZIP")

# CPACK_SYSTEM_NAME: appended to the package filename
# e.g., practice_018b-1.0.0-win64.zip
set(CPACK_SYSTEM_NAME "win64")

# CPACK_RESOURCE_FILE_LICENSE / CPACK_RESOURCE_FILE_README:
#   Files displayed during installation (NSIS) or included in the package.
#   Optional — omitted here since this is a practice project.

# include(CPack) MUST be the last command — it reads all CPACK_* variables
# and generates CPackConfig.cmake in the build directory.
include(CPack)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary of targets
# ═══════════════════════════════════════════════════════════════════════════════
#
# Library targets:
#   mymath           — STATIC library (src/mymath.cpp)
#
# Executable targets:
#   main_app         — Main app using mymath
#   codegen_demo     — Phase 5: code generation demo (codegen/)
#   test_mymath      — Phase 4: unit tests (tests/)
#   test_advanced    — Phase 4: advanced CTest features (tests/)
#
# Dependency graph:
#   main_app ──PRIVATE──▶ mymath
#   test_mymath ──PRIVATE──▶ mymath
#   test_advanced ──PRIVATE──▶ mymath
#   codegen_demo ──▶ (generated embedded_data.h ← data.txt)
