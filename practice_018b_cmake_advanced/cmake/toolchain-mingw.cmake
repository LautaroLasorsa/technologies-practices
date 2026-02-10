# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 2: MinGW Toolchain File                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# A toolchain file tells CMake HOW to compile — which compiler, linker, and
# system libraries to use. It's loaded BEFORE the project() command via:
#   cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-mingw.cmake
#
# Or, with presets, the "toolchainFile" field references this file automatically.
#
# This file configures MinGW GCC as the compiler, allowing you to build the
# same project with a different compiler than MSVC — useful for portability
# testing, CI matrices, and eventual Linux cross-compilation.
#
# Docs: https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html

# ─── System identification ───────────────────────────────────────────────────
#
# CMAKE_SYSTEM_NAME:
#   Tells CMake what OS the TARGET platform is.
#   - "Windows" → building FOR Windows (even if compiling ON Windows)
#   - "Linux"   → cross-compiling for Linux
#   - "Generic" → bare-metal / embedded (no OS)
#
#   When CMAKE_SYSTEM_NAME differs from the host OS, CMake enters
#   "cross-compiling mode" and sets CMAKE_CROSSCOMPILING=TRUE.
#   Since we're using MinGW on Windows to target Windows, we set "Windows".
set(CMAKE_SYSTEM_NAME Windows)

# CMAKE_SYSTEM_PROCESSOR:
#   Tells CMake the target CPU architecture.
#   Common values: "x86_64", "AMD64", "i686", "aarch64", "arm"
set(CMAKE_SYSTEM_PROCESSOR x86_64)


# ─── Compiler paths ─────────────────────────────────────────────────────────
#
# TODO(human): Set the paths to your MinGW compilers.
#
# MinGW is typically installed at C:\MinGW\bin on your system.
# The compilers are:
#   - C compiler:   C:/MinGW/bin/gcc.exe
#   - C++ compiler: C:/MinGW/bin/g++.exe
#
# Fill in the paths below. Use forward slashes (CMake style) even on Windows.
#
# Hint: Run `where gcc` or `C:\MinGW\bin\gcc.exe --version` to verify the path.

# set(CMAKE_C_COMPILER   "C:/MinGW/bin/gcc.exe")    # <-- uncomment and verify path
# set(CMAKE_CXX_COMPILER "C:/MinGW/bin/g++.exe")    # <-- uncomment and verify path


# ─── Find root path ─────────────────────────────────────────────────────────
#
# CMAKE_FIND_ROOT_PATH:
#   When cross-compiling, this tells find_package(), find_library(), find_path()
#   where to look for TARGET-platform libraries and headers.
#
#   For MinGW on Windows targeting Windows, the MinGW installation itself
#   contains the target libraries (libstdc++, etc.).
#
# CMAKE_FIND_ROOT_PATH_MODE_PROGRAM:
#   NEVER  → Look for programs (compilers, tools) on the HOST, not in the sysroot.
#            You want to run cmake, make, etc. from the host.
#
# CMAKE_FIND_ROOT_PATH_MODE_LIBRARY / _INCLUDE:
#   ONLY   → Only search the target sysroot for libraries and headers.
#            Prevents accidentally linking against host (MSVC) libraries.
#   BOTH   → Search both sysroot and host paths (less strict, sometimes needed).
#
# For MinGW-on-Windows-targeting-Windows, BOTH is usually fine since host=target OS.
# For true cross-compilation (Windows→Linux), you'd use ONLY.

set(CMAKE_FIND_ROOT_PATH "C:/MinGW")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)


# ─── Optional: Compiler flags ───────────────────────────────────────────────
#
# You can set default compiler flags for this toolchain here.
# These apply to ALL targets built with this toolchain.
#
# Example (not required for this practice):
#   set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra")
#   set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O2 -DNDEBUG")
#   set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -O0")
