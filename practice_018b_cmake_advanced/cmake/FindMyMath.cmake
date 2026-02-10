# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Phase 3: Custom Find Module — FindMyMath.cmake                           ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#
# A Find module is a CMake script that locates a third-party library that does
# NOT ship its own CMake config files. When you call:
#   find_package(MyMath REQUIRED)
#
# CMake searches CMAKE_MODULE_PATH for FindMyMath.cmake and runs it.
#
# The module's job:
#   1. Find the header(s) with find_path()
#   2. Find the library file(s) with find_library()
#   3. Validate with find_package_handle_standard_args()
#   4. Create an IMPORTED target so consumers can use target_link_libraries()
#
# IMPORTANT: Find modules (FindXxx.cmake) vs Config packages (XxxConfig.cmake):
#   - Find modules: YOU write them, for libs that don't provide CMake support
#   - Config packages: the LIBRARY AUTHORS generate them via install(EXPORT ...)
#   - find_package() checks for Config first, then falls back to Find modules
#   - Prefer Config packages when available (more accurate, maintained by authors)
#
# Docs: https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules
# Docs: https://cmake.org/cmake/help/latest/command/find_package.html

# ─── Step 1: Find the header ────────────────────────────────────────────────
#
# find_path(VARIABLE_NAME header_filename PATHS ... HINTS ...)
#
# Searches for a directory containing the given header file.
# On success, VARIABLE_NAME is set to the directory path.
# On failure, VARIABLE_NAME is set to "VARIABLE_NAME-NOTFOUND".
#
# HINTS: preferred search paths (user-provided or pkg-config paths)
# PATHS: fallback search paths
# PATH_SUFFIXES: subdirectories to check within each search path
#
# TODO(human): Use find_path to locate "mymath.h"
#
# The installed header will be at: <install_prefix>/include/mymath/mymath.h
# So we need PATH_SUFFIXES "mymath" and HINTS "${MYMATH_ROOT}/include"
# (MYMATH_ROOT can be passed by the user: -DMYMATH_ROOT=/path/to/install)
#
# Pattern:
#   find_path(MyMath_INCLUDE_DIR
#       mymath.h
#       HINTS ${MYMATH_ROOT}/include ENV MYMATH_ROOT
#       PATH_SUFFIXES mymath
#   )
#
# Fill in below:

# find_path(MyMath_INCLUDE_DIR
#     mymath.h
#     HINTS ${MYMATH_ROOT}/include ENV MYMATH_ROOT
#     PATH_SUFFIXES mymath
# )


# ─── Step 2: Find the library ───────────────────────────────────────────────
#
# find_library(VARIABLE_NAME library_name PATHS ... HINTS ...)
#
# Searches for a library file (mymath.lib on Windows, libmymath.a on Linux).
# CMake automatically adds platform-specific prefixes (lib) and suffixes (.lib/.a/.so).
#
# TODO(human): Use find_library to locate the mymath library.
#
# The installed library will be at: <install_prefix>/lib/mymath.lib (or libmymath.a)
#
# Pattern:
#   find_library(MyMath_LIBRARY
#       NAMES mymath
#       HINTS ${MYMATH_ROOT}/lib ENV MYMATH_ROOT
#   )
#
# Fill in below:

# find_library(MyMath_LIBRARY
#     NAMES mymath
#     HINTS ${MYMATH_ROOT}/lib ENV MYMATH_ROOT
# )


# ─── Step 3: Validate ───────────────────────────────────────────────────────
#
# find_package_handle_standard_args():
#   Standard CMake macro that:
#   - Checks if all REQUIRED_VARS are found (not *-NOTFOUND)
#   - Sets MyMath_FOUND to TRUE/FALSE
#   - Prints a nice status message
#   - Handles REQUIRED/QUIET flags from find_package()
#
# This is what makes find_package(MyMath REQUIRED) produce a clear error
# message when the library is not found, instead of a cryptic CMake error.

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(MyMath
    REQUIRED_VARS
        MyMath_LIBRARY
        MyMath_INCLUDE_DIR
)


# ─── Step 4: Create imported target ─────────────────────────────────────────
#
# An IMPORTED target wraps the found library into a proper CMake target that
# consumers can use with target_link_libraries(). Without this, users would
# have to manually set include directories and link paths.
#
# TODO(human): Create an IMPORTED target MyMath::MyMath
#
# Pattern:
#   if(MyMath_FOUND AND NOT TARGET MyMath::MyMath)
#       add_library(MyMath::MyMath UNKNOWN IMPORTED)
#       set_target_properties(MyMath::MyMath PROPERTIES
#           IMPORTED_LOCATION "${MyMath_LIBRARY}"
#           INTERFACE_INCLUDE_DIRECTORIES "${MyMath_INCLUDE_DIR}"
#       )
#   endif()
#
# UNKNOWN IMPORTED: lets CMake figure out if it's STATIC or SHARED from the file.
# IMPORTED_LOCATION: path to the actual .lib/.a/.so file.
# INTERFACE_INCLUDE_DIRECTORIES: propagated to consumers via target_link_libraries.
#
# Fill in below:

# if(MyMath_FOUND AND NOT TARGET MyMath::MyMath)
#     add_library(MyMath::MyMath UNKNOWN IMPORTED)
#     set_target_properties(MyMath::MyMath PROPERTIES
#         IMPORTED_LOCATION "${MyMath_LIBRARY}"
#         INTERFACE_INCLUDE_DIRECTORIES "${MyMath_INCLUDE_DIR}"
#     )
# endif()


# ─── Cleanup: Hide internal variables from cmake-gui ────────────────────────
mark_as_advanced(MyMath_INCLUDE_DIR MyMath_LIBRARY)
