@echo off
REM ============================================================================
REM Build script for Practice 012a: C++17 Features & abseil-cpp
REM
REM Prerequisites:
REM   - CMake 3.16+ (cmake --version)
REM   - A C++17 compiler (MSVC 2019+, GCC 9+, or Clang 10+)
REM   - Git (for FetchContent to clone abseil-cpp)
REM   - Internet connection (first build only, to fetch abseil-cpp)
REM
REM Usage:
REM   build.bat          Build all targets (Release)
REM   build.bat debug    Build all targets (Debug)
REM   build.bat clean    Remove build directory
REM ============================================================================

set BUILD_DIR=build
set BUILD_TYPE=Release

if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo Done.
    exit /b 0
)

echo.
echo === Practice 012a: C++17 Features ^& abseil-cpp ===
echo.
echo Build type: %BUILD_TYPE%
echo.

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure (first time fetches abseil-cpp -- may take a few minutes)
echo [1/2] Configuring with CMake...
cmake -S . -B %BUILD_DIR% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
    echo.
    echo CMake configuration failed!
    echo Make sure you have CMake 3.16+ and a C++17 compiler installed.
    exit /b 1
)

REM Build all targets
echo.
echo [2/2] Building all phases...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% --target all_phases
if errorlevel 1 (
    echo.
    echo Build failed! Check the errors above.
    exit /b 1
)

echo.
echo === Build successful! ===
echo.
echo Run individual phases:
echo   %BUILD_DIR%\%BUILD_TYPE%\phase1_bindings.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase2_optional_variant.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase3_strings.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase4_containers.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase5_errors.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase6_constexpr.exe
echo.
