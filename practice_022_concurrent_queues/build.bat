@echo off
REM ============================================================================
REM Build script for Practice 022: Concurrent Queues (moodycamel)
REM
REM Prerequisites:
REM   - CMake 3.16+ (bundled with VS 2022 or standalone)
REM   - MSVC 2019+ (VS 2022 recommended)
REM   - Git (for FetchContent to clone concurrentqueue)
REM   - Internet connection (first build only, to fetch moodycamel)
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
echo === Practice 022: Concurrent Queues (moodycamel::ConcurrentQueue) ===
echo.
echo Build type: %BUILD_TYPE%
echo.

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure (first time fetches concurrentqueue -- should be quick, header-only)
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
echo   %BUILD_DIR%\%BUILD_TYPE%\phase1_basics.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase2_multithreaded.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase3_tokens.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase4_bulk.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase5_blocking.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase6_patterns.exe
echo.
