@echo off
REM ============================================================================
REM Build script for Practice 020a: HFT Low-Latency C++ Patterns
REM
REM Prerequisites:
REM   - CMake 3.16+ (cmake --version)
REM   - MSVC via Visual Studio 2022 (cl.exe available in PATH)
REM   - No external dependencies -- pure C++17 + Windows API
REM
REM Usage:
REM   build.bat              Build all targets (Release)
REM   build.bat debug        Build all targets (Debug)
REM   build.bat clean        Remove build directory
REM   build.bat phase1       Build only phase1_cache (Release)
REM   build.bat phase2       Build only phase2_spsc (Release)
REM   build.bat phase3       Build only phase3_memory (Release)
REM   build.bat phase4       Build only phase4_dispatch (Release)
REM   build.bat phase5       Build only phase5_timing (Release)
REM   build.bat phase6       Build only phase6_pipeline (Release)
REM
REM TIP: Run from a "Developer Command Prompt for VS 2022" or
REM      "x64 Native Tools Command Prompt" so cl.exe is on PATH.
REM ============================================================================

set BUILD_DIR=build
set BUILD_TYPE=Release
set TARGET=all_phases

if "%1"=="debug" (
    set BUILD_TYPE=Debug
    if not "%2"=="" set TARGET=phase%2
    goto :build
)
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo Done.
    exit /b 0
)
if "%1"=="phase1" set TARGET=phase1_cache
if "%1"=="phase2" set TARGET=phase2_spsc
if "%1"=="phase3" set TARGET=phase3_memory
if "%1"=="phase4" set TARGET=phase4_dispatch
if "%1"=="phase5" set TARGET=phase5_timing
if "%1"=="phase6" set TARGET=phase6_pipeline

:build
echo.
echo === Practice 020a: HFT Low-Latency C++ Patterns ===
echo.
echo Build type: %BUILD_TYPE%
echo Target:     %TARGET%
echo.

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure
echo [1/2] Configuring with CMake...
cmake -S . -B %BUILD_DIR% -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
if errorlevel 1 (
    echo.
    echo CMake configuration failed!
    echo Make sure you have CMake 3.16+ and MSVC installed.
    echo TIP: Run from "Developer Command Prompt for VS 2022"
    exit /b 1
)

REM Build
echo.
echo [2/2] Building %TARGET%...
cmake --build %BUILD_DIR% --config %BUILD_TYPE% --target %TARGET%
if errorlevel 1 (
    echo.
    echo Build failed! Check the errors above.
    exit /b 1
)

echo.
echo === Build successful! ===
echo.
echo Run individual phases:
echo   %BUILD_DIR%\%BUILD_TYPE%\phase1_cache.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase2_spsc.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase3_memory.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase4_dispatch.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase5_timing.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase6_pipeline.exe
echo.
