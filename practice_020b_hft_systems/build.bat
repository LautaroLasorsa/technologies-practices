@echo off
REM ============================================================================
REM Build script for Practice 020b: HFT Systems
REM
REM Prerequisites:
REM   - CMake 3.16+ (cmake --version)
REM   - MSVC 2022 (VS Developer Command Prompt or vcvarsall.bat)
REM   - Git (for FetchContent to clone abseil-cpp)
REM   - Internet connection (first build only, to fetch abseil-cpp)
REM
REM Usage:
REM   build.bat          Build all targets (Release)
REM   build.bat debug    Build all targets (Debug)
REM   build.bat clean    Remove build directory
REM   build.bat phase1   Build only phase1_orderbook (Release)
REM   build.bat phase2   Build only phase2_matching (Release)
REM   build.bat phase3   Build only phase3_feed (Release)
REM   build.bat phase4   Build only phase4_signal (Release)
REM   build.bat phase5   Build only phase5_oms (Release)
REM   build.bat phase6   Build only phase6_simulation (Release)
REM ============================================================================

set BUILD_DIR=build
set BUILD_TYPE=Release
set TARGET=all_phases

if "%1"=="debug" set BUILD_TYPE=Debug
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo Done.
    exit /b 0
)

if "%1"=="phase1" set TARGET=phase1_orderbook
if "%1"=="phase2" set TARGET=phase2_matching
if "%1"=="phase3" set TARGET=phase3_feed
if "%1"=="phase4" set TARGET=phase4_signal
if "%1"=="phase5" set TARGET=phase5_oms
if "%1"=="phase6" set TARGET=phase6_simulation

echo.
echo === Practice 020b: HFT Systems -- Order Book, Matching Engine ^& Feed Handler ===
echo.
echo Build type: %BUILD_TYPE%
echo Target:     %TARGET%
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

REM Build target(s)
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
echo   %BUILD_DIR%\%BUILD_TYPE%\phase1_orderbook.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase2_matching.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase3_feed.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase4_signal.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase5_oms.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\phase6_simulation.exe
echo.
