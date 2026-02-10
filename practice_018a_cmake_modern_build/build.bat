@echo off
REM ============================================================================
REM Build script for Practice 018a: Modern CMake
REM
REM Prerequisites:
REM   - CMake 3.16+ (cmake --version)
REM   - MSVC 19.39+ (VS 2022)
REM   - Git (for FetchContent to clone dependencies)
REM   - Internet connection (first build only, to fetch nlohmann/json + fmt)
REM
REM Usage:
REM   build.bat              Build all targets (Release)
REM   build.bat debug        Build all targets (Debug)
REM   build.bat clean        Remove build directory
REM   build.bat test         Build and run tests
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
echo === Practice 018a: Modern CMake ===
echo.
echo Build type: %BUILD_TYPE%
echo.

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure (first run fetches nlohmann/json + fmt — may take a minute)
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
echo [2/2] Building all targets...
cmake --build %BUILD_DIR% --config %BUILD_TYPE%
if errorlevel 1 (
    echo.
    echo Build failed! Check the errors above.
    exit /b 1
)

echo.
echo === Build successful! ===
echo.
echo Run individual targets:
echo   %BUILD_DIR%\%BUILD_TYPE%\main_app.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\json_demo.exe
echo   %BUILD_DIR%\%BUILD_TYPE%\test_greeter.exe
echo.

REM If user asked for tests, run them
if "%1"=="test" (
    echo === Running tests ===
    echo.
    %BUILD_DIR%\%BUILD_TYPE%\test_greeter.exe
    echo.
)
