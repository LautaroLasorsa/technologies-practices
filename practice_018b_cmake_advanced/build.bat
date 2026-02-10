@echo off
REM ============================================================================
REM Build script for Practice 018b: Advanced CMake
REM
REM Uses CMake presets when available, falls back to manual cmake invocation.
REM
REM Usage:
REM   build.bat              Build with MSVC Release (default)
REM   build.bat debug        Build with MSVC Debug
REM   build.bat mingw        Build with MinGW Release
REM   build.bat test         Build MSVC Debug + run tests
REM   build.bat package      Build MSVC Release + create ZIP package
REM   build.bat clean        Remove all build directories
REM   build.bat install      Build MSVC Release + install to _install/
REM ============================================================================

set PRESET=msvc-release
set CONFIG=Release

if "%1"=="debug" (
    set PRESET=msvc-debug
    set CONFIG=Debug
)

if "%1"=="mingw" (
    set PRESET=mingw-release
    set CONFIG=Release
)

if "%1"=="clean" (
    echo Cleaning build directories...
    if exist build rmdir /s /q build
    if exist _install rmdir /s /q _install
    echo Done.
    exit /b 0
)

echo.
echo === Practice 018b: Advanced CMake ===
echo.
echo Preset: %PRESET%
echo.

REM ─── Configure ─────────────────────────────────────────────────────────────

echo [1/2] Configuring with preset: %PRESET%
cmake --preset=%PRESET%
if errorlevel 1 (
    echo.
    echo CMake configuration failed!
    echo Make sure CMake 3.21+ is installed and the preset is valid.
    echo.
    echo Available presets:
    cmake --list-presets
    exit /b 1
)

REM ─── Build ─────────────────────────────────────────────────────────────────

echo.
echo [2/2] Building with preset: %PRESET%
cmake --build --preset=%PRESET%
if errorlevel 1 (
    echo.
    echo Build failed! Check the errors above.
    exit /b 1
)

echo.
echo === Build successful! ===
echo.

REM ─── Optional: test ────────────────────────────────────────────────────────

if "%1"=="test" (
    echo === Running tests ===
    echo.
    ctest --preset=msvc-debug
    echo.
)

REM ─── Optional: package ─────────────────────────────────────────────────────

if "%1"=="package" (
    echo === Creating package ===
    echo.
    cpack --config build/%PRESET%/CPackConfig.cmake -G ZIP
    echo.
)

REM ─── Optional: install ─────────────────────────────────────────────────────

if "%1"=="install" (
    echo === Installing ===
    echo.
    cmake --install build/%PRESET%
    echo.
    echo Installed to: _install/%PRESET%/
    echo.
)

REM ─── Print run instructions ────────────────────────────────────────────────

echo Run individual targets:
if "%1"=="mingw" (
    echo   build\%PRESET%\main_app.exe
    echo   build\%PRESET%\codegen\codegen_demo.exe
) else (
    echo   build\%PRESET%\%CONFIG%\main_app.exe
    echo   build\%PRESET%\codegen\%CONFIG%\codegen_demo.exe
)
echo.
