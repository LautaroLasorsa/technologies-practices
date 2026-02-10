@echo off
REM build.bat — Configure and build all CUDA practice phases.
REM Run from the practice_019a_cuda_fundamentals directory.

set CMAKE="C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set BUILD_DIR=build

if not exist %BUILD_DIR% mkdir %BUILD_DIR%

echo === Configuring with CMake ===
%CMAKE% -S . -B %BUILD_DIR% -G "Visual Studio 17 2022"
if errorlevel 1 (
    echo CMake configuration failed.
    exit /b 1
)

echo === Building (Release) ===
%CMAKE% --build %BUILD_DIR% --config Release
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo === Build complete ===
echo Executables are in %BUILD_DIR%\Release\
echo Run: %BUILD_DIR%\Release\phase1_hello.exe
