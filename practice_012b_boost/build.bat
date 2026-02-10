@echo off
REM Build script for practice_012b_boost
REM
REM Usage:
REM   build.bat                          -- configure + build (Debug)
REM   build.bat Release                  -- configure + build (Release)
REM   build.bat clean                    -- remove build directory
REM
REM Prerequisites:
REM   - CMake >= 3.16 on PATH
REM   - vcpkg installed with Boost: vcpkg install boost
REM   - Set VCPKG_ROOT env variable or edit the toolchain path below

setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Debug
if /i "%BUILD_TYPE%"=="clean" (
    echo Removing build directory...
    rmdir /s /q build 2>nul
    echo Done.
    exit /b 0
)

REM --- Resolve vcpkg toolchain ---
if defined VCPKG_ROOT (
    set TOOLCHAIN=-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake
) else (
    echo WARNING: VCPKG_ROOT not set. CMake will search default paths for Boost.
    echo          Set VCPKG_ROOT or pass -DCMAKE_TOOLCHAIN_FILE manually.
    set TOOLCHAIN=
)

echo === Configuring [%BUILD_TYPE%] ===
cmake -B build -DCMAKE_BUILD_TYPE=%BUILD_TYPE% %TOOLCHAIN%
if errorlevel 1 (
    echo CMake configure failed.
    exit /b 1
)

echo === Building ===
cmake --build build --config %BUILD_TYPE%
if errorlevel 1 (
    echo Build failed.
    exit /b 1
)

echo === Build succeeded ===
echo Run: build\%BUILD_TYPE%\boost_practice.exe --help
