@echo off
REM Requires VCPKG_ROOT environment variable. Set with: setx VCPKG_ROOT "C:\path\to\vcpkg"
if not defined VCPKG_ROOT (
    echo ERROR: VCPKG_ROOT is not set. Run: setx VCPKG_ROOT "C:\path\to\vcpkg"
    exit /b 1
)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1

set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "VCPKG_TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
set "SRCDIR=%~dp0."
set "BLDDIR=%~dp0build_clangd"

"%CMAKE%" -S "%SRCDIR%" -B "%BLDDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=cl "-DCMAKE_MAKE_PROGRAM=%NINJA%" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG_TOOLCHAIN%"

if errorlevel 1 (
    echo CMake configure failed.
    exit /b 1
)

copy /y "%BLDDIR%\compile_commands.json" "%SRCDIR%\compile_commands.json"
echo compile_commands.json copied to project root.
