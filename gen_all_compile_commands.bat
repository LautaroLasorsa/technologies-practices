@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 >nul 2>&1

set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "NINJA=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
set "VCPKG=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
set "ROOT=%~dp0."

REM ---- Helper function ----
goto :start

:gen_cc
set "PROJ=%~1"
set "EXTRA=%~2"
set "SRCDIR=%ROOT%\%PROJ%"
set "BLDDIR=%ROOT%\%PROJ%\build_clangd"
echo.
echo === %PROJ% ===
if exist "%BLDDIR%" rmdir /s /q "%BLDDIR%"
"%CMAKE%" -S "%SRCDIR%" -B "%BLDDIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER=cl "-DCMAKE_MAKE_PROGRAM=%NINJA%" %EXTRA% >nul 2>&1
if errorlevel 1 (
    echo   FAILED: CMake configure error
    goto :eof
)
copy /y "%BLDDIR%\compile_commands.json" "%SRCDIR%\compile_commands.json" >nul
echo   OK: compile_commands.json generated
goto :eof

:start

call :gen_cc "practice_018a_cmake_modern_build"
call :gen_cc "practice_018b_cmake_advanced"
call :gen_cc "practice_020a_hft_lowlatency_cpp"
call :gen_cc "practice_020b_hft_systems"
call :gen_cc "practice_022_concurrent_queues"
call :gen_cc "practice_013_financial_cpp" "-DCMAKE_TOOLCHAIN_FILE=%VCPKG%"
call :gen_cc "practice_019a_cuda_fundamentals"
call :gen_cc "practice_019b_cuda_hpc_hft"

echo.
echo === Done ===
