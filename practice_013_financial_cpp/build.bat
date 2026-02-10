@echo off
REM Build script for Practice 013: Financial C++ (QuickFIX & QuantLib)
REM
REM Prerequisites:
REM   1. vcpkg installed and integrated
REM   2. vcpkg install quickfix:x64-windows quantlib:x64-windows
REM   3. VCPKG_ROOT environment variable set (or pass -DCMAKE_TOOLCHAIN_FILE manually)
REM
REM Usage:
REM   build.bat             -- configure + build (Release)
REM   build.bat configure   -- only configure
REM   build.bat build       -- only build
REM   build.bat clean       -- remove build directory

setlocal

set PROJECT_DIR=%~dp0
set BUILD_DIR=%PROJECT_DIR%build

if "%VCPKG_ROOT%"=="" (
    echo [ERROR] VCPKG_ROOT environment variable is not set.
    echo Set it to your vcpkg installation directory, e.g.:
    echo   set VCPKG_ROOT=C:\src\vcpkg
    exit /b 1
)

set TOOLCHAIN=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake

if "%1"=="clean" (
    echo Cleaning build directory...
    if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
    echo Done.
    exit /b 0
)

if "%1"=="build" goto :build
if "%1"=="configure" goto :configure

:configure
echo === Configuring with CMake ===
cmake -B "%BUILD_DIR%" -S "%PROJECT_DIR%" -DCMAKE_TOOLCHAIN_FILE="%TOOLCHAIN%"
if errorlevel 1 (
    echo [ERROR] CMake configuration failed.
    exit /b 1
)
if "%1"=="configure" exit /b 0

:build
echo === Building (Release) ===
cmake --build "%BUILD_DIR%" --config Release
if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo === Build successful ===
echo Executables:
echo   %BUILD_DIR%\Release\fix_acceptor.exe
echo   %BUILD_DIR%\Release\fix_initiator.exe
echo   %BUILD_DIR%\Release\option_pricing.exe
echo   %BUILD_DIR%\Release\bond_pricing.exe
echo.
echo To run the FIX demo (run from project root directory):
echo   1. Start the acceptor:  build\Release\fix_acceptor.exe
echo   2. Start the initiator: build\Release\fix_initiator.exe
echo      (in a separate terminal)
echo.
echo To run QuantLib demos (from project root directory):
echo   build\Release\option_pricing.exe
echo   build\Release\bond_pricing.exe

endlocal
