# Practice 018b: Advanced CMake — Toolchains, Presets & Cross-Compilation

## Technologies

- **CMake 3.21+** — Presets (CMakePresets.json), toolchain files, CTest, CPack, custom commands, Find modules
- **MSVC 19.39+** — Visual Studio 2022 C++17 compiler
- **MinGW GCC** — Cross-compilation target from VS dev environment
- **CTest** — CMake's test runner with labels, fixtures, and preset integration
- **CPack** — CMake's packaging tool (ZIP, NSIS)

## Stack

- C++17
- CMake 3.21+ (bundled with VS 2022)
- MSVC + MinGW GCC toolchains

## Theoretical Context

### What CMake Presets, Toolchains, and Advanced Features Solve

CMake 3.21+ introduced **CMakePresets.json**, a standardized way to define build configurations. Before presets, teams documented long cmake commands in wikis (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=...`), leading to "works on my machine" issues. Presets replace these with version-controlled JSON files that define configure, build, and test workflows. This solves onboarding friction, CI/CD complexity, and cross-platform reproducibility.

**Toolchain files** (`CMAKE_TOOLCHAIN_FILE`) specify the compiler, linker, sysroot, and search paths for cross-compilation. They're loaded before `project()`, allowing CMake to set up the compilation environment before detecting features or building targets. Toolchains are essential for embedded systems (ARM bare-metal), mobile (Android NDK), and multi-compiler CI matrices (MSVC + GCC + Clang).

**Custom find modules** (`FindXxx.cmake`) fill the gap when third-party libraries don't ship CMake config files. While modern libraries generate `XxxConfig.cmake` via `install(EXPORT ...)`, older libraries (pre-CMake era) or system libraries (OpenSSL, Boost pre-1.70) require custom find modules. Writing these modules teaches how `find_package()` works under the hood.

**CTest** (CMake's test driver) and **CPack** (CMake's packaging tool) complete the build-test-package lifecycle. CTest runs tests with labels, fixtures, and timeouts, integrating with CI dashboards. CPack generates installers (ZIP, NSIS, DEB, RPM) from install rules, ensuring what you build can be deployed.

### How These Systems Work Internally

**CMake Presets** use a two-layer architecture: `CMakePresets.json` (checked into version control, team-wide defaults) and `CMakeUserPresets.json` (local overrides, gitignored). Presets support **inheritance** (`"inherits": ["base"]`) to avoid duplication, and **condition expressions** to enable platform-specific configurations. When you run `cmake --preset=msvc-release`, CMake loads the preset, resolves inherited fields, evaluates conditions, and then invokes the equivalent `cmake -S ... -B ... -G ... -D...` command. Build presets (`cmake --build --preset=...`) and test presets (`ctest --preset=...`) layer on top, referencing configure presets via `configurePreset`.

**Toolchain files** are executed at the start of configuration, before compiler detection. They set variables like `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER`, `CMAKE_SYSTEM_NAME`, and `CMAKE_FIND_ROOT_PATH`. Setting `CMAKE_SYSTEM_NAME` to a value different from the host OS triggers **cross-compiling mode** (`CMAKE_CROSSCOMPILING=TRUE`), which alters how CMake searches for libraries (only in sysroot, not host paths) and how it handles try_compile tests (compiles for target, runs on host via emulator or skips).

**Custom find modules** follow a four-step pattern: (1) `find_path()` locates headers, (2) `find_library()` locates compiled libraries, (3) `find_package_handle_standard_args()` validates results and sets `XXX_FOUND`, (4) `add_library(... IMPORTED)` creates a target wrapping the found artifacts. The IMPORTED target mimics a regular CMake target, allowing `target_link_libraries(app PRIVATE XXX::XXX)` to propagate includes and linkage automatically.

**CTest** registers tests via `add_test(NAME ... COMMAND ...)` and stores them in `CTestTestfile.cmake`. When you run `ctest`, it parses these files, applies filters (`-L label`, `-R regex`), executes tests in parallel (with dependency ordering via fixtures), captures output, and reports pass/fail. CTest integrates with CDash (CMake's test dashboard) for tracking test history across CI runs.

**CPack** reads `install(TARGETS ...)` and `install(FILES ...)` rules, along with `CPACK_*` variables, to generate platform-specific installers. The `include(CPack)` command at the end of `CMakeLists.txt` creates a `package` target. Running `cmake --build build --target package` invokes CPack, which stages installed files into a temporary directory, then compresses (ZIP), creates an installer (NSIS/WiX on Windows), or builds a package (DEB/RPM on Linux).

### Key Concepts

| Concept | Description |
|---------|-------------|
| **CMakePresets.json** | Version-controlled build configuration file. Defines configure/build/test presets with inheritance, conditions, and cacheVariables. |
| **Toolchain File** | CMake script loaded before `project()` to set compiler, sysroot, and cross-compilation settings. Enables multi-compiler builds and cross-compilation. |
| **Cross-Compiling Mode** | Activated when `CMAKE_SYSTEM_NAME` differs from host OS. Changes library search paths, disables host-executable execution in try_compile. |
| **Find Module** | CMake script (`FindXxx.cmake`) that locates third-party libraries without CMake config. Implements `find_path`, `find_library`, creates IMPORTED target. |
| **IMPORTED Target** | CMake target representing a pre-built library. Created by find modules or `find_package` config mode. Used with `target_link_libraries` like regular targets. |
| **CTest** | CMake's test runner. Executes tests registered via `add_test`, supports labels, fixtures, parallel execution, and CDash integration. |
| **Test Fixtures** | CTest feature for setup/teardown dependencies. Tests marked with `FIXTURES_SETUP` run before, `FIXTURES_CLEANUP` run after fixture-dependent tests. |
| **CPack** | CMake's packaging tool. Generates installers (ZIP, NSIS, DEB, RPM) from install rules and `CPACK_*` variables. Invoked via `package` target. |
| **Generator Expression** | `$<...>` syntax evaluated at generation time. Used in toolchains for conditional flags (e.g., `$<$<CONFIG:Debug>:-g>`). |

### Ecosystem Context

**Presets vs build scripts**: Before presets, teams wrote shell scripts (`build.sh`, `build.bat`) wrapping cmake commands. Presets embed this logic in JSON, making it cross-platform and IDE-integrated (VS Code, CLion, Visual Studio all parse `CMakePresets.json`). However, presets require CMake 3.21+, so older projects still use scripts.

**Toolchains vs Docker**: Cross-compilation toolchains (e.g., Windows → Linux ARM) require target sysroots and cross-compilers. Docker simplifies this by providing pre-configured environments, but toolchains are more lightweight (no container overhead). For embedded (Yocto, Buildroot), toolchains are standard; for Linux → Windows, Wine + MinGW toolchains are common.

**Find modules vs vcpkg/Conan**: Modern package managers (vcpkg, Conan) generate CMake config files, eliminating the need for custom find modules. However, system libraries (Threads, OpenGL) and legacy libraries (old Boost versions) still require find modules. Understanding them is essential for integrating non-CMake libraries.

**CTest vs gtest/pytest**: CTest is a test *runner*, not a framework. You write tests using Google Test, Catch2, or plain asserts, then register them with `add_test()`. CTest handles discovery, parallel execution, and reporting, while the framework provides assertions and fixtures. For pure CMake projects (no external frameworks), CTest suffices.

**CPack vs platform installers**: CPack generates installers but lacks advanced UI customization (e.g., WiX's complex dialog flows). For production Windows installers, teams often use CPack for quick internal builds and switch to WiX/InstallShield for customer-facing releases. For Linux, CPack's DEB/RPM generators are production-ready.

## Description

Go beyond `add_executable` and `target_link_libraries` into the **build system configuration layer** of CMake. This session covers the tools that make C++ projects portable, reproducible, and CI/CD-ready: presets that replace long command lines, toolchain files that enable cross-compilation, custom find modules for dependency management, CTest for structured testing, custom commands for code generation, and CPack for packaging.

This is a direct follow-up to 018a (targets, FetchContent, install/export). The C++ code is intentionally simple — a small math library — because the **CMake files ARE the learning material**.

### What you'll learn

1. **CMake Presets** — Replace `cmake -S . -B build -G "..." -DCMAKE_BUILD_TYPE=... -DCMAKE_TOOLCHAIN_FILE=...` with `cmake --preset=msvc-release`
2. **Toolchain files** — How to target MinGW GCC from the same project that builds with MSVC
3. **Custom Find modules** — Write `FindMyMath.cmake`, understand Find vs Config packages
4. **CTest** — Labels, fixtures, timeouts, `ctest --preset=...`
5. **Custom commands & code generation** — Generate C++ source at build time from data files
6. **CPack** — Create distributable ZIP/NSIS packages

## Instructions

### Prerequisites

- CMake 3.21+ (bundled with VS 2022)
- MSVC 19.39+ (VS 2022)
- MinGW GCC at `C:\MinGW\bin` (for Phase 2 toolchain)
- Git

### Phase 1: CMake Presets (~15 min)

1. **Read** `CMakePresets.json` — understand the structure: `configurePresets`, `buildPresets`, `testPresets`
2. **Concepts:** Preset inheritance, generator selection, cacheVariables, condition expressions
3. **Try:** `cmake --list-presets` to see available presets, then `cmake --preset=msvc-debug` to configure
4. **Key insight:** Presets standardize the configure/build/test workflow — eliminates "works on my machine" and simplifies CI/CD scripts
5. **Question:** Why does `CMakePresets.json` go into version control but `CMakeUserPresets.json` does not?

Docs: [cmake-presets(7)](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)

### Phase 2: Toolchain Files (~20 min)

1. **Read** `cmake/toolchain-mingw.cmake` — understand each variable
2. **TODO(human):** Fill in `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER` paths for MinGW
3. **Concepts:** `CMAKE_SYSTEM_NAME`, `CMAKE_FIND_ROOT_PATH`, `CMAKE_FIND_ROOT_PATH_MODE_*`
4. **Try:** `cmake --preset=mingw-release` to configure with the MinGW toolchain, then build
5. **Compare:** Build the same project with MSVC and MinGW — observe different output directories
6. **Question:** What happens if `CMAKE_SYSTEM_NAME` is set to "Linux" when cross-compiling on Windows?

Docs: [cmake-toolchains(7)](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)

### Phase 3: Custom Find Modules & Config Packages (~20 min)

1. **Read** `cmake/FindMyMath.cmake` — understand the canonical Find module pattern
2. **TODO(human):** Complete the `find_path`, `find_library`, and imported target creation
3. **Concepts:** `find_path`, `find_library`, `find_package_handle_standard_args`, `IMPORTED` targets
4. **Key insight:** Find modules (`FindXxx.cmake`) are for third-party libs that don't ship CMake config. Config packages (`XxxConfig.cmake`) are generated by `install(EXPORT ...)` — prefer Config when available.
5. **Try:** Install the mymath library, then consume it from a hypothetical external project via `find_package(MyMath)`
6. **Question:** In what order does `find_package(Foo)` search for `FooConfig.cmake` vs `FindFoo.cmake`?

Docs: [find_package](https://cmake.org/cmake/help/latest/command/find_package.html), [cmake-developer: Find Modules](https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules)

### Phase 4: Testing with CTest (~15 min)

1. **Read** `tests/CMakeLists.txt` — understand `add_test`, test properties, labels, fixtures
2. **TODO(human):** Write test cases in `test_mymath.cpp` for add, multiply, factorial, fibonacci
3. **Concepts:** `enable_testing()`, `add_test()`, `set_tests_properties()`, LABELS, TIMEOUT, WILL_FAIL, FIXTURES_SETUP/CLEANUP
4. **Try:** `ctest --preset=msvc-debug`, `ctest -L unit`, `ctest -L edge_case`
5. **Question:** How do CTest fixtures differ from test framework fixtures (e.g., Google Test SetUp/TearDown)?

Docs: [ctest(1)](https://cmake.org/cmake/help/latest/manual/ctest.1.html), [add_test](https://cmake.org/cmake/help/latest/command/add_test.html), [set_tests_properties](https://cmake.org/cmake/help/latest/command/set_tests_properties.html)

### Phase 5: Custom Commands & Code Generation (~15 min)

1. **Read** `codegen/CMakeLists.txt` and `codegen/embed_file.cmake`
2. **Concepts:** `add_custom_command(OUTPUT ...)`, `add_custom_target`, dependency tracking between generated and compiled files
3. **TODO(human):** Complete `codegen_demo.cpp` to use the generated embedded string
4. **Key insight:** `add_custom_command` output files become dependencies — CMake regenerates them when inputs change
5. **Question:** When would you use `add_custom_command` vs `add_custom_target`? Can a custom target have OUTPUT?

Docs: [add_custom_command](https://cmake.org/cmake/help/latest/command/add_custom_command.html), [add_custom_target](https://cmake.org/cmake/help/latest/command/add_custom_target.html)

### Phase 6: CPack & Packaging (~10 min)

1. **Read** the CPack section at the bottom of `CMakeLists.txt`
2. **Concepts:** `include(CPack)`, `CPACK_*` variables, generator selection (ZIP, NSIS)
3. **Try:** `cmake --build --preset=msvc-release --target package` or `cpack --config build/msvc-release/CPackConfig.cmake`
4. **Observe:** The generated ZIP in the build directory containing binaries + headers
5. **Question:** How does CPack know which files to include? What's the relationship between `install()` rules and CPack?

Docs: [cpack(1)](https://cmake.org/cmake/help/latest/manual/cpack.1.html), [CPack module](https://cmake.org/cmake/help/latest/module/CPack.html)

### Build & Run

```bat
REM With presets (preferred):
cmake --preset=msvc-debug
cmake --build --preset=msvc-debug
ctest --preset=msvc-debug

REM Or use the helper script:
build.bat                   REM MSVC Release (default)
build.bat debug             REM MSVC Debug
build.bat mingw             REM MinGW Release
build.bat test              REM Build + run tests
build.bat package           REM Build + create package
build.bat clean             REM Remove all build directories
```

## Motivation

- **Presets** are how modern C++ teams standardize builds across developers and CI — replacing wiki pages of cmake flags with a single JSON file checked into version control.
- **Toolchain files** are the gateway to cross-compilation, embedded systems, and multi-compiler CI matrices. Understanding them is essential for any C++ role beyond single-platform desktop apps.
- **Find modules** and the `find_package` ecosystem are how large C++ codebases (100+ dependencies) manage linking. Knowing how to write and debug them saves hours of "undefined reference" frustration.
- **CTest + CPack** close the build-test-package loop that CI/CD pipelines require.

These are the skills that separate "I can build my project" from "I can ship a portable, reproducible C++ build system."

## References

- [cmake-presets(7)](https://cmake.org/cmake/help/latest/manual/cmake-presets.7.html)
- [cmake-toolchains(7)](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)
- [cmake-developer(7) — Find Modules](https://cmake.org/cmake/help/latest/manual/cmake-developer.7.html#find-modules)
- [find_package](https://cmake.org/cmake/help/latest/command/find_package.html)
- [add_test](https://cmake.org/cmake/help/latest/command/add_test.html)
- [add_custom_command](https://cmake.org/cmake/help/latest/command/add_custom_command.html)
- [cpack(1)](https://cmake.org/cmake/help/latest/manual/cpack.1.html)
- [Professional CMake: A Practical Guide (Craig Scott)](https://crascit.com/professional-cmake/)
- [Modern CMake — An Introduction (Henry Schreiner)](https://cliutils.gitlab.io/modern-cmake/)

## Commands

All commands are run from `practice_018b_cmake_advanced/`. This practice uses CMake presets defined in `CMakePresets.json`.

### List Available Presets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --list-presets 2>&1"` | List all available configure presets |

### Phase 1: Configure with Presets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --preset=msvc-debug 2>&1"` | Configure with MSVC Debug preset (builds to `build/msvc-debug/`) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --preset=msvc-release 2>&1"` | Configure with MSVC Release preset (builds to `build/msvc-release/`) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --preset=mingw-release 2>&1"` | Configure with MinGW toolchain preset (Phase 2, builds to `build/mingw-release/`) |

### Build with Presets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-debug 2>&1"` | Build all targets (MSVC Debug) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release 2>&1"` | Build all targets (MSVC Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=mingw-release 2>&1"` | Build all targets (MinGW Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release --target main_app 2>&1"` | Build only main_app (MSVC Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release --target codegen_demo 2>&1"` | Build only codegen_demo (MSVC Release, Phase 5) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release --target test_mymath 2>&1"` | Build only test_mymath (MSVC Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release --target test_advanced 2>&1"` | Build only test_advanced (MSVC Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build --preset=msvc-release --target package 2>&1"` | Build + create ZIP package (Phase 6, CPack) |

### Run Executables (MSVC)

| Command | Description |
|---------|-------------|
| `.\build\msvc-release\Release\main_app.exe` | Run main app — uses mymath library |
| `.\build\msvc-debug\Debug\main_app.exe` | Run main app (Debug build) |
| `.\build\msvc-release\codegen\Release\codegen_demo.exe` | Run codegen demo — prints embedded data from data.txt (Phase 5) |
| `.\build\msvc-debug\codegen\Debug\codegen_demo.exe` | Run codegen demo (Debug build) |

### Run Executables (MinGW)

| Command | Description |
|---------|-------------|
| `.\build\mingw-release\main_app.exe` | Run main app built with MinGW (Phase 2) |
| `.\build\mingw-release\codegen\codegen_demo.exe` | Run codegen demo built with MinGW |

### Phase 4: CTest with Presets

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "ctest --preset=msvc-debug 2>&1"` | Run all CTest tests (MSVC Debug) |
| `powershell.exe -Command "ctest --preset=msvc-release 2>&1"` | Run all CTest tests (MSVC Release) |
| `powershell.exe -Command "ctest --preset=mingw-release 2>&1"` | Run all CTest tests (MinGW Release) |
| `powershell.exe -Command "ctest --preset=msvc-debug -L unit 2>&1"` | Run only unit-labeled tests |
| `powershell.exe -Command "ctest --preset=msvc-debug -L edge_case 2>&1"` | Run only edge_case-labeled tests |
| `powershell.exe -Command "ctest --preset=msvc-debug -L fixture 2>&1"` | Run only fixture tests (setup/main/cleanup in order) |
| `powershell.exe -Command "ctest --preset=msvc-debug -L output 2>&1"` | Run only output-verification tests |
| `powershell.exe -Command "ctest --preset=msvc-debug -N 2>&1"` | List all tests without running them |

### Phase 6: CPack Packaging

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "cpack --config build/msvc-release/CPackConfig.cmake -G ZIP 2>&1"` | Create ZIP package from MSVC Release build |

### Install (Phase 3 / find_package consumption)

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --install build/msvc-release 2>&1"` | Install mymath library to `_install/msvc-release/` for find_package consumption |

### Cleanup

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "if (Test-Path build) { Remove-Item -Recurse -Force build }; if (Test-Path _install) { Remove-Item -Recurse -Force _install }"` | Remove all build and install directories |

### Helper Script (alternative)

| Command | Description |
|---------|-------------|
| `.\build.bat` | Build with MSVC Release (default) |
| `.\build.bat debug` | Build with MSVC Debug |
| `.\build.bat mingw` | Build with MinGW Release |
| `.\build.bat test` | Build MSVC Debug + run tests |
| `.\build.bat package` | Build MSVC Release + create ZIP package |
| `.\build.bat install` | Build MSVC Release + install to `_install/` |
| `.\build.bat clean` | Remove all build directories |

## State

`not-started`
