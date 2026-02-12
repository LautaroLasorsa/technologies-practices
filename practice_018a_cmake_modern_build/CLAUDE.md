# Practice 018a: Modern CMake — Target-Based Builds & Dependency Management

## Technologies

- **CMake 3.16+** — Modern target-based build system (cmake_minimum_required, project, add_executable, add_library, target_link_libraries, FetchContent, install, configure_file, generator expressions)
- **nlohmann/json** — Header-only JSON library (fetched via FetchContent)
- **fmt** — Fast formatting library (fetched via FetchContent)
- **MSVC 19.39+** — Visual Studio 2022 C++17 compiler

## Stack

- C++17
- CMake 3.16+ (bundled with VS 2022)
- nlohmann/json + fmt (fetched via FetchContent)

## Theoretical Context

### What CMake Is and What Problem It Solves

CMake is a cross-platform build system generator. Unlike traditional build systems (Make, Ninja, MSBuild), CMake doesn't build your project directly—it generates build files for other build systems. You write platform-agnostic `CMakeLists.txt` files describing your targets and dependencies, and CMake generates Visual Studio solutions on Windows, Makefiles on Linux, or Xcode projects on macOS. This solves the "works on my machine" problem: one set of build instructions that works across all platforms and toolchains.

Modern CMake (3.0+) introduced the **target-based paradigm**, a fundamental shift from variable-based configuration. Pre-3.0 CMake relied on global variables (`include_directories()`, `link_directories()`, `add_definitions()`) that polluted the entire project. Modern CMake attaches properties (includes, flags, definitions) directly to targets via `target_*` commands, enabling precise dependency propagation and avoiding mysterious "why is this flag set?" questions.

### How CMake Works Internally

CMake operates in two phases: **configuration** and **generation**. During configuration (`cmake -S . -B build`), CMake executes your `CMakeLists.txt` scripts top-to-bottom, resolving variables, downloading FetchContent dependencies, running find_package searches, and building an internal representation of all targets and their properties. During generation, it translates this representation into native build files (`.vcxproj`, `Makefile`, `build.ninja`). This two-phase model is why FetchContent happens at configure time (dependencies must be known before generation) while ExternalProject_Add happens at build time (for non-CMake projects that don't participate in the dependency graph).

The target-based model relies on **transitive property propagation**. When you write `target_link_libraries(app PRIVATE lib)` where `lib` has `PUBLIC` include directories, CMake automatically propagates lib's includes to app. This is implemented via generator expressions (`$<BUILD_INTERFACE:...>`, `$<TARGET_PROPERTY:...>`) that are evaluated during generation, not configuration—allowing context-sensitive properties (e.g., different includes for build vs install).

CMake's **export system** enables libraries to be consumable by other CMake projects. `install(EXPORT ...)` generates `*-targets.cmake` files describing imported targets with all their usage requirements (includes, flags, dependencies). When another project calls `find_package(YourLib)`, CMake loads these files and reconstructs the targets as `IMPORTED` targets, making the library look like a built-in CMake target.

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Target** | A buildable entity (executable, library, custom target). The fundamental unit in modern CMake—all properties attach to targets. |
| **Generator** | Backend build system (Visual Studio, Ninja, Unix Makefiles). CMake generates files for the selected generator. |
| **Generator Expression** | `$<...>` syntax evaluated at generation time (not configuration). Enables conditional properties (e.g., `$<CONFIG:Debug>`, `$<BUILD_INTERFACE:include>`). |
| **FetchContent** | Configure-time dependency fetching from Git/URL. Downloads and adds external CMake projects to your build tree via `add_subdirectory`. |
| **PUBLIC/PRIVATE/INTERFACE** | Scope keywords for `target_link_libraries` and `target_*` commands. PUBLIC = target + consumers, PRIVATE = target only, INTERFACE = consumers only. |
| **Install/Export** | Mechanism to make your library consumable by others. Install copies artifacts; export generates `find_package`-compatible config files. |
| **configure_file** | Template substitution at configure time. Replaces `@VAR@` placeholders with CMake variables, typically for version headers. |
| **compile_commands.json** | JSON database of all compilation commands (file → compiler flags). Used by clangd, LSP servers, and static analyzers. |

### Ecosystem Context

**CMake vs alternatives**: CMake dominates C++ (90%+ of open-source projects), but alternatives exist. **Meson** offers faster configuration and simpler syntax but has a smaller ecosystem. **Bazel** excels at monorepo builds with hermetic caching but has steep learning curves. **xmake** is Lua-based and modern but lacks CMake's maturity. For production C++, CMake is the de facto standard—understanding it is non-negotiable.

**Dependency management**: CMake's built-in FetchContent works for CMake projects but lacks version resolution (no semver, no lockfiles). **vcpkg** (Microsoft's package manager) integrates via `CMAKE_TOOLCHAIN_FILE` and provides binary caching + thousands of pre-built packages, making it ideal for large projects. **Conan** is Python-based with better version resolution but requires separate configuration. For this practice, FetchContent teaches the core concepts without external tooling.

## Description

Build a **multi-target CMake project from scratch** that demonstrates every core Modern CMake concept: target-based dependency management, FetchContent for pulling GitHub libraries, library targets with proper visibility scopes, configure_file for version injection, install/export rules, and compile_commands.json generation for editor tooling.

This is NOT about writing complex C++ — the C++ code is intentionally simple. The **CMakeLists.txt IS the learning material**. Every section teaches a CMake concept, and the C++ files exist only to exercise that concept.

### What you'll learn

1. **Target-based CMake** — Why `target_link_libraries(PRIVATE)` replaced `include_directories()` / `link_directories()`
2. **FetchContent** — Pull libraries from GitHub with pinned tags, understand FetchContent vs find_package vs ExternalProject
3. **Multi-target projects** — Library targets, executable targets, test targets with proper PUBLIC/PRIVATE/INTERFACE scopes
4. **configure_file** — Generate headers at configure time with project version, options, git hash
5. **Install & export** — Make your library consumable by other CMake projects via find_package
6. **compile_commands.json** — IDE/clangd integration via CMAKE_EXPORT_COMPILE_COMMANDS

## Instructions

### Prerequisites

- CMake 3.16+ (bundled with VS 2022, invoked via `build.bat`)
- MSVC 19.39+ (VS 2022)
- Git (for FetchContent to clone dependencies)
- Internet connection (first build only)

### Phase 1: CMake Fundamentals (~15 min)

1. **Read** the root `CMakeLists.txt` top section — understand `cmake_minimum_required`, `project()`, C++ standard settings
2. **Concepts:** `add_executable`, `target_link_libraries`, PUBLIC/PRIVATE/INTERFACE link scopes
3. **Key insight:** Modern CMake is "target-based" — every property (includes, flags, definitions) attaches to a target, NOT globally. No `include_directories()`, no `link_directories()`, no `add_definitions()`.
4. **User implements:** `src/main.cpp` — read the generated config header and print version info
5. **Question:** What's the difference between `PUBLIC`, `PRIVATE`, and `INTERFACE` in `target_link_libraries`? When would you use each?

### Phase 2: FetchContent (~20 min)

1. **Read** `cmake/FetchDeps.cmake` — understand FetchContent_Declare/FetchContent_MakeAvailable
2. **Concepts:** GIT_TAG pinning, GIT_SHALLOW, why FetchContent > manual submodules
3. **User implements:** `apps/json_demo.cpp` — parse JSON and format output using nlohmann/json + fmt
4. **Comparison:** FetchContent (configure-time, in-tree) vs find_package (pre-installed) vs ExternalProject_Add (build-time, out-of-tree)
5. **Question:** What happens if two FetchContent dependencies both pull the same transitive dependency at different versions?

### Phase 3: Multi-Target Project (~20 min)

1. **Read** the library section of `CMakeLists.txt` — `add_library`, `target_include_directories`, `target_compile_features`
2. **Concepts:** STATIC vs SHARED vs INTERFACE libraries, generator expressions (`$<BUILD_INTERFACE:...>`, `$<INSTALL_INTERFACE:...>`)
3. **User implements:** `src/greeter.h` + `src/greeter.cpp` — a simple Greeter class
4. **User implements:** `tests/test_greeter.cpp` — basic assert-based tests
5. **Question:** Why do we need `$<BUILD_INTERFACE:...>` vs `$<INSTALL_INTERFACE:...>` for include directories?

### Phase 4: Options, configure_file & Compile Definitions (~15 min)

1. **Read** the configure_file section of `CMakeLists.txt` and `src/config.h.in`
2. **Concepts:** `option()`, `configure_file()`, `@VAR@` substitution, `target_compile_definitions`
3. **User implements:** extend `src/main.cpp` to use the generated config header values
4. **Question:** When would you use `configure_file` vs `target_compile_definitions`? What are the tradeoffs?

### Phase 5: Install Rules & Export (~15 min)

1. **Read** the install section of `CMakeLists.txt`
2. **Concepts:** `install(TARGETS)`, `install(EXPORT)`, `install(FILES)`, how `find_package` discovers installed packages
3. **Observe:** After `cmake --install`, the library is consumable by other projects
4. **Question:** What is the difference between `install(TARGETS ... EXPORT ...)` and `install(EXPORT ...)`?

### Phase 6: compile_commands.json & Tooling (~5 min)

1. **Concepts:** `CMAKE_EXPORT_COMPILE_COMMANDS`, what compile_commands.json contains, clangd integration
2. **Note:** MSVC generators don't natively produce compile_commands.json — use Ninja generator or a cmake-file-api workaround
3. **Try:** Configure with `-G Ninja` if Ninja is available, or observe the CMake file API under `build/.cmake/api/`

### Build & Run

```bat
build.bat              REM Build all targets (Release)
build.bat debug        REM Build all targets (Debug)
build.bat clean        REM Remove build directory

REM Run individual targets after building:
build\Release\main_app.exe
build\Release\json_demo.exe
build\Release\test_greeter.exe
```

## Motivation

- **CMake is THE build system for production C++.** Understanding it deeply means you can set up any C++ project from scratch, pull dependencies cleanly, and structure multi-target builds — skills assumed but rarely taught.
- **FetchContent mastery** eliminates dependency headaches. No more manual submodule management, vcpkg mismatches, or "works on my machine" issues.
- **Install/export knowledge** is what separates "I can build my project" from "I can ship a library that others consume via find_package."
- **Direct follow-up to 012a** — you saw FetchContent pull abseil automatically and wanted to understand how it actually works. This practice answers that.

## References

- [Modern CMake — An Introduction (Henry Schreiner)](https://cliutils.gitlab.io/modern-cmake/)
- [Professional CMake: A Practical Guide (Craig Scott)](https://crascit.com/professional-cmake/)
- [CMake Official Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [FetchContent Module Docs](https://cmake.org/cmake/help/latest/module/FetchContent.html)
- [target_link_libraries Docs](https://cmake.org/cmake/help/latest/command/target_link_libraries.html)
- [configure_file Docs](https://cmake.org/cmake/help/latest/command/configure_file.html)
- [install() Docs](https://cmake.org/cmake/help/latest/command/install.html)
- [Generator Expressions Docs](https://cmake.org/cmake/help/latest/manual/cmake-generator-expressions.7.html)
- [nlohmann/json GitHub](https://github.com/nlohmann/json)
- [fmt GitHub](https://github.com/fmtlib/fmt)

## Commands

All commands are run from `practice_018a_cmake_modern_build/`. The cmake path assumes VS 2022 bundled CMake.

### Configure & Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Release 2>&1"` | Configure the project (Release). First run fetches nlohmann/json + fmt via FetchContent. |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Debug 2>&1"` | Configure the project (Debug) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_GREETING_EMOJI=ON 2>&1"` | Configure with emoji option enabled (Phase 4) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release 2>&1"` | Build all targets (Release) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Debug 2>&1"` | Build all targets (Debug) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target main_app 2>&1"` | Build only the main_app target |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target json_demo 2>&1"` | Build only the json_demo target |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target test_greeter 2>&1"` | Build only the test_greeter target |

### Run Executables

| Command | Description |
|---------|-------------|
| `.\build\Release\main_app.exe` | Run main app — prints version info from config.h + uses greeter library (Phase 1+3+4) |
| `.\build\Release\json_demo.exe` | Run JSON demo — parses JSON with nlohmann/json, formats with fmt (Phase 2) |
| `.\build\Release\test_greeter.exe` | Run greeter tests directly (Phase 3) |

### CTest

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "ctest --test-dir build -C Release 2>&1"` | Run all registered CTest tests (Release) |
| `powershell.exe -Command "ctest --test-dir build -C Debug 2>&1"` | Run all registered CTest tests (Debug) |

### Install & Export (Phase 5)

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --install build --config Release --prefix _install 2>&1"` | Install greeter library + headers to `_install/` for consumption by other projects |

### Cleanup

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "if (Test-Path build) { Remove-Item -Recurse -Force build }"` | Remove build directory |

### Helper Script (alternative)

| Command | Description |
|---------|-------------|
| `.\build.bat` | Build all targets (Release) |
| `.\build.bat debug` | Build all targets (Debug) |
| `.\build.bat test` | Build + run tests |
| `.\build.bat clean` | Remove build directory |

## State

`not-started`
