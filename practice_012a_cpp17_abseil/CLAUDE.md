# Practice 012a: C++17 Features & abseil-cpp

## Technologies

- **C++17** -- Structured bindings, std::optional, std::variant, std::visit, std::string_view, if-constexpr, fold expressions, CTAD, init-statements in if/switch
- **abseil-cpp** -- Google's open-source C++ library: absl::flat_hash_map, absl::StatusOr, absl::StrCat/StrJoin/StrSplit, absl::Span, absl::Time
- **CMake 3.16+** -- Build system with FetchContent for dependency management

## Stack

- C++17
- abseil-cpp (fetched via CMake FetchContent)

## Description

Build a **mini log analytics engine**: parse structured log lines, extract fields, store them in fast containers, query/filter/aggregate, and handle errors gracefully. This teaches modern C++17 features you rarely use in competitive programming alongside abseil utilities that are standard in production C++ (Google, Chromium, gRPC).

### What you'll learn

1. **Structured bindings** -- Destructure maps, pairs, tuples, and custom structs
2. **std::optional & std::variant** -- Type-safe nullable values and sum types (like Rust's Option/enum)
3. **std::string_view** -- Zero-copy string references (like Rust's &str)
4. **if-constexpr & fold expressions** -- Compile-time branching and variadic template magic
5. **abseil string utilities** -- absl::StrSplit, absl::StrJoin, absl::StrCat for ergonomic string processing
6. **absl::flat_hash_map** -- Swiss table: 2-3x faster than std::unordered_map
7. **absl::StatusOr** -- Error handling without exceptions (like Rust's Result<T, E>)

## Instructions

### Phase 1: Structured Bindings & Modern Control Flow (~15 min)

1. **Concepts:** Structured bindings (auto [a, b] = ...), init-statements in if/switch
2. **User implements:** Parse a log line "2024-03-15T10:30:00 ERROR auth-service Connection refused" into a LogEntry struct using structured bindings
3. **User implements:** A function that uses if-with-init to look up a key in a map and handle presence/absence in one statement
4. Key question: How do structured bindings differ from std::tie? When would you prefer one over the other?

### Phase 2: std::optional, std::variant & std::visit (~20 min)

1. **Concepts:** std::optional as a nullable value, std::variant as a tagged union, std::visit for pattern matching
2. **User implements:** A function returning std::optional<LogEntry> that parses a log line or returns std::nullopt on malformed input
3. **User implements:** A LogValue variant (int, double, std::string) and a visitor that formats each type differently for display
4. Key question: How does std::visit compare to if-else chains with std::holds_alternative? What about the overloaded lambda pattern?

### Phase 3: Abseil String Processing (~20 min)

1. **Concepts:** absl::StrSplit, absl::StrJoin, absl::StrCat, and how they interplay with std::string_view
2. **User implements:** Parse CSV-like log metadata using absl::StrSplit with different delimiters
3. **User implements:** Build a formatted summary report using absl::StrJoin with a custom formatter
4. **User implements:** Efficient string building with absl::StrCat (vs. repeated operator+)
5. Key question: Why does absl::StrCat outperform std::string concatenation chains?

### Phase 4: Abseil Containers & Hashing (~15 min)

1. **Concepts:** absl::flat_hash_map (Swiss table), absl::Hash, why it beats std::unordered_map
2. **User implements:** Store LogEntry objects in an absl::flat_hash_map keyed by timestamp
3. **User implements:** Aggregate log counts by severity level, comparing flat_hash_map vs unordered_map performance
4. Key question: What is the Swiss table layout and why does it have better cache behavior?

### Phase 5: Error Handling with absl::StatusOr (~15 min)

1. **Concepts:** absl::Status, absl::StatusOr<T>, canonical error codes (like gRPC status codes)
2. **User implements:** A file-loading function that returns absl::StatusOr<std::vector<LogEntry>>
3. **User implements:** Chain multiple StatusOr-returning functions with early returns
4. Key question: How does absl::StatusOr compare to exceptions? To std::expected (C++23)?

### Phase 6: if-constexpr & Fold Expressions (~15 min)

1. **Concepts:** Compile-time branching, parameter pack expansion, fold expressions
2. **User implements:** A generic print_all(args...) using fold expressions
3. **User implements:** A type-aware serializer using if-constexpr to handle int/double/string differently at compile time
4. Key question: What code would the compiler generate without if-constexpr? Why would it fail?

## Motivation

- **Production C++ gap**: Competitive programming uses C++17 syntax minimally (mostly auto, structured bindings). Production code at Google, Meta, Bloomberg heavily uses abseil, std::optional, std::variant, and StatusOr patterns.
- **Industry standard**: abseil-cpp underpins gRPC, Protobuf, TensorFlow, and Chromium -- understanding it signals production readiness.
- **Rust parallel**: Many C++17 + abseil patterns (StatusOr, variant, string_view, flat_hash_map) mirror Rust idioms you already know (Result, enum, &str, HashMap). This session makes the bridge explicit.
- **Error handling maturity**: Moving from exception-based to value-based error handling (StatusOr) is a key professional skill.

## References

- [C++17 Features - cppreference](https://en.cppreference.com/w/cpp/17.html)
- [Modern C++ Features Cheatsheet](https://github.com/AnthonyCalandra/modern-cpp-features)
- [Abseil C++ Official](https://abseil.io/)
- [Abseil Strings Guide](https://abseil.io/docs/cpp/guides/strings)
- [Abseil Status Guide](https://abseil.io/docs/cpp/guides/status)
- [Abseil Containers Guide](https://abseil.io/docs/cpp/guides/container)
- [Abseil CMake Quickstart](https://abseil.io/docs/cpp/quickstart-cmake.html)

## Commands

All commands are run from the `practice_012a_cpp17_abseil/` folder root. The cmake binary on this machine is at `C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`.

### Configure

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build 2>&1"` | Configure the project (fetches abseil-cpp via FetchContent on first run) |

### Build

| Command | Description |
|---------|-------------|
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target all_phases 2>&1"` | Build all six phase executables at once |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target practice_info 2>&1"` | Build the info target only (src/main.cpp) |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase1_bindings 2>&1"` | Build Phase 1: Structured bindings & modern control flow |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase2_optional_variant 2>&1"` | Build Phase 2: std::optional, std::variant, std::visit |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase3_strings 2>&1"` | Build Phase 3: Abseil string processing |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase4_containers 2>&1"` | Build Phase 4: Abseil containers & hashing |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase5_errors 2>&1"` | Build Phase 5: Error handling with absl::StatusOr |
| `powershell.exe -Command "& 'C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target phase6_constexpr 2>&1"` | Build Phase 6: if-constexpr & fold expressions |

### Run

| Command | Description |
|---------|-------------|
| `build\Release\practice_info.exe` | Run the info/overview target |
| `build\Release\phase1_bindings.exe` | Run Phase 1: Structured bindings & modern control flow |
| `build\Release\phase2_optional_variant.exe` | Run Phase 2: std::optional, std::variant, std::visit |
| `build\Release\phase3_strings.exe` | Run Phase 3: Abseil string processing |
| `build\Release\phase4_containers.exe` | Run Phase 4: Abseil containers & hashing |
| `build\Release\phase5_errors.exe` | Run Phase 5: Error handling with absl::StatusOr |
| `build\Release\phase6_constexpr.exe` | Run Phase 6: if-constexpr & fold expressions |

## State

`not-started`
